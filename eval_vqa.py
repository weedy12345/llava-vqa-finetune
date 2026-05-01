import os
import torch
import json
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
data_dir = "/root/autodl-tmp/vqa_data"
data = json.load(open(os.path.join(data_dir, "vqa_1k.json")))

print("加载模型...")
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
base_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, os.path.join(data_dir, "vqa_lora_weights"))
model.eval()

# 只测前 20 条
correct = 0
total = 20

for item in data[:total]:
    image = Image.open(item["image"])
    question = item["conversations"][0]["value"].replace("<image>\n", "")
    ground_truth = item["conversations"][1]["value"]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question}
            ]
        }
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)

    answer = processor.decode(output[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
    is_correct = ground_truth.lower() in answer.lower()
    if is_correct:
        correct += 1

    print(f"问题: {question}")
    print(f"正确答案: {ground_truth}")
    print(f"模型回答: {answer}")
    print(f"正确: {is_correct}")
    print("---")

print(f"\n准确率: {correct}/{total} = {correct/total*100:.1f}%")
