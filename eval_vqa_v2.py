import os
import torch
import json
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import PeftModel
from PIL import Image

os.environ["TRANSFORMERS_OFFLINE"] = "1"
data_dir = "/root/autodl-tmp/vqa_data"
data = json.load(open(os.path.join(data_dir, "vqa_5k.json")))

print("加载模型...")
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
base_model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, os.path.join(data_dir, "vqa_lora_weights"))
model.eval()

# 同义词表
synonyms = {
    "resting": ["tired", "sleeping", "lying"],
    "eating": ["feeding", "having food"],
    "running": ["jogging", "sprinting"],
    "yes": ["yeah", "yep", "correct"],
    "no": ["nope", "not"],
}

def soft_match(pred, gt):
    pred = pred.lower().strip()
    gt = gt.lower().strip()
    # 完全匹配
    if pred == gt:
        return True
    # 包含匹配
    if gt in pred or pred in gt:
        return True
    # 同义词匹配
    for key, vals in synonyms.items():
        group = [key] + vals
        if pred in group and gt in group:
            return True
    return False

total = 50
exact_correct = 0
soft_correct = 0

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

    exact = ground_truth.lower() in answer.lower()
    soft = soft_match(answer, ground_truth)

    if exact:
        exact_correct += 1
    if soft:
        soft_correct += 1

print(f"\n评估 {total} 条样本")
print(f"精确匹配准确率: {exact_correct}/{total} = {exact_correct/total*100:.1f}%")
print(f"宽松匹配准确率: {soft_correct}/{total} = {soft_correct/total*100:.1f}%")
