import os
import torch
import json
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import get_peft_model, LoraConfig
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

data_dir = "/root/autodl-tmp/vqa_data"
data = json.load(open(os.path.join(data_dir, "vqa_1k.json")))

print("加载模型...")
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("开始训练...")
model.train()
for epoch in range(1):
    total_loss = 0
    for i, item in enumerate(data):
        image = Image.open(item["image"])
        human_text = item["conversations"][0]["value"]
        gpt_text = item["conversations"][1]["value"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": human_text.replace("<image>\n", "")}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": gpt_text}]
            }
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")

       
        labels = inputs["input_ids"].clone()
        # 找到 [/INST] 的位置，之前的 token 全部设为 -100
        input_ids = inputs["input_ids"][0]
        inst_token_id = processor.tokenizer.convert_tokens_to_ids("[/INST]")
        inst_pos = (input_ids == inst_token_id).nonzero()
        if len(inst_pos) > 0:
            labels[0, :inst_pos[-1].item()+1] = -100

        outputs = model(**inputs, labels=labels)	
        	
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Step {i+1}/1000, Loss: {total_loss/(i+1):.4f}")

    print(f"Epoch {epoch+1} 完成, 平均 Loss: {total_loss/len(data):.4f}")

print("保存模型...")
model.save_pretrained(os.path.join(data_dir, "vqa_lora_weights"))
print("完成！")
