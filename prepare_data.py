import json
import os

data_dir = "/root/autodl-tmp/vqa_data"

questions = json.load(open(os.path.join(data_dir, "v2_OpenEnded_mscoco_val2014_questions.json")))
annotations = json.load(open(os.path.join(data_dir, "v2_mscoco_val2014_annotations.json")))

# 建立 question_id 到 answer 的映射
ann_map = {a["question_id"]: a["multiple_choice_answer"] for a in annotations["annotations"]}

# 取前 1000 条，过滤掉图片不存在的
samples = []
for q in questions["questions"]:
    image_id = q["image_id"]
    image_file = f"COCO_val2014_{image_id:012d}.jpg"
    image_path = os.path.join(data_dir, "val2014", image_file)
    
    if not os.path.exists(image_path):
        continue
    
    samples.append({
        "id": str(q["question_id"]),
        "image": image_path,
        "conversations": [
            {"from": "human", "value": f"<image>\n{q['question']}"},
            {"from": "gpt", "value": ann_map[q["question_id"]]}
        ]
    })
    
    if len(samples) >= 1000:
        break

output_path = os.path.join(data_dir, "vqa_1k.json")
json.dump(samples, open(output_path, "w"))
print(f"保存了 {len(samples)} 条数据")
print("样例:", samples[0])
