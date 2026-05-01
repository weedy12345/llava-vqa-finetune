# LLaVA VQA Fine-tuning

基于 LLaVA-1.6 在 VQAv2 数据集上进行 LoRA 微调的实验项目。

## 项目结构

- `prepare_data.py` - 数据预处理，从 VQAv2 提取 1000 条样本
- `finetune_vqa.py` - LoRA 微调脚本
- `eval_vqa.py` - 评估脚本

## 环境

- Python 3.10
- PyTorch 2.4.0
- Transformers 4.40.0
- PEFT (LoRA)
- GPU: RTX 4090 24GB

## 训练细节

- 基础模型: llava-hf/llava-v1.6-mistral-7b-hf
- 微调方法: LoRA (r=16, alpha=32)
- 可训练参数: 8.4M / 7.5B (0.11%)
- 训练数据: VQAv2 val2014 子集 1000 条
- Batch size: 1
- Learning rate: 2e-4
- Epochs: 3

## 实验结果

| Epoch | Loss |
|-------|------|
| 1 | 4.399 |
| 2 | 4.527 |
| 3 | 4.851 |

20 条样本评估准确率: 15%

## 问题分析

- 数据量不足（1000 条）导致过拟合
- Epoch 2、3 的 loss 上升说明模型开始记忆训练数据
- 模型倾向于输出"yes"，说明需要屏蔽问题部分的 loss
- 下一步：扩大数据量至 10K+，降低学习率至 2e-5，添加 label masking

## 数据集

VQAv2: https://visualqa.org/download.html
COCO val2014 images
