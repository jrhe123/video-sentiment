# pip install transformers datasets peft accelerate
"""
LoRA 通过 插入低秩矩阵 来减少训练参数数量，只对少数参数微调，大幅节省资源；

适合在资源受限环境中进行大模型的微调；

peft 是目前主流的 LoRA 应用库，适配 HuggingFace 的模型；

"""


from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

# 1. 加载数据
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset = encoded_dataset.rename_column("label", "labels")
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 2. 加载模型
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 3. 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,        # 任务类型：文本分类
    inference_mode=False,
    r=8,                                # rank 值
    lora_alpha=32,
    lora_dropout=0.1
)

# 4. 注入 LoRA 到模型
model = get_peft_model(base_model, peft_config)

# 5. 训练参数
training_args = TrainingArguments(
    output_dir="./lora-imdb",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# 6. 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"].shuffle(seed=42).select(range(2000)),  # 选一小部分做demo
    eval_dataset=encoded_dataset["test"].select(range(500)),
)

# 7. 开始训练
trainer.train()
