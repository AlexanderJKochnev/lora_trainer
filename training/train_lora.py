# training/train_lora.py
import os
import json
import torch
import argparse
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          Trainer, DataCollatorForLanguageModeling, AutoConfig, BitsAndBytesConfig)
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType)
from datasets import Dataset
from auto_gptq import AutoGPTQForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("LoRA TRAINING FOR QWEN2.5-7B-GPTQ (OPTIMIZED FOR 12GB VRAM)")
    print("=" * 50)

    print("\nЗагрузка модели (GPTQ Native Mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map={"": 0},  # Явно сажаем всю модель на 0-ю карту
        trust_remote_code=True, load_in_4bit=True,  # ФОРСИРУЕМ 4 бита через bitsandbytes
        torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Подготовка модели (важные флаги для экономии памяти)
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=8,  # Снизил с 16 до 8 для экономии памяти (на качество почти не влияет)
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Оставили основные слои
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Загрузка и форматирование датасета
    data = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    def format_example(example):
        text = f"<|im_start|>user\n{example['instruction']}\n{example.get('input', '')}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return {"text": text}

    dataset = Dataset.from_list(data).map(format_example)

    # 3. УМЕНЬШЕНИЕ КОНТЕКСТА (Самый важный пункт)
    # 2048 токенов НЕ влезут в 12ГБ. Ставим 512 или 768.
    MAX_LENGTH = 512

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Не паддим сразу, сделаем это в коллаторе
            max_length=MAX_LENGTH
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # 4. Настройки обучения (Критически важные для OOM)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Только 1!
        gradient_accumulation_steps=16,  # Увеличил, чтобы компенсировать батч-сайз
        warmup_steps=20,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=1,
        save_steps=100,
        save_total_limit=1,
        report_to="none",
        # ИСПОЛЬЗУЕМ PAGED ОПТИМИЗАТОР (выносит часть данных в RAM)
        optim="paged_adamw_32bit",
        # Отключаем лишнее
        gradient_checkpointing=True,
        max_grad_norm=0.3,
    )

    # Используем DataCollator для эффективного паддинга
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nНАЧАЛО ОБУЧЕНИЯ (Artemis 2 is Go!)")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Готово! Адаптер в: {args.output_dir}")


if __name__ == "__main__":
    main()
