# training/train_lora.py
import os
import json
import torch
import argparse
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig)
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType)
from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("LoRA TRAINING FOR QWEN2.5-7B-GPTQ")
    print("=" * 50)
    print(f"Модель: {args.model_path}")
    print(f"Датасет: {args.dataset_path}")
    print(f"Выход: {args.output_dir}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # 4-bit quantization конфигурация с отключением Exllama
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_use_quantization_config=True, )

    print("\nЗагрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb_config, device_map="cuda:0",  # явно указываем GPU, а не "auto"
        trust_remote_code=True, torch_dtype=torch.float16, use_cache=False  # отключаем кэш для обучения
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Подготовка для LoRA
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # LoRA конфигурация
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Загрузка датасета
    print(f"\nЗагрузка датасета...")
    data = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Загружено {len(data)} записей")

    # Форматирование для Qwen2.5
    def format_example(example):
        if example.get('input') and example['input'] != "Информация о вине":
            text = f"<|im_start|>user\n{example['instruction']}\n\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        else:
            text = f"<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        return {"text": text}

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_example)

    # Токенизация
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=2048
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Обучение
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=3, per_device_train_batch_size=1,
        # уменьшаем для стабильности
        gradient_accumulation_steps=8,  # компенсируем batch size
        warmup_steps=100, learning_rate=2e-4, fp16=True, logging_steps=10, save_steps=500,
        save_total_limit=2, report_to="none", remove_unused_columns=False, )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, )

    print("\n" + "=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50)

    trainer.train()

    # Сохранение LoRA адаптера
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nОбучение завершено!")
    print(f"LoRA адаптер сохранен в: {args.output_dir}")


if __name__ == "__main__":
    main()
