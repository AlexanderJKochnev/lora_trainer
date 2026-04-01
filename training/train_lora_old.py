# training/train_lora.py
import os
import json
import torch
import argparse
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, )
from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType)
from datasets import Dataset
from auto_gptq import AutoGPTQForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Путь к GPTQ модели"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Путь к JSONL датасету"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Куда сохранить LoRA адаптер"
    )
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("LoRA TRAINING FOR QWEN2.5-7B-GPTQ")
    print("=" * 50)

    # Проверка путей
    if not os.path.exists(args.model_path):
        print(f"ОШИБКА: Модель не найдена: {args.model_path}")
        return

    if not os.path.exists(args.dataset_path):
        print(f"ОШИБКА: Датасет не найден: {args.dataset_path}")
        return

    print(f"Модель: {args.model_path}")
    print(f"Датасет: {args.dataset_path}")
    print(f"Выход: {args.output_dir}")

    # Проверка GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("GPU не обнаружена!")
        return

    # Загрузка GPTQ модели (локально, без интернета)
    print("\nЗагрузка GPTQ модели...")
    try:
        model = AutoGPTQForCausalLM.from_quantized(
            args.model_path, device="cuda:0", use_triton=False, inject_fused_attention=False,
            inject_fused_mlp=False, trust_remote_code=True, use_cuda_fp16=True
        )
        print("Модель загружена")
    except Exception as e:
        print(f"Ошибка загрузки GPTQ: {e}")
        print("Пробуем стандартную загрузку...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16,
            local_files_only=True
        )

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Подготовка модели для LoRA
    model = prepare_model_for_kbit_training(model)

    # LoRA конфигурация
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)

    # Подсчет параметров
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters: {trainable_params:,} ({trainable_params / total_params:.2f}%)")

    # Загрузка датасета
    print(f"\nЗагрузка датасета...")
    data = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Загружено {len(data)} записей")

    # Форматирование для Qwen2.5
    def format_example(example):
        if example['input'] and example['input'] != "Информация о вине":
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

    # Настройка обучения
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4, warmup_steps=100, learning_rate=args.learning_rate, fp16=True,
        logging_steps=10, save_steps=500, save_total_limit=2, report_to="none",
        remove_unused_columns=False, )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_dataset, tokenizer=tokenizer, )

    # Запуск обучения
    print("\n" + "=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50)

    trainer.train()

    # Сохранение LoRA адаптера
    print(f"\nСохранение LoRA адаптера в {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 50)
    print(f"LoRA адаптер сохранен в: {args.output_dir}")

    # Вывод размера адаптера
    total_size = 0
    for root, dirs, files in os.walk(args.output_dir):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f"Размер адаптера: {total_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
