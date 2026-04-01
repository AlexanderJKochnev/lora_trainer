import os
import argparse
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # 1. Загрузка модели через Unsloth (4-bit по умолчанию)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=2048,  # Можно ставить даже 2048 на 12ГБ!
        load_in_4bit=True,
        trust_remote_code=True,
    )

    # 2. Добавление LoRA адаптеров
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Unsloth оптимален при 0
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # 3. Подготовка датасета (используем ваш формат Qwen)
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            if input_text and input_text != "Информация о вине":
                text = f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            else:
                text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 4. Обучение через SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=2,  # На Unsloth можно даже 2!
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=100,  # Для теста
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=args.output_dir,
            optim="adamw_8bit",
        ),
    )

    trainer.train()

    # 5. Сохранение (Unsloth сохраняет очень быстро)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Обучение завершено. Адаптеры в {args.output_dir}")


if __name__ == "__main__":
    main()
