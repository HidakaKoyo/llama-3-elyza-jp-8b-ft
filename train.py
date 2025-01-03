import os
import time
import logging
import argparse
from datetime import datetime

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)

try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
    )

    GPU_MONITORING_AVAILABLE = True
    nvmlInit()  # NOTE: NVML初期化
    handle = nvmlDeviceGetHandleByIndex(0)  # NOTE: マルチGPUの場合は調整
except Exception as e:
    GPU_MONITORING_AVAILABLE = False
    print("[WARNING] pynvmlが読み込めないため、GPU利用率ログは出力されません。")
    print(str(e))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GpuUsageCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if GPU_MONITORING_AVAILABLE:
            usage = nvmlDeviceGetUtilizationRates(handle)
            logger.info(f"[GPU] usage: {usage.gpu}%, memory: {usage.memory}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Llama-3-ELYZA-JP-8B with LoRA")
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="hf://datasets/KisanVaani/agriculture-qa-english-only/data/train-00000-of-00001.parquet",
        help="Parquet file path or HF dataset path.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="elyza/Llama-3-ELYZA-JP-8B",
        help="Base model name on Hugging Face.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_output",
        help="Directory to save the finetuned model (full model).",
    )
    parser.add_argument(
        "--lora_save_dir",
        type=str,
        default="./lora_weights",
        help="Directory to save LoRA-only weights.",
    )
    parser.add_argument(
        "--save_lora_only",
        action="store_true",
        help="If set, only LoRA weights are saved.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("===== Llama-3-ELYZA-JP-8B Fine-tuning with LoRA (PEFT) =====")
    logger.info(f"Arguments: {args}")

    # NOTE: ----- 1. データ読み込み -----
    parquet_path = args.parquet_path
    logger.info(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    data_list = []
    for idx, row in df.iterrows():
        instruction_text = str(row["question"])
        answer_text = str(row["answers"])
        data_list.append(
            {"instruction": instruction_text, "input": "", "output": answer_text}
        )
    raw_dataset = Dataset.from_list(data_list)
    train_test = raw_dataset.train_test_split(test_size=0.01, shuffle=True, seed=42)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    # NOTE: ----- 2. モデル・トークナイザー読み込み -----
    model_name = args.model_name
    logger.info(f"Loading tokenizer and base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,  # v4.31+ のtransformersでは quantization_config
        torch_dtype="auto",
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # NOTE: pad_token がない場合に登録する
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model.config.pad_token_id = tokenizer.pad_token_id

    model.resize_token_embeddings(len(tokenizer))

    # NOTE: ----- 3. LoRA設定 -----
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA model ready.")

    # gradient checkpointing と LoRA + 量子化モデルを併用する場合は use_cache=False が必要
    # ただし Trainer の実行時に自動で use_cache=False に変更されるときもある
    model.config.use_cache = False

    # NOTE: ----- 4. データのトークナイズ -----
    def tokenize_fn(examples):
        sources = [f"Q: {ins}\n\nA: " for ins in examples["instruction"]]
        targets = [out for out in examples["output"]]
        model_inputs = [s + t for s, t in zip(sources, targets)]
        tokenized = tokenizer(
            model_inputs, max_length=256, padding="max_length", truncation=True
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    logger.info("Tokenizing dataset...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    # NOTE: ----- 5. Trainer設定 -----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        report_to="none",
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[GpuUsageCallback()],
    )

    # NOTE: ----- 6. 学習開始 -----
    logger.info("Starting training...")

    # === ここから追加 ===
    # 全体実行時間計測用（CPU側）
    start_time = time.time()

    # GPU 実行時間計測用: torch.cuda.Event
    gpu_start, gpu_end = None, None
    if torch.cuda.is_available():
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        gpu_start.record()
    # === ここまで追加 ===

    trainer.train()

    # === ここから追加 ===
    # GPU 実行時間計測の終了
    gpu_time_sec = None
    if gpu_end is not None:
        gpu_end.record()
        torch.cuda.synchronize()
        gpu_time_ms = gpu_start.elapsed_time(gpu_end)
        gpu_time_sec = gpu_time_ms / 1000.0

    end_time = time.time()
    total_train_time = end_time - start_time
    logger.info(
        f"Training complete. Elapsed total time: {total_train_time:.2f} seconds"
    )

    if gpu_time_sec is not None:
        logger.info(f"Training complete. Elapsed GPU time: {gpu_time_sec:.2f} seconds")
    # === ここまで追加 ===

    # NOTE: ----- 7. 学習結果の保存 -----
    if args.save_lora_only:
        # NOTE: LoRAのみ保存
        logger.info(f"Saving LoRA weights to: {args.lora_save_dir}")
        model.save_pretrained(args.lora_save_dir)
        logger.info("LoRA weights saved.")
    else:
        # NOTE: フルモデルを保存
        logger.info(f"Saving full model to: {args.output_dir}")
        trainer.save_model(args.output_dir)
        logger.info("Full model checkpoint saved.")

    logger.info("All done.")


if __name__ == "__main__":
    main()
