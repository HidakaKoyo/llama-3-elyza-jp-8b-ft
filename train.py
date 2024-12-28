import os
import time
import logging
from datetime import datetime

import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# NOTE: GPU利用率の取得のためのライブラリ読み込み
try:
    from pynvml import (
        nvmlInit,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetUtilizationRates,
    )

    GPU_MONITORING_AVAILABLE = True
    nvmlInit()  # NOTE: NVML初期化
    handle = nvmlDeviceGetHandleByIndex(0)  # NOTE: マルチGPU環境なら適宜変更
except Exception as e:
    GPU_MONITORING_AVAILABLE = False
    print("[WARNING] pynvmlが読み込めないため、GPU利用率ログは出力されません。")
    print(str(e))

# NOTE: ロガーのセットアップ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# NOTE: TrainerCallback: ロギングでGPU使用率を出力
class GpuUsageCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if GPU_MONITORING_AVAILABLE:
            usage = nvmlDeviceGetUtilizationRates(handle)
            # NOTE: usage.gpu は GPU使用率(%), usage.memory は GPUメモリ使用率(%)
            logger.info(f"[GPU] usage: {usage.gpu}%, memory: {usage.memory}%")


def main():
    # NOTE: 1. パラメータ設定
    model_name = "elyza/Llama-3-ELYZA-JP-8B"
    parquet_path = "hf://datasets/KisanVaani/agriculture-qa-english-only/data/train-00000-of-00001.parquet"

    # NOTE: 学習ハイパーパラメータ (例)
    num_train_epochs = 1
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 4
    learning_rate = 1e-4
    output_dir = "./lora-output"

    logger.info("===== Llama-3-ELYZA-JP-8B Fine-tuning with LoRA (PEFT) =====")

    # NOTE: 2. データの読み込み＆前処理
    logger.info("Loading dataset...")
    df = pd.read_parquet(parquet_path)

    # NOTE: 例：DataFrame の列名を想定して 'question' と 'answer' を使用
    # NOTE: 必要に応じて列名を合わせてください
    data_list = []
    for idx, row in df.iterrows():
        instruction_text = str(row["question"])
        answer_text = str(row["answer"])

        data_list.append(
            {
                "instruction": instruction_text,
                "input": "",  # NOTE: 必要なら補足情報を入れる
                "output": answer_text,
            }
        )

    # NOTE: Hugging Face datasets 形式に変換
    raw_dataset = Dataset.from_list(data_list)

    # NOTE: 学習用データと検証用データに分割 (例: 99:1 split)
    # NOTE: プロジェクトに応じてスプリットは調整してください
    train_test = raw_dataset.train_test_split(test_size=0.01, shuffle=True, seed=42)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    # NOTE: 3. トークナイザー＆モデルの読み込み
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    # NOTE: 4. LoRA(PEFT) 設定
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj"],  # NOTE: モデルに合わせて要調整
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    logger.info("LoRA model is ready.")

    # NOTE: 5. データのトークナイズ
    # NOTE: 一般的には "instruction"+"input" からプロンプトを作り、"output" を学習ターゲットにする
    # NOTE: 下記は非常に簡易的な例です
    def tokenize_fn(examples):
        # NOTE: シンプルに「Q: ... / A: ...」の形式で連結
        sources = [f"Q: {ins}\n\nA: " for ins in examples["instruction"]]
        # NOTE: 出力
        targets = [out for out in examples["output"]]

        # NOTE: 入力と出力を連結して一つのテキストに
        model_inputs = [s + t for s, t in zip(sources, targets)]
        # NOTE: トークナイズ
        tokenized = tokenizer(
            model_inputs, max_length=512, padding="max_length", truncation=True
        )
        # NOTE: ラベル（出力部分）を設定 (本来はオフセットを計算してマスクするとより厳密)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True)

    # NOTE: 6. Trainer の設定
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",  # NOTE: 定期的にevalする場合
        eval_steps=50,
        logging_steps=10,
        save_steps=200,
        fp16=True,
        learning_rate=learning_rate,
        optim="adamw_torch",
        report_to="none",  # NOTE: wandb 等を使わない場合
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=[
            GpuUsageCallback()
        ],  # NOTE: GPU使用率をロギングするコールバックを追加
    )

    # NOTE: 7. 学習の実行と時間計測
    logger.info("Starting training...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()

    # NOTE: 8. 結果の保存
    trainer.save_model(output_dir)
    logger.info(f"Model is saved at: {output_dir}")

    # NOTE: LoRAの差分だけを保存したい場合は下記のように
    # NOTE: model.save_pretrained("lora-only-weights")

    total_train_time = end_time - start_time
    logger.info(f"Training complete. Elapsed time: {total_train_time:.2f} seconds")


if __name__ == "__main__":
    main()
