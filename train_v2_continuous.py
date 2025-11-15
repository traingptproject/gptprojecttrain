#!/usr/bin/env python3
"""
dLNk GPT V2 - Continuous Training Script
Designed for Google Colab Pro+ with A100 GPU
Uses 1M+ training dataset from Google Drive
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers

# Configuration
MODEL_NAME = "EleutherAI/gpt-j-6b"
OUTPUT_DIR = "./dLNk-gpt-j-6b-exploit-v2-continuous"
DATASET_PATH = "/content/drive/MyDrive/Colab Notebooks/training_data_1m_final.jsonl"
HF_REPO = "traingptproject/dLNk-gpt-j-6b-exploit-v2"

# V2 Anti-Overfitting Configuration
LEARNING_RATE = 5e-6  # 4x lower than V1
WEIGHT_DECAY = 0.1    # 10x higher than V1
LORA_RANK = 16        # 2x higher than V1
LORA_ALPHA = 32
LORA_DROPOUT = 0.05   # Added regularization
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
MAX_LENGTH = 512
NUM_EPOCHS = 3
EVAL_STEPS = 500      # 10x more frequent than V1
SAVE_STEPS = 500
WARMUP_STEPS = 100

def send_line_notification(message):
    """Send notification via LINE MCP"""
    try:
        cmd = [
            "manus-mcp-cli", "tool", "call", "push_text_message",
            "--server", "line",
            "--input", json.dumps({"message": {"text": message}})
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except Exception as e:
        print(f"LINE notification failed: {e}")
        return False

def format_time(seconds):
    """Format seconds to readable time"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def load_and_prepare_dataset():
    """Load and prepare the 1M+ training dataset"""
    print("="*80)
    print("Loading Dataset")
    print("="*80)
    
    if not os.path.exists(DATASET_PATH):
        error_msg = f"❌ Dataset not found: {DATASET_PATH}"
        print(error_msg)
        send_line_notification(error_msg)
        sys.exit(1)
    
    print(f"📥 Loading dataset: {DATASET_PATH}")
    
    try:
        # Load dataset
        dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
        total_samples = len(dataset)
        print(f"✅ Loaded {total_samples:,} samples")
        
        # Check columns
        print(f"📋 Columns: {dataset.column_names}")
        
        # Filter to keep only required columns
        required_cols = ['instruction', 'input', 'output']
        if all(col in dataset.column_names for col in required_cols):
            # Remove extra columns if they exist
            extra_cols = [col for col in dataset.column_names if col not in required_cols]
            if extra_cols:
                print(f"🗑️  Removing extra columns: {extra_cols}")
                dataset = dataset.remove_columns(extra_cols)
        else:
            error_msg = f"❌ Missing required columns. Found: {dataset.column_names}"
            print(error_msg)
            send_line_notification(error_msg)
            sys.exit(1)
        
        # Split dataset
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
        print(f"📊 Train samples: {len(train_dataset):,}")
        print(f"📊 Eval samples: {len(eval_dataset):,}")
        
        # Send notification
        send_line_notification(
            f"📊 ข้อมูลเทรนนิ่งพร้อม!\n\n"
            f"📈 Total: {total_samples:,} samples\n"
            f"🎯 Train: {len(train_dataset):,}\n"
            f"✅ Eval: {len(eval_dataset):,}\n\n"
            f"🚀 กำลังเริ่มการเทรน..."
        )
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        error_msg = f"❌ Error loading dataset: {str(e)}"
        print(error_msg)
        send_line_notification(error_msg)
        sys.exit(1)

def load_model_and_tokenizer():
    """Load GPT-J model with LoRA configuration"""
    print("="*80)
    print("Loading Model")
    print("="*80)
    
    print(f"📥 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"📥 Loading model: {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # V2 LoRA Configuration
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("✅ Model loaded with V2 LoRA configuration")
    
    return model, tokenizer

def preprocess_function(examples, tokenizer):
    """Format examples for training"""
    texts = []
    for instruction, input_text, output in zip(
        examples['instruction'],
        examples['input'],
        examples['output']
    ):
        if input_text and input_text.strip():
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        texts.append(prompt)
    
    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )

class LineNotificationCallback(transformers.TrainerCallback):
    """Custom callback for LINE notifications"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_notification = 0
        self.notification_interval = 300  # 5 minutes
    
    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps
        send_line_notification(
            f"🚀 เริ่มการเทรน dLNk GPT V2\n\n"
            f"📊 Total Steps: {total_steps:,}\n"
            f"📈 Epochs: {NUM_EPOCHS}\n"
            f"⏰ เวลาเริ่ม: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"💡 ระบบจะรายงานทุก 5 นาที"
        )
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        current_time = time.time()
        
        # Send notification every 5 minutes
        if current_time - self.last_notification >= self.notification_interval:
            elapsed = current_time - self.start_time
            progress = (state.global_step / state.max_steps) * 100
            
            # Calculate progress bar
            bar_length = 20
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # Estimate remaining time
            if state.global_step > 0:
                time_per_step = elapsed / state.global_step
                remaining_steps = state.max_steps - state.global_step
                remaining_time = time_per_step * remaining_steps
            else:
                remaining_time = 0
            
            message = (
                f"📈 ความคืบหน้าการเทรน\n\n"
                f"{bar} {progress:.1f}%\n\n"
                f"🔢 Step: {state.global_step:,}/{state.max_steps:,}\n"
                f"📉 Loss: {logs.get('loss', 0):.4f}\n"
                f"⚡ LR: {logs.get('learning_rate', 0):.2e}\n\n"
                f"⏱️ เวลาที่ใช้: {format_time(elapsed)}\n"
                f"🕐 เวลาคงเหลือ: {format_time(remaining_time)}"
            )
            
            send_line_notification(message)
            self.last_notification = current_time
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            message = (
                f"🎯 ผลการประเมิน (Step {state.global_step:,})\n\n"
                f"📉 Eval Loss: {metrics.get('eval_loss', 0):.4f}\n"
                f"📊 Train Loss: {metrics.get('loss', 0):.4f}\n\n"
                f"✅ โมเดลกำลังเรียนรู้อย่างต่อเนื่อง"
            )
            send_line_notification(message)
    
    def on_save(self, args, state, control, **kwargs):
        message = (
            f"💾 บันทึก Checkpoint\n\n"
            f"🔢 Step: {state.global_step:,}\n"
            f"📁 Location: {OUTPUT_DIR}\n\n"
            f"✅ พร้อมกู้คืนได้ทันที"
        )
        send_line_notification(message)
    
    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        message = (
            f"✅ การเทรนเสร็จสมบูรณ์!\n\n"
            f"⏱️ เวลาทั้งหมด: {format_time(elapsed)}\n"
            f"🔢 Total Steps: {state.global_step:,}\n\n"
            f"💾 โมเดลถูกบันทึกที่:\n{OUTPUT_DIR}\n\n"
            f"🎉 พร้อมใช้งาน!"
        )
        send_line_notification(message)

def main():
    """Main training function"""
    print("="*80)
    print("dLNk GPT V2 - Continuous Training")
    print("="*80)
    print(f"🔥 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"💾 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"🎯 Model: {MODEL_NAME}")
    print(f"📊 Dataset: {DATASET_PATH}")
    print(f"📈 Learning Rate: {LEARNING_RATE}")
    print(f"⚖️  Weight Decay: {WEIGHT_DECAY}")
    print(f"🔢 LoRA Rank: {LORA_RANK}")
    print(f"💧 LoRA Dropout: {LORA_DROPOUT}")
    print("="*80)
    
    # Load dataset
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Preprocess datasets
    print("\n📝 Formatting dataset...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )
    print("✅ Dataset formatted")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=10,
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        push_to_hub=False,
        report_to="none",
        save_total_limit=3,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LineNotificationCallback()],
    )
    
    # Start training
    print("\n🚀 Starting training...")
    print("="*80)
    
    try:
        trainer.train()
        
        # Save final model
        print("\n💾 Saving final model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"✅ Model saved to {OUTPUT_DIR}")
        
        # Optional: Push to Hugging Face Hub
        # Uncomment if you want to upload
        # print("\n📤 Uploading to Hugging Face Hub...")
        # trainer.push_to_hub(HF_REPO)
        # print(f"✅ Model uploaded to {HF_REPO}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        send_line_notification("⚠️ การเทรนถูกหยุดโดยผู้ใช้")
    except Exception as e:
        error_msg = f"❌ Training error: {str(e)}"
        print(error_msg)
        send_line_notification(error_msg)
        raise

if __name__ == "__main__":
    main()
