#!/usr/bin/env python3
"""
dLNk GPT Training Script for Linux (Ubuntu/Kali) and Windows
Optimized for GPU training with proper error handling
Supports RTX 5060 Ti and other Blackwell architecture GPUs
"""

import os
import sys
import warnings
# Suppress CUDA capability warnings for newer GPUs (e.g., RTX 5060 Ti with sm_120)
warnings.filterwarnings("ignore", message=".*CUDA capability.*")
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import transformers

# Suppress warnings
transformers.logging.set_verbosity_error()

print("="*80)
print("dLNk GPT - Local Training Script")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

# Hugging Face token (REQUIRED)
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Read from environment for security
PUSH_TO_HUB = bool(HF_TOKEN)

# Model configuration
MODEL_NAME = "EleutherAI/gpt-j-6B"
DATASET_NAME = "dlnkgpt/dlnkgpt-uncensored-dataset"
OUTPUT_DIR = "./dlnkgpt-model-output-orig"
HUB_MODEL_ID = "dlnkgpt/dlnkgpt-uncensored"

# Training parameters
EPOCHS = 3
BATCH_SIZE = 1  # Adjusted for 16GB-class VRAM to reduce OOM risk
GRADIENT_ACCUMULATION = 16
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
MAX_LENGTH = 512
SAVE_STEPS = 250  # Save more frequently for better checkpointing
EVAL_STEPS = 250  # Evaluate more frequently to catch overfitting early
LOGGING_STEPS = 10

# LoRA parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1  # Increased dropout to prevent overfitting

# System parameters (can override via environment variables)
import platform as _platform
_default_8bit = True
USE_FP16 = os.getenv("USE_FP16", "0") == "1"  # 1 to enable FP16
USE_8BIT = os.getenv("USE_8BIT", "1") == "1"  # default to 8-bit unless explicitly disabled
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

print("Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Gradient Accumulation: {GRADIENT_ACCUMULATION}")
print(f"  Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Max Length: {MAX_LENGTH}")
print(f"  Eval Steps: {EVAL_STEPS} (monitoring for overfitting)")
print(f"  Save Steps: {SAVE_STEPS}")
print(f"  LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"  Weight Decay: 0.15 (overfitting prevention)")
print(f"  Early Stopping: patience=5, threshold=0.001")
print(f"  FP16: {USE_FP16}")
print(f"  8-bit: {USE_8BIT}")
print()

# ============================================================================
# Check Requirements
# ============================================================================

print("[1/9] Checking system requirements...")

# Check CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    print("Please install CUDA-enabled PyTorch")
    sys.exit(1)

print(f"  OK PyTorch: {torch.__version__}")
print(f"  OK CUDA: {torch.version.cuda}")
gpu_name = torch.cuda.get_device_name(0)
print(f"  OK GPU: {gpu_name}")
print(f"  OK GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Check for RTX 5060 Ti or other Blackwell architecture GPUs
if "5060" in gpu_name or "5080" in gpu_name or "5090" in gpu_name:
    compute_capability = torch.cuda.get_device_properties(0).major * 10 + torch.cuda.get_device_properties(0).minor
    if compute_capability >= 120:
        print(f"  INFO: Detected GPU with compute capability {compute_capability} (sm_{compute_capability})")
        print(f"  INFO: CUDA capability warnings are suppressed - GPU should work despite warnings")
        print(f"  INFO: For best compatibility, consider upgrading to PyTorch 2.8.0+ with CUDA 12.8+")
print()

# Determine data source
DATA_FILE = os.getenv("DATA_FILE", "")
USING_LOCAL = False
if DATA_FILE and os.path.exists(DATA_FILE):
    USING_LOCAL = True
else:
    # also consider default local files
    for p in ["training_data_filtered.jsonl", "training_data_1.1m_final.jsonl"]:
        if os.path.exists(os.path.abspath(p)):
            USING_LOCAL = True
            break

# Check HF token only if not using local data
if not HF_TOKEN and not USING_LOCAL:
    print("ERROR: Hugging Face token is required for downloading dataset/model from Hub!")
    print("Please set HF_TOKEN environment variable, or set DATA_FILE to a local JSONL.")
    sys.exit(1)

if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN

# ============================================================================
# Load Dataset
# ============================================================================

print("[2/9] Loading dataset...")

# Determine data source
DATA_FILE = os.getenv("DATA_FILE", "")

try:
    # Priority 1: DATA_FILE environment variable
    if DATA_FILE and os.path.exists(DATA_FILE):
        print(f"  -> Loading from DATA_FILE: {DATA_FILE}")
        dataset = load_dataset('json', data_files={'train': DATA_FILE})
        dataset = dataset['train'].train_test_split(test_size=0.05, seed=42)
        dataset = DatasetDict({'train': dataset['train'], 'validation': dataset['test']})
    
    # Priority 2: Local filtered file
    elif os.path.exists("training_data_filtered.jsonl"):
        print("  -> Found and loading: training_data_filtered.jsonl")
        local_path = os.path.abspath("training_data_filtered.jsonl")
        dataset = load_dataset('json', data_files={'train': local_path})
        dataset = dataset['train'].train_test_split(test_size=0.05, seed=42)
        dataset = DatasetDict({'train': dataset['train'], 'validation': dataset['test']})

    # Priority 3: Default local file
    elif os.path.exists("training_data_1.1m_final.jsonl"):
        print("  -> Found and loading: training_data_1.1m_final.jsonl")
        local_path = os.path.abspath("training_data_1.1m_final.jsonl")
        dataset = load_dataset('json', data_files={'train': local_path})
        dataset = dataset['train'].train_test_split(test_size=0.05, seed=42)
        dataset = DatasetDict({'train': dataset['train'], 'validation': dataset['test']})

    # Priority 4: Fallback to Hugging Face Hub
    else:
        print(f"  -> No local files found. Loading from Hugging Face Hub: {DATASET_NAME}")
        if not HF_TOKEN:
            print("ERROR: HF_TOKEN is required to download from the Hub.")
            sys.exit(1)
        dataset = load_dataset(DATASET_NAME, token=HF_TOKEN)

    print(f"  OK Train: {len(dataset['train']):,} examples")
    print(f"  OK Validation: {len(dataset['validation']):,} examples")

except Exception as e:
    print(f"ERROR: Failed to load dataset: {e}")
    sys.exit(1)
print()

# ============================================================================
# Load Tokenizer
# ============================================================================

print("[3/9] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  OK Tokenizer loaded")
    print(f"  OK Vocab size: {len(tokenizer):,}")
except Exception as e:
    print(f"ERROR loading tokenizer: {e}")
    sys.exit(1)
print()

# ============================================================================
# Prepare text field and Tokenize Dataset
# ============================================================================

print("[4/9] Preparing dataset text field and tokenizing...")
print("  This may take a few minutes...")

# Ensure there is a 'text' field. If not, try to construct it from common columns.

def _format_example(ex):
    # If 'text' already exists and is non-empty, use it
    if isinstance(ex.get('text'), str) and ex.get('text').strip():
        return {'text': ex['text']}
    # Common instruction-tuning formats
    instr = ex.get('instruction', '')
    inp = ex.get('input', '')
    out = ex.get('output', '')
    parts = []
    if instr:
        parts.append(f"Instruction:\n{instr}")
    if inp:
        parts.append(f"Input:\n{inp}")
    if out:
        parts.append(f"Response:\n{out}")
    if parts:
        return {'text': "\n\n".join(parts)}
    # Fallback: join all string-like fields
    try:
        strings = [str(v) for k, v in ex.items() if isinstance(v, (str, int, float))]
        if strings:
            return {'text': "\n".join(strings)}
    except Exception:
        pass
    return {'text': ''}

try:
    # Some environments return a plain dict; normalize to HF DatasetDict
    if isinstance(dataset, dict) and not hasattr(dataset, 'map'):
        dataset = DatasetDict(dataset)

    # Map to ensure 'text' exists
    dataset = dataset.map(_format_example, desc="Constructing text field")

    def tokenize_function(examples):
        return tokenizer(
            examples.get("text", [""] * len(next(iter(examples.values())))),
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )

    # Remove all original columns except 'text' to avoid collisions
    remove_cols = [c for c in dataset["train"].column_names if c != 'text']

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=remove_cols,
        desc="Tokenizing"
    )
    print(f"  OK Dataset tokenized")
    # Optional fast exit to avoid GPU cost while validating data pipeline
    if os.getenv("DRY_RUN", "0") == "1":
        print("  DRY RUN: completed preprocessing (load/tokenize). Exiting before model download/training.")
        sys.exit(0)
except Exception as e:
    print(f"ERROR tokenizing dataset: {e}")
    sys.exit(1)
print()

# ============================================================================
# Load Model
# ============================================================================

print("[5/9] Loading model...")
print(f"  Downloading {MODEL_NAME} (~24GB)...")
print("  This may take 10-30 minutes depending on your internet speed...")

try:
    if USE_8BIT:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                load_in_8bit=True,
                device_map="auto",
                low_cpu_mem_usage=True,
                token=HF_TOKEN
            )
            model = prepare_model_for_kbit_training(model)
            print("  -> Using 8-bit quantization (bitsandbytes)")
        except Exception as be:
            print(f"WARNING: 8-bit load failed: {be}")
            print("  -> Falling back to FP16 if available, otherwise FP32")
            fallback_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            device_map = "auto" if torch.cuda.is_available() else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=fallback_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                token=HF_TOKEN
            )
    else:
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if USE_FP16 else torch.float32,
            device_map=device_map,
            low_cpu_mem_usage=True,
            token=HF_TOKEN
        )
    # Reduce memory footprint during training
    try:
        model.config.use_cache = False
    except Exception:
        pass
    
    # Enable gradient checkpointing to reduce VRAM
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    
    print(f"  OK Model loaded")
    print(f"  OK Parameters: {model.num_parameters():,}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)
print()

# ============================================================================
# Apply LoRA
# ============================================================================

print("[6/9] Applying LoRA...")

try:
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],  # GPT-J specific
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  OK LoRA applied")
except Exception as e:
    print(f"ERROR applying LoRA: {e}")
    sys.exit(1)
print()

# ============================================================================
# Setup Training
# ============================================================================

print("[7/9] Setting up training...")

try:
    # Reproducibility
    set_seed(42)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_total_limit=3,  # Keep more checkpoints for comparison
        fp16=USE_FP16 and not USE_8BIT,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        push_to_hub=PUSH_TO_HUB,
        hub_model_id=HUB_MODEL_ID,
        hub_token=HF_TOKEN,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        weight_decay=0.15,  # Increased weight decay to prevent overfitting
        lr_scheduler_type="cosine",
        save_strategy="steps",
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Enhanced early stopping to prevent overfitting
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,  # Wait for 5 evaluation steps without improvement
        early_stopping_threshold=0.001,  # Minimum improvement threshold
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        callbacks=[early_stopping]
    )
    
    print(f"  OK Trainer configured")
except Exception as e:
    print(f"ERROR setting up training: {e}")
    sys.exit(1)
print()

# ============================================================================
# Start Training
# ============================================================================

print("[8/9] Starting training...")
print("="*80)
print()
print(f"Training will take approximately 8-12 hours on RTX 3090/4090")
print(f"You can monitor progress below")
print()
print("="*80)
print()

try:
    resume_flag = os.getenv("RESUME", "1") == "1"
    resume_ckpt = None
    if resume_flag and os.path.isdir(OUTPUT_DIR):
        # Try to detect latest checkpoint
        cks = [os.path.join(OUTPUT_DIR, d) for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if cks:
            resume_ckpt = max(cks, key=lambda p: int(p.split("checkpoint-")[-1]))
            print(f"  -> Resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt if resume_ckpt else False)
    
    print()
    print("="*80)
    print("Training Complete!")
    print("="*80)
    print()
    
except KeyboardInterrupt:
    print()
    print("="*80)
    print("Training interrupted by user")
    print("="*80)
    print()
except RuntimeError as re:
    # Commonly CUDA OOM
    print()
    print("="*80)
    print(f"ERROR during training (runtime): {re}")
    print("HINT: Reduce per_device_train_batch_size or increase GRADIENT_ACCUMULATION. Training state is saved; will resume on next run.")
    print("="*80)
    print()
    sys.exit(1)
except Exception as e:
    print()
    print("="*80)
    print(f"ERROR during training: {e}")
    print("="*80)
    print()
    sys.exit(1)

# ============================================================================
# Save Model
# ============================================================================

print("[9/9] Saving model...")

try:
    # Save locally
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  OK Model saved to {OUTPUT_DIR}")
    
    # Push to Hub
    print(f"  Pushing to Hugging Face Hub...")
    model.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    tokenizer.push_to_hub(HUB_MODEL_ID, token=HF_TOKEN)
    print(f"  OK Model pushed to https://huggingface.co/{HUB_MODEL_ID}")
    
except Exception as e:
    print(f"WARNING: Error saving model: {e}")
    print(f"Model is still available in {OUTPUT_DIR}")

print()
print("="*80)
print("All Done!")
print("="*80)
print()
print(f"Model location:")
print(f"  Local: {OUTPUT_DIR}")
print(f"  Hub: https://huggingface.co/{HUB_MODEL_ID}")
print()
