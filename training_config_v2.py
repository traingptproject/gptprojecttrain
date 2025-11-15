"""
Training Configuration V2 - With Regularization to Prevent Overfitting

Changes from V1:
1. Added weight_decay = 0.1 (L2 regularization)
2. Reduced learning_rate from 2e-5 to 5e-6 (4x slower)
3. Changed eval_strategy to "steps" with eval_steps = 500
4. Increased save_steps to 500 for better monitoring
5. Added gradient_accumulation_steps = 8 for more stable training
"""

# Model Configuration
MODEL_NAME = "EleutherAI/gpt-j-6b"
OUTPUT_DIR = "./dLNk-gpt-uncensored-v2"

# LoRA Configuration (same as V1)
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Training Arguments V2 - With Regularization
TRAINING_ARGS = {
    # Basic Settings
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8,  # Increased from 4 to 8 for stability
    
    # Learning Rate - REDUCED for better generalization
    "learning_rate": 5e-6,  # Changed from 2e-5 to 5e-6 (4x slower)
    "weight_decay": 0.1,    # NEW: L2 regularization to prevent overfitting
    
    # Learning Rate Scheduler (same as V1)
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    
    # Evaluation Strategy - IMPROVED for frequent monitoring
    "eval_strategy": "steps",  # Changed from "epoch" to "steps"
    "eval_steps": 500,         # NEW: Evaluate every 500 steps
    "save_strategy": "steps",
    "save_steps": 500,         # Changed from 1000 to 500
    "save_total_limit": 5,     # Increased from 3 to 5
    
    # Logging - More frequent
    "logging_steps": 10,       # Same as V1
    "logging_dir": f"{OUTPUT_DIR}/logs",
    
    # Early Stopping (same as V1)
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    
    # Optimization
    "optim": "adamw_torch",
    "max_grad_norm": 1.0,
    
    # Mixed Precision (same as V1)
    "fp16": True,
    
    # Other Settings
    "report_to": "tensorboard",
    "dataloader_num_workers": 4,
    "dataloader_pin_memory": True,
    "remove_unused_columns": False,
}

# Early Stopping Configuration (same as V1)
EARLY_STOPPING = {
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.0
}

# Dataset Configuration (same as V1)
DATASET_CONFIG = {
    "train_file": "training_data_1m_final.jsonl",
    "train_samples": 54000,
    "eval_samples": 6000,
    "max_length": 512,
    "test_size": 0.1
}

# QA Test Configuration (same as V1)
QA_TESTS = [
    {
        "prompt": "Write a Python function to calculate factorial:",
        "expected_keywords": ["def", "factorial", "return"]
    },
    {
        "prompt": "Explain quantum computing in simple terms:",
        "expected_keywords": ["quantum", "bit", "computer"]
    },
    {
        "prompt": "What are the main causes of climate change?",
        "expected_keywords": ["carbon", "greenhouse", "emission"]
    }
]

# Monitoring Configuration - ENHANCED
MONITORING_CONFIG = {
    "check_interval": 300,  # Check every 5 minutes
    "metrics_to_track": [
        "loss",
        "eval_loss",
        "learning_rate",
        "grad_norm",
        "epoch"
    ],
    "alert_thresholds": {
        "loss_spike": 2.0,  # Alert if loss increases by 2x
        "grad_norm_max": 10.0,  # Alert if gradient norm > 10
        "eval_loss_divergence": 1.5  # Alert if eval_loss > 1.5x train_loss
    }
}

# Improvement Notes
IMPROVEMENTS = """
V2 Improvements:
1. Weight Decay (0.1): Prevents model from memorizing training data
2. Lower Learning Rate (5e-6): Slower but more stable learning
3. Frequent Evaluation (every 500 steps): Early detection of overfitting
4. More Checkpoints (every 500 steps): Better model selection
5. Gradient Accumulation (8): More stable gradients

Expected Results:
- Loss will decrease slower (this is GOOD!)
- Training and validation loss will be closer
- Better generalization to unseen data
- Model will learn patterns, not memorize data
"""

print("✅ Training Configuration V2 loaded with regularization")
print(f"📊 Learning Rate: {TRAINING_ARGS['learning_rate']}")
print(f"📊 Weight Decay: {TRAINING_ARGS['weight_decay']}")
print(f"📊 Eval Steps: {TRAINING_ARGS['eval_steps']}")
