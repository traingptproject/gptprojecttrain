"""
Enhanced Training Script for dLNk GPT with Advanced Workflow
Includes overfitting prevention, monitoring, and quality assurance
"""

import os
import sys
import json
import torch
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Import custom modules
from training_config import WorkflowConfig
from training_callbacks import (
    QualityAssuranceCallback,
    MetricsLoggingCallback,
    ResourceMonitorCallback,
)


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def setup_environment():
    """Setup training environment and check GPU availability"""
    print_header("Environment Setup")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  No GPU detected. Training will be VERY slow on CPU.")
        print("   Consider using Google Colab with GPU enabled.")
    
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"✅ Python Version: {sys.version.split()[0]}")


def load_and_prepare_data(config: WorkflowConfig):
    """Load and prepare dataset"""
    print_header("Data Loading")
    
    print(f"📥 Loading dataset: {config.data.dataset_name}")
    
    try:
        dataset = load_dataset(config.data.dataset_name)
        
        train_dataset = dataset[config.data.train_split]
        eval_dataset = dataset[config.data.validation_split]
        
        print(f"✅ Training samples: {len(train_dataset):,}")
        print(f"✅ Validation samples: {len(eval_dataset):,}")
        
        # Show sample
        print(f"\n📄 Sample from training set:")
        print(f"   {train_dataset[0][config.data.text_column][:200]}...")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise


def setup_model_and_tokenizer(config: WorkflowConfig):
    """Setup model, tokenizer, and apply LoRA"""
    print_header("Model Setup")
    
    print(f"📥 Loading tokenizer: {config.model.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model_name)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer loaded")
    print(f"   Vocab size: {len(tokenizer):,}")
    print(f"   Max length: {config.model.model_max_length}")
    
    # Quantization config
    quantization_config = None
    if config.quantization.use_quantization:
        print(f"\n⚙️  Setting up quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.quantization.load_in_4bit,
            load_in_8bit=config.quantization.load_in_8bit,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.quantization.bnb_4bit_use_double_quant,
        )
        quant_type = "4-bit" if config.quantization.load_in_4bit else "8-bit"
        print(f"✅ {quant_type} quantization enabled")
    
    # Load model
    print(f"\n📥 Loading base model: {config.model.base_model_name}")
    print(f"   This may take several minutes...")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=config.model.use_cache,
    )
    
    print(f"✅ Base model loaded")
    
    # Prepare model for training
    if config.quantization.use_quantization:
        model = prepare_model_for_kbit_training(model)
        print(f"✅ Model prepared for k-bit training")
    
    # Apply LoRA
    if config.lora.use_lora:
        print(f"\n⚙️  Applying LoRA...")
        lora_config = LoraConfig(
            r=config.lora.lora_r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"✅ LoRA applied")
    
    return model, tokenizer


def create_trainer(model, tokenizer, train_dataset, eval_dataset, config: WorkflowConfig):
    """Create Trainer with all configurations and callbacks"""
    print_header("Trainer Setup")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        adam_beta1=config.training.adam_beta1,
        adam_beta2=config.training.adam_beta2,
        adam_epsilon=config.training.adam_epsilon,
        max_grad_norm=config.training.max_grad_norm,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_steps=config.training.warmup_steps,
        fp16=config.training.fp16 and torch.cuda.is_available(),
        bf16=config.training.bf16,
        logging_dir=config.training.logging_dir,
        logging_strategy=config.training.logging_strategy,
        logging_steps=config.training.logging_steps,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        report_to=config.training.report_to,
        seed=config.training.seed,
        dataloader_num_workers=config.training.dataloader_num_workers,
        remove_unused_columns=config.training.remove_unused_columns,
        push_to_hub=config.huggingface.push_to_hub,
        hub_model_id=config.huggingface.hub_model_id if config.huggingface.push_to_hub else None,
        hub_strategy=config.huggingface.hub_strategy if config.huggingface.push_to_hub else None,
        hub_private_repo=config.huggingface.hub_private_repo,
    )
    
    print(f"✅ Training arguments configured")
    print(f"   Output directory: {config.training.output_dir}")
    print(f"   Logging directory: {config.training.logging_dir}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Callbacks
    callbacks = [
        MetricsLoggingCallback(),
        ResourceMonitorCallback(check_interval=100),
        EarlyStoppingCallback(
            early_stopping_patience=config.training.early_stopping_patience,
            early_stopping_threshold=config.training.early_stopping_threshold,
        ),
    ]
    
    # Add QA callback if enabled
    if config.qa.run_qa_tests:
        callbacks.append(
            QualityAssuranceCallback(
                tokenizer=tokenizer,
                test_prompts=config.qa.qa_test_prompts,
                max_new_tokens=config.qa.qa_max_new_tokens,
                temperature=config.qa.qa_temperature,
                top_p=config.qa.qa_top_p,
                do_sample=config.qa.qa_do_sample,
            )
        )
    
    print(f"✅ {len(callbacks)} callbacks registered")
    for cb in callbacks:
        print(f"   - {cb.__class__.__name__}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    print(f"✅ Trainer created")
    
    return trainer


def save_config(config: WorkflowConfig, output_dir: str):
    """Save configuration to file"""
    config_file = os.path.join(output_dir, "training_config.json")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    print(f"✅ Configuration saved to: {config_file}")


def main():
    """Main training function"""
    print("\n" + "="*80)
    print("  dLNk GPT - Enhanced Training Workflow")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Load configuration
    config = WorkflowConfig()
    
    # Setup environment
    setup_environment()
    
    # Save configuration
    save_config(config, config.training.output_dir)
    
    # Load data
    train_dataset, eval_dataset = load_and_prepare_data(config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Create trainer
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, config)
    
    # Start training
    print_header("Training Started")
    print(f"🚀 Beginning training with {config.training.num_train_epochs} epochs...")
    print(f"📊 Effective batch size: {config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps}")
    print(f"📈 Total training steps: {len(train_dataset) // (config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps) * config.training.num_train_epochs}")
    
    try:
        trainer.train()
        
        print_header("Training Completed Successfully")
        
        # Save final model
        print(f"💾 Saving final model...")
        trainer.save_model(os.path.join(config.training.output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(config.training.output_dir, "final_model"))
        print(f"✅ Model saved to: {os.path.join(config.training.output_dir, 'final_model')}")
        
        # Push to hub if configured
        if config.huggingface.push_to_hub:
            print(f"\n📤 Pushing model to Hugging Face Hub...")
            trainer.push_to_hub()
            print(f"✅ Model pushed to: {config.huggingface.hub_model_id}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        print(f"💾 Saving checkpoint...")
        trainer.save_model(os.path.join(config.training.output_dir, "interrupted_checkpoint"))
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    
    print("\n" + "="*80)
    print("  Training workflow completed!")
    print(f"  Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
