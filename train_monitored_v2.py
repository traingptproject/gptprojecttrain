"""
Monitored Training Script V2 with Regularization
Full training with improved workflow to prevent overfitting
"""

import os
import sys
import json
import torch
import traceback
from datetime import datetime

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import our monitoring system
from line_monitor import monitor
from training_controller import TrainingController
from training_callbacks_enhanced import (
    LINEIntegratedCallback,
    QualityAssuranceLINECallback,
    ResourceMonitorLINECallback,
    MetricsLoggingLINECallback,
)

# Import V2 configuration
from training_config_v2 import (
    MODEL_NAME,
    OUTPUT_DIR,
    LORA_CONFIG,
    TRAINING_ARGS,
    EARLY_STOPPING,
    DATASET_CONFIG,
    QA_TESTS,
    MONITORING_CONFIG,
)


def print_header(text: str):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def setup_environment():
    print_header("Environment Setup")
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        error_msg = "❌ No GPU detected!"
        print(error_msg)
        monitor.send_error(error_msg, "GPU is required for training")
        sys.exit(1)


def load_data(tokenizer):
    print_header("Loading Dataset")
    
    try:
        print(f"📥 Loading dataset: {DATASET_CONFIG['train_file']}")
        
        # Load JSONL dataset
        dataset = load_dataset('json', data_files=DATASET_CONFIG['train_file'])
        dataset = dataset['train']
        
        # Limit samples
        train_samples = DATASET_CONFIG['train_samples']
        eval_samples = DATASET_CONFIG['eval_samples']
        total_samples = train_samples + eval_samples
        
        if len(dataset) > total_samples:
            dataset = dataset.select(range(total_samples))
        
        # Split into train/eval
        split_dataset = dataset.train_test_split(
            test_size=DATASET_CONFIG['test_size'], 
            seed=42
        )
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
        # Tokenize
        print(f"\n📝 Tokenizing datasets...")
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                max_length=DATASET_CONFIG['max_length'],
                padding=False,
            )
        
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train",
        )
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing eval",
        )
        
        print(f"✅ Training samples: {len(train_dataset):,}")
        print(f"✅ Validation samples: {len(eval_dataset):,}")
        
        return train_dataset, eval_dataset
        
    except Exception as e:
        error_msg = f"Failed to load dataset: {e}"
        print(f"❌ {error_msg}")
        monitor.send_error(error_msg, traceback.format_exc())
        raise


def setup_model_and_tokenizer():
    print_header("Model Setup")
    
    try:
        print(f"📥 Loading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer):,})")
        
        # Quantization config
        print(f"\n⚙️  Setting up 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model
        print(f"\n📥 Loading model: {MODEL_NAME}")
        print(f"   This may take several minutes...")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )
        
        print(f"✅ Model loaded")
        
        model = prepare_model_for_kbit_training(model)
        print(f"✅ Model prepared for training")
        
        # Apply LoRA
        print(f"\n⚙️  Applying LoRA...")
        lora_config = LoraConfig(
            r=LORA_CONFIG['r'],
            lora_alpha=LORA_CONFIG['lora_alpha'],
            target_modules=LORA_CONFIG['target_modules'],
            lora_dropout=LORA_CONFIG['lora_dropout'],
            bias=LORA_CONFIG['bias'],
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(f"✅ LoRA applied")
        
        return model, tokenizer
        
    except Exception as e:
        error_msg = f"Failed to setup model: {e}"
        print(f"❌ {error_msg}")
        monitor.send_error(error_msg, traceback.format_exc())
        raise



def create_trainer(model, tokenizer, train_dataset, eval_dataset, controller):
    print_header("Trainer Setup")
    
    try:
        # Data already tokenized in load_data()
        
        # Use TRAINING_ARGS from config_v2
        training_args = TrainingArguments(**TRAINING_ARGS)
        
        print(f"✅ Training arguments configured")
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Create callbacks with LINE integration
        callbacks = [
            LINEIntegratedCallback(controller),
            MetricsLoggingLINECallback(),
            ResourceMonitorLINECallback(check_interval=50),
            QualityAssuranceLINECallback(
                tokenizer=tokenizer,
                test_prompts=[
                    "Explain quantum computing:",
                    "Write a Python function:",
                    "What is the meaning of life?",
                ],
                max_new_tokens=100,
            ),
        ]
        
        print(f"✅ {len(callbacks)} callbacks registered (with LINE integration)")
        
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
        
    except Exception as e:
        error_msg = f"Failed to create trainer: {e}"
        print(f"❌ {error_msg}")
        monitor.send_error(error_msg, traceback.format_exc())
        raise


def main():
    print("\n" + "="*80)
    print("  dLNk GPT - Training V2 with Enhanced Regularization")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # Setup
        setup_environment()
        
        # Setup model and tokenizer first
        model, tokenizer = setup_model_and_tokenizer()
        
        # Load and tokenize data
        train_dataset, eval_dataset = load_data(tokenizer)
        
        # Create controller with V2 config
        config = {
            'learning_rate': TRAINING_ARGS['learning_rate'],
            'num_epochs': TRAINING_ARGS['num_train_epochs'],
        }
        controller = TrainingController(config)
        
        # Create trainer
        trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, controller)
        
        print_header("Starting Training V2")
        print(f"🚀 Full Training: {TRAINING_ARGS['num_train_epochs']} epochs")
        print(f"📊 Training samples: {len(train_dataset):,}")
        print(f"📊 Validation samples: {len(eval_dataset):,}")
        print(f"📉 Learning Rate: {TRAINING_ARGS['learning_rate']}")
        print(f"⚖️  Weight Decay: {TRAINING_ARGS['weight_decay']}")
        print(f"\n✅ Monitoring active (console output)\n")
        
        # Train
        trainer.train()
        
        print_header("Training Completed!")
        
        # Save model
        print(f"💾 Saving model...")
        output_path = os.path.join(trainer.args.output_dir, "final_model")
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"✅ Model saved to: {output_path}")
        
        print("\n" + "="*80)
        print("  ✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("  🎯 Model ready for deployment")
        print("="*80 + "\n")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        monitor.send_line_message("⚠️ การเทรนถูกหยุดโดยผู้ใช้")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        monitor.send_error(str(e), traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
