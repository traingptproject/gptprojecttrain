"""
Monitored Test Training Script with LINE Integration
Runs 2-epoch test with real-time LINE updates
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


def load_data():
    print_header("Loading Dataset")
    
    try:
        # For testing, create a small synthetic dataset
        print("📥 Creating test dataset (1000 samples)...")
        
        # You can replace this with actual dataset loading
        from datasets import Dataset
        
        # Create synthetic data for testing
        data = {
            'text': [
                f"This is a test sample number {i}. " * 20
                for i in range(1000)
            ]
        }
        
        dataset = Dataset.from_dict(data)
        
        # Split into train/eval
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
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
        model_name = "EleutherAI/gpt-j-6b"
        
        print(f"📥 Loading tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
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
        print(f"\n📥 Loading model: {model_name}")
        print(f"   This may take several minutes...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
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


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the examples"""
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=None,
    )


def create_trainer(model, tokenizer, train_dataset, eval_dataset, controller):
    print_header("Trainer Setup")
    
    try:
        # Tokenize datasets
        print("📝 Tokenizing datasets...")
        train_dataset = train_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        eval_dataset = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names,
        )
        print("✅ Tokenization complete")
        
        output_dir = "./training_output_test"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,  # 2 epochs for testing
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            fp16=True,
            logging_dir="./logs_test",
            logging_strategy="steps",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",  # Disable default reporting
            seed=42,
            dataloader_num_workers=2,
            remove_unused_columns=False,
        )
        
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
    print("  dLNk GPT - Monitored Test Training (2 Epochs)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # Setup
        setup_environment()
        
        # Create controller
        config = {
            'learning_rate': 2e-5,
            'num_epochs': 2,
        }
        controller = TrainingController(config)
        
        # Load data and model
        train_dataset, eval_dataset = load_data()
        model, tokenizer = setup_model_and_tokenizer()
        
        # Create trainer
        trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, controller)
        
        print_header("Starting Training")
        print(f"🧪 Test Mode: 2 epochs")
        print(f"📊 Training samples: {len(train_dataset):,}")
        print(f"📊 Validation samples: {len(eval_dataset):,}")
        print(f"\n✅ LINE monitoring active")
        print(f"📱 You will receive real-time updates via LINE\n")
        
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
        print("  ✅ TEST PASSED - Workflow is working correctly!")
        print("  📱 Check your LINE for detailed reports")
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
