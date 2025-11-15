"""
Custom Training Callbacks for dLNk GPT Enhanced Workflow
Includes quality assurance, monitoring, and automated testing
"""

import os
import json
import time
from typing import Dict, List
from datetime import datetime

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers import AutoTokenizer


class QualityAssuranceCallback(TrainerCallback):
    """
    Callback to perform quality assurance tests at the end of each epoch.
    Generates sample outputs from test prompts to verify model behavior.
    """
    
    def __init__(self, tokenizer: AutoTokenizer, test_prompts: List[str], 
                 max_new_tokens: int = 200, temperature: float = 0.7, 
                 top_p: float = 0.9, do_sample: bool = True):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.qa_results = []
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, model=None, **kwargs):
        """Run QA tests at the end of each epoch"""
        if model is None:
            return
        
        print("\n" + "="*80)
        print(f"🔍 Quality Assurance Tests - Epoch {int(state.epoch)}")
        print("="*80)
        
        model.eval()
        epoch_results = {
            "epoch": int(state.epoch),
            "step": state.global_step,
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        for i, prompt in enumerate(self.test_prompts, 1):
            print(f"\n[Test {i}/{len(self.test_prompts)}] Prompt: {prompt[:100]}...")
            
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                # Generate output
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=self.do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                generation_time = time.time() - start_time
                
                # Decode output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the new tokens (remove input prompt)
                response = generated_text[len(prompt):].strip()
                
                print(f"Response: {response[:200]}...")
                print(f"Generation time: {generation_time:.2f}s")
                
                # Store results
                test_result = {
                    "prompt": prompt,
                    "response": response,
                    "generation_time": generation_time,
                    "tokens_generated": len(outputs[0]) - len(inputs["input_ids"][0])
                }
                epoch_results["tests"].append(test_result)
                
            except Exception as e:
                print(f"❌ Error during generation: {e}")
                epoch_results["tests"].append({
                    "prompt": prompt,
                    "error": str(e)
                })
        
        self.qa_results.append(epoch_results)
        
        # Save QA results to file
        qa_output_dir = os.path.join(args.output_dir, "qa_results")
        os.makedirs(qa_output_dir, exist_ok=True)
        qa_file = os.path.join(qa_output_dir, f"epoch_{int(state.epoch)}.json")
        
        with open(qa_file, "w", encoding="utf-8") as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ QA results saved to: {qa_file}")
        print("="*80 + "\n")
        
        model.train()
        return control


class MetricsLoggingCallback(TrainerCallback):
    """
    Enhanced logging callback to track additional metrics and provide
    better visibility into training progress.
    """
    
    def __init__(self):
        self.training_start_time = None
        self.epoch_start_time = None
        self.best_eval_loss = float('inf')
        self.metrics_history = []
        
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, 
                       control: TrainerControl, **kwargs):
        """Record training start time"""
        self.training_start_time = time.time()
        print("\n" + "="*80)
        print("🚀 Training Started")
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Total epochs: {args.num_train_epochs}")
        print(f"📊 Batch size: {args.per_device_train_batch_size} x {args.gradient_accumulation_steps} = {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
        print(f"📈 Learning rate: {args.learning_rate}")
        print("="*80 + "\n")
        
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, 
                       control: TrainerControl, **kwargs):
        """Record epoch start time"""
        self.epoch_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"📖 Epoch {int(state.epoch) + 1}/{args.num_train_epochs} Started")
        print(f"{'='*80}\n")
        
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, **kwargs):
        """Log epoch summary"""
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.training_start_time
        
        print(f"\n{'='*80}")
        print(f"✅ Epoch {int(state.epoch)} Completed")
        print(f"⏱️  Epoch time: {epoch_time/60:.2f} minutes")
        print(f"⏱️  Total time: {total_time/60:.2f} minutes")
        
        # Get latest metrics
        if state.log_history:
            latest_metrics = state.log_history[-1]
            if 'eval_loss' in latest_metrics:
                eval_loss = latest_metrics['eval_loss']
                print(f"📉 Validation Loss: {eval_loss:.4f}")
                
                if eval_loss < self.best_eval_loss:
                    improvement = self.best_eval_loss - eval_loss
                    print(f"🎉 New best validation loss! (improved by {improvement:.4f})")
                    self.best_eval_loss = eval_loss
                else:
                    print(f"⚠️  No improvement (best: {self.best_eval_loss:.4f})")
        
        print(f"{'='*80}\n")
        
    def on_log(self, args: TrainingArguments, state: TrainerState, 
               control: TrainerControl, logs: Dict = None, **kwargs):
        """Log metrics during training"""
        if logs:
            self.metrics_history.append({
                "step": state.global_step,
                "epoch": state.epoch,
                **logs
            })
            
    def on_train_end(self, args: TrainingArguments, state: TrainerState, 
                     control: TrainerControl, **kwargs):
        """Log training completion summary"""
        total_time = time.time() - self.training_start_time
        
        print("\n" + "="*80)
        print("🎉 Training Completed!")
        print(f"⏱️  Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"📊 Total steps: {state.global_step}")
        print(f"📉 Best validation loss: {self.best_eval_loss:.4f}")
        print("="*80 + "\n")
        
        # Save metrics history
        metrics_file = os.path.join(args.output_dir, "metrics_history.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"📊 Metrics history saved to: {metrics_file}\n")


class ResourceMonitorCallback(TrainerCallback):
    """
    Monitor GPU/CPU memory and provide warnings if resources are running low.
    """
    
    def __init__(self, check_interval: int = 100):
        self.check_interval = check_interval
        self.step_count = 0
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, **kwargs):
        """Check resources periodically"""
        self.step_count += 1
        
        if self.step_count % self.check_interval == 0:
            if torch.cuda.is_available():
                # GPU memory
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    
                    usage_percent = (allocated / total) * 100
                    
                    if usage_percent > 90:
                        print(f"\n⚠️  WARNING: GPU {i} memory usage at {usage_percent:.1f}%")
                        print(f"   Allocated: {allocated:.2f}GB / {total:.2f}GB\n")


class CheckpointCleanupCallback(TrainerCallback):
    """
    Clean up old checkpoints to save disk space, keeping only the best ones.
    """
    
    def __init__(self, keep_best_n: int = 3):
        self.keep_best_n = keep_best_n
        
    def on_save(self, args: TrainingArguments, state: TrainerState, 
                control: TrainerControl, **kwargs):
        """Clean up old checkpoints after saving"""
        # This is handled by save_total_limit in TrainingArguments
        # But we can add custom logic here if needed
        pass
