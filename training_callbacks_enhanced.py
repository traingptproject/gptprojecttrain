"""
Enhanced Training Callbacks with LINE Integration
"""

import os
import json
import time
import torch
from datetime import datetime
from transformers import TrainerCallback, TrainerState, TrainerControl
from line_monitor import monitor
from training_controller import TrainingController


class LINEIntegratedCallback(TrainerCallback):
    """Callback that integrates with LINE monitoring system"""
    
    def __init__(self, controller: TrainingController):
        self.controller = controller
        self.epoch_start_time = None
        self.step_count = 0
        
    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training"""
        config = {
            'dataset_name': 'dLNk GPT Uncensored Dataset',
            'num_samples': state.max_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps,
            'num_epochs': args.num_train_epochs,
            'batch_size': args.per_device_train_batch_size,
            'learning_rate': args.learning_rate,
        }
        monitor.send_start_notification(config)
    
    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each epoch"""
        self.epoch_start_time = time.time()
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        monitor.send_epoch_start(current_epoch, int(args.num_train_epochs))
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each epoch"""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        # Get metrics
        metrics = {
            'train_loss': state.log_history[-1].get('loss', 0) if state.log_history else 0,
            'eval_loss': state.log_history[-1].get('eval_loss', 0) if state.log_history else 0,
            'learning_rate': state.log_history[-1].get('learning_rate', 0) if state.log_history else 0,
            'epoch_time': epoch_time,
        }
        
        # Send epoch end notification
        monitor.send_epoch_end(current_epoch, int(args.num_train_epochs), metrics)
        
        # Analyze and adjust
        analysis = self.controller.analyze_epoch(current_epoch, metrics)
        
        if analysis.get('stop_training', False):
            monitor.send_early_stopping(
                "Validation loss ไม่ดีขึ้นต่อเนื่อง 3 epochs",
                current_epoch - 3,
                self.controller.best_loss
            )
            control.should_training_stop = True
        
        # Get recommendations
        recommendations = self.controller.get_recommendations()
        if recommendations:
            rec_text = "\n".join(recommendations)
            monitor.send_line_message(f"💡 คำแนะนำ:\n\n{rec_text}")
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Called when logging"""
        if logs and 'loss' in logs:
            self.step_count += 1
            
            # Send periodic updates
            if state.max_steps:
                monitor.send_periodic_update(
                    state.global_step,
                    state.max_steps,
                    logs.get('loss', 0)
                )
    
    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training"""
        total_time = time.time() - monitor.start_time
        
        final_metrics = {
            'best_loss': self.controller.best_loss,
            'total_epochs': int(state.epoch) if state.epoch else 0,
        }
        
        monitor.send_completion(total_time, final_metrics)


class QualityAssuranceLINECallback(TrainerCallback):
    """QA callback with LINE reporting"""
    
    def __init__(self, tokenizer, test_prompts, max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.output_dir = "./training_output_test/qa_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Run QA tests at end of epoch"""
        if model is None:
            return
        
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        
        print(f"\n{'='*80}")
        print(f"  🔍 Quality Assurance Tests - Epoch {current_epoch}")
        print(f"{'='*80}\n")
        
        qa_results = []
        
        for i, prompt in enumerate(self.test_prompts, 1):
            print(f"[Test {i}/{len(self.test_prompts)}] {prompt[:50]}...")
            
            try:
                start_time = time.time()
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=self.do_sample,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generation_time = time.time() - start_time
                
                qa_results.append({
                    'prompt': prompt,
                    'response': response,
                    'generation_time': generation_time,
                })
                
                print(f"✅ Response: {response[:100]}...")
                print(f"⏱️  Time: {generation_time:.2f}s\n")
                
            except Exception as e:
                print(f"❌ Error: {e}\n")
                qa_results.append({
                    'prompt': prompt,
                    'response': f"ERROR: {str(e)}",
                    'generation_time': 0,
                })
        
        # Save results
        result_file = os.path.join(self.output_dir, f"qa_epoch_{current_epoch}.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'epoch': current_epoch,
                'timestamp': datetime.now().isoformat(),
                'tests': qa_results,
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✅ QA results saved to: {result_file}\n")
        
        # Send to LINE
        monitor.send_qa_results(current_epoch, qa_results)


class ResourceMonitorLINECallback(TrainerCallback):
    """Monitor GPU/Memory with LINE alerts"""
    
    def __init__(self, check_interval=100):
        self.check_interval = check_interval
        self.step_count = 0
    
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Check resources periodically"""
        self.step_count += 1
        
        if self.step_count % self.check_interval != 0:
            return
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            usage_pct = (memory_allocated / memory_total) * 100
            
            # Alert if memory usage is high
            if usage_pct > 90:
                monitor.send_warning(
                    'memory_high',
                    f"GPU Memory: {memory_allocated:.2f}GB / {memory_total:.2f}GB ({usage_pct:.1f}%)\n"
                    f"Reserved: {memory_reserved:.2f}GB"
                )


class MetricsLoggingLINECallback(TrainerCallback):
    """Log metrics with LINE integration"""
    
    def __init__(self):
        self.metrics_history = []
        self.output_file = "./training_output_test/metrics_history.json"
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
    
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Save metrics to file"""
        if logs:
            self.metrics_history.append({
                'step': state.global_step,
                'epoch': state.epoch,
                'timestamp': datetime.now().isoformat(),
                **logs
            })
            
            # Save to file
            with open(self.output_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
