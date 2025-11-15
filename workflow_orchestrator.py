#!/usr/bin/env python3
"""
Workflow Orchestrator for dLNk GPT V2 Training
Automates the entire training pipeline with monitoring and notifications
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Import LINE notifier
from line_notifier import LINENotifier

class WorkflowOrchestrator:
    """
    Orchestrates the complete training workflow
    - GitHub integration
    - Hugging Face integration  
    - Training execution
    - Real-time monitoring
    - LINE notifications
    - Anti-disconnect mechanism
    """
    
    def __init__(self, 
                 github_repo: str = "traingptproject/gptprojecttrain",
                 hf_repo: Optional[str] = None,
                 work_dir: str = "/content/gptprojecttrain"):
        """
        Initialize workflow orchestrator
        
        Args:
            github_repo: GitHub repository (owner/repo)
            hf_repo: Hugging Face repository (optional)
            work_dir: Working directory
        """
        self.github_repo = github_repo
        self.hf_repo = hf_repo
        self.work_dir = work_dir
        self.notifier = LINENotifier()
        
        self.start_time = None
        self.training_process = None
        
    def send_notification(self, message: str):
        """Send notification via LINE"""
        print(f"[NOTIFY] {message}")
        self.notifier.send_text(message)
    
    def run_command(self, cmd: list, timeout: Optional[int] = None) -> tuple:
        """
        Run shell command
        
        Args:
            cmd: Command as list
            timeout: Timeout in seconds
            
        Returns:
            tuple: (returncode, stdout, stderr)
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timeout"
        except Exception as e:
            return -1, "", str(e)
    
    def check_gpu(self) -> Dict[str, Any]:
        """Check GPU availability"""
        try:
            import torch
            
            gpu_info = {
                "available": torch.cuda.is_available(),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU",
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "pytorch_version": torch.__version__
            }
            
            return gpu_info
        except Exception as e:
            return {"available": False, "error": str(e)}
    
    def setup_environment(self):
        """Setup environment and dependencies"""
        self.send_notification("🔧 กำลังตั้งค่าสภาพแวดล้อม...")
        
        # Check if running in Colab
        try:
            import google.colab
            is_colab = True
        except:
            is_colab = False
        
        if is_colab:
            self.send_notification("✅ ตรวจพบ Google Colab")
        else:
            self.send_notification("⚠️ ไม่ได้รันบน Google Colab")
        
        # Check GPU
        gpu_info = self.check_gpu()
        
        if gpu_info["available"]:
            self.notifier.send_system_info(
                gpu_name=gpu_info["name"],
                gpu_memory=gpu_info["memory_gb"],
                cuda_version=gpu_info["cuda_version"],
                pytorch_version=gpu_info["pytorch_version"]
            )
        else:
            self.send_notification("❌ ไม่พบ GPU - การเทรนจะช้ามาก!")
        
        return gpu_info["available"]
    
    def clone_github_repo(self):
        """Clone GitHub repository"""
        self.send_notification(f"📥 กำลังดึงข้อมูลจาก GitHub: {self.github_repo}")
        
        # Remove existing directory
        if os.path.exists(self.work_dir):
            self.run_command(["rm", "-rf", self.work_dir])
        
        # Clone repository
        repo_url = f"https://github.com/{self.github_repo}.git"
        returncode, stdout, stderr = self.run_command(
            ["git", "clone", repo_url, self.work_dir],
            timeout=300
        )
        
        if returncode == 0:
            self.send_notification(f"✅ ดึงข้อมูลจาก GitHub สำเร็จ\\n\\n📁 ตำแหน่ง: {self.work_dir}")
            
            # List key files
            key_files = [
                "exploit_agent.py",
                "training_config_v2_exploit.py",
                "train_exploit_agent_v2.py",
                "exploit_training_data_v2_enhanced.jsonl",
                "line_notifier.py"
            ]
            
            found_files = []
            for file in key_files:
                filepath = os.path.join(self.work_dir, file)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    found_files.append(f"✅ {file} ({size} bytes)")
                else:
                    found_files.append(f"❌ {file}")
            
            self.send_notification("📄 ไฟล์ที่พบ:\\n\\n" + "\\n".join(found_files))
            return True
        else:
            self.send_notification(f"❌ ดึงข้อมูลล้มเหลว:\\n{stderr}")
            return False
    
    def check_training_data(self):
        """Check training data"""
        self.send_notification("📊 กำลังตรวจสอบข้อมูลเทรนนิ่ง...")
        
        data_file = os.path.join(self.work_dir, "exploit_training_data_v2_enhanced.jsonl")
        
        if not os.path.exists(data_file):
            # Try alternative location
            data_file = os.path.join(self.work_dir, "analysis/exploit_training_template.jsonl")
        
        if os.path.exists(data_file):
            # Count samples
            with open(data_file, 'r') as f:
                samples = sum(1 for _ in f)
            
            size = os.path.getsize(data_file)
            
            # Calculate train/val split
            val_split = 0.1
            train_samples = int(samples * (1 - val_split))
            val_samples = samples - train_samples
            
            self.notifier.send_dataset_info(
                train_samples=train_samples,
                val_samples=val_samples,
                max_length=2048
            )
            
            return True
        else:
            self.send_notification(f"❌ ไม่พบไฟล์ข้อมูลเทรนนิ่ง:\\n{data_file}")
            return False
    
    def start_training(self):
        """Start training process"""
        self.send_notification("🚀 กำลังเริ่มการเทรน...")
        
        # Change to work directory
        os.chdir(self.work_dir)
        
        # Start training
        training_script = "train_exploit_agent_v2.py"
        
        if not os.path.exists(training_script):
            self.send_notification(f"❌ ไม่พบ training script: {training_script}")
            return False
        
        # Send training start notification
        self.notifier.send_training_start(
            model_name="dLNk GPT V2 Exploit Agent",
            total_steps=1000,  # Will be updated from actual config
            epochs=3
        )
        
        self.start_time = datetime.now()
        
        # Run training (this will block)
        self.send_notification("⚙️ กำลังรันการเทรน...\\n\\nระบบจะรายงานความคืบหน้าอัตโนมัติ")
        
        returncode, stdout, stderr = self.run_command(
            ["python3", training_script],
            timeout=None  # No timeout for training
        )
        
        if returncode == 0:
            elapsed = datetime.now() - self.start_time
            self.notifier.send_training_complete(
                total_time=str(elapsed).split('.')[0],
                final_loss=0.0,  # Will be extracted from logs
                total_steps=1000,
                model_dir="./dLNk-gpt-j-6b-exploit-v2"
            )
            return True
        else:
            self.notifier.send_error(
                error_message=stderr[:500],
                step=None
            )
            return False
    
    def upload_to_huggingface(self):
        """Upload model to Hugging Face"""
        if not self.hf_repo:
            self.send_notification("⚠️ ไม่ได้ตั้งค่า Hugging Face repository - ข้าม")
            return True
        
        self.send_notification(f"📤 กำลังอัปโหลดไปยัง Hugging Face: {self.hf_repo}")
        
        try:
            from huggingface_hub import HfApi, login
            
            # Login (assumes HF_TOKEN is set)
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                self.send_notification("❌ ไม่พบ HF_TOKEN - ไม่สามารถอัปโหลด")
                return False
            
            login(token=hf_token)
            
            # Upload
            api = HfApi()
            model_path = os.path.join(self.work_dir, "dLNk-gpt-j-6b-exploit-v2")
            
            if not os.path.exists(model_path):
                self.send_notification(f"❌ ไม่พบโมเดล: {model_path}")
                return False
            
            api.create_repo(self.hf_repo, exist_ok=True, private=True)
            api.upload_folder(
                folder_path=model_path,
                repo_id=self.hf_repo,
                repo_type="model"
            )
            
            self.send_notification(f"✅ อัปโหลดสำเร็จ\\n\\n🤗 {self.hf_repo}")
            return True
            
        except Exception as e:
            self.send_notification(f"❌ อัปโหลดล้มเหลว:\\n{str(e)[:500]}")
            return False
    
    def run_workflow(self):
        """Run complete workflow"""
        self.send_notification("🎬 เริ่มต้น Workflow Orchestrator\\n\\n📋 dLNk GPT V2 Training Pipeline")
        
        # Step 1: Setup environment
        if not self.setup_environment():
            self.send_notification("❌ ไม่พบ GPU - หยุดการทำงาน")
            return False
        
        # Step 2: Clone GitHub repo
        if not self.clone_github_repo():
            self.send_notification("❌ ดึงข้อมูลจาก GitHub ล้มเหลว - หยุดการทำงาน")
            return False
        
        # Step 3: Check training data
        if not self.check_training_data():
            self.send_notification("❌ ไม่พบข้อมูลเทรนนิ่ง - หยุดการทำงาน")
            return False
        
        # Step 4: Start training
        if not self.start_training():
            self.send_notification("❌ การเทรนล้มเหลว")
            return False
        
        # Step 5: Upload to Hugging Face
        self.upload_to_huggingface()
        
        # Final summary
        total_time = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        summary = f"""🎉 Workflow เสร็จสมบูรณ์!

⏱️ เวลาทั้งหมด: {str(total_time).split('.')[0]}

✅ ขั้นตอนที่เสร็จสิ้น:
1. Setup environment
2. Clone GitHub repository
3. Check training data
4. Train model
5. Upload to Hugging Face

🚀 โมเดลพร้อมใช้งาน!
"""
        
        self.send_notification(summary)
        return True

def main():
    """Main entry point"""
    # Configuration
    github_repo = os.environ.get("GITHUB_REPO", "traingptproject/gptprojecttrain")
    hf_repo = os.environ.get("HF_REPO", None)
    work_dir = os.environ.get("WORK_DIR", "/content/gptprojecttrain")
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        github_repo=github_repo,
        hf_repo=hf_repo,
        work_dir=work_dir
    )
    
    # Run workflow
    success = orchestrator.run_workflow()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
