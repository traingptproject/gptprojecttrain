#!/usr/bin/env python3
"""
LINE Notification Helper for Google Colab
Sends real-time training progress to LINE via MCP
"""

import subprocess
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any

class LINENotifier:
    """
    Helper class for sending LINE notifications from Google Colab
    Uses manus-mcp-cli to send messages via LINE MCP server
    """
    
    def __init__(self, user_id: Optional[str] = None):
        """
        Initialize LINE notifier
        
        Args:
            user_id: LINE user ID (optional, uses default if not provided)
        """
        self.user_id = user_id
        self.mcp_cli = "manus-mcp-cli"
        
    def send_text(self, message: str) -> bool:
        """
        Send simple text message to LINE
        
        Args:
            message: Text message to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare MCP command
            payload = {
                "message": {
                    "text": message
                }
            }
            
            if self.user_id:
                payload["userId"] = self.user_id
            
            # Call manus-mcp-cli
            cmd = [
                self.mcp_cli,
                "tool", "call",
                "push_text_message",
                "--server", "line",
                "--input", json.dumps(payload)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"[LINE] ✅ Sent: {message[:50]}...")
                return True
            else:
                print(f"[LINE] ❌ Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[LINE] ❌ Exception: {e}")
            return False
    
    def send_flex(self, alt_text: str, contents: Dict[str, Any]) -> bool:
        """
        Send flex message to LINE (rich formatting)
        
        Args:
            alt_text: Alternative text for notification
            contents: Flex message contents (bubble or carousel)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            payload = {
                "message": {
                    "altText": alt_text,
                    "contents": contents
                }
            }
            
            if self.user_id:
                payload["userId"] = self.user_id
            
            cmd = [
                self.mcp_cli,
                "tool", "call",
                "push_flex_message",
                "--server", "line",
                "--input", json.dumps(payload)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"[LINE FLEX] ✅ Sent: {alt_text}")
                return True
            else:
                print(f"[LINE FLEX] ❌ Error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[LINE FLEX] ❌ Exception: {e}")
            return False
    
    def send_training_start(self, model_name: str, total_steps: int, epochs: int):
        """Send training start notification"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"""🚀 เริ่มการเทรนโมเดล

📊 โมเดล: {model_name}
🔢 Total Steps: {total_steps:,}
📈 Epochs: {epochs}
⏰ เวลาเริ่ม: {timestamp}

💡 ระบบจะรายงานความคืบหน้าทุก 5 นาที
"""
        return self.send_text(message)
    
    def send_training_progress(self, 
                             current_step: int,
                             total_steps: int,
                             loss: float,
                             learning_rate: float,
                             elapsed_time: str,
                             eta: str):
        """Send training progress notification"""
        progress = (current_step / total_steps) * 100
        
        # Create progress bar
        bar_length = 20
        filled = int(bar_length * current_step / total_steps)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        message = f"""📈 ความคืบหน้าการเทรน

{bar} {progress:.1f}%

🔢 Step: {current_step:,}/{total_steps:,}
📉 Loss: {loss:.4f}
⚡ Learning Rate: {learning_rate:.2e}

⏱️ เวลาที่ใช้: {elapsed_time}
🕐 เวลาคงเหลือ: {eta}
"""
        return self.send_text(message)
    
    def send_evaluation_result(self, 
                              step: int,
                              eval_loss: float,
                              train_loss: Optional[float] = None):
        """Send evaluation result notification"""
        message = f"""🎯 ผลการประเมิน

🔢 Step: {step:,}
📉 Eval Loss: {eval_loss:.4f}
"""
        if train_loss is not None:
            message += f"📊 Train Loss: {train_loss:.4f}\n"
            message += f"📈 Difference: {abs(eval_loss - train_loss):.4f}\n"
        
        return self.send_text(message)
    
    def send_checkpoint_saved(self, step: int, checkpoint_dir: str):
        """Send checkpoint saved notification"""
        message = f"""💾 บันทึก Checkpoint

🔢 Step: {step:,}
📁 ตำแหน่ง: {checkpoint_dir}
⏰ เวลา: {datetime.now().strftime("%H:%M:%S")}
"""
        return self.send_text(message)
    
    def send_training_complete(self,
                             total_time: str,
                             final_loss: float,
                             total_steps: int,
                             model_dir: str):
        """Send training completion notification"""
        message = f"""✅ การเทรนเสร็จสมบูรณ์!

⏱️ เวลาทั้งหมด: {total_time}
📉 Final Loss: {final_loss:.4f}
🔢 Total Steps: {total_steps:,}

💾 โมเดลถูกบันทึกที่:
{model_dir}

🎉 พร้อมใช้งาน!
"""
        return self.send_text(message)
    
    def send_error(self, error_message: str, step: Optional[int] = None):
        """Send error notification"""
        message = f"""❌ เกิดข้อผิดพลาด

"""
        if step is not None:
            message += f"🔢 Step: {step:,}\n"
        
        message += f"""⚠️ Error: {error_message}

กรุณาตรวจสอบ Colab notebook
"""
        return self.send_text(message)
    
    def send_system_info(self, 
                        gpu_name: str,
                        gpu_memory: float,
                        cuda_version: str,
                        pytorch_version: str):
        """Send system information"""
        message = f"""💻 ข้อมูลระบบ

🎮 GPU: {gpu_name}
💾 GPU Memory: {gpu_memory:.1f} GB
🔥 CUDA: {cuda_version}
🐍 PyTorch: {pytorch_version}

✅ ระบบพร้อมเทรน
"""
        return self.send_text(message)
    
    def send_dataset_info(self,
                         train_samples: int,
                         val_samples: int,
                         max_length: int):
        """Send dataset information"""
        message = f"""📊 ข้อมูลชุดเทรนนิ่ง

📚 Training: {train_samples:,} samples
📖 Validation: {val_samples:,} samples
📏 Max Length: {max_length} tokens

✅ ข้อมูลพร้อมแล้ว
"""
        return self.send_text(message)

# Example usage
if __name__ == "__main__":
    notifier = LINENotifier()
    
    # Test notifications
    notifier.send_text("🧪 ทดสอบ LINE Notifier")
    
    notifier.send_system_info(
        gpu_name="Tesla T4",
        gpu_memory=15.0,
        cuda_version="11.8",
        pytorch_version="2.0.1"
    )
    
    notifier.send_dataset_info(
        train_samples=2700,
        val_samples=300,
        max_length=2048
    )
    
    notifier.send_training_start(
        model_name="dLNk GPT V2",
        total_steps=1000,
        epochs=3
    )
    
    notifier.send_training_progress(
        current_step=500,
        total_steps=1000,
        loss=1.234,
        learning_rate=5e-6,
        elapsed_time="01:23:45",
        eta="01:20:00"
    )
    
    notifier.send_evaluation_result(
        step=500,
        eval_loss=1.456,
        train_loss=1.234
    )
    
    notifier.send_checkpoint_saved(
        step=500,
        checkpoint_dir="./checkpoints/step-500"
    )
    
    notifier.send_training_complete(
        total_time="02:45:30",
        final_loss=0.987,
        total_steps=1000,
        model_dir="./dLNk-gpt-j-6b-exploit-v2"
    )
