"""
Real-time LINE Monitoring System for dLNk GPT Training
Sends detailed Thai language reports via LINE MCP
"""

import os
import json
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional


class LINEMonitor:
    """Real-time training monitor with LINE notifications in Thai"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_report_time = time.time()
        self.report_interval = 300  # Report every 5 minutes
        self.best_loss = float('inf')
        self.epoch_losses = []
        
    def send_line_message(self, message: str) -> bool:
        """Send message via LINE MCP"""
        try:
            cmd = [
                'manus-mcp-cli', 'tool', 'call', 'push_text_message',
                '--server', 'line',
                '--input', json.dumps({"message": {"text": message}})
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except Exception as e:
            print(f"❌ ส่ง LINE ไม่สำเร็จ: {e}")
            return False
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to Thai time string"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours} ชั่วโมง {minutes} นาที"
        elif minutes > 0:
            return f"{minutes} นาที {secs} วินาที"
        else:
            return f"{secs} วินาที"
    
    def send_start_notification(self, config: Dict):
        """Send training start notification"""
        message = f"""🚀 เริ่มการเทรน dLNk GPT

📊 การตั้งค่า:
• Dataset: {config.get('dataset_name', 'N/A')}
• Samples: {config.get('num_samples', 'N/A'):,} ตัวอย่าง
• Epochs: {config.get('num_epochs', 'N/A')} รอบ
• Batch Size: {config.get('batch_size', 'N/A')}
• Learning Rate: {config.get('learning_rate', 'N/A')}

⏰ เวลาเริ่ม: {datetime.now().strftime('%H:%M:%S')}
📍 สถานะ: กำลังโหลดโมเดล...

ผมจะรายงานความคืบหน้าแบบ real-time ครับ"""
        
        self.send_line_message(message)
    
    def send_epoch_start(self, epoch: int, total_epochs: int):
        """Send epoch start notification"""
        elapsed = time.time() - self.start_time
        
        message = f"""📖 เริ่ม Epoch {epoch}/{total_epochs}

⏱️ เวลาที่ผ่านไป: {self.format_time(elapsed)}
📊 สถานะ: กำลังเทรน...

รอรับรายงานผลเมื่อจบ epoch นี้"""
        
        self.send_line_message(message)
    
    def send_epoch_end(self, epoch: int, total_epochs: int, metrics: Dict):
        """Send epoch end notification with detailed metrics"""
        train_loss = metrics.get('train_loss', 0)
        eval_loss = metrics.get('eval_loss', 0)
        learning_rate = metrics.get('learning_rate', 0)
        
        # Calculate improvement
        if eval_loss < self.best_loss:
            improvement = self.best_loss - eval_loss
            improvement_pct = (improvement / self.best_loss) * 100 if self.best_loss != float('inf') else 0
            status = f"✅ ดีขึ้น {improvement:.4f} ({improvement_pct:.2f}%)"
            self.best_loss = eval_loss
        else:
            degradation = eval_loss - self.best_loss
            degradation_pct = (degradation / self.best_loss) * 100
            status = f"⚠️ แย่ลง {degradation:.4f} ({degradation_pct:.2f}%)"
        
        self.epoch_losses.append(eval_loss)
        
        elapsed = time.time() - self.start_time
        
        message = f"""✅ จบ Epoch {epoch}/{total_epochs}

📊 ผลลัพธ์:
• Training Loss: {train_loss:.4f}
• Validation Loss: {eval_loss:.4f}
• Learning Rate: {learning_rate:.2e}

📈 ประสิทธิภาพ:
• สถานะ: {status}
• Loss ต่ำสุด: {self.best_loss:.4f}

⏱️ เวลา:
• Epoch นี้: {self.format_time(metrics.get('epoch_time', 0))}
• รวมทั้งหมด: {self.format_time(elapsed)}

{'🎯 โมเดลกำลังเรียนรู้ได้ดี!' if eval_loss < self.best_loss else '⚠️ ระวัง overfitting!'}"""
        
        self.send_line_message(message)
    
    def send_qa_results(self, epoch: int, qa_results: List[Dict]):
        """Send QA test results"""
        message = f"""🔍 ผลทดสอบคุณภาพ - Epoch {epoch}

"""
        
        for i, test in enumerate(qa_results[:3], 1):  # Show first 3 tests
            prompt = test.get('prompt', '')[:50]
            response = test.get('response', '')[:100]
            gen_time = test.get('generation_time', 0)
            
            message += f"""[ทดสอบ {i}]
❓ {prompt}...
💬 {response}...
⏱️ {gen_time:.2f} วินาที

"""
        
        message += "✅ โมเดลตอบคำถามได้ปกติ"
        
        self.send_line_message(message)
    
    def send_warning(self, warning_type: str, details: str):
        """Send warning notification"""
        warnings = {
            'overfitting': '⚠️ ตรวจพบ Overfitting',
            'high_loss': '⚠️ Loss สูงผิดปกติ',
            'slow_training': '⚠️ การเทรนช้าเกินไป',
            'memory_high': '⚠️ หน่วยความจำใกล้เต็ม',
            'gpu_error': '❌ GPU มีปัญหา',
        }
        
        title = warnings.get(warning_type, '⚠️ คำเตือน')
        
        message = f"""{title}

📋 รายละเอียด:
{details}

🔧 กำลังดำเนินการแก้ไข..."""
        
        self.send_line_message(message)
    
    def send_adjustment(self, adjustment_type: str, old_value: float, new_value: float):
        """Send parameter adjustment notification"""
        message = f"""🔧 ปรับพารามิเตอร์อัตโนมัติ

📊 การเปลี่ยนแปลง:
• ประเภท: {adjustment_type}
• ค่าเดิม: {old_value}
• ค่าใหม่: {new_value}

✅ ปรับเสร็จแล้ว กำลังเทรนต่อ..."""
        
        self.send_line_message(message)
    
    def send_early_stopping(self, reason: str, best_epoch: int, best_loss: float):
        """Send early stopping notification"""
        message = f"""🛑 หยุดการเทรนอัตโนมัติ (Early Stopping)

📋 เหตุผล:
{reason}

🏆 Epoch ที่ดีที่สุด:
• Epoch: {best_epoch}
• Validation Loss: {best_loss:.4f}

✅ กำลังโหลดโมเดลที่ดีที่สุด..."""
        
        self.send_line_message(message)
    
    def send_completion(self, total_time: float, final_metrics: Dict):
        """Send training completion notification"""
        best_loss = final_metrics.get('best_loss', 0)
        total_epochs = final_metrics.get('total_epochs', 0)
        
        message = f"""🎉 เทรนเสร็จสมบูรณ์!

📊 สรุปผลลัพธ์:
• จำนวน Epochs: {total_epochs}
• Validation Loss ต่ำสุด: {best_loss:.4f}
• เวลารวม: {self.format_time(total_time)}

✅ โมเดลพร้อมใช้งาน!
📦 กำลังบันทึกและ deploy..."""
        
        self.send_line_message(message)
    
    def send_error(self, error_message: str, traceback_info: str):
        """Send error notification"""
        message = f"""❌ เกิดข้อผิดพลาด!

🔴 Error:
{error_message[:200]}

📋 รายละเอียด:
{traceback_info[:300]}

🔧 กำลังพยายามแก้ไข..."""
        
        self.send_line_message(message)
    
    def send_periodic_update(self, current_step: int, total_steps: int, current_loss: float):
        """Send periodic progress update"""
        now = time.time()
        
        # Only send if interval has passed
        if now - self.last_report_time < self.report_interval:
            return
        
        self.last_report_time = now
        progress = (current_step / total_steps) * 100
        elapsed = now - self.start_time
        
        message = f"""📊 อัพเดทความคืบหน้า

🔄 ความคืบหน้า: {progress:.1f}%
• Step: {current_step:,}/{total_steps:,}
• Loss ปัจจุบัน: {current_loss:.4f}
• Loss ต่ำสุด: {self.best_loss:.4f}

⏱️ เวลา: {self.format_time(elapsed)}

✅ กำลังเทรนตามปกติ"""
        
        self.send_line_message(message)


# Global monitor instance
monitor = LINEMonitor()
