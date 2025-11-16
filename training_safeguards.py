"""
ระบบป้องกันปัญหาการเทรนครบวงจร
- ป้องกัน Overfitting
- ตรวจจับ Model ไม่เรียนรู้
- ตรวจสอบข้อมูลขัดแย้ง
- Auto-adjustment สำหรับ hyperparameters
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from collections import defaultdict
import warnings

class TrainingSafeguards:
    """ระบบป้องกันปัญหาการเทรนแบบครบวงจร"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.warnings = []
        
    def check_overfitting(self, train_loss: float, eval_loss: float) -> Tuple[bool, str]:
        """
        ตรวจจับ Overfitting
        
        Returns:
            (is_overfitting, message)
        """
        self.train_losses.append(train_loss)
        self.eval_losses.append(eval_loss)
        
        if len(self.train_losses) < 3:
            return False, "ข้อมูลยังไม่เพียงพอสำหรับการตรวจจับ"
        
        # ตรวจสอบว่า train loss ลดลง แต่ eval loss เพิ่มขึ้น
        train_trend = np.mean(np.diff(self.train_losses[-3:]))
        eval_trend = np.mean(np.diff(self.eval_losses[-3:]))
        
        # Gap ระหว่าง train และ eval loss
        gap = eval_loss - train_loss
        gap_threshold = 0.5  # ถ้า gap มากกว่า 0.5 ถือว่า overfitting
        
        is_overfitting = False
        message = ""
        
        if train_trend < 0 and eval_trend > 0:
            is_overfitting = True
            message = "⚠️ ตรวจพบ Overfitting: Train loss ลดลง แต่ Eval loss เพิ่มขึ้น"
        elif gap > gap_threshold:
            is_overfitting = True
            message = f"⚠️ ตรวจพบ Overfitting: Gap ระหว่าง train และ eval = {gap:.4f} (มากกว่า {gap_threshold})"
        else:
            message = f"✅ ไม่พบ Overfitting (Gap = {gap:.4f})"
        
        if is_overfitting:
            self.warnings.append({
                'type': 'overfitting',
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'gap': gap,
                'message': message
            })
        
        return is_overfitting, message
    
    def check_learning_progress(self, current_loss: float) -> Tuple[bool, str]:
        """
        ตรวจสอบว่าโมเดลกำลังเรียนรู้หรือไม่
        
        Returns:
            (is_learning, message)
        """
        if len(self.train_losses) < 2:
            return True, "เริ่มต้นการเทรน"
        
        recent_losses = self.train_losses[-5:]  # ดู 5 epochs ล่าสุด
        
        # ตรวจสอบว่า loss ลดลงหรือไม่
        if len(recent_losses) >= 2:
            improvement = recent_losses[0] - recent_losses[-1]
            
            if improvement < self.min_delta:
                self.patience_counter += 1
                message = f"⚠️ Loss ไม่ลดลง ({self.patience_counter}/{self.patience})"
                
                if self.patience_counter >= self.patience:
                    self.warnings.append({
                        'type': 'no_learning',
                        'recent_losses': recent_losses,
                        'message': "โมเดลหยุดเรียนรู้"
                    })
                    return False, "❌ โมเดลหยุดเรียนรู้! แนะนำให้ปรับ learning rate หรือหยุดการเทรน"
                
                return True, message
            else:
                self.patience_counter = 0
                return True, f"✅ โมเดลกำลังเรียนรู้ (ลดลง {improvement:.4f})"
        
        return True, "กำลังเทรน..."
    
    def check_data_conflicts(self, dataset: List[Dict]) -> Tuple[bool, List[str]]:
        """
        ตรวจสอบข้อมูลที่ขัดแย้งกัน
        
        Args:
            dataset: รายการของ dict ที่มี 'instruction', 'input', 'output'
        
        Returns:
            (has_conflicts, conflict_messages)
        """
        conflicts = []
        instruction_outputs = defaultdict(set)
        
        # รวบรวม outputs สำหรับแต่ละ instruction+input
        for idx, item in enumerate(dataset):
            key = f"{item.get('instruction', '')}|||{item.get('input', '')}"
            output = item.get('output', '')
            instruction_outputs[key].add(output)
        
        # ตรวจสอบว่ามี instruction+input เดียวกันที่มี output ต่างกันหรือไม่
        for key, outputs in instruction_outputs.items():
            if len(outputs) > 1:
                instruction, inp = key.split('|||')
                conflicts.append(
                    f"⚠️ ข้อมูลขัดแย้ง: '{instruction[:50]}...' มี {len(outputs)} outputs ที่แตกต่างกัน"
                )
        
        if conflicts:
            self.warnings.append({
                'type': 'data_conflicts',
                'count': len(conflicts),
                'examples': conflicts[:5]  # เก็บแค่ 5 ตัวอย่างแรก
            })
        
        return len(conflicts) > 0, conflicts
    
    def suggest_learning_rate(self, current_lr: float, current_loss: float) -> Tuple[float, str]:
        """
        แนะนำ learning rate ที่เหมาะสม
        
        Returns:
            (suggested_lr, reason)
        """
        self.learning_rates.append(current_lr)
        
        if len(self.train_losses) < 3:
            return current_lr, "ข้อมูลยังไม่เพียงพอ"
        
        recent_losses = self.train_losses[-3:]
        loss_change = recent_losses[-1] - recent_losses[0]
        
        # ถ้า loss ไม่ลดลงเลย -> ลด learning rate
        if loss_change >= 0:
            new_lr = current_lr * 0.5
            return new_lr, f"Loss ไม่ลดลง -> ลด LR เหลือ {new_lr:.2e}"
        
        # ถ้า loss ลดลงเร็วมาก -> อาจเพิ่ม learning rate ได้
        if loss_change < -0.5:
            new_lr = min(current_lr * 1.2, 5e-5)  # ไม่เกิน 5e-5
            return new_lr, f"Loss ลดลงเร็ว -> เพิ่ม LR เป็น {new_lr:.2e}"
        
        return current_lr, "Learning rate เหมาะสมแล้ว"
    
    def should_stop_training(self) -> Tuple[bool, str]:
        """
        ตัดสินใจว่าควรหยุดการเทรนหรือไม่
        
        Returns:
            (should_stop, reason)
        """
        if len(self.eval_losses) < self.patience:
            return False, "ยังเทรนไม่พอ"
        
        # ตรวจสอบว่า eval loss ไม่ดีขึ้นในช่วง patience epochs
        recent_eval = self.eval_losses[-self.patience:]
        if min(recent_eval) >= self.best_eval_loss - self.min_delta:
            return True, f"Eval loss ไม่ดีขึ้นใน {self.patience} epochs ล่าสุด"
        
        # อัปเดต best eval loss
        if self.eval_losses[-1] < self.best_eval_loss:
            self.best_eval_loss = self.eval_losses[-1]
        
        return False, "ยังเทรนต่อได้"
    
    def get_training_summary(self) -> Dict:
        """
        สรุปผลการเทรนและคำเตือนทั้งหมด
        
        Returns:
            dict ที่มีสรุปข้อมูล
        """
        summary = {
            'total_epochs': len(self.train_losses),
            'best_train_loss': min(self.train_losses) if self.train_losses else None,
            'best_eval_loss': self.best_eval_loss if self.best_eval_loss != float('inf') else None,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_eval_loss': self.eval_losses[-1] if self.eval_losses else None,
            'warnings': self.warnings,
            'warning_count': len(self.warnings)
        }
        
        return summary
    
    def generate_report(self) -> str:
        """
        สร้างรายงานสรุปเป็นข้อความ
        
        Returns:
            รายงานในรูปแบบข้อความ
        """
        summary = self.get_training_summary()
        
        report = "=" * 60 + "\n"
        report += "📊 สรุปผลการเทรน\n"
        report += "=" * 60 + "\n\n"
        
        report += f"จำนวน Epochs: {summary['total_epochs']}\n"
        report += f"Train Loss ที่ดีที่สุด: {summary['best_train_loss']:.4f}\n" if summary['best_train_loss'] else ""
        report += f"Eval Loss ที่ดีที่สุด: {summary['best_eval_loss']:.4f}\n" if summary['best_eval_loss'] else ""
        report += f"Train Loss สุดท้าย: {summary['final_train_loss']:.4f}\n" if summary['final_train_loss'] else ""
        report += f"Eval Loss สุดท้าย: {summary['final_eval_loss']:.4f}\n" if summary['final_eval_loss'] else ""
        
        report += f"\n⚠️ คำเตือนทั้งหมด: {summary['warning_count']} รายการ\n"
        
        if summary['warnings']:
            report += "\nรายละเอียดคำเตือน:\n"
            for i, warning in enumerate(summary['warnings'], 1):
                report += f"\n{i}. {warning['type'].upper()}\n"
                report += f"   {warning.get('message', 'ไม่มีข้อความ')}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report


# ฟังก์ชันสำหรับใช้งานง่าย
def create_safeguards(patience=3, min_delta=0.001):
    """สร้าง TrainingSafeguards instance"""
    return TrainingSafeguards(patience=patience, min_delta=min_delta)


if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    safeguards = create_safeguards()
    
    # จำลองการเทรน
    print("🧪 ทดสอบระบบป้องกันปัญหา\n")
    
    # Epoch 1
    is_overfitting, msg = safeguards.check_overfitting(train_loss=2.5, eval_loss=2.6)
    print(f"Epoch 1: {msg}")
    
    # Epoch 2
    is_overfitting, msg = safeguards.check_overfitting(train_loss=2.0, eval_loss=2.1)
    print(f"Epoch 2: {msg}")
    
    # Epoch 3 - Overfitting
    is_overfitting, msg = safeguards.check_overfitting(train_loss=1.5, eval_loss=2.5)
    print(f"Epoch 3: {msg}")
    
    # ตรวจสอบการเรียนรู้
    is_learning, msg = safeguards.check_learning_progress(1.5)
    print(f"\nการเรียนรู้: {msg}")
    
    # สรุปผล
    print("\n" + safeguards.generate_report())
