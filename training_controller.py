"""
Automated Training Controller
Monitors training in real-time and makes adjustments automatically
"""

import time
from typing import Dict, List, Optional
from line_monitor import monitor


class TrainingController:
    """Automated controller for training process"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.history = []
        self.consecutive_bad_epochs = 0
        self.best_loss = float('inf')
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.original_lr = self.learning_rate
        
    def check_overfitting(self, train_loss: float, eval_loss: float) -> bool:
        """Check if model is overfitting"""
        # Overfitting if eval loss is significantly higher than train loss
        gap = eval_loss - train_loss
        threshold = 0.5  # Threshold for overfitting detection
        
        if gap > threshold:
            return True
        return False
    
    def check_loss_explosion(self, current_loss: float) -> bool:
        """Check if loss is exploding"""
        if len(self.history) < 2:
            return False
        
        prev_loss = self.history[-1].get('eval_loss', 0)
        
        # Loss explosion if current loss is 2x previous
        if current_loss > prev_loss * 2:
            return True
        return False
    
    def check_plateau(self, window_size: int = 3) -> bool:
        """Check if training has plateaued"""
        if len(self.history) < window_size:
            return False
        
        recent_losses = [h.get('eval_loss', 0) for h in self.history[-window_size:]]
        
        # Check if losses are very similar (< 0.1% change)
        max_loss = max(recent_losses)
        min_loss = min(recent_losses)
        
        if max_loss > 0:
            change_pct = ((max_loss - min_loss) / max_loss) * 100
            if change_pct < 0.1:
                return True
        
        return False
    
    def adjust_learning_rate(self, factor: float, reason: str):
        """Adjust learning rate"""
        old_lr = self.learning_rate
        self.learning_rate = self.learning_rate * factor
        
        # Clamp learning rate
        self.learning_rate = max(1e-7, min(1e-3, self.learning_rate))
        
        monitor.send_adjustment(
            f"Learning Rate ({reason})",
            old_lr,
            self.learning_rate
        )
        
        return self.learning_rate
    
    def handle_overfitting(self, train_loss: float, eval_loss: float):
        """Handle overfitting situation"""
        monitor.send_warning(
            'overfitting',
            f"Train Loss: {train_loss:.4f}\nEval Loss: {eval_loss:.4f}\nGap: {eval_loss - train_loss:.4f}"
        )
        
        # Reduce learning rate
        new_lr = self.adjust_learning_rate(0.5, "ลด overfitting")
        
        self.consecutive_bad_epochs += 1
        
        return {
            'action': 'reduce_lr',
            'new_lr': new_lr,
            'stop_training': self.consecutive_bad_epochs >= 3
        }
    
    def handle_loss_explosion(self, current_loss: float):
        """Handle loss explosion"""
        prev_loss = self.history[-1].get('eval_loss', 0)
        
        monitor.send_warning(
            'high_loss',
            f"Loss เพิ่มขึ้นอย่างรวดเร็ว!\nก่อนหน้า: {prev_loss:.4f}\nปัจจุบัน: {current_loss:.4f}"
        )
        
        # Significantly reduce learning rate
        new_lr = self.adjust_learning_rate(0.1, "แก้ loss explosion")
        
        return {
            'action': 'reduce_lr_aggressive',
            'new_lr': new_lr,
            'stop_training': False
        }
    
    def handle_plateau(self):
        """Handle training plateau"""
        monitor.send_warning(
            'slow_training',
            f"Loss ไม่เปลี่ยนแปลงมาหลาย epochs\nพิจารณาปรับ learning rate"
        )
        
        # Slightly increase learning rate to escape plateau
        new_lr = self.adjust_learning_rate(1.2, "หนี plateau")
        
        return {
            'action': 'increase_lr',
            'new_lr': new_lr,
            'stop_training': False
        }
    
    def analyze_epoch(self, epoch: int, metrics: Dict) -> Dict:
        """Analyze epoch results and decide actions"""
        train_loss = metrics.get('train_loss', 0)
        eval_loss = metrics.get('eval_loss', 0)
        
        # Store in history
        self.history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'learning_rate': self.learning_rate
        })
        
        # Check for improvement
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.consecutive_bad_epochs = 0
            return {'action': 'continue', 'status': 'improving'}
        
        # Check for problems
        if self.check_loss_explosion(eval_loss):
            return self.handle_loss_explosion(eval_loss)
        
        if self.check_overfitting(train_loss, eval_loss):
            return self.handle_overfitting(train_loss, eval_loss)
        
        if self.check_plateau():
            return self.handle_plateau()
        
        # No issues, continue
        return {'action': 'continue', 'status': 'normal'}
    
    def should_stop_training(self) -> bool:
        """Determine if training should stop"""
        return self.consecutive_bad_epochs >= 3
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations for improving training"""
        recommendations = []
        
        if len(self.history) < 2:
            return recommendations
        
        recent_eval_losses = [h.get('eval_loss', 0) for h in self.history[-3:]]
        
        # Check trend
        if len(recent_eval_losses) >= 2:
            if all(recent_eval_losses[i] > recent_eval_losses[i-1] for i in range(1, len(recent_eval_losses))):
                recommendations.append("⚠️ Eval loss กำลังเพิ่มขึ้นต่อเนื่อง - พิจารณาหยุดหรือลด LR")
        
        # Check if LR has been reduced too much
        if self.learning_rate < self.original_lr * 0.1:
            recommendations.append("⚠️ Learning rate ถูกลดมากเกินไป - อาจต้อง restart")
        
        # Check overfitting
        last_metrics = self.history[-1]
        if self.check_overfitting(last_metrics.get('train_loss', 0), last_metrics.get('eval_loss', 0)):
            recommendations.append("⚠️ มีสัญญาณ overfitting - เพิ่ม regularization หรือลด epochs")
        
        return recommendations
