# dLNk GPT V2 - Google Colab Training Setup Guide

## 🎯 Overview

This guide explains how to run continuous training for dLNk GPT V2 Exploit Agent on Google Colab with real-time LINE notifications.

## 📋 Prerequisites

1. **Google Account** with Google Colab access
2. **LINE Official Account** (already configured via MCP)
3. **GitHub Repository** (traingptproject/gptprojecttrain)
4. **Hugging Face Account** (optional, for model upload)

## 🚀 Quick Start

### Method 1: Using Jupyter Notebook (Recommended)

1. **Upload to Google Colab:**
   - Open [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Upload `dLNk_GPT_V2_Training_Colab.ipynb`

2. **Select GPU Runtime:**
   - Click `Runtime` → `Change runtime type`
   - Select `T4 GPU` or `A100 GPU` (if available)
   - Click `Save`

3. **Run All Cells:**
   - Click `Runtime` → `Run all`
   - Or press `Ctrl+F9`

4. **Monitor via LINE:**
   - You will receive real-time notifications on LINE
   - Progress updates every 5 minutes
   - Evaluation results
   - Completion notification

### Method 2: Using Python Script

1. **Create New Colab Notebook:**
   ```python
   # Cell 1: Clone repository
   !git clone https://github.com/traingptproject/gptprojecttrain.git
   %cd gptprojecttrain
   
   # Cell 2: Install dependencies
   !pip install -q torch transformers datasets peft accelerate bitsandbytes
   
   # Cell 3: Run workflow orchestrator
   !python3 workflow_orchestrator.py
   ```

2. **Run the cells** and monitor via LINE

## 📁 File Structure

```
gptprojecttrain/
├── dLNk_GPT_V2_Training_Colab.ipynb    # Main Colab notebook
├── workflow_orchestrator.py             # Automated workflow
├── line_notifier.py                     # LINE notification helper
├── train_exploit_agent_v2.py           # Training script
├── training_config_v2_exploit.py       # Training configuration
├── exploit_agent.py                     # Agent system
├── exploit_training_data_v2_enhanced.jsonl  # Training data
└── COLAB_SETUP_GUIDE.md                # This file
```

## ⚙️ Configuration

### Environment Variables (Optional)

Set these in Colab if needed:

```python
import os

# GitHub repository
os.environ['GITHUB_REPO'] = 'traingptproject/gptprojecttrain'

# Hugging Face (for model upload)
os.environ['HF_TOKEN'] = 'your_huggingface_token'
os.environ['HF_REPO'] = 'your-username/dLNk-GPT-J-6B-Exploit-V2'

# Working directory
os.environ['WORK_DIR'] = '/content/gptprojecttrain'
```

### Hugging Face Token Setup

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with `write` access
3. In Colab, go to `Secrets` (🔑 icon on left sidebar)
4. Add secret: `HF_TOKEN` = `your_token_here`

## 🔧 Features

### 1. Anti-Disconnect Mechanism

The notebook includes JavaScript code to prevent Colab from disconnecting:

```javascript
function ClickConnect(){
  console.log("Keeping Colab alive...");
  document.querySelector("colab-toolbar-button#connect").click()
}
setInterval(ClickConnect, 60000)
```

This runs every 60 seconds to keep the connection alive.

### 2. Real-time LINE Notifications

You will receive notifications for:

- ✅ Training start
- 📈 Progress updates (every 5 minutes)
- 🎯 Evaluation results (every 500 steps)
- 💾 Checkpoint saves
- ✅ Training completion
- ❌ Errors (if any)

### 3. Automatic Checkpointing

The training saves checkpoints every 500 steps to prevent data loss.

### 4. GPU Monitoring

System automatically checks:
- GPU availability
- GPU memory
- CUDA version
- PyTorch version

## 📊 Training Configuration

### V2 Settings (Anti-Overfitting)

```python
Learning Rate: 5e-6          # 4x lower than V1
Weight Decay: 0.1            # 10x higher than V1
LoRA Rank: 16                # 2x higher than V1
LoRA Dropout: 0.05           # Added regularization
Evaluation: Every 500 steps  # 10x more frequent
```

### Expected Training Time

| GPU Type | Estimated Time | Notes |
|----------|---------------|-------|
| T4 | 4-6 hours | Free tier |
| A100 | 1-2 hours | Colab Pro |
| V100 | 2-3 hours | Colab Pro |

## 📱 LINE Notification Examples

### Training Start
```
🚀 เริ่มการเทรนโมเดล

📊 โมเดล: dLNk GPT V2 Exploit Agent
🔢 Total Steps: 1,000
📈 Epochs: 3
⏰ เวลาเริ่ม: 2025-11-15 12:00:00

💡 ระบบจะรายงานความคืบหน้าทุก 5 นาที
```

### Progress Update
```
📈 ความคืบหน้าการเทรน

██████████░░░░░░░░░░ 50.0%

🔢 Step: 500/1,000
📉 Loss: 1.2345
⚡ Learning Rate: 5.00e-06

⏱️ เวลาที่ใช้: 01:23:45
🕐 เวลาคงเหลือ: 01:20:00
```

### Evaluation Result
```
🎯 ผลการประเมิน

🔢 Step: 500
📉 Eval Loss: 1.4567
📊 Train Loss: 1.2345
📈 Difference: 0.2222
```

### Training Complete
```
✅ การเทรนเสร็จสมบูรณ์!

⏱️ เวลาทั้งหมด: 02:45:30
📉 Final Loss: 0.9876
🔢 Total Steps: 1,000

💾 โมเดลถูกบันทึกที่:
./dLNk-gpt-j-6b-exploit-v2

🎉 พร้อมใช้งาน!
```

## 🐛 Troubleshooting

### Issue: Colab Disconnects

**Solution:**
- Make sure the anti-disconnect JavaScript is running (Cell 1)
- Keep the browser tab active
- Use Colab Pro for longer sessions

### Issue: Out of Memory

**Solution:**
```python
# Reduce batch size
per_device_train_batch_size: 2
gradient_accumulation_steps: 8

# Enable gradient checkpointing
gradient_checkpointing: True
```

### Issue: LINE Notifications Not Working

**Solution:**
1. Check if LINE MCP is configured
2. Verify `manus-mcp-cli` is available
3. Test with:
```bash
manus-mcp-cli tool call push_text_message --server line --input '{"message": {"text": "Test"}}'
```

### Issue: Training Too Slow

**Solution:**
- Use A100 GPU (Colab Pro)
- Reduce max_length from 2048 to 1024
- Reduce training data size

### Issue: Model Not Saving

**Solution:**
- Check disk space: `!df -h`
- Verify output directory exists
- Check permissions

## 📈 Monitoring Training

### Via LINE
- Automatic notifications every 5 minutes
- Real-time progress updates
- Error alerts

### Via Colab Console
- Watch training logs in real-time
- Monitor GPU usage: `!nvidia-smi`
- Check memory: `!free -h`

### Via TensorBoard (Optional)
```python
%load_ext tensorboard
%tensorboard --logdir ./dLNk-gpt-j-6b-exploit-v2/runs
```

## 💾 Saving Results

### Download Model from Colab
```python
from google.colab import files

# Zip model directory
!zip -r model.zip ./dLNk-gpt-j-6b-exploit-v2

# Download
files.download('model.zip')
```

### Upload to Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy model
!cp -r ./dLNk-gpt-j-6b-exploit-v2 /content/drive/MyDrive/
```

### Upload to Hugging Face
Automatically done if `HF_TOKEN` is set (Cell 9 in notebook)

## 🔄 Continuous Training

### Keep Training Running
1. Use Colab Pro for longer sessions (up to 24 hours)
2. Enable anti-disconnect mechanism
3. Monitor via LINE notifications
4. Set up automatic checkpointing

### Resume from Checkpoint
```python
# In training configuration
resume_from_checkpoint: "./dLNk-gpt-j-6b-exploit-v2/checkpoint-500"
```

## 📊 Expected Results

### V2 Training Metrics

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Initial Loss | 3.0 - 4.0 | Normal for GPT-J |
| Final Loss | 0.8 - 1.2 | Good convergence |
| Eval Loss | 1.0 - 1.5 | Should be close to train loss |
| Training Time | 2-6 hours | Depends on GPU |

### Signs of Good Training
- ✅ Loss decreases steadily
- ✅ Eval loss close to train loss (< 0.5 difference)
- ✅ No sudden spikes in loss
- ✅ Learning rate decreases smoothly

### Signs of Problems
- ❌ Loss stuck or increasing
- ❌ Eval loss >> train loss (overfitting)
- ❌ Loss becomes NaN
- ❌ Out of memory errors

## 🎯 Next Steps After Training

1. **Test the Model:**
   ```python
   from exploit_agent import ExploitAgent
   agent = ExploitAgent()
   # Test exploit generation
   ```

2. **Deploy API:**
   - Use FastAPI or Flask
   - Deploy to Hugging Face Spaces
   - Or use Google Cloud Run

3. **Integrate with Tools:**
   - Metasploit
   - Burp Suite
   - Custom penetration testing workflows

## 📞 Support

If you encounter issues:
1. Check LINE notifications for error messages
2. Review Colab console logs
3. Verify GPU is available
4. Check GitHub repository for updates

## ⚠️ Important Notes

1. **Free Colab Limitations:**
   - 12-hour session limit
   - May disconnect if idle
   - Limited GPU availability

2. **Colab Pro Benefits:**
   - 24-hour sessions
   - Priority GPU access
   - More powerful GPUs (A100)

3. **Data Privacy:**
   - Training happens in Google's infrastructure
   - Models are saved to your Google Drive or Hugging Face
   - No data is shared without your permission

## 🚀 Ready to Start?

1. Open `dLNk_GPT_V2_Training_Colab.ipynb` in Google Colab
2. Select GPU runtime
3. Run all cells
4. Monitor via LINE
5. Wait for completion notification

**Estimated total time: 2-6 hours**

---

**Version:** 2.0  
**Last Updated:** 2025-11-15  
**Status:** Production Ready
