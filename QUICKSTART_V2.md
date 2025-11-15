# Quick Start Guide - Training Workflow V2

## Overview

This guide provides step-by-step instructions for using the improved Training Workflow V2 in Google Colab Pro+.

## What's New in V2?

Training Workflow V2 addresses the overfitting issue observed in the initial training run. Key improvements include:

- **Lower Learning Rate:** Reduced from `2e-5` to `5e-6` for more stable convergence
- **Strong Regularization:** Added weight decay of `0.1` to prevent overfitting
- **Frequent Evaluation:** Evaluate every 500 steps instead of once per epoch
- **Enhanced LoRA:** Increased LoRA rank from 8 to 16 for better model expressiveness

## Prerequisites

1. Google Colab Pro+ account
2. Training data file: `training_data_1m_final.jsonl` (should be in your Google Drive)
3. GitHub repository access: `traingptproject/gptprojecttrain`

## Step-by-Step Instructions

### Step 1: Stop Current Training (If Running)

If you have a training run currently in progress:

1. Go to your Google Colab notebook
2. Click **Runtime → Interrupt execution** or press `Ctrl+M I`
3. Wait for the training to stop completely

### Step 2: Pull Latest Changes from GitHub

In your Colab notebook, run:

```python
!cd /content/gptprojecttrain && git pull origin main
```

This will download the new V2 files:
- `training_config_v2.py`
- `train_monitored_v2.py`
- `training_analysis_and_recommendations.md`
- `IMPROVED_TRAINING_STRATEGY.md`

### Step 3: Verify the Training Data

Make sure your training data is accessible:

```python
import os
data_path = "/content/gptprojecttrain/training_data_1m_final.jsonl"
if os.path.exists(data_path):
    print(f"✅ Training data found: {os.path.getsize(data_path) / (1024**3):.2f} GB")
else:
    print("❌ Training data not found. Please upload it to your Colab environment.")
```

### Step 4: Install Dependencies

```python
!pip install -q transformers datasets accelerate bitsandbytes peft
```

### Step 5: Run the V2 Training Script

```python
%cd /content/gptprojecttrain
!python3 train_monitored_v2.py
```

### Step 6: Monitor the Training

The training will now run with the improved configuration. You should observe:

- **Slower loss decrease:** The training loss should decrease more gradually compared to the previous run
- **Regular evaluation:** The model will be evaluated every 500 steps
- **Checkpoint saving:** Best models will be saved based on validation loss
- **Early stopping:** Training will stop automatically if validation loss doesn't improve for 3 consecutive evaluations

### Expected Training Time

With the new configuration on Google Colab Pro+ (A100 GPU):
- **Per Epoch:** Approximately 4-6 hours
- **Total (3 epochs):** Approximately 12-18 hours

## Monitoring Progress

The script will print progress updates to the console, including:

- Current step and epoch
- Training loss
- Validation loss (every 500 steps)
- GPU memory usage
- Estimated time remaining

## What to Look For

### Good Signs:
- Training loss decreases gradually (not rapidly)
- Validation loss tracks training loss closely
- No sudden spikes in loss values

### Warning Signs:
- Training loss drops too quickly (like before)
- Validation loss increases while training loss decreases (overfitting)
- Out of memory errors

## After Training Completes

The final model will be saved to:
```
/content/gptprojecttrain/training_output_v2/final_model/
```

You can then test the model or deploy it as needed.

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size in `training_config_v2.py`:
```python
TRAINING_ARGS = {
    ...
    "per_device_train_batch_size": 1,  # Reduce from 2 to 1
    "gradient_accumulation_steps": 16,  # Increase from 8 to 16
    ...
}
```

### Issue: Training Too Slow

**Solution:** The slower training is intentional to prevent overfitting. However, if you need faster training, you can slightly increase the learning rate (but not above `1e-5`).

### Issue: Early Stopping Triggered Too Soon

**Solution:** Increase patience in `training_config_v2.py`:
```python
EARLY_STOPPING = {
    "patience": 5,  # Increase from 3 to 5
    ...
}
```

## Need Help?

If you encounter any issues:
1. Check the console output for error messages
2. Review the `training_analysis_and_recommendations.md` document
3. Check the GitHub repository for updates

## Summary

Training Workflow V2 is designed to produce a more robust and generalizable model by preventing overfitting. The training will be slower, but the results should be significantly better. Let the training run to completion (all 3 epochs) unless early stopping is triggered.

Good luck with your training! 🚀
