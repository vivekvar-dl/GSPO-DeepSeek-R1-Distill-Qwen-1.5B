# GSPO Implementation: Group Sequence Policy Optimization

This repository contains a complete implementation of **Group Sequence Policy Optimization (GSPO)** based on the paper by Zheng et al. (2025) from Alibaba's Qwen team.

## 🎯 What is GSPO?

GSPO solves fundamental stability issues in large language model RL training by:
- **Sequence-level optimization** instead of token-level
- **Proper importance sampling** with length-normalized ratios  
- **Stable training** for large models, especially MoE architectures
- **Superior performance** compared to GRPO and other token-level methods

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
python setup.py

# Or manually:
pip install -r requirements.txt
```

### 2. Validate Implementation

**IMPORTANT: Always run validation first!**

```bash
# Quick validation (2-3 minutes)
python validate_gspo.py
```

This will test:
- ✅ Technical correctness of GSPO implementation
- ✅ Training dynamics and clipping behavior  
- ✅ Performance improvement over training steps
- ✅ Comparison of different clipping ranges

### 3. Run GSPO Training

```bash
# Quick test run
python train_gspo.py --dataset mixed --num_train_samples 50 --num_epochs 3

# Math-focused training  
python train_gspo.py --dataset math --num_train_samples 200 --num_epochs 5

# Full training with larger model
python train_gspo.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --num_train_samples 1000 \
  --num_epochs 10 \
  --use_wandb
```

## 📊 How to Verify GSPO is Working

### Method 1: Run Validation Script
```bash
python validate_gspo.py
```

**Expected Output:**
```
✅ All technical tests PASSED!
✅ Training dynamics look good!  
✅ Performance test: PASS - Clear improvement
🎉 GSPO implementation looks GOOD!
```

### Method 2: Check Key Indicators

#### **1. Training Metrics Should Look Like:**
- **Clip Fraction**: 10-30% (much higher than GRPO's ~0.1%)
- **Importance Ratios**: Stay positive, around 0.8-1.2 range
- **Loss**: Decreases over time (negative values are normal)
- **Rewards**: Increase during training

#### **2. Performance Indicators:**
- Model responses get better quality over training
- Average rewards increase from initial to final evaluation
- No catastrophic model collapse (responses stay coherent)

#### **3. GSPO-Specific Behaviors:**
- High clip fractions (this is good! Unlike GRPO)
- Stable importance ratios (no extreme values)
- Smooth training curves without spikes

### Method 3: Compare Before/After

The validation script automatically tests this:
1. **Before training**: Baseline model performance
2. **After training**: Improved performance  
3. **Key metric**: Reward improvement > 0.01 for short training

## 🏗️ Architecture Overview

```
📁 GSPO Implementation
├── gspo_implementation.py   # Core GSPO algorithm
├── data_loader.py          # Datasets and reward functions  
├── train_gspo.py           # Main training script
├── validate_gspo.py        # Validation and testing
├── setup.py               # Environment setup
└── requirements.txt       # Dependencies
```

## 🔧 Configuration Options

### Model Options
- **Qwen/Qwen2.5-7B-Instruct** (default, recommended)
- **Qwen/Qwen2.5-14B-Instruct** (larger, fits H100)
- **meta-llama/Llama-3.1-8B-Instruct** (alternative)
- **Qwen/Qwen2.5-MoE-A2.7B-Instruct** (test MoE benefits)

### Dataset Options
- **`mixed`**: Math + Code + Reasoning (recommended start)
- **`math`**: Pure mathematical problems 
- **`code`**: Programming/coding tasks
- **`gsm8k`**: Real GSM8K benchmark dataset

### GSPO Parameters (from paper)
```python
left_clip_range=3e-4      # Left clipping bound
right_clip_range=4e-4     # Right clipping bound  
group_size=4             # Number of responses per query
learning_rate=1e-6       # Conservative learning rate
```

## 🎛️ Training Commands

### Basic Testing
```bash
# Quick functionality test
python train_gspo.py --dataset mixed --num_train_samples 20 --num_epochs 2

# Test with different models
python train_gspo.py --model_name Qwen/Qwen2.5-14B-Instruct --dataset math
```

### Serious Training
```bash
# Math reasoning training
python train_gspo.py \
  --dataset gsm8k \
  --num_train_samples 1000 \
  --num_epochs 10 \
  --batch_size 4 \
  --learning_rate 1e-6 \
  --use_wandb

# Code generation training  
python train_gspo.py \
  --dataset code \
  --num_train_samples 500 \
  --num_epochs 8 \
  --group_size 6
```

### Advanced Options
```bash
# Custom clipping ranges
python train_gspo.py \
  --left_clip_range 2e-4 \
  --right_clip_range 5e-4 \
  --update_frequency 3

# Larger context and batch sizes
python train_gspo.py \
  --max_length 1024 \
  --batch_size 8 \
  --mini_batch_size 2
```

## 📈 Expected Results

### Validation Results
- **Technical tests**: 100% pass rate
- **Training dynamics**: Stable loss curves, reasonable clip fractions
- **Performance**: Clear improvement or at least stability
- **Clipping**: Higher clip fractions than GRPO (this is correct!)

### Training Results
- **Reward curves**: Generally increasing trend
- **Loss curves**: Decreasing (negative values are normal)
- **Model quality**: Better responses to math/code problems
- **Stability**: No catastrophic collapse

## ⚠️ Troubleshooting

### Common Issues

#### **1. Validation Fails**
```bash
# Check setup first
python setup.py

# Run with smaller model
python validate_gspo.py  # Already uses small model
```

#### **2. Memory Issues**
```bash
# Use smaller model
python train_gspo.py --model_name microsoft/DialoGPT-medium

# Reduce batch size
python train_gspo.py --batch_size 2 --group_size 2
```

#### **3. Training Unstable**
```bash
# Reduce learning rate
python train_gspo.py --learning_rate 5e-7

# Update old model more frequently  
python train_gspo.py --update_frequency 2
```

#### **4. Poor Performance**
```bash
# Check reward function is working
python data_loader.py  # Test reward functions

# Try higher learning rate
python train_gspo.py --learning_rate 2e-6

# Use more training data
python train_gspo.py --num_train_samples 500
```

## 🔬 Technical Details

### GSPO vs GRPO Key Differences

| Aspect | GRPO | GSPO |
|--------|------|------|
| **Importance Ratio** | Token-level: `π(y_t\|x,y<t) / π_old(y_t\|x,y<t)` | Sequence-level: `(π(y\|x) / π_old(y\|x))^(1/\|y\|)` |
| **Optimization Unit** | Individual tokens | Complete sequences |  
| **Clipping** | ~0.1% of tokens | ~15% of tokens (higher is better) |
| **Stability** | Prone to collapse | Stable training |
| **MoE Support** | Needs "Routing Replay" | Works naturally |

### Mathematical Foundation

**GSPO Objective:**
```
J_GSPO = E[1/G * Σ min(s_i(θ) * A_i, clip(s_i(θ)) * A_i)]
```

**Sequence Importance Ratio:**
```
s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)
```

This is the **geometric mean** of token-level ratios, providing much more stable training.

## 📊 Monitoring Training

### Key Metrics to Watch

#### **Good Signs:**
- Clip fraction: 10-30% (high is good for GSPO!)
- Importance ratios: 0.5-2.0 range, not extreme
- Loss: Generally decreasing  
- Rewards: Increasing trend
- Model responses: Getting more coherent/correct

#### **Warning Signs:**
- Importance ratios > 10 or < 0.1 (extreme values)
- Clip fraction > 80% (too much clipping)
- Loss increasing consistently
- Model generating nonsense

#### **Emergency Stops:**
- NaN in any metric
- Model outputs becoming gibberish
- Importance ratios exploding (>100)

## 🎯 Replicating Paper Results

To replicate the paper's findings:

```bash
# Use paper's hyperparameters
python train_gspo.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --dataset gsm8k \
  --group_size 4 \
  --left_clip_range 3e-4 \
  --right_clip_range 4e-4 \
  --learning_rate 1e-6 \
  --num_train_samples 2000 \
  --num_epochs 15 \
  --update_frequency 5 \
  --use_wandb
```

Expected improvements:
- **Training Stability**: No collapse, smooth curves
- **Higher Clip Fractions**: 15% vs GRPO's 0.13%
- **Better Performance**: Superior rewards on math/code tasks
- **MoE Compatibility**: Stable training without "Routing Replay"

## 📚 References

- **Paper**: Group Sequence Policy Optimization (Zheng et al., 2025)
- **Original Work**: Used in Qwen3 models
- **Key Innovation**: Sequence-level importance sampling for RL training

## 🤝 Contributing

Feel free to:
- Report issues with validation results
- Suggest improvements to reward functions
- Add support for new datasets
- Optimize for different hardware configurations

---

**Ready to train with GSPO? Start with validation:**

```bash
python validate_gspo.py
```

If you see **"🎉 GSPO implementation looks GOOD!"**, you're ready to scale up! 🚀 