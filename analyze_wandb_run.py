#!/usr/bin/env python3
"""
Analyze GSPO wandb run to diagnose performance decline
"""

def analyze_gspo_run():
    """Analyze the most recent GSPO training run"""
    
    try:
        import wandb
        import pandas as pd
        import numpy as np
    except ImportError:
        print("❌ wandb or pandas not available")
        print("Run: pip install wandb pandas")
        return
    
    try:
        # Initialize wandb API
        api = wandb.Api()
        
        # Get the most recent run (you can replace with specific run ID)
        project = "gspo-Qwen2.5-7B-Instruct" 
        runs = api.runs(project)
        
        if not runs:
            print("❌ No runs found")
            return
            
        # Get the most recent run
        run = runs[0]
        print(f"📊 Analyzing run: {run.id} ({run.name})")
        print(f"🕐 Started: {run.created_at}")
        print(f"⏱️ Duration: {run.summary.get('_runtime', 'Unknown')}")
        print()
        
        # Get hyperparameters
        print("=" * 50)
        print("🎛️ HYPERPARAMETERS")
        print("=" * 50)
        config = run.config
        key_params = [
            'learning_rate', 'batch_size', 'group_size', 'num_epochs', 
            'max_length', 'left_clip_range', 'right_clip_range'
        ]
        
        for param in key_params:
            if param in config:
                print(f"{param:20}: {config[param]}")
        print()
        
        # Get training history
        history = run.history()
        print(f"📈 Total logged steps: {len(history)}")
        print()
        
        # Analyze performance trends
        print("=" * 50)
        print("📉 PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        # Eval rewards
        if 'eval/reward' in history.columns:
            eval_rewards = history['eval/reward'].dropna()
            if len(eval_rewards) > 1:
                start_eval = eval_rewards.iloc[0]
                end_eval = eval_rewards.iloc[-1]
                change_eval = end_eval - start_eval
                
                print(f"Eval reward (start):     {start_eval:.4f}")
                print(f"Eval reward (end):       {end_eval:.4f}")
                print(f"Eval reward (change):    {change_eval:+.4f}")
                
                if change_eval < -0.01:
                    print("🚨 PROBLEM: Eval performance DECLINING!")
                elif change_eval > 0.01:
                    print("✅ GOOD: Eval performance improving")
                else:
                    print("⚠️ WARNING: Eval performance stagnant")
        
        # Train rewards
        if 'train/reward_mean' in history.columns:
            train_rewards = history['train/reward_mean'].dropna()
            if len(train_rewards) > 1:
                start_train = train_rewards.iloc[0]
                end_train = train_rewards.iloc[-1]
                change_train = end_train - start_train
                
                print(f"Train reward (start):    {start_train:.4f}")
                print(f"Train reward (end):      {end_train:.4f}")
                print(f"Train reward (change):   {change_train:+.4f}")
        
        print()
        
        # Analyze GSPO-specific metrics
        print("=" * 50)
        print("🔧 GSPO ALGORITHM ANALYSIS")
        print("=" * 50)
        
        # Importance ratios
        if 'train/importance_ratio_mean' in history.columns:
            ratios = history['train/importance_ratio_mean'].dropna()
            ratio_mean = ratios.mean()
            ratio_std = ratios.std()
            ratio_drift = ratios.iloc[-1] - ratios.iloc[0] if len(ratios) > 1 else 0
            
            print(f"Importance ratio (mean): {ratio_mean:.4f}")
            print(f"Importance ratio (std):  {ratio_std:.4f}")
            print(f"Importance ratio (drift): {ratio_drift:+.4f}")
            
            if abs(ratio_mean - 1.0) > 0.2:
                print("🚨 PROBLEM: Importance ratios drifting from 1.0!")
            elif ratio_std > 0.5:
                print("⚠️ WARNING: High importance ratio variance")
            else:
                print("✅ GOOD: Importance ratios stable")
        
        # Clipping behavior
        if 'train/clip_fraction' in history.columns:
            clipping = history['train/clip_fraction'].dropna()
            clip_mean = clipping.mean()
            clip_max = clipping.max()
            
            print(f"Clip fraction (mean):    {clip_mean:.4f}")
            print(f"Clip fraction (max):     {clip_max:.4f}")
            
            if clip_mean > 0.5:
                print("🚨 PROBLEM: Too much clipping (>50%)!")
            elif clip_mean < 0.05:
                print("⚠️ WARNING: Very little clipping (<5%)")
            else:
                print("✅ GOOD: Reasonable clipping levels")
        
        # Loss behavior
        if 'train/loss' in history.columns:
            losses = history['train/loss'].dropna()
            loss_trend = "decreasing" if losses.iloc[-1] < losses.iloc[0] else "increasing"
            loss_std = losses.std()
            
            print(f"Loss trend:              {loss_trend}")
            print(f"Loss stability (std):    {loss_std:.6f}")
            
            if loss_std > 1.0:
                print("⚠️ WARNING: Loss very unstable")
        
        print()
        
        # Recommendations
        print("=" * 50)
        print("💡 RECOMMENDATIONS")
        print("=" * 50)
        
        recommendations = []
        
        # Check eval performance
        if 'eval/reward' in history.columns:
            eval_rewards = history['eval/reward'].dropna()
            if len(eval_rewards) > 1 and (eval_rewards.iloc[-1] - eval_rewards.iloc[0]) < -0.01:
                recommendations.append("🔧 REDUCE learning rate (try 1e-6 instead of 3e-6)")
                recommendations.append("🔧 ADD early stopping when eval performance degrades")
        
        # Check importance ratios
        if 'train/importance_ratio_mean' in history.columns:
            ratios = history['train/importance_ratio_mean'].dropna()
            if len(ratios) > 0 and abs(ratios.mean() - 1.0) > 0.2:
                recommendations.append("🔧 CHECK old model update frequency")
                recommendations.append("🔧 VERIFY importance ratio computation")
        
        # Check clipping
        if 'train/clip_fraction' in history.columns:
            clipping = history['train/clip_fraction'].dropna()
            if len(clipping) > 0 and clipping.mean() > 0.5:
                recommendations.append("🔧 INCREASE clip ranges (less aggressive clipping)")
        
        if not recommendations:
            recommendations.append("✅ Training appears stable - consider longer training")
        
        for rec in recommendations:
            print(f"  {rec}")
        
        print()
        print("🚀 Next step: Restart training with improved hyperparameters")
        
    except Exception as e:
        print(f"❌ Error analyzing run: {e}")
        print("Make sure you're logged into wandb: wandb login")

if __name__ == "__main__":
    analyze_gspo_run() 