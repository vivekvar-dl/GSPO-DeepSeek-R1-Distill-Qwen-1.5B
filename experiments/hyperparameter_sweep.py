#!/usr/bin/env python3
"""
GSPO Hyperparameter Sweep
Systematic optimization to achieve target performance metrics
"""

import subprocess
import itertools
import time
from pathlib import Path

def run_gspo_experiment(config, experiment_id):
    """Run a single GSPO experiment with given configuration"""
    
    cmd = [
        "python", "train_gspo.py",
        "--dataset", "gsm8k",
        "--num_train_samples", str(config["samples"]),
        "--num_epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--group_size", str(config["group_size"]),
        "--learning_rate", str(config["learning_rate"]),
        "--left_clip_range", str(config["left_clip"]),
        "--right_clip_range", str(config["right_clip"]),
        "--update_frequency", str(config["update_freq"]),
        "--max_length", str(config["max_length"]),
        "--output_dir", f"./experiments/gspo_exp_{experiment_id}",
        "--use_wandb"
    ]
    
    print(f"\n🚀 Running Experiment {experiment_id}:")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Clipping: [{config['left_clip']}, {config['right_clip']}]")
    print(f"   Update Freq: {config['update_freq']}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        return {
            "experiment_id": experiment_id,
            "config": config,
            "success": result.returncode == 0,
            "output": result.stdout[-1000:] if result.stdout else "",  # Last 1000 chars
            "error": result.stderr[-500:] if result.stderr else ""
        }
    except subprocess.TimeoutExpired:
        return {
            "experiment_id": experiment_id,
            "config": config,
            "success": False,
            "error": "Timeout after 2 hours"
        }
    except Exception as e:
        return {
            "experiment_id": experiment_id,
            "config": config,
            "success": False,
            "error": str(e)
        }

def gspo_hyperparameter_sweep():
    """Run systematic hyperparameter sweep for GSPO"""
    
    # Create experiments directory
    Path("experiments").mkdir(exist_ok=True)
    
    # Hyperparameter grid - focused on key parameters affecting clipping
    param_grid = {
        "learning_rate": [2e-8, 5e-8, 1e-7],           # Ultra-conservative to moderate
        "left_clip": [1e-3, 2e-3, 3e-3],               # Wider clipping ranges
        "right_clip": [1e-3, 2e-3, 3e-3],              # Symmetric clipping
        "update_freq": [1, 2],                          # Frequent old model updates
        "samples": [200],                               # Fixed for speed
        "epochs": [3],                                  # Fixed for speed
        "batch_size": [2],                              # Fixed for memory
        "group_size": [4],                              # Fixed - standard GSPO
        "max_length": [512]                             # Fixed for memory
    }
    
    # Generate all combinations (optimized subset)
    keys = ["learning_rate", "left_clip", "right_clip", "update_freq"]
    values = [param_grid[k] for k in keys]
    
    # Add fixed parameters
    fixed_params = {
        "samples": param_grid["samples"][0],
        "epochs": param_grid["epochs"][0], 
        "batch_size": param_grid["batch_size"][0],
        "group_size": param_grid["group_size"][0],
        "max_length": param_grid["max_length"][0]
    }
    
    experiments = []
    experiment_id = 1
    
    # Priority experiments - most likely to achieve target performance
    priority_configs = [
        # Conservative clipping + ultra-low LR
        {"learning_rate": 2e-8, "left_clip": 2e-3, "right_clip": 2e-3, "update_freq": 1},
        {"learning_rate": 5e-8, "left_clip": 2e-3, "right_clip": 2e-3, "update_freq": 1},
        {"learning_rate": 2e-8, "left_clip": 3e-3, "right_clip": 3e-3, "update_freq": 1},
        
        # Moderate settings
        {"learning_rate": 5e-8, "left_clip": 3e-3, "right_clip": 3e-3, "update_freq": 1},
        {"learning_rate": 1e-7, "left_clip": 3e-3, "right_clip": 3e-3, "update_freq": 1},
        
        # More frequent updates
        {"learning_rate": 5e-8, "left_clip": 2e-3, "right_clip": 2e-3, "update_freq": 2},
    ]
    
    print("🔬 GSPO HYPERPARAMETER SWEEP")
    print("=" * 50)
    print(f"Total priority experiments: {len(priority_configs)}")
    print("Target metrics:")
    print("  - Clip fraction: 15-25%")
    print("  - Improvement: 8-12%")
    print("  - Smooth curves")
    print("=" * 50)
    
    results = []
    
    for config_subset in priority_configs:
        # Merge with fixed parameters
        full_config = {**fixed_params, **config_subset}
        
        print(f"\n⏰ Starting experiment {experiment_id} at {time.strftime('%H:%M:%S')}")
        
        result = run_gspo_experiment(full_config, experiment_id)
        results.append(result)
        
        # Log result
        if result["success"]:
            print(f"✅ Experiment {experiment_id} completed successfully")
        else:
            print(f"❌ Experiment {experiment_id} failed: {result['error'][:100]}")
        
        experiment_id += 1
        
        # Short break between experiments
        time.sleep(10)
    
    # Save results summary
    print("\n📊 SWEEP RESULTS SUMMARY:")
    print("=" * 50)
    
    success_count = sum(1 for r in results if r["success"])
    print(f"Successful experiments: {success_count}/{len(results)}")
    
    for result in results:
        if result["success"]:
            config = result["config"]
            print(f"✅ Exp {result['experiment_id']}: LR={config['learning_rate']:.0e}, "
                  f"Clip=[{config['left_clip']:.0e}, {config['right_clip']:.0e}], "
                  f"Update={config['update_freq']}")
        else:
            print(f"❌ Exp {result['experiment_id']}: {result['error'][:50]}")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Check wandb for clip_fraction and eval/reward metrics")
    print("2. Identify best performing configuration")
    print("3. Run extended training with optimal hyperparameters")
    
    return results

if __name__ == "__main__":
    results = gspo_hyperparameter_sweep() 