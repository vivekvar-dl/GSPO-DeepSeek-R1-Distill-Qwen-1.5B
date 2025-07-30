#!/usr/bin/env python3
"""
GSPO Training Script

This script demonstrates how to train a language model using Group Sequence Policy Optimization.
Designed for H100 GPU with appropriate model sizes and datasets.
"""

import os
import sys
import argparse
import json
import torch
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed
)

# Make wandb optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Import our GSPO implementation
from gspo_implementation import GSPOTrainer, GSPOConfig
from data_loader import DatasetLoader, create_reward_evaluator

def setup_logging(config):
    """Setup wandb logging if enabled"""
    if hasattr(config, 'use_wandb') and config.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="gspo-training",
            name=f"gspo-{config.model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(config)
        )

def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer with appropriate configuration for H100"""
    print(f"Loading model: {model_name}")
    
    # Memory optimization settings
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        
        # Set memory fraction to leave room for optimizer states
        torch.cuda.set_per_process_memory_fraction(0.85)
    
    # Model loading configuration based on size with memory optimization
    if "7B" in model_name or "8B" in model_name:
        torch_dtype = torch.bfloat16  # More memory efficient than float16
        device_map = "auto"
    elif "14B" in model_name or "13B" in model_name:
        torch_dtype = torch.bfloat16
        device_map = "auto" 
    elif "30B" in model_name:
        torch_dtype = torch.bfloat16
        device_map = "auto"
    else:
        torch_dtype = torch.bfloat16
        device_map = "auto"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Try with FlashAttention2 first, fall back if not available
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                low_cpu_mem_usage=True,
                use_cache=False  # Disable KV cache to save memory during training
            )
            print("✓ Using FlashAttention2 for optimization")
        except Exception as fa_error:
            print(f"FlashAttention2 not available: {fa_error}")
            print("Loading without FlashAttention2...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_cache=False
            )
            print("✓ Model loaded successfully without FlashAttention2")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("✓ Gradient checkpointing enabled")
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Added pad token")
        
        print(f"Model loaded successfully!")
        print(f"Model parameters: {model.num_parameters() / 1e9:.2f}B")
        
        # Clear cache after loading
        torch.cuda.empty_cache()
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to smaller model...")
        # Fallback to a smaller model if the requested one fails
        fallback_model = "microsoft/DialoGPT-medium"
        return load_model_and_tokenizer(fallback_model, device)

def create_training_config(args):
    """Create GSPO training configuration with memory optimizations"""
    return GSPOConfig(
        # Core GSPO parameters - from paper
        left_clip_range=args.left_clip_range,
        right_clip_range=args.right_clip_range,
        group_size=min(args.group_size, 2),  # Reduce group size for memory
        
        # Training parameters - memory optimized
        learning_rate=args.learning_rate,
        batch_size=min(args.batch_size, 2),  # Reduce batch size for memory
        mini_batch_size=1,  # Force mini batch size to 1
        max_length=min(args.max_length, 512),  # Limit sequence length
        
        # Logging
        log_frequency=args.log_frequency,
        eval_frequency=args.eval_frequency
    )

def evaluate_model(trainer, eval_data, reward_function):
    """Evaluate model performance on held-out data"""
    print("Running evaluation...")
    
    total_reward = 0.0
    num_samples = min(len(eval_data), 20)  # Evaluate on subset
    
    for i, item in enumerate(eval_data[:num_samples]):
        query = item['query']
        reference = item['reference_answer']
        
        # Generate response
        responses = trainer.generate_responses([query], max_new_tokens=128)
        response = responses[0]
        
        # Compute reward
        reward = reward_function(query, response, item)
        total_reward += reward
        
        if i < 3:  # Print first few examples
            print(f"\nExample {i+1}:")
            print(f"Query: {query[:100]}...")
            print(f"Response: {response[:100]}...")
            print(f"Reference: {reference[:100]}...")
            print(f"Reward: {reward:.3f}")
    
    avg_reward = total_reward / num_samples
    print(f"\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.3f}")
    
    return avg_reward

def main():
    parser = argparse.ArgumentParser(description="Train a language model using GSPO")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    # GSPO parameters (from paper) - OPTIMIZED FOR TARGET PERFORMANCE
    parser.add_argument("--left_clip_range", type=float, default=2e-3,
                       help="Left clipping range for GSPO")
    parser.add_argument("--right_clip_range", type=float, default=2e-3,
                       help="Right clipping range for GSPO")
    parser.add_argument("--group_size", type=int, default=4,
                       help="Group size G for GSPO")
    
    # Training parameters - OPTIMIZED FOR STABILITY
    parser.add_argument("--learning_rate", type=float, default=5e-8,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size (number of queries per step)")
    parser.add_argument("--mini_batch_size", type=int, default=1,
                       help="Mini batch size for gradient accumulation")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--update_frequency", type=int, default=1,
                       help="How often to update old model")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="mixed",
                       choices=["mixed", "math", "code", "gsm8k"],
                       help="Dataset to use for training")
    parser.add_argument("--num_train_samples", type=int, default=100,
                       help="Number of training samples")
    parser.add_argument("--num_eval_samples", type=int, default=20,
                       help="Number of evaluation samples")
    
    # Logging and evaluation
    parser.add_argument("--log_frequency", type=int, default=5,
                       help="How often to log training stats")
    parser.add_argument("--eval_frequency", type=int, default=20,
                       help="How often to run evaluation")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--output_dir", type=str, default="./gspo_outputs",
                       help="Output directory for checkpoints")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging(args)
    
    print("=" * 60)
    print("GSPO Training Script")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Group Size: {args.group_size}")
    print(f"Clipping Range: [{args.left_clip_range}, {args.right_clip_range}]")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
    
    # Create GSPO configuration
    config = create_training_config(args)
    
    # Initialize GSPO trainer
    trainer = GSPOTrainer(model, tokenizer, config, args.device)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    loader = DatasetLoader(seed=args.seed)
    
    if args.dataset == "mixed":
        train_data = loader.create_mixed_dataset(total_samples=args.num_train_samples)
        eval_data = loader.create_mixed_dataset(total_samples=args.num_eval_samples)
    elif args.dataset == "math":
        train_data = (loader.load_math_problems("easy", args.num_train_samples // 2) + 
                     loader.load_math_problems("medium", args.num_train_samples // 2))
        eval_data = loader.load_math_problems("easy", args.num_eval_samples)
    elif args.dataset == "code":
        train_data = (loader.load_code_problems("easy", args.num_train_samples // 2) + 
                     loader.load_code_problems("medium", args.num_train_samples // 2))
        eval_data = loader.load_code_problems("easy", args.num_eval_samples)
    elif args.dataset == "gsm8k":
        train_data = loader.load_gsm8k("train", args.num_train_samples)
        eval_data = loader.load_gsm8k("test", args.num_eval_samples)
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(eval_data)} evaluation samples")
    
    # Create reward function
    reward_function = create_reward_evaluator()
    
    # Create a closure that includes the data item for proper reward evaluation
    def training_reward_function(query: str, response: str):
        # Find the corresponding data item for this query
        for item in train_data:
            if item['query'] == query:
                return reward_function(query, response, item)
        # Fallback to simple math reward if not found
        return reward_function(query, response, {'type': 'math', 'reference_answer': ''})
    
    # Initial evaluation
    print("\n" + "=" * 60)
    print("INITIAL EVALUATION")
    print("=" * 60)
    initial_reward = evaluate_model(trainer, eval_data, reward_function)
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    best_reward = initial_reward
    training_stats = []
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*20} EPOCH {epoch + 1}/{args.num_epochs} {'='*20}")
        
        # Create batches of queries
        queries = [item['query'] for item in train_data]
        
        # Shuffle and batch queries
        import random
        random.shuffle(queries)
        
        epoch_stats = []
        
        for step in range(0, len(queries), args.batch_size):
            batch_queries = queries[step:step + args.batch_size]
            
            # Training step
            stats = trainer.train_step(batch_queries, training_reward_function)
            epoch_stats.append(stats)
            
            # Log to wandb if enabled
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": stats["loss"],
                    "train/clip_fraction": stats["clip_fraction"],
                    "train/importance_ratio_mean": stats["importance_ratio_mean"],
                    "train/reward_mean": stats["reward_mean"],
                    "epoch": epoch,
                    "step": trainer.step
                })
            
            # Evaluation
            if trainer.step % args.eval_frequency == 0:
                eval_reward = evaluate_model(trainer, eval_data, reward_function)
                
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({"eval/reward": eval_reward, "step": trainer.step})
                
                # Save best model
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    checkpoint_path = os.path.join(args.output_dir, "best_model")
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    print(f"New best model saved! Reward: {eval_reward:.3f}")
            
            # Clear GPU cache periodically
            if trainer.step % 5 == 0:
                torch.cuda.empty_cache()
        
        # Update old model periodically - only after sufficient training steps
        # Update every update_frequency epochs AND only if we've had enough steps
        if (epoch + 1) % args.update_frequency == 0 and trainer.step >= 20:
            # Also ensure some actual parameter changes have occurred by checking step count
            trainer.update_old_model()
            print(f"Updated old model at epoch {epoch + 1}, step {trainer.step}")
        elif (epoch + 1) % args.update_frequency == 0 and trainer.step < 20:
            print(f"Skipping old model update at epoch {epoch + 1} - only {trainer.step} steps completed")
        
        # Epoch summary
        avg_loss = sum(s["loss"] for s in epoch_stats) / len(epoch_stats)
        avg_clip_frac = sum(s["clip_fraction"] for s in epoch_stats) / len(epoch_stats)
        avg_reward = sum(s["reward_mean"] for s in epoch_stats) / len(epoch_stats)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average Clip Fraction: {avg_clip_frac:.4f}")
        print(f"  Average Reward: {avg_reward:.4f}")
        
        training_stats.append({
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_clip_fraction": avg_clip_frac,
            "avg_reward": avg_reward
        })
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_reward = evaluate_model(trainer, eval_data, reward_function)
    
    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)
    
    # Save training statistics
    stats_path = os.path.join(args.output_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            "args": vars(args),
            "initial_reward": initial_reward,
            "final_reward": final_reward,
            "best_reward": best_reward,
            "training_stats": training_stats
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Initial Reward: {initial_reward:.3f}")
    print(f"Final Reward: {final_reward:.3f}")
    print(f"Best Reward: {best_reward:.3f}")
    print(f"Improvement: {final_reward - initial_reward:.3f}")
    print(f"Models saved to: {args.output_dir}")
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "final/initial_reward": initial_reward,
            "final/final_reward": final_reward,
            "final/best_reward": best_reward,
            "final/improvement": final_reward - initial_reward
        })
        wandb.finish()
    
    # Final cleanup
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 