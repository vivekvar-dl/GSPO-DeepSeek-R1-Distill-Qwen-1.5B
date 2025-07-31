#!/usr/bin/env python3
"""
GSPO Training with Custom Dataset
Designed for guaranteed success and clear results
"""

import os
import json
import torch
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from gspo_implementation import GSPOTrainer, GSPOConfig
from custom_dataset import GSPOCustomDataset

def load_custom_dataset():
    """Load or create custom dataset"""
    if not os.path.exists("custom_train_dataset.json"):
        print("📊 Creating custom dataset...")
        generator = GSPOCustomDataset()
        train_data = generator.generate_dataset(200, {"easy": 0.6, "medium": 0.3, "hard": 0.1})
        eval_data = generator.generate_dataset(50, {"easy": 0.5, "medium": 0.3, "hard": 0.2})
        
        with open("custom_train_dataset.json", "w") as f:
            json.dump(train_data, f, indent=2)
        with open("custom_eval_dataset.json", "w") as f:
            json.dump(eval_data, f, indent=2)
    else:
        print("📊 Loading existing custom dataset...")
        with open("custom_train_dataset.json", "r") as f:
            train_data = json.load(f)
        with open("custom_eval_dataset.json", "r") as f:
            eval_data = json.load(f)
    
    return train_data, eval_data

def create_custom_reward_function():
    """Create reward function for custom dataset"""
    generator = GSPOCustomDataset()
    return generator.create_reward_function()

def evaluate_model(trainer, eval_data, reward_function):
    """Evaluate model on custom dataset"""
    print("🔍 Running evaluation...")
    
    total_reward = 0.0
    num_samples = min(len(eval_data), 20)
    correct_answers = 0
    
    for i, item in enumerate(eval_data[:num_samples]):
        query = item['query']
        target_answer = item['target_answer']
        
        # Generate response
        responses = trainer.generate_responses([query], max_new_tokens=128)
        response = responses[0]
        
        # Compute reward
        reward = reward_function(query, response, item)
        total_reward += reward
        
        # Check for correct answer
        if target_answer.lower() in response.lower():
            correct_answers += 1
        
        if i < 3:  # Print first few examples
            print(f"\n📝 Example {i+1} ({item['type']}):")
            print(f"Query: {query}")
            print(f"Response: {response[:150]}...")
            print(f"Target: {target_answer}")
            print(f"Reward: {reward:.3f}")
            print(f"Correct: {'✅' if target_answer.lower() in response.lower() else '❌'}")
    
    avg_reward = total_reward / num_samples
    accuracy = correct_answers / num_samples
    
    print(f"\n📊 Evaluation Results:")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Accuracy: {accuracy:.3f} ({correct_answers}/{num_samples})")
    
    return avg_reward, accuracy

def main():
    parser = argparse.ArgumentParser(description="Train GSPO on custom dataset")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="microsoft/DialoGPT-medium",
                       help="HuggingFace model name (smaller model for faster iteration)")
    
    # GSPO parameters - OPTIMIZED for custom dataset
    parser.add_argument("--learning_rate", type=float, default=2e-7,
                       help="Learning rate")
    parser.add_argument("--left_clip_range", type=float, default=3e-3,
                       help="Left clipping range")
    parser.add_argument("--right_clip_range", type=float, default=3e-3,
                       help="Right clipping range")
    parser.add_argument("--group_size", type=int, default=4,
                       help="Group size")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size")
    parser.add_argument("--update_frequency", type=int, default=1,
                       help="Old model update frequency")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Max sequence length")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./custom_gspo_results",
                       help="Output directory")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use wandb logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="gspo-custom-dataset",
            name=f"gspo-custom-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    print("🚀 GSPO Custom Dataset Training")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Clipping: [{args.left_clip_range}, {args.right_clip_range}]")
    print(f"Update Frequency: {args.update_frequency}")
    print("=" * 60)
    
    # Load model and tokenizer
    print(f"📥 Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # Use float32 for stability
        low_cpu_mem_usage=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"✅ Model loaded: {model.num_parameters() / 1e6:.1f}M parameters")
    print(f"🔧 Device: {device}")
    
    # Create GSPO configuration
    config = GSPOConfig(
        left_clip_range=args.left_clip_range,
        right_clip_range=args.right_clip_range,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gradient_accumulation_steps=2,
        use_8bit_optimizer=False,  # Use regular optimizer for stability
        use_gradient_checkpointing=False  # Disable for smaller models
    )
    
    # Initialize trainer
    trainer = GSPOTrainer(model, tokenizer, config, device)
    
    # Load custom dataset
    train_data, eval_data = load_custom_dataset()
    reward_function = create_custom_reward_function()
    
    print(f"📊 Dataset loaded:")
    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")
    
    # Show sample problems
    print("\n📝 Sample problems:")
    for i, sample in enumerate(train_data[:3]):
        print(f"{i+1}. {sample['type']} ({sample['difficulty']}):")
        print(f"   Query: {sample['query']}")
        print(f"   Target: {sample['target_answer']}")
    
    # Training reward function wrapper
    def training_reward_function(query: str, response: str):
        # Find corresponding data item
        for item in train_data:
            if item['query'] == query:
                return reward_function(query, response, item)
        # Fallback
        return 0.5
    
    # Initial evaluation
    print("\n" + "=" * 60)
    print("🔍 INITIAL EVALUATION")
    print("=" * 60)
    initial_reward, initial_accuracy = evaluate_model(trainer, eval_data, reward_function)
    
    # Training loop
    print("\n" + "=" * 60)
    print("🏋️ STARTING TRAINING")
    print("=" * 60)
    
    best_reward = initial_reward
    training_stats = []
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*20} EPOCH {epoch + 1}/{args.num_epochs} {'='*20}")
        
        # Shuffle training data
        import random
        queries = [item['query'] for item in train_data]
        random.shuffle(queries)
        
        epoch_stats = []
        
        # Training steps
        for step in range(0, len(queries), args.batch_size):
            batch_queries = queries[step:step + args.batch_size]
            
            # Training step
            stats = trainer.train_step(batch_queries, training_reward_function)
            epoch_stats.append(stats)
            
            # Log to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train/loss": stats["loss"],
                    "train/clip_fraction": stats["clip_fraction"],
                    "train/importance_ratio_mean": stats["importance_ratio_mean"],
                    "train/reward_mean": stats["reward_mean"],
                    "epoch": epoch,
                    "step": trainer.step
                })
            
            # Periodic evaluation
            if trainer.step % 10 == 0:
                eval_reward, eval_accuracy = evaluate_model(trainer, eval_data, reward_function)
                
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "eval/reward": eval_reward,
                        "eval/accuracy": eval_accuracy,
                        "step": trainer.step
                    })
                
                # Save best model
                if eval_reward > best_reward:
                    best_reward = eval_reward
                    checkpoint_path = os.path.join(args.output_dir, "best_model")
                    model.save_pretrained(checkpoint_path)
                    tokenizer.save_pretrained(checkpoint_path)
                    print(f"💾 New best model saved! Reward: {eval_reward:.3f}, Accuracy: {eval_accuracy:.3f}")
        
        # Update old model
        if (epoch + 1) % args.update_frequency == 0:
            trainer.update_old_model()
            print(f"🔄 Updated old model at epoch {epoch + 1}")
        
        # Epoch summary
        if epoch_stats:
            avg_loss = sum(s["loss"] for s in epoch_stats) / len(epoch_stats)
            avg_clip_frac = sum(s["clip_fraction"] for s in epoch_stats) / len(epoch_stats)
            avg_reward = sum(s["reward_mean"] for s in epoch_stats) / len(epoch_stats)
            
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Average Clip Fraction: {avg_clip_frac:.3f}")
            print(f"  Average Reward: {avg_reward:.3f}")
            
            training_stats.append({
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
                "avg_clip_fraction": avg_clip_frac,
                "avg_reward": avg_reward
            })
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("🎯 FINAL EVALUATION")
    print("=" * 60)
    final_reward, final_accuracy = evaluate_model(trainer, eval_data, reward_function)
    
    # Save final model
    final_checkpoint_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_checkpoint_path)
    tokenizer.save_pretrained(final_checkpoint_path)
    
    # Save training stats
    stats_path = os.path.join(args.output_dir, "training_stats.json")
    with open(stats_path, 'w') as f:
        json.dump({
            "args": vars(args),
            "initial_reward": initial_reward,
            "initial_accuracy": initial_accuracy,
            "final_reward": final_reward,
            "final_accuracy": final_accuracy,
            "best_reward": best_reward,
            "improvement": final_reward - initial_reward,
            "training_stats": training_stats
        }, f, indent=2)
    
    # Final results
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETED")
    print("=" * 60)
    print(f"Initial Reward: {initial_reward:.3f}")
    print(f"Final Reward: {final_reward:.3f}")
    print(f"Best Reward: {best_reward:.3f}")
    print(f"Improvement: {final_reward - initial_reward:.3f}")
    print(f"Initial Accuracy: {initial_accuracy:.3f}")
    print(f"Final Accuracy: {final_accuracy:.3f}")
    print(f"Models saved to: {args.output_dir}")
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "final/initial_reward": initial_reward,
            "final/final_reward": final_reward,
            "final/best_reward": best_reward,
            "final/improvement": final_reward - initial_reward,
            "final/initial_accuracy": initial_accuracy,
            "final/final_accuracy": final_accuracy
        })
        wandb.finish()
    
    return final_reward > initial_reward

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SUCCESS: Custom GSPO training improved performance!")
    else:
        print("\n⚠️ Training completed but no improvement detected.") 