#!/usr/bin/env python3
"""
Robust GSPO Training with Scaled Dataset
500 training + validation split + balanced evaluation
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

def create_balanced_eval_set(generator, num_samples=100):
    """Create evaluation set with balanced problem types"""
    
    problem_types = ["arithmetic_chain", "pattern_completion", "logical_reasoning", "word_problems"]
    samples_per_type = num_samples // len(problem_types)
    
    eval_data = []
    eval_id = 1
    
    for problem_type in problem_types:
        for difficulty in ["easy", "medium", "hard"]:
            # Distribute difficulties: 50% easy, 30% medium, 20% hard
            if difficulty == "easy":
                count = int(samples_per_type * 0.5)
            elif difficulty == "medium":
                count = int(samples_per_type * 0.3)
            else:  # hard
                count = int(samples_per_type * 0.2)
            
            for _ in range(count):
                if problem_type == "arithmetic_chain":
                    problem = generator.generate_arithmetic_chain(difficulty)
                elif problem_type == "pattern_completion":
                    problem = generator.generate_pattern_completion(difficulty)
                elif problem_type == "logical_reasoning":
                    problem = generator.generate_logical_reasoning(difficulty)
                else:  # word_problems
                    problem = generator.generate_word_problems(difficulty)
                
                problem["id"] = eval_id
                eval_data.append(problem)
                eval_id += 1
    
    return eval_data

def load_robust_dataset(train_size=500, eval_size=100, val_split=0.1):
    """Load or create robust dataset with validation split"""
    
    cache_file = f"robust_dataset_{train_size}_{eval_size}.json"
    
    if os.path.exists(cache_file):
        print(f"📊 Loading cached robust dataset: {cache_file}")
        with open(cache_file, "r") as f:
            data = json.load(f)
        return data["train_data"], data["val_data"], data["eval_data"]
    
    print(f"📊 Creating robust dataset: {train_size} train + {eval_size} eval...")
    generator = GSPOCustomDataset()
    
    # Generate training data with curriculum-friendly mix
    # More easy problems initially, balanced overall
    train_data = generator.generate_dataset(
        train_size, 
        {"easy": 0.5, "medium": 0.3, "hard": 0.2}
    )
    
    # Create validation split from training data
    val_size = int(train_size * val_split)
    val_data = train_data[-val_size:]
    train_data = train_data[:-val_size]
    
    # Create balanced evaluation set
    eval_data = create_balanced_eval_set(generator, eval_size)
    
    # Cache for reuse
    cache_data = {
        "train_data": train_data,
        "val_data": val_data,
        "eval_data": eval_data,
        "created": datetime.now().isoformat(),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "eval_size": len(eval_data)
    }
    
    with open(cache_file, "w") as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"✅ Dataset created and cached:")
    print(f"   Training: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples") 
    print(f"   Evaluation: {len(eval_data)} samples (balanced)")
    
    return train_data, val_data, eval_data

def evaluate_model_robust(trainer, eval_data, reward_function, name="Evaluation"):
    """Robust evaluation with detailed breakdown"""
    print(f"🔍 Running {name.lower()}...")
    
    total_reward = 0.0
    correct_answers = 0
    problem_type_stats = {}
    difficulty_stats = {}
    
    for item in eval_data:
        query = item['query']
        target_answer = item['target_answer']
        problem_type = item['type']
        difficulty = item['difficulty']
        
        # Generate response
        try:
            responses = trainer.generate_responses([query], max_new_tokens=128)
            response = responses[0]
            
            # Compute reward
            reward = reward_function(query, response, item)
            total_reward += reward
            
            # Check for correct answer
            is_correct = target_answer.lower() in response.lower()
            if is_correct:
                correct_answers += 1
            
            # Track by problem type
            if problem_type not in problem_type_stats:
                problem_type_stats[problem_type] = {'total': 0, 'correct': 0, 'reward_sum': 0.0}
            problem_type_stats[problem_type]['total'] += 1
            problem_type_stats[problem_type]['reward_sum'] += reward
            if is_correct:
                problem_type_stats[problem_type]['correct'] += 1
            
            # Track by difficulty
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'total': 0, 'correct': 0, 'reward_sum': 0.0}
            difficulty_stats[difficulty]['total'] += 1
            difficulty_stats[difficulty]['reward_sum'] += reward
            if is_correct:
                difficulty_stats[difficulty]['correct'] += 1
                
        except Exception as e:
            print(f"❌ Error in {name.lower()}: {e}")
            continue
    
    # Calculate metrics
    avg_reward = total_reward / len(eval_data) if eval_data else 0.0
    accuracy = correct_answers / len(eval_data) if eval_data else 0.0
    
    print(f"\n📊 {name} Results:")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Accuracy: {accuracy:.3f} ({correct_answers}/{len(eval_data)})")
    
    # Detailed breakdown
    print(f"\n📊 By Problem Type:")
    type_metrics = {}
    for ptype, stats in problem_type_stats.items():
        type_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        type_avg_reward = stats['reward_sum'] / stats['total'] if stats['total'] > 0 else 0.0
        type_metrics[ptype] = {'accuracy': type_accuracy, 'reward': type_avg_reward}
        print(f"  {ptype.replace('_', ' ').title()}: "
              f"Acc={type_accuracy:.3f}, Reward={type_avg_reward:.3f} ({stats['total']} samples)")
    
    return {
        "avg_reward": avg_reward,
        "accuracy": accuracy,
        "problem_type_stats": problem_type_stats,
        "difficulty_stats": difficulty_stats,
        "type_metrics": type_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Robust GSPO training")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    
    # Dataset scaling
    parser.add_argument("--train_size", type=int, default=500,
                       help="Training dataset size")
    parser.add_argument("--eval_size", type=int, default=100,
                       help="Evaluation dataset size")
    parser.add_argument("--val_split", type=float, default=0.1,
                       help="Validation split fraction")
    
    # GSPO parameters - optimized for robust training
    parser.add_argument("--learning_rate", type=float, default=1e-7,
                       help="Learning rate")
    parser.add_argument("--left_clip_range", type=float, default=2e-3,
                       help="Left clipping range")
    parser.add_argument("--right_clip_range", type=float, default=2e-3,
                       help="Right clipping range")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=4,
                       help="Gradient accumulation steps for stability")
    parser.add_argument("--update_frequency", type=int, default=1)
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="Number of epochs")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="Early stopping patience")
    
    # Curriculum learning
    parser.add_argument("--use_curriculum", action="store_true",
                       help="Enable curriculum learning")
    parser.add_argument("--curriculum_threshold", type=float, default=0.6,
                       help="Accuracy threshold to advance curriculum")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./robust_gspo_results")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="gspo-robust-training",
            name=f"robust-gspo-{args.train_size}-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    print("🚀 ROBUST GSPO TRAINING")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.train_size} train + {args.eval_size} eval")
    print(f"Validation split: {args.val_split:.1%}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Curriculum Learning: {'✅' if args.use_curriculum else '❌'}")
    print("=" * 70)
    
    # Load model
    print(f"📥 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"✅ Model loaded: {model.num_parameters() / 1e6:.1f}M parameters")
    
    # Create GSPO configuration
    config = GSPOConfig(
        left_clip_range=args.left_clip_range,
        right_clip_range=args.right_clip_range,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gradient_accumulation_steps=args.grad_accum_steps,
        use_8bit_optimizer=False,
        use_gradient_checkpointing=True  # Enable for larger datasets
    )
    
    trainer = GSPOTrainer(model, tokenizer, config, device)
    
    # Load robust dataset
    train_data, val_data, eval_data = load_robust_dataset(
        args.train_size, args.eval_size, args.val_split
    )
    
    generator = GSPOCustomDataset()
    reward_function = generator.create_reward_function()
    
    def training_reward_function(query: str, response: str):
        for item in train_data + val_data:
            if item['query'] == query:
                return reward_function(query, response, item)
        return 0.5
    
    # Initial evaluation
    print("\n" + "=" * 70)
    print("🔍 INITIAL BASELINE")
    print("=" * 70)
    initial_results = evaluate_model_robust(trainer, eval_data, reward_function, "Initial Evaluation")
    
    # Training with validation and early stopping
    print("\n" + "=" * 70)
    print("🏋️ ROBUST TRAINING START")
    print("=" * 70)
    
    best_val_reward = 0.0
    patience_counter = 0
    training_history = []
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*25} EPOCH {epoch + 1}/{args.num_epochs} {'='*25}")
        
        # Curriculum learning
        if args.use_curriculum:
            current_accuracy = initial_results["accuracy"] if epoch == 0 else training_history[-1]["val_accuracy"]
            if current_accuracy < args.curriculum_threshold:
                # Use easier problems
                epoch_queries = [item['query'] for item in train_data if item['difficulty'] != 'hard']
                print(f"📚 Curriculum: Using easy+medium problems (current acc: {current_accuracy:.3f})")
            else:
                # Use all problems
                epoch_queries = [item['query'] for item in train_data]
                print(f"🎓 Curriculum: Using all problems (advanced)")
        else:
            epoch_queries = [item['query'] for item in train_data]
        
        # Shuffle training data
        import random
        random.shuffle(epoch_queries)
        
        # Training loop
        epoch_stats = []
        for step in range(0, len(epoch_queries), args.batch_size):
            batch_queries = epoch_queries[step:step + args.batch_size]
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
        
        # Validation evaluation
        val_results = evaluate_model_robust(trainer, val_data, reward_function, "Validation")
        
        # Test evaluation
        test_results = evaluate_model_robust(trainer, eval_data, reward_function, "Test")
        
        # Epoch summary
        avg_loss = sum(s["loss"] for s in epoch_stats) / len(epoch_stats) if epoch_stats else 0.0
        avg_clip_frac = sum(s["clip_fraction"] for s in epoch_stats) / len(epoch_stats) if epoch_stats else 0.0
        
        epoch_summary = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "avg_clip_fraction": avg_clip_frac,
            "val_reward": val_results["avg_reward"],
            "val_accuracy": val_results["accuracy"],
            "test_reward": test_results["avg_reward"],
            "test_accuracy": test_results["accuracy"]
        }
        training_history.append(epoch_summary)
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_loss:.6f}")
        print(f"  Train Clip: {avg_clip_frac:.3f}")
        print(f"  Val Reward: {val_results['avg_reward']:.3f}")
        print(f"  Test Reward: {test_results['avg_reward']:.3f}")
        
        # Early stopping check
        if val_results["avg_reward"] > best_val_reward:
            best_val_reward = val_results["avg_reward"]
            patience_counter = 0
            # Save best model
            best_path = os.path.join(args.output_dir, "best_model")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            print(f"💾 New best model saved! Val reward: {best_val_reward:.3f}")
        else:
            patience_counter += 1
            print(f"⏳ Patience: {patience_counter}/{args.early_stopping_patience}")
        
        # Update old model
        if (epoch + 1) % args.update_frequency == 0:
            trainer.update_old_model()
            print(f"🔄 Updated old model")
        
        # Log to wandb
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "val/reward": val_results["avg_reward"],
                "val/accuracy": val_results["accuracy"],
                "test/reward": test_results["avg_reward"],
                "test/accuracy": test_results["accuracy"],
                "epoch": epoch + 1
            })
        
        # Early stopping
        if patience_counter >= args.early_stopping_patience:
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("🎯 FINAL ROBUST EVALUATION")
    print("=" * 70)
    final_results = evaluate_model_robust(trainer, eval_data, reward_function, "Final Evaluation")
    
    # Save comprehensive results
    results = {
        "args": vars(args),
        "initial_results": initial_results,
        "final_results": final_results,
        "training_history": training_history,
        "improvement": {
            "reward": final_results["avg_reward"] - initial_results["avg_reward"],
            "accuracy": final_results["accuracy"] - initial_results["accuracy"]
        }
    }
    
    results_path = os.path.join(args.output_dir, "robust_training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print(f"\n🏆 ROBUST TRAINING COMPLETED")
    print("=" * 70)
    print(f"Initial → Final Reward: {initial_results['avg_reward']:.3f} → {final_results['avg_reward']:.3f}")
    print(f"Initial → Final Accuracy: {initial_results['accuracy']:.3f} → {final_results['accuracy']:.3f}")
    print(f"Improvement: +{results['improvement']['reward']:.3f} reward, +{results['improvement']['accuracy']:.3f} accuracy")
    print(f"Best validation reward: {best_val_reward:.3f}")
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "final/reward_improvement": results['improvement']['reward'],
            "final/accuracy_improvement": results['improvement']['accuracy'],
            "final/best_val_reward": best_val_reward
        })
        wandb.finish()
    
    return results

if __name__ == "__main__":
    main() 