#!/usr/bin/env python3
"""
Initial Evaluation Test for Custom GSPO Dataset
Check baseline performance before training
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from custom_dataset import GSPOCustomDataset

def test_initial_evaluation():
    """Test initial model performance on custom dataset"""
    
    print("🔍 INITIAL EVALUATION TEST")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Model configuration
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"📥 Loading model: {model_name}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded: {model.num_parameters() / 1e6:.1f}M parameters")
    print(f"🔧 Device: {device}")
    
    # Create custom dataset
    print("\n📊 Creating custom dataset...")
    generator = GSPOCustomDataset()
    
    # Generate small test set
    test_data = generator.generate_dataset(20, {"easy": 0.6, "medium": 0.3, "hard": 0.1})
    reward_function = generator.create_reward_function()
    
    print(f"Generated {len(test_data)} test problems")
    
    # Show sample problems
    print("\n📝 Sample Problems:")
    for i, sample in enumerate(test_data[:3]):
        print(f"\n{i+1}. {sample['type'].replace('_', ' ').title()} ({sample['difficulty']}):")
        print(f"   Query: {sample['query']}")
        print(f"   Target Answer: {sample['target_answer']}")
    
    # Test generation function
    def generate_response(query, max_new_tokens=128):
        """Generate response for a query"""
        inputs = tokenizer(
            query, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=384
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("🎯 BASELINE EVALUATION RESULTS")
    print("=" * 60)
    
    total_reward = 0.0
    correct_answers = 0
    problem_type_stats = {}
    difficulty_stats = {}
    
    for i, item in enumerate(test_data):
        query = item['query']
        target_answer = item['target_answer']
        problem_type = item['type']
        difficulty = item['difficulty']
        
        # Generate response
        try:
            response = generate_response(query)
            
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
            
            # Show first few examples
            if i < 5:
                print(f"\n📝 Example {i+1} ({problem_type}, {difficulty}):")
                print(f"Query: {query}")
                print(f"Response: {response[:200]}{'...' if len(response) > 200 else ''}")
                print(f"Target: {target_answer}")
                print(f"Reward: {reward:.3f}")
                print(f"Correct: {'✅' if is_correct else '❌'}")
                
        except Exception as e:
            print(f"❌ Error processing item {i+1}: {e}")
            continue
    
    # Overall results
    avg_reward = total_reward / len(test_data)
    accuracy = correct_answers / len(test_data)
    
    print(f"\n📊 OVERALL BASELINE PERFORMANCE:")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Accuracy: {accuracy:.3f} ({correct_answers}/{len(test_data)})")
    
    # By problem type
    print(f"\n📊 BY PROBLEM TYPE:")
    for ptype, stats in problem_type_stats.items():
        type_accuracy = stats['correct'] / stats['total']
        type_avg_reward = stats['reward_sum'] / stats['total']
        print(f"  {ptype.replace('_', ' ').title()}: "
              f"Accuracy={type_accuracy:.3f} ({stats['correct']}/{stats['total']}), "
              f"Avg Reward={type_avg_reward:.3f}")
    
    # By difficulty
    print(f"\n📊 BY DIFFICULTY:")
    for diff, stats in difficulty_stats.items():
        diff_accuracy = stats['correct'] / stats['total']
        diff_avg_reward = stats['reward_sum'] / stats['total']
        print(f"  {diff.title()}: "
              f"Accuracy={diff_accuracy:.3f} ({stats['correct']}/{stats['total']}), "
              f"Avg Reward={diff_avg_reward:.3f}")
    
    print(f"\n🎯 TRAINING POTENTIAL:")
    if avg_reward < 0.6:
        print("✅ Good baseline - lots of room for GSPO improvement!")
        potential_improvement = (0.8 - avg_reward) / avg_reward * 100
        print(f"📈 Potential improvement: ~{potential_improvement:.0f}%")
    else:
        print("⚠️ High baseline - may need harder problems")
    
    if accuracy < 0.4:
        print("✅ Good accuracy baseline for demonstrating GSPO benefits")
    else:
        print("⚠️ High accuracy baseline - GSPO improvements may be subtle")
    
    print(f"\n🚀 NEXT STEP:")
    print("Run full GSPO training with:")
    print("python train_custom_gspo.py --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --use_wandb")
    
    return {
        "avg_reward": avg_reward,
        "accuracy": accuracy,
        "problem_type_stats": problem_type_stats,
        "difficulty_stats": difficulty_stats
    }

if __name__ == "__main__":
    test_initial_evaluation() 