#!/usr/bin/env python3
"""
Test script to evaluate the trained GSPO model
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from data_loader import create_reward_evaluator

def load_model(model_path, device="cuda"):
    """Load model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    """Generate response for a given prompt"""
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3
        )
    
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.size(1):], 
        skip_special_tokens=True
    )
    return response.strip()

def evaluate_model_on_tasks(model, tokenizer, num_samples=5):
    """Evaluate model on different tasks"""
    
    # Test prompts for different capabilities
    test_prompts = {
        "Math": [
            {"query": "Problem: What is 15 × 23?\nSolution:", "reference_answer": "345", "type": "math"},
            {"query": "Problem: If a rectangle has length 12 and width 8, what is its area?\nSolution:", "reference_answer": "96", "type": "math"},
            {"query": "Problem: Solve for x: 3x + 7 = 22\nSolution:", "reference_answer": "5", "type": "math"},
            {"query": "Problem: What is the square root of 144?\nSolution:", "reference_answer": "12", "type": "math"},
            {"query": "Problem: Calculate 25% of 80.\nSolution:", "reference_answer": "20", "type": "math"}
        ],
        "Reasoning": [
            {"query": "Context: A person is holding an umbrella and walking quickly. The ground is wet.\n\nWhat can we conclude?", "reference_answer": "It is raining", "type": "reasoning"},
            {"query": "Context: All cats have fur. Fluffy is a cat.\n\nWhat can we conclude about Fluffy?", "reference_answer": "Fluffy has fur", "type": "reasoning"},
            {"query": "Context: If it's raining, then the ground gets wet. The ground is wet.\n\nWhat can we conclude?", "reference_answer": "It might be raining", "type": "reasoning"},
            {"query": "Context: Sarah scored higher than John. John scored higher than Mike.\n\nWho scored the lowest?", "reference_answer": "Mike", "type": "reasoning"},
            {"query": "Context: A library closes at 5 PM. It's currently 6 PM.\n\nIs the library open?", "reference_answer": "No", "type": "reasoning"}
        ],
        "Code": [
            {"query": "Write a Python function to find the maximum number in a list:", "reference_answer": "def max_num(lst): return max(lst)", "type": "code"},
            {"query": "Write a function to check if a number is prime:", "reference_answer": "def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))", "type": "code"},
            {"query": "Write a Python function to reverse a string:", "reference_answer": "def reverse_string(s): return s[::-1]", "type": "code"},
            {"query": "Write code to calculate the factorial of a number:", "reference_answer": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)", "type": "code"},
            {"query": "Write a function to find the sum of all even numbers in a list:", "reference_answer": "def sum_even(lst): return sum(x for x in lst if x % 2 == 0)", "type": "code"}
        ]
    }
    
    # Create reward evaluator (no arguments needed)
    reward_evaluator = create_reward_evaluator()
    
    results = {}
    total_reward = 0
    total_samples = 0
    
    for task_type, prompts in test_prompts.items():
        print(f"\n{'='*50}")
        print(f"Testing {task_type} Tasks")
        print(f"{'='*50}")
        
        task_rewards = []
        
        for i, prompt_data in enumerate(prompts[:num_samples]):
            print(f"\n--- {task_type} Example {i+1} ---")
            print(f"Prompt: {prompt_data['query']}")
            
            response = generate_response(model, tokenizer, prompt_data['query'])
            print(f"Response: {response}")
            
            # Calculate reward using the correct signature
            reward = reward_evaluator(prompt_data['query'], response, prompt_data)
            task_rewards.append(reward)
            print(f"Reward: {reward:.3f}")
            
            total_reward += reward
            total_samples += 1
        
        avg_task_reward = sum(task_rewards) / len(task_rewards)
        results[task_type] = {
            "average_reward": avg_task_reward,
            "rewards": task_rewards
        }
        print(f"\n{task_type} Average Reward: {avg_task_reward:.3f}")
    
    overall_avg = total_reward / total_samples
    results["overall_average"] = overall_avg
    
    print(f"\n{'='*50}")
    print(f"OVERALL RESULTS")
    print(f"{'='*50}")
    print(f"Overall Average Reward: {overall_avg:.3f}")
    
    return results

def compare_models():
    """Compare base model vs trained GSPO model"""
    print("🔬 GSPO Model Evaluation")
    print("=" * 60)
    
    # Check if trained models exist
    best_model_path = "gspo_outputs/best_model"
    final_model_path = "gspo_outputs/final_model"
    
    if os.path.exists(best_model_path):
        print("✅ Found best GSPO model")
        model_path = best_model_path
    elif os.path.exists(final_model_path):
        print("✅ Found final GSPO model")
        model_path = final_model_path
    else:
        print("❌ No trained GSPO model found!")
        return
    
    # Load trained model
    try:
        trained_model, tokenizer = load_model(model_path)
        print(f"✅ Successfully loaded trained model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading trained model: {e}")
        return
    
    # Evaluate trained model
    print("\n🧠 Evaluating GSPO-trained model...")
    trained_results = evaluate_model_on_tasks(trained_model, tokenizer, num_samples=3)
    
    # Save results
    results_file = "gspo_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "trained_model_results": trained_results,
            "model_path": model_path
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {results_file}")
    
    return trained_results

if __name__ == "__main__":
    results = compare_models()
    
    if results:
        print("\n🎯 GSPO Training Assessment:")
        print(f"Overall Performance: {results['overall_average']:.3f}")
        
        for task_type in ["Math", "Reasoning", "Code"]:
            if task_type in results:
                avg = results[task_type]["average_reward"]
                print(f"{task_type} Performance: {avg:.3f}")
        
        # Performance interpretation
        overall = results['overall_average']
        if overall >= 0.8:
            print("🌟 Excellent performance!")
        elif overall >= 0.7:
            print("✅ Good performance!")
        elif overall >= 0.6:
            print("📈 Decent performance, room for improvement")
        else:
            print("📉 Performance needs improvement") 