#!/usr/bin/env python3
"""
Custom Dataset for GSPO Validation
Designed to highlight sequence-level optimization benefits
"""

import random
import json
from typing import List, Dict, Tuple
import re

class GSPOCustomDataset:
    """Custom dataset designed to showcase GSPO effectiveness"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        self.templates = self._create_templates()
    
    def _create_templates(self):
        """Create problem templates that benefit from sequence-level optimization"""
        return {
            "arithmetic_chain": {
                "description": "Multi-step arithmetic requiring consistent logic",
                "difficulty_levels": ["easy", "medium", "hard"]
            },
            "pattern_completion": {
                "description": "Sequence patterns requiring holistic understanding", 
                "difficulty_levels": ["easy", "medium", "hard"]
            },
            "logical_reasoning": {
                "description": "Step-by-step logical deduction",
                "difficulty_levels": ["easy", "medium", "hard"]
            },
            "word_problems": {
                "description": "Simple word problems with clear answers",
                "difficulty_levels": ["easy", "medium", "hard"]
            }
        }
    
    def generate_arithmetic_chain(self, difficulty="easy"):
        """Generate multi-step arithmetic problems"""
        
        if difficulty == "easy":
            # 2-step problems
            a, b, c = random.randint(1, 20), random.randint(1, 20), random.randint(1, 20)
            operation1 = random.choice(["+", "-"])
            operation2 = random.choice(["+", "-"])
            
            # Calculate correct answer
            if operation1 == "+":
                intermediate = a + b
            else:
                intermediate = a - b
                
            if operation2 == "+":
                final_answer = intermediate + c
            else:
                final_answer = intermediate - c
            
            query = f"Calculate step by step: {a} {operation1} {b} {operation2} {c} = ?"
            reference = f"Step 1: {a} {operation1} {b} = {intermediate}\nStep 2: {intermediate} {operation2} {c} = {final_answer}\nAnswer: {final_answer}"
            
        elif difficulty == "medium":
            # 3-step problems with parentheses
            a, b, c, d = [random.randint(1, 15) for _ in range(4)]
            
            # (a + b) * c - d
            intermediate1 = a + b
            intermediate2 = intermediate1 * c
            final_answer = intermediate2 - d
            
            query = f"Calculate step by step: ({a} + {b}) × {c} - {d} = ?"
            reference = f"Step 1: ({a} + {b}) = {intermediate1}\nStep 2: {intermediate1} × {c} = {intermediate2}\nStep 3: {intermediate2} - {d} = {final_answer}\nAnswer: {final_answer}"
            
        else:  # hard
            # 4-step problems
            a, b, c, d, e = [random.randint(1, 10) for _ in range(5)]
            
            # ((a + b) * c - d) / e (rounded)
            intermediate1 = a + b
            intermediate2 = intermediate1 * c
            intermediate3 = intermediate2 - d
            final_answer = round(intermediate3 / e, 1)
            
            query = f"Calculate step by step: (({a} + {b}) × {c} - {d}) ÷ {e} = ? (round to 1 decimal)"
            reference = f"Step 1: ({a} + {b}) = {intermediate1}\nStep 2: {intermediate1} × {c} = {intermediate2}\nStep 3: {intermediate2} - {d} = {intermediate3}\nStep 4: {intermediate3} ÷ {e} = {final_answer}\nAnswer: {final_answer}"
        
        return {
            "query": query,
            "reference_answer": reference,
            "type": "arithmetic_chain",
            "difficulty": difficulty,
            "target_answer": str(final_answer)
        }
    
    def generate_pattern_completion(self, difficulty="easy"):
        """Generate pattern completion problems"""
        
        if difficulty == "easy":
            # Simple arithmetic sequences
            start = random.randint(1, 10)
            step = random.randint(2, 5)
            sequence = [start + i * step for i in range(4)]
            next_val = start + 4 * step
            
            query = f"Complete the pattern: {', '.join(map(str, sequence))}, ?"
            reference = f"Pattern: Add {step} each time\n{sequence[0]} + {step} = {sequence[1]}\n{sequence[1]} + {step} = {sequence[2]}\n{sequence[2]} + {step} = {sequence[3]}\n{sequence[3]} + {step} = {next_val}\nAnswer: {next_val}"
            
        elif difficulty == "medium":
            # Geometric sequences
            start = random.randint(2, 5)
            ratio = random.randint(2, 3)
            sequence = [start * (ratio ** i) for i in range(4)]
            next_val = start * (ratio ** 4)
            
            query = f"Complete the pattern: {', '.join(map(str, sequence))}, ?"
            reference = f"Pattern: Multiply by {ratio} each time\n{sequence[0]} × {ratio} = {sequence[1]}\n{sequence[1]} × {ratio} = {sequence[2]}\n{sequence[2]} × {ratio} = {sequence[3]}\n{sequence[3]} × {ratio} = {next_val}\nAnswer: {next_val}"
            
        else:  # hard
            # Mixed operations
            a, b = random.randint(1, 5), random.randint(2, 4)
            sequence = [a, a*b, a*b+a, (a*b+a)*b]
            next_val = (a*b+a)*b + a
            
            query = f"Complete the complex pattern: {', '.join(map(str, sequence))}, ?"
            reference = f"Pattern analysis:\n{sequence[0]} → ×{b} → {sequence[1]}\n{sequence[1]} → +{a} → {sequence[2]}\n{sequence[2]} → ×{b} → {sequence[3]}\n{sequence[3]} → +{a} → {next_val}\nAnswer: {next_val}"
        
        return {
            "query": query,
            "reference_answer": reference,
            "type": "pattern_completion", 
            "difficulty": difficulty,
            "target_answer": str(next_val)
        }
    
    def generate_logical_reasoning(self, difficulty="easy"):
        """Generate logical reasoning problems"""
        
        names = ["Alice", "Bob", "Carol", "Dave", "Emma"]
        colors = ["red", "blue", "green", "yellow", "purple"]
        
        if difficulty == "easy":
            # Simple deduction
            person1, person2 = random.sample(names, 2)
            color1, color2 = random.sample(colors, 2)
            
            query = f"Logic puzzle: {person1} likes {color1}. {person2} doesn't like {color1}. If {person2} likes either {color1} or {color2}, what color does {person2} like?"
            reference = f"Given:\n1. {person1} likes {color1}\n2. {person2} doesn't like {color1}\n3. {person2} likes either {color1} or {color2}\n\nSince {person2} doesn't like {color1}, and must like either {color1} or {color2}, {person2} must like {color2}.\nAnswer: {color2}"
            target = color2
            
        elif difficulty == "medium":
            # 3-person deduction
            person1, person2, person3 = random.sample(names, 3)
            color1, color2, color3 = random.sample(colors, 3)
            
            query = f"Logic puzzle: {person1} likes {color1}. {person2} likes {color2}. {person3} doesn't like {color1} or {color2}. If {person3} must choose from {color1}, {color2}, or {color3}, what does {person3} choose?"
            reference = f"Given:\n1. {person1} likes {color1}\n2. {person2} likes {color2}\n3. {person3} doesn't like {color1} or {color2}\n4. {person3} must choose from {color1}, {color2}, or {color3}\n\nSince {person3} doesn't like {color1} or {color2}, and must choose from the three options, {person3} must choose {color3}.\nAnswer: {color3}"
            target = color3
            
        else:  # hard
            # 4-step deduction
            person1, person2, person3, person4 = random.sample(names, 4)
            
            query = f"Logic puzzle: In a race, {person1} finished before {person2}. {person2} finished before {person3}. {person4} finished after {person1} but before {person2}. What is the finishing order?"
            
            # Order: person1, person4, person2, person3
            reference = f"Given:\n1. {person1} before {person2}\n2. {person2} before {person3}\n3. {person4} after {person1} but before {person2}\n\nDeduction:\n- From 1&2: {person1} → {person2} → {person3}\n- From 3: {person1} → {person4} → {person2}\n- Combined: {person1} → {person4} → {person2} → {person3}\n\nAnswer: {person1}, {person4}, {person2}, {person3}"
            target = f"{person1}, {person4}, {person2}, {person3}"
        
        return {
            "query": query,
            "reference_answer": reference,
            "type": "logical_reasoning",
            "difficulty": difficulty,
            "target_answer": target
        }
    
    def generate_word_problems(self, difficulty="easy"):
        """Generate simple word problems"""
        
        if difficulty == "easy":
            # Simple addition/subtraction
            items = ["apples", "books", "toys", "cookies", "marbles"]
            item = random.choice(items)
            start = random.randint(5, 20)
            change = random.randint(1, 10)
            operation = random.choice(["gained", "lost", "bought", "gave away"])
            
            if operation in ["gained", "bought"]:
                result = start + change
                query = f"Sarah had {start} {item}. She {operation} {change} more {item}. How many {item} does she have now?"
                reference = f"Starting amount: {start} {item}\nChange: +{change} {item} ({operation})\nFinal amount: {start} + {change} = {result} {item}\nAnswer: {result}"
            else:
                result = start - change
                query = f"Sarah had {start} {item}. She {operation} {change} {item}. How many {item} does she have now?"
                reference = f"Starting amount: {start} {item}\nChange: -{change} {item} ({operation})\nFinal amount: {start} - {change} = {result} {item}\nAnswer: {result}"
        
        elif difficulty == "medium":
            # Two-step word problems
            item = random.choice(["dollars", "stickers", "cards", "points"])
            start = random.randint(10, 50)
            earn = random.randint(5, 20)
            spend = random.randint(3, 15)
            
            after_earn = start + earn
            final = after_earn - spend
            
            query = f"Tom started with {start} {item}. He earned {earn} more {item}, then spent {spend} {item}. How many {item} does he have left?"
            reference = f"Starting: {start} {item}\nAfter earning: {start} + {earn} = {after_earn} {item}\nAfter spending: {after_earn} - {spend} = {final} {item}\nAnswer: {final}"
        
        else:  # hard
            # Three-step word problems
            people = ["friends", "students", "workers", "players"]
            group = random.choice(people)
            start = random.randint(8, 20)
            join = random.randint(2, 8)
            leave = random.randint(1, 5)
            multiply = random.randint(2, 3)
            
            after_join = start + join
            after_leave = after_join - leave
            final = after_leave * multiply
            
            query = f"A team started with {start} {group}. {join} more {group} joined, then {leave} {group} left. If each remaining person brings {multiply-1} more {group}, how many {group} are there total?"
            reference = f"Step 1: Started with {start} {group}\nStep 2: After {join} joined: {start} + {join} = {after_join} {group}\nStep 3: After {leave} left: {after_join} - {leave} = {after_leave} {group}\nStep 4: Each brings {multiply-1} more: {after_leave} × {multiply} = {final} {group}\nAnswer: {final}"
        
        return {
            "query": query,
            "reference_answer": reference,
            "type": "word_problems",
            "difficulty": difficulty,
            "target_answer": str(final if 'final' in locals() else result)
        }
    
    def generate_dataset(self, num_samples=100, difficulty_mix=None):
        """Generate a balanced custom dataset"""
        
        if difficulty_mix is None:
            difficulty_mix = {"easy": 0.5, "medium": 0.3, "hard": 0.2}
        
        dataset = []
        problem_types = ["arithmetic_chain", "pattern_completion", "logical_reasoning", "word_problems"]
        
        for i in range(num_samples):
            # Choose problem type
            problem_type = random.choice(problem_types)
            
            # Choose difficulty based on mix
            rand_val = random.random()
            if rand_val < difficulty_mix["easy"]:
                difficulty = "easy"
            elif rand_val < difficulty_mix["easy"] + difficulty_mix["medium"]:
                difficulty = "medium"
            else:
                difficulty = "hard"
            
            # Generate problem
            if problem_type == "arithmetic_chain":
                problem = self.generate_arithmetic_chain(difficulty)
            elif problem_type == "pattern_completion":
                problem = self.generate_pattern_completion(difficulty)
            elif problem_type == "logical_reasoning":
                problem = self.generate_logical_reasoning(difficulty)
            else:  # word_problems
                problem = self.generate_word_problems(difficulty)
            
            problem["id"] = i + 1
            dataset.append(problem)
        
        return dataset
    
    def create_reward_function(self):
        """Create a reward function optimized for this dataset"""
        
        def custom_reward_function(query: str, response: str, data_item: Dict):
            """Reward function for custom dataset"""
            
            target_answer = data_item.get("target_answer", "")
            problem_type = data_item.get("type", "")
            difficulty = data_item.get("difficulty", "easy")
            
            # Base reward structure
            base_rewards = {"easy": 0.3, "medium": 0.5, "hard": 0.7}
            base_reward = base_rewards.get(difficulty, 0.3)
            
            # Check for exact answer match
            if target_answer.lower() in response.lower():
                exact_match_bonus = 0.4
            else:
                exact_match_bonus = 0.0
            
            # Check for step-by-step reasoning (sequence-level quality)
            reasoning_indicators = ["step", "first", "then", "next", "finally", "therefore", "so", "because"]
            reasoning_count = sum(1 for indicator in reasoning_indicators if indicator.lower() in response.lower())
            reasoning_bonus = min(reasoning_count * 0.05, 0.2)
            
            # Check for mathematical expressions (for arithmetic problems)
            if problem_type == "arithmetic_chain":
                math_patterns = ["+", "-", "×", "÷", "=", "(", ")"]
                math_count = sum(1 for pattern in math_patterns if pattern in response)
                math_bonus = min(math_count * 0.02, 0.1)
            else:
                math_bonus = 0.0
            
            # Length appropriateness (not too short, not too long)
            response_length = len(response.split())
            if 10 <= response_length <= 100:
                length_bonus = 0.1
            elif 5 <= response_length <= 150:
                length_bonus = 0.05
            else:
                length_bonus = 0.0
            
            # Final reward calculation
            total_reward = base_reward + exact_match_bonus + reasoning_bonus + math_bonus + length_bonus
            total_reward = min(total_reward, 1.0)  # Cap at 1.0
            
            return total_reward
        
        return custom_reward_function

def create_custom_gspo_dataset():
    """Create and save custom dataset for GSPO training"""
    
    generator = GSPOCustomDataset()
    
    # Generate training and evaluation sets
    train_data = generator.generate_dataset(200, {"easy": 0.6, "medium": 0.3, "hard": 0.1})
    eval_data = generator.generate_dataset(50, {"easy": 0.5, "medium": 0.3, "hard": 0.2})
    
    # Save datasets
    with open("custom_train_dataset.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open("custom_eval_dataset.json", "w") as f:
        json.dump(eval_data, f, indent=2)
    
    print("📊 Custom Dataset Created!")
    print(f"Training samples: {len(train_data)}")
    print(f"Evaluation samples: {len(eval_data)}")
    print("\nSample problems:")
    
    for i, sample in enumerate(train_data[:3]):
        print(f"\n{i+1}. {sample['type']} ({sample['difficulty']}):")
        print(f"Query: {sample['query']}")
        print(f"Target: {sample['target_answer']}")
    
    return train_data, eval_data

if __name__ == "__main__":
    create_custom_gspo_dataset() 