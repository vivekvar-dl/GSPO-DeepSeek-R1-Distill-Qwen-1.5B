#!/usr/bin/env python3
"""
Test Trained GSPO Model on Real Reasoning Benchmarks
Focus on ZebraLogic for logical reasoning validation
"""

import torch
import json
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import re

class ZebraLogicTester:
    """Test GSPO model on ZebraLogic benchmark"""
    
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading trained GSPO model from: {model_path}")
        
        # Load your trained GSPO model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.model.eval()
        print(f"✅ GSPO model loaded: {self.model.num_parameters() / 1e6:.1f}M parameters")
    
    def load_zebra_logic_samples(self):
        """Load ZebraLogic-style problems"""
        
        # Sample ZebraLogic problems for testing
        zebra_problems = [
            {
                "id": 1,
                "problem": """Logic Puzzle: There are 5 houses in a row, each painted a different color. In each house lives a person of different nationality. Each person drinks a different beverage, smokes a different brand of cigar, and keeps a different pet.

Clues:
1. The Brit lives in the red house
2. The Swede keeps dogs as pets
3. The Dane drinks tea
4. The green house is on the left of the white house
5. The green house's owner drinks coffee
6. The person who smokes Pall Mall rears birds
7. The owner of the yellow house smokes Dunhill
8. The man living in the center house drinks milk
9. The Norwegian lives in the first house
10. The man who smokes Blends lives next to the one who keeps cats
11. The man who keeps horses lives next to the man who smokes Dunhill
12. The owner who smokes BlueMaster drinks beer
13. The German smokes Prince
14. The Norwegian lives next to the blue house
15. The man who smokes Blends has a neighbor who drinks water

Question: Who keeps the fish?""",
                "answer": "German",
                "type": "classic_zebra"
            },
            {
                "id": 2,
                "problem": """Logic Puzzle: Three friends Alice, Bob, and Carol have different favorite colors (red, blue, green) and different pets (cat, dog, bird).

Clues:
1. Alice does not like red
2. The person who likes blue has a cat
3. Bob does not have a dog
4. Carol's favorite color is not green
5. The person with the bird likes green

Question: What pet does Alice have?""",
                "answer": "dog",
                "type": "simple_logic"
            },
            {
                "id": 3,
                "problem": """Logic Puzzle: Four students (Amy, Ben, Carl, Dana) study different subjects (Math, Science, Art, History) and live in different dorms (North, South, East, West).

Clues:
1. Amy doesn't study Math or Science
2. The student in North dorm studies Science
3. Ben lives in East dorm
4. Carl doesn't live in North or South
5. The student studying Art lives in South dorm
6. Dana studies Math

Question: Where does Amy live?""",
                "answer": "South",
                "type": "assignment_logic"
            },
            {
                "id": 4,
                "problem": """Logic Puzzle: Five people sit in a row. From left to right, determine their order.

Clues:
1. John is not at either end
2. Mary is somewhere to the left of Peter
3. Susan is immediately to the right of David
4. Peter is not next to John
5. David is not at the right end

Question: If Mary is in position 1, what is the complete seating order from left to right?""",
                "answer": "Mary, David, Susan, John, Peter",
                "type": "sequence_logic"
            },
            {
                "id": 5,
                "problem": """Logic Puzzle: Three boxes contain different items.

Clues:
1. Box A is not empty
2. If Box B contains apples, then Box C contains oranges
3. Box C does not contain apples
4. Either Box A or Box B contains apples (but not both)
5. If Box A contains apples, then Box B is empty

Question: Which box contains apples?""",
                "answer": "Box A",
                "type": "conditional_logic"
            }
        ]
        
        return zebra_problems
    
    def generate_solution(self, problem_text, max_new_tokens=256):
        """Generate solution using trained GSPO model"""
        
        # Create step-by-step prompt
        prompt = f"""Solve this logic puzzle step by step:

{problem_text}

Let me work through this systematically:

Step 1:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def extract_answer(self, response, expected_answer):
        """Extract final answer from model response"""
        
        # Look for various answer patterns
        answer_patterns = [
            r"(?:answer|Answer|ANSWER)(?:\s*:|\s+is|\s*=)\s*([^.\n]+)",
            r"(?:solution|Solution|SOLUTION)(?:\s*:|\s+is|\s*=)\s*([^.\n]+)",
            r"(?:therefore|Therefore|THEREFORE),?\s*([^.\n]+)",
            r"(?:so|So|SO),?\s*([^.\n]+)",
            expected_answer.lower()  # Direct match
        ]
        
        response_lower = response.lower()
        expected_lower = expected_answer.lower()
        
        # Check direct answer presence
        if expected_lower in response_lower:
            return expected_answer, True
        
        # Extract structured answer
        for pattern in answer_patterns[:-1]:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                # Check if extracted answer matches expected
                if expected_lower in extracted.lower():
                    return extracted, True
                return extracted, False
        
        return "No clear answer found", False
    
    def run_benchmark(self):
        """Run full ZebraLogic benchmark"""
        
        print("\n🧩 GSPO MODEL ZEBRALOGIC BENCHMARK")
        print("=" * 60)
        
        problems = self.load_zebra_logic_samples()
        results = []
        
        correct = 0
        total = len(problems)
        
        for i, problem in enumerate(problems):
            print(f"\n📝 Problem {i+1}/{total} ({problem['type']}):")
            print(f"Problem preview: {problem['problem'][:150]}...")
            
            # Generate solution
            try:
                response = self.generate_solution(problem['problem'])
                extracted_answer, is_correct = self.extract_answer(response, problem['answer'])
                
                if is_correct:
                    correct += 1
                    status = "✅ CORRECT"
                else:
                    status = "❌ INCORRECT"
                
                print(f"Expected: {problem['answer']}")
                print(f"Extracted: {extracted_answer}")
                print(f"Status: {status}")
                
                # Show reasoning excerpt
                reasoning_preview = response[:200] + "..." if len(response) > 200 else response
                print(f"Reasoning: {reasoning_preview}")
                
                results.append({
                    "problem_id": problem['id'],
                    "type": problem['type'],
                    "expected": problem['answer'],
                    "extracted": extracted_answer,
                    "correct": is_correct,
                    "response": response
                })
                
            except Exception as e:
                print(f"❌ Error processing problem {i+1}: {e}")
                results.append({
                    "problem_id": problem['id'],
                    "type": problem['type'],
                    "expected": problem['answer'],
                    "extracted": "Error",
                    "correct": False,
                    "response": f"Error: {str(e)}"
                })
        
        # Final results
        accuracy = correct / total
        print(f"\n🏆 ZEBRALOGIC BENCHMARK RESULTS:")
        print("=" * 60)
        print(f"Correct: {correct}/{total}")
        print(f"Accuracy: {accuracy:.1%}")
        
        # Breakdown by problem type
        type_stats = {}
        for result in results:
            ptype = result['type']
            if ptype not in type_stats:
                type_stats[ptype] = {'correct': 0, 'total': 0}
            type_stats[ptype]['total'] += 1
            if result['correct']:
                type_stats[ptype]['correct'] += 1
        
        print(f"\n📊 By Problem Type:")
        for ptype, stats in type_stats.items():
            type_acc = stats['correct'] / stats['total']
            print(f"  {ptype}: {stats['correct']}/{stats['total']} ({type_acc:.1%})")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"zebralogic_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "model_info": "GSPO Trained Model",
                "total_problems": total,
                "correct": correct,
                "accuracy": accuracy,
                "type_breakdown": type_stats,
                "detailed_results": results,
                "timestamp": timestamp
            }, f, indent=2)
        
        print(f"\n💾 Detailed results saved: {results_file}")
        
        return accuracy, results

def main():
    """Main benchmark testing function"""
    
    # Test your trained GSPO model
    model_path = "./robust_gspo_results/best_model"  # Adjust path as needed
    
    print("🚀 TESTING GSPO MODEL ON ZEBRALOGIC BENCHMARK")
    print("=" * 60)
    
    try:
        tester = ZebraLogicTester(model_path)
        accuracy, results = tester.run_benchmark()
        
        print(f"\n🎯 GSPO BENCHMARK PERFORMANCE:")
        print(f"ZebraLogic Accuracy: {accuracy:.1%}")
        
        if accuracy >= 0.6:
            print("🏆 EXCELLENT: Strong logical reasoning performance!")
        elif accuracy >= 0.4:
            print("✅ GOOD: Solid reasoning with GSPO optimization!")
        elif accuracy >= 0.2:
            print("🔶 DECENT: GSPO showing some reasoning capability")
        else:
            print("⚠️ NEEDS IMPROVEMENT: Consider larger model or more training")
            
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}")
        print("Make sure your GSPO training completed and saved the best model")
    except Exception as e:
        print(f"❌ Error running benchmark: {e}")

if __name__ == "__main__":
    main() 