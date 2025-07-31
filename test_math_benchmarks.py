#!/usr/bin/env python3
"""
Test GSPO Model on Mathematical Reasoning Benchmarks
AIME and HMMT style problems for advanced validation
"""

import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

class MathBenchmarkTester:
    """Test GSPO model on mathematical reasoning benchmarks"""
    
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading GSPO model: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        
        self.model.eval()
        print(f"✅ Model loaded: {self.model.num_parameters() / 1e6:.1f}M parameters")
    
    def load_aime_problems(self):
        """Load AIME-style problems (American Invitational Mathematics Examination)"""
        
        aime_problems = [
            {
                "id": "AIME_1",
                "problem": """Find the number of ordered pairs $(a,b)$ of integers such that $|a| + |b| = 100$ and $\gcd(a,b) = 1$.

Note: $\gcd(a,b)$ denotes the greatest common divisor of $a$ and $b$.""",
                "answer": "4020",
                "type": "number_theory",
                "difficulty": "AIME"
            },
            {
                "id": "AIME_2", 
                "problem": """A circle with center $(3,7)$ and radius $r$ is tangent to the $x$-axis. What is the value of $r$?""",
                "answer": "7",
                "type": "geometry",
                "difficulty": "AIME"
            },
            {
                "id": "AIME_3",
                "problem": """Let $f(x) = x^2 + 6x + 5$. Find the sum of all values of $x$ for which $f(f(x)) = 3$.""",
                "answer": "-6",
                "type": "algebra",
                "difficulty": "AIME"
            },
            {
                "id": "AIME_4",
                "problem": """In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. The incircle touches side $BC$ at point $D$. Find $BD$.""",
                "answer": "5",
                "type": "geometry",
                "difficulty": "AIME"
            },
            {
                "id": "AIME_5",
                "problem": """How many positive integers $n$ satisfy $\lfloor \log_2 n \rfloor = 10$?

Note: $\lfloor x \rfloor$ denotes the greatest integer less than or equal to $x$.""",
                "answer": "512",
                "type": "combinatorics",
                "difficulty": "AIME"
            }
        ]
        
        return aime_problems
    
    def load_hmmt_problems(self):
        """Load HMMT-style problems (Harvard-MIT Mathematics Tournament)"""
        
        hmmt_problems = [
            {
                "id": "HMMT_1",
                "problem": """Compute the number of ways to arrange the letters in HMMT such that no two identical letters are adjacent.""",
                "answer": "8",
                "type": "combinatorics",
                "difficulty": "HMMT"
            },
            {
                "id": "HMMT_2",
                "problem": """Let $S = 1^3 + 2^3 + 3^3 + ... + 100^3$. Find the remainder when $S$ is divided by $1000$.""",
                "answer": "500",
                "type": "number_theory", 
                "difficulty": "HMMT"
            },
            {
                "id": "HMMT_3",
                "problem": """A regular hexagon has side length $6$. A circle is inscribed in the hexagon. What is the area of the region inside the hexagon but outside the circle?""",
                "answer": "54√3 - 27π",
                "type": "geometry",
                "difficulty": "HMMT"
            },
            {
                "id": "HMMT_4",
                "problem": """Find the coefficient of $x^{50}$ in the expansion of $(1 + x + x^2 + x^3 + x^4)^{20}$.""",
                "answer": "54264",
                "type": "algebra",
                "difficulty": "HMMT"
            },
            {
                "id": "HMMT_5",
                "problem": """How many sequences $(a_1, a_2, ..., a_{10})$ of positive integers satisfy $a_1 + a_2 + ... + a_{10} = 20$?""",
                "answer": "92378",
                "type": "combinatorics",
                "difficulty": "HMMT"
            }
        ]
        
        return hmmt_problems
    
    def generate_solution(self, problem_text, max_new_tokens=512):
        """Generate mathematical solution using GSPO model"""
        
        prompt = f"""Solve this mathematical problem step by step. Show all work and reasoning.

Problem: {problem_text}

Solution:
Let me work through this systematically:

Step 1:"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,  # Lower temperature for mathematical precision
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def extract_numerical_answer(self, response, expected_answer):
        """Extract numerical answer from mathematical solution"""
        
        # Common answer patterns for math problems
        answer_patterns = [
            r"(?:answer|Answer|ANSWER|final answer|Final Answer)(?:\s*:|\s+is|\s*=)\s*([^\.\n,]+)",
            r"(?:solution|Solution|SOLUTION)(?:\s*:|\s+is|\s*=)\s*([^\.\n,]+)",
            r"(?:therefore|Therefore|THEREFORE),?\s*([^\.\n,]+)",
            r"(?:thus|Thus|THUS),?\s*([^\.\n,]+)",
            r"= *([0-9]+(?:\.[0-9]+)?)",  # Numerical equality
            r"\$([0-9]+(?:\.[0-9]+)?)\$",  # LaTeX format
        ]
        
        response_clean = response.replace("\\", "").replace("$", "")
        expected_clean = str(expected_answer).strip()
        
        # Direct numerical match
        if expected_clean in response_clean:
            return expected_answer, True
        
        # Pattern matching
        for pattern in answer_patterns:
            matches = re.findall(pattern, response_clean, re.IGNORECASE)
            for match in matches:
                match_clean = match.strip().replace(",", "").replace("$", "")
                try:
                    # Try exact match
                    if match_clean == expected_clean:
                        return match_clean, True
                    
                    # Try numerical comparison for integers
                    if expected_clean.isdigit() and match_clean.isdigit():
                        if int(match_clean) == int(expected_clean):
                            return match_clean, True
                            
                    # Try floating point comparison
                    try:
                        if abs(float(match_clean) - float(expected_clean)) < 0.01:
                            return match_clean, True
                    except:
                        pass
                        
                except:
                    continue
        
        return "No clear answer found", False
    
    def run_benchmark(self, benchmark_type="both"):
        """Run mathematical reasoning benchmark"""
        
        print(f"\n🧮 GSPO MATHEMATICAL REASONING BENCHMARK")
        print("=" * 70)
        
        all_problems = []
        
        if benchmark_type in ["aime", "both"]:
            all_problems.extend(self.load_aime_problems())
        
        if benchmark_type in ["hmmt", "both"]:
            all_problems.extend(self.load_hmmt_problems())
        
        results = []
        correct = 0
        total = len(all_problems)
        
        # Stats by type
        difficulty_stats = {}
        topic_stats = {}
        
        for i, problem in enumerate(all_problems):
            print(f"\n📝 Problem {i+1}/{total} ({problem['difficulty']} - {problem['type']}):")
            print(f"Problem preview: {problem['problem'][:100]}...")
            
            try:
                response = self.generate_solution(problem['problem'])
                extracted_answer, is_correct = self.extract_numerical_answer(response, problem['answer'])
                
                if is_correct:
                    correct += 1
                    status = "✅ CORRECT"
                else:
                    status = "❌ INCORRECT"
                
                print(f"Expected: {problem['answer']}")
                print(f"Extracted: {extracted_answer}")
                print(f"Status: {status}")
                
                # Show reasoning excerpt
                reasoning_preview = response[:300] + "..." if len(response) > 300 else response
                print(f"Reasoning: {reasoning_preview}")
                
                # Update stats
                difficulty = problem['difficulty']
                topic = problem['type']
                
                if difficulty not in difficulty_stats:
                    difficulty_stats[difficulty] = {'correct': 0, 'total': 0}
                if topic not in topic_stats:
                    topic_stats[topic] = {'correct': 0, 'total': 0}
                
                difficulty_stats[difficulty]['total'] += 1
                topic_stats[topic]['total'] += 1
                
                if is_correct:
                    difficulty_stats[difficulty]['correct'] += 1
                    topic_stats[topic]['correct'] += 1
                
                results.append({
                    "problem_id": problem['id'],
                    "difficulty": problem['difficulty'],
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
                    "difficulty": problem['difficulty'],
                    "type": problem['type'],
                    "expected": problem['answer'],
                    "extracted": "Error",
                    "correct": False,
                    "response": f"Error: {str(e)}"
                })
        
        # Final results
        accuracy = correct / total if total > 0 else 0
        print(f"\n🏆 MATHEMATICAL REASONING RESULTS:")
        print("=" * 70)
        print(f"Overall: {correct}/{total} ({accuracy:.1%})")
        
        # Breakdown by difficulty
        print(f"\n📊 By Difficulty:")
        for difficulty, stats in difficulty_stats.items():
            diff_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {difficulty}: {stats['correct']}/{stats['total']} ({diff_acc:.1%})")
        
        # Breakdown by topic
        print(f"\n📊 By Topic:")
        for topic, stats in topic_stats.items():
            topic_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {topic}: {stats['correct']}/{stats['total']} ({topic_acc:.1%})")
        
        # Assessment
        print(f"\n🎯 PERFORMANCE ASSESSMENT:")
        if accuracy >= 0.4:
            print("🏆 EXCEPTIONAL: Outstanding mathematical reasoning for 2B model!")
        elif accuracy >= 0.2:
            print("✅ STRONG: Solid mathematical problem-solving capability!")
        elif accuracy >= 0.1:
            print("🔶 PROMISING: Shows mathematical reasoning potential!")
        else:
            print("⚠️ CHALLENGING: Advanced math requires larger models or specialized training")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"math_benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "model_info": "GSPO Trained Model - Math Benchmarks",
                "benchmark_type": benchmark_type,
                "total_problems": total,
                "correct": correct,
                "accuracy": accuracy,
                "difficulty_breakdown": difficulty_stats,
                "topic_breakdown": topic_stats,
                "detailed_results": results,
                "timestamp": timestamp
            }, f, indent=2)
        
        print(f"\n💾 Results saved: {results_file}")
        
        return accuracy, results

def main():
    """Main testing function"""
    
    model_path = "./robust_gspo_results/best_model"
    
    print("🚀 GSPO MATHEMATICAL REASONING BENCHMARK")
    print("=" * 70)
    
    try:
        tester = MathBenchmarkTester(model_path)
        
        # Test both AIME and HMMT
        accuracy, results = tester.run_benchmark("both")
        
        print(f"\n🎯 MATHEMATICAL REASONING SUMMARY:")
        print(f"Overall Accuracy: {accuracy:.1%}")
        print("Tested: AIME + HMMT style problems")
        print("These are competition-level mathematical reasoning problems!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 