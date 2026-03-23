import json
import random
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import os
import re

class DatasetLoader:
    """Unified data loader for various RL training datasets"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
    
    def load_gsm8k(self, split: str = "train", max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load GSM8K dataset (grade school math)"""
        dataset = load_dataset("gsm8k", "main", split=split)
        
        data = []
        for item in dataset:
            if max_samples and len(data) >= max_samples:
                break
                
            query = f"Question: {item['question']}\nAnswer:"
            answer = item['answer']
            
            data.append({
                "query": query,
                "reference_answer": answer,
                "type": "math"
            })
        
        return data
    
    def load_humaneval(self, max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load HumanEval dataset (code generation)"""
        dataset = load_dataset("openai_humaneval", split="test")
        
        data = []
        for item in dataset:
            if max_samples and len(data) >= max_samples:
                break
            
            # Format as a coding problem
            query = f"Complete the following Python function:\n\n{item['prompt']}"
            
            data.append({
                "query": query,
                "reference_answer": item['canonical_solution'],
                "test_cases": item['test'],
                "entry_point": item['entry_point'],
                "type": "code"
            })
        
        return data
    
    def load_hellaswag(self, split: str = "validation", max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load HellaSwag dataset (commonsense reasoning)"""
        dataset = load_dataset("hellaswag", split=split)
        
        data = []
        for item in dataset:
            if max_samples and len(data) >= max_samples:
                break
            
            # Format as multiple choice
            context = item['ctx']
            endings = item['endings']
            correct_idx = int(item['label'])
            
            query = f"Context: {context}\n\nChoose the best ending:\n"
            for i, ending in enumerate(endings):
                query += f"{chr(65+i)}) {ending}\n"
            query += "\nAnswer:"
            
            correct_answer = chr(65 + correct_idx)
            
            data.append({
                "query": query,
                "reference_answer": correct_answer,
                "type": "reasoning"
            })
        
        return data
    
    def load_math_problems(self, difficulty: str = "easy", max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load custom math problems of varying difficulty"""
        
        if difficulty == "easy":
            problems = [
                {"question": "What is 15 + 27?", "answer": "42"},
                {"question": "Solve for x: 3x = 21", "answer": "x = 7"},
                {"question": "What is the area of a rectangle with length 8 and width 5?", "answer": "40"},
                {"question": "If a train travels 60 km in 2 hours, what is its speed?", "answer": "30 km/h"},
                {"question": "What is 12 × 8?", "answer": "96"},
                {"question": "Solve: 2x + 4 = 12. What is x?", "answer": "x = 4"},
                {"question": "What is 144 ÷ 12?", "answer": "12"},
                {"question": "Find the perimeter of a square with side length 6.", "answer": "24"},
                {"question": "What is 25% of 80?", "answer": "20"},
                {"question": "Solve: x - 5 = 12. What is x?", "answer": "x = 17"}
            ]
        elif difficulty == "medium":
            problems = [
                {"question": "Solve the quadratic equation: x² - 5x + 6 = 0", "answer": "x = 2 or x = 3"},
                {"question": "Find the derivative of f(x) = 3x² + 2x - 1", "answer": "f'(x) = 6x + 2"},
                {"question": "What is the sum of the first 10 positive integers?", "answer": "55"},
                {"question": "Calculate the area of a circle with radius 5.", "answer": "25π ≈ 78.54"},
                {"question": "Solve the system: 2x + y = 7, x - y = 2", "answer": "x = 3, y = 1"},
                {"question": "What is log₂(32)?", "answer": "5"},
                {"question": "Find the slope of the line passing through (2,3) and (6,11).", "answer": "2"},
                {"question": "What is the factorial of 6?", "answer": "720"},
                {"question": "Solve: 2^x = 16. What is x?", "answer": "x = 4"},
                {"question": "Find the volume of a cylinder with radius 3 and height 8.", "answer": "72π ≈ 226.19"}
            ]
        elif difficulty == "hard":
            problems = [
                {"question": "Find the limit: lim(x→0) (sin(x)/x)", "answer": "1"},
                {"question": "Solve the differential equation: dy/dx = y, with y(0) = 1", "answer": "y = e^x"},
                {"question": "Calculate the integral: ∫(2x + 3)dx from 0 to 2", "answer": "10"},
                {"question": "Find the eigenvalues of the matrix [[2,1],[1,2]]", "answer": "λ₁ = 3, λ₂ = 1"},
                {"question": "What is the sum of the infinite series: 1 + 1/2 + 1/4 + 1/8 + ...?", "answer": "2"},
                {"question": "Solve: x³ - 6x² + 11x - 6 = 0", "answer": "x = 1, x = 2, x = 3"},
                {"question": "Find the Taylor series of e^x around x = 0 (first 4 terms)", "answer": "1 + x + x²/2! + x³/3! + ..."},
                {"question": "What is the determinant of [[1,2,3],[4,5,6],[7,8,9]]?", "answer": "0"},
                {"question": "Find the critical points of f(x) = x³ - 3x² + 2", "answer": "x = 0, x = 2"},
                {"question": "Calculate: ∫₀^π sin(x)dx", "answer": "2"}
            ]
        
        data = []
        for problem in problems:
            if max_samples and len(data) >= max_samples:
                break
            
            query = f"Problem: {problem['question']}\nSolution:"
            
            data.append({
                "query": query,
                "reference_answer": problem['answer'],
                "type": "math",
                "difficulty": difficulty
            })
        
        return data
    
    def load_medical_qa(self, difficulty: str = "easy", max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load medical QA problems. Attempts to load from HuggingFace datasets first,
        then falls back to built-in curated medical questions."""

        # Try to load from HuggingFace medical datasets
        try:
            dataset = load_dataset("medmcqa", split="train", trust_remote_code=True)
            difficulty_map = {"easy": 0, "medium": 1, "hard": 2}
            target_subject = difficulty_map.get(difficulty, 0)

            data = []
            for item in dataset:
                if max_samples and len(data) >= max_samples:
                    break
                # medmcqa has opa/opb/opc/opd options and cop (correct option index 0-3)
                options = [
                    item.get("opa", ""),
                    item.get("opb", ""),
                    item.get("opc", ""),
                    item.get("opd", ""),
                ]
                correct_idx = item.get("cop", 0)
                correct_letter = chr(65 + correct_idx)
                option_text = "\n".join(
                    f"{chr(65+i)}) {opt}" for i, opt in enumerate(options) if opt
                )
                query = (
                    f"Medical Question: {item['question']}\n\n"
                    f"Options:\n{option_text}\n\nAnswer:"
                )
                data.append({
                    "query": query,
                    "reference_answer": correct_letter,
                    "explanation": item.get("exp", ""),
                    "type": "medical",
                    "difficulty": difficulty,
                })
            if data:
                return data
        except Exception:
            pass  # Fall through to built-in problems

        # Built-in curated medical questions
        if difficulty == "easy":
            problems = [
                {
                    "question": "Which organ is responsible for pumping blood throughout the body?",
                    "options": ["A) Liver", "B) Lungs", "C) Heart", "D) Kidneys"],
                    "answer": "C",
                    "explanation": "The heart is the muscular organ that pumps blood through the circulatory system.",
                },
                {
                    "question": "What is the normal resting heart rate for adults?",
                    "options": ["A) 20-40 bpm", "B) 60-100 bpm", "C) 120-160 bpm", "D) 150-200 bpm"],
                    "answer": "B",
                    "explanation": "A normal resting heart rate for adults is 60-100 beats per minute.",
                },
                {
                    "question": "Which vitamin is produced when skin is exposed to sunlight?",
                    "options": ["A) Vitamin A", "B) Vitamin B12", "C) Vitamin C", "D) Vitamin D"],
                    "answer": "D",
                    "explanation": "Vitamin D is synthesized in the skin upon exposure to ultraviolet B (UVB) sunlight.",
                },
                {
                    "question": "What is the function of red blood cells?",
                    "options": [
                        "A) Fight infection",
                        "B) Transport oxygen",
                        "C) Produce antibodies",
                        "D) Regulate blood pressure",
                    ],
                    "answer": "B",
                    "explanation": "Red blood cells (erythrocytes) carry oxygen from the lungs to tissues via hemoglobin.",
                },
                {
                    "question": "Which part of the brain controls balance and coordination?",
                    "options": ["A) Cerebrum", "B) Medulla oblongata", "C) Cerebellum", "D) Thalamus"],
                    "answer": "C",
                    "explanation": "The cerebellum coordinates voluntary movements, posture, balance, and coordination.",
                },
                {
                    "question": "What is the largest organ of the human body?",
                    "options": ["A) Liver", "B) Brain", "C) Skin", "D) Intestine"],
                    "answer": "C",
                    "explanation": "The skin is the largest organ, covering the entire external surface of the body.",
                },
                {
                    "question": "Which blood type is the universal donor?",
                    "options": ["A) AB+", "B) O-", "C) A+", "D) B-"],
                    "answer": "B",
                    "explanation": "O- (O negative) is the universal donor for red blood cells as it lacks A, B, and Rh antigens.",
                },
                {
                    "question": "What does the acronym 'BMI' stand for?",
                    "options": [
                        "A) Blood Mass Index",
                        "B) Body Mass Index",
                        "C) Bone Mineral Index",
                        "D) Blood Mineral Indicator",
                    ],
                    "answer": "B",
                    "explanation": "BMI stands for Body Mass Index, a measure of body fat based on height and weight.",
                },
                {
                    "question": "How many bones are in the adult human body?",
                    "options": ["A) 106", "B) 206", "C) 306", "D) 406"],
                    "answer": "B",
                    "explanation": "The adult human body has 206 bones.",
                },
                {
                    "question": "Which hormone regulates blood sugar levels?",
                    "options": ["A) Estrogen", "B) Cortisol", "C) Insulin", "D) Thyroxine"],
                    "answer": "C",
                    "explanation": "Insulin, produced by the pancreas, regulates blood glucose by facilitating cellular uptake.",
                },
            ]
        elif difficulty == "medium":
            problems = [
                {
                    "question": "A patient presents with fever, productive cough, and lobar consolidation on chest X-ray. What is the most likely diagnosis?",
                    "options": [
                        "A) Pulmonary tuberculosis",
                        "B) Community-acquired pneumonia",
                        "C) Pulmonary embolism",
                        "D) Lung cancer",
                    ],
                    "answer": "B",
                    "explanation": "Fever, productive cough, and lobar consolidation are classic features of community-acquired pneumonia.",
                },
                {
                    "question": "Which enzyme is elevated in myocardial infarction and is most cardiac-specific?",
                    "options": ["A) AST", "B) LDH", "C) Troponin I", "D) CK-MM"],
                    "answer": "C",
                    "explanation": "Cardiac troponin I (cTnI) is highly specific to cardiac muscle and is the gold-standard biomarker for MI.",
                },
                {
                    "question": "What is the mechanism of action of beta-blockers?",
                    "options": [
                        "A) Block calcium channels",
                        "B) Block beta-adrenergic receptors",
                        "C) Inhibit ACE enzyme",
                        "D) Block sodium channels",
                    ],
                    "answer": "B",
                    "explanation": "Beta-blockers competitively block beta-adrenergic receptors, reducing heart rate and blood pressure.",
                },
                {
                    "question": "Which cranial nerve is responsible for the pupillary light reflex?",
                    "options": [
                        "A) CN II only",
                        "B) CN III only",
                        "C) CN II (afferent) and CN III (efferent)",
                        "D) CN V and CN VII",
                    ],
                    "answer": "C",
                    "explanation": "Light detected by CN II triggers CN III (oculomotor) to constrict the pupil via the Edinger-Westphal nucleus.",
                },
                {
                    "question": "What is the first-line treatment for type 2 diabetes mellitus?",
                    "options": ["A) Insulin", "B) Metformin", "C) Glibenclamide", "D) Sitagliptin"],
                    "answer": "B",
                    "explanation": "Metformin is the recommended first-line pharmacological treatment for type 2 DM due to its efficacy and safety profile.",
                },
                {
                    "question": "Which electrolyte imbalance is most commonly associated with cardiac arrhythmias?",
                    "options": ["A) Hyponatremia", "B) Hypokalemia", "C) Hypercalcemia", "D) Hypermagnesemia"],
                    "answer": "B",
                    "explanation": "Hypokalemia decreases the resting membrane potential of cardiac cells, predisposing to arrhythmias.",
                },
                {
                    "question": "A patient with jaundice, pale stools, and dark urine most likely has which type of jaundice?",
                    "options": [
                        "A) Pre-hepatic (hemolytic)",
                        "B) Hepatic (hepatocellular)",
                        "C) Post-hepatic (obstructive)",
                        "D) Neonatal physiological",
                    ],
                    "answer": "C",
                    "explanation": "Pale stools (lack of urobilinogen) and dark urine (bilirubin) with jaundice indicate biliary obstruction.",
                },
                {
                    "question": "What does the P wave on an ECG represent?",
                    "options": [
                        "A) Ventricular depolarization",
                        "B) Ventricular repolarization",
                        "C) Atrial depolarization",
                        "D) Atrial repolarization",
                    ],
                    "answer": "C",
                    "explanation": "The P wave represents depolarization of the atria, triggering atrial contraction.",
                },
                {
                    "question": "Which bacteria is the most common cause of community-acquired urinary tract infections?",
                    "options": [
                        "A) Staphylococcus aureus",
                        "B) Escherichia coli",
                        "C) Klebsiella pneumoniae",
                        "D) Pseudomonas aeruginosa",
                    ],
                    "answer": "B",
                    "explanation": "E. coli causes approximately 80% of uncomplicated community-acquired UTIs.",
                },
                {
                    "question": "What is the antidote for paracetamol (acetaminophen) overdose?",
                    "options": [
                        "A) Naloxone",
                        "B) Flumazenil",
                        "C) N-acetylcysteine",
                        "D) Atropine",
                    ],
                    "answer": "C",
                    "explanation": "N-acetylcysteine replenishes glutathione stores, preventing toxic metabolite accumulation in paracetamol overdose.",
                },
            ]
        else:  # hard
            problems = [
                {
                    "question": "A 45-year-old presents with proximal muscle weakness, heliotrope rash, and elevated CK. Anti-Jo-1 antibodies are positive. What is the diagnosis?",
                    "options": [
                        "A) Systemic lupus erythematosus",
                        "B) Dermatomyositis with antisynthetase syndrome",
                        "C) Polymyalgia rheumatica",
                        "D) Myasthenia gravis",
                    ],
                    "answer": "B",
                    "explanation": "Heliotrope rash, proximal weakness, elevated CK, and anti-Jo-1 (antisynthetase antibody) indicate dermatomyositis with antisynthetase syndrome.",
                },
                {
                    "question": "Which mutation in CFTR is most commonly responsible for cystic fibrosis?",
                    "options": [
                        "A) G551D",
                        "B) ΔF508 (F508del)",
                        "C) W1282X",
                        "D) R117H",
                    ],
                    "answer": "B",
                    "explanation": "The ΔF508 deletion accounts for approximately 70% of CF alleles worldwide.",
                },
                {
                    "question": "A patient with nephrotic syndrome has hypoalbuminemia and marked proteinuria. Which pathological finding on electron microscopy is most consistent with minimal change disease?",
                    "options": [
                        "A) Subepithelial immune deposits ('humps')",
                        "B) Mesangial IgA deposits",
                        "C) Diffuse effacement of podocyte foot processes",
                        "D) Thickened GBM with 'tram-track' appearance",
                    ],
                    "answer": "C",
                    "explanation": "Minimal change disease is characterized by diffuse effacement (fusion) of podocyte foot processes on EM with normal LM.",
                },
                {
                    "question": "Which enzyme deficiency causes phenylketonuria (PKU)?",
                    "options": [
                        "A) Homogentisate oxidase",
                        "B) Phenylalanine hydroxylase",
                        "C) Tyrosinase",
                        "D) Fumarylacetoacetate hydrolase",
                    ],
                    "answer": "B",
                    "explanation": "PKU is caused by deficiency of phenylalanine hydroxylase, which converts phenylalanine to tyrosine.",
                },
                {
                    "question": "A child presents with recurrent infections, partial albinism, and giant lysosomal granules in leukocytes. What syndrome does this describe?",
                    "options": [
                        "A) Wiskott-Aldrich syndrome",
                        "B) Chédiak-Higashi syndrome",
                        "C) Chronic granulomatous disease",
                        "D) Leukocyte adhesion deficiency",
                    ],
                    "answer": "B",
                    "explanation": "Chédiak-Higashi syndrome involves a LYST gene mutation causing giant lysosomes, partial albinism, and neutrophil dysfunction.",
                },
                {
                    "question": "In the RAAS pathway, which enzyme converts angiotensin I to angiotensin II?",
                    "options": [
                        "A) Renin",
                        "B) Aldosterone synthase",
                        "C) Angiotensin-converting enzyme (ACE)",
                        "D) Neprilysin",
                    ],
                    "answer": "C",
                    "explanation": "ACE (found mainly in pulmonary endothelium) cleaves two amino acids from angiotensin I to produce the active angiotensin II.",
                },
                {
                    "question": "Which HLA type is most strongly associated with ankylosing spondylitis?",
                    "options": ["A) HLA-DR4", "B) HLA-B27", "C) HLA-DR3", "D) HLA-A3"],
                    "answer": "B",
                    "explanation": "HLA-B27 is present in >90% of patients with ankylosing spondylitis.",
                },
                {
                    "question": "A patient on warfarin therapy has a supratherapeutic INR of 8.0 with minor bleeding. What is the appropriate management?",
                    "options": [
                        "A) Administer IV vitamin K only",
                        "B) Hold warfarin, give oral vitamin K, and monitor",
                        "C) Administer fresh frozen plasma immediately",
                        "D) Administer protamine sulfate",
                    ],
                    "answer": "B",
                    "explanation": "For supratherapeutic INR with minor bleeding, holding warfarin and giving oral vitamin K with monitoring is appropriate.",
                },
                {
                    "question": "Which cell type is primarily responsible for delayed-type hypersensitivity (Type IV) reactions?",
                    "options": [
                        "A) B lymphocytes",
                        "B) Mast cells",
                        "C) CD4+ T helper cells (Th1)",
                        "D) Eosinophils",
                    ],
                    "answer": "C",
                    "explanation": "Type IV hypersensitivity is mediated by sensitized CD4+ Th1 cells that activate macrophages upon re-exposure to antigen.",
                },
                {
                    "question": "What is the mechanism of action of heparin as an anticoagulant?",
                    "options": [
                        "A) Inhibits vitamin K epoxide reductase",
                        "B) Directly inhibits thrombin only",
                        "C) Activates antithrombin III to inhibit thrombin and factor Xa",
                        "D) Blocks platelet ADP receptor",
                    ],
                    "answer": "C",
                    "explanation": "Heparin binds and activates antithrombin III, which then rapidly inhibits thrombin (IIa) and factor Xa.",
                },
            ]

        data = []
        for problem in problems:
            if max_samples and len(data) >= max_samples:
                break

            option_text = "\n".join(problem["options"])
            query = (
                f"Medical Question: {problem['question']}\n\n"
                f"Options:\n{option_text}\n\nAnswer:"
            )

            data.append({
                "query": query,
                "reference_answer": problem["answer"],
                "explanation": problem["explanation"],
                "type": "medical",
                "difficulty": difficulty,
            })

        return data

    def load_code_problems(self, difficulty: str = "easy", max_samples: Optional[int] = None) -> List[Dict[str, str]]:
        """Load custom coding problems"""
        
        if difficulty == "easy":
            problems = [
                {
                    "prompt": "def add_numbers(a, b):\n    # Complete this function to return the sum of a and b\n    pass",
                    "solution": "def add_numbers(a, b):\n    return a + b"
                },
                {
                    "prompt": "def is_even(n):\n    # Return True if n is even, False otherwise\n    pass",
                    "solution": "def is_even(n):\n    return n % 2 == 0"
                },
                {
                    "prompt": "def max_of_three(a, b, c):\n    # Return the maximum of three numbers\n    pass",
                    "solution": "def max_of_three(a, b, c):\n    return max(a, b, c)"
                },
                {
                    "prompt": "def reverse_string(s):\n    # Return the reverse of string s\n    pass",
                    "solution": "def reverse_string(s):\n    return s[::-1]"
                },
                {
                    "prompt": "def count_vowels(text):\n    # Count the number of vowels in text\n    pass",
                    "solution": "def count_vowels(text):\n    vowels = 'aeiouAEIOU'\n    return sum(1 for char in text if char in vowels)"
                }
            ]
        elif difficulty == "medium":
            problems = [
                {
                    "prompt": "def fibonacci(n):\n    # Return the nth Fibonacci number\n    pass",
                    "solution": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
                },
                {
                    "prompt": "def binary_search(arr, target):\n    # Implement binary search\n    pass",
                    "solution": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
                },
                {
                    "prompt": "def merge_sorted_arrays(arr1, arr2):\n    # Merge two sorted arrays\n    pass",
                    "solution": "def merge_sorted_arrays(arr1, arr2):\n    result = []\n    i, j = 0, 0\n    while i < len(arr1) and j < len(arr2):\n        if arr1[i] <= arr2[j]:\n            result.append(arr1[i])\n            i += 1\n        else:\n            result.append(arr2[j])\n            j += 1\n    result.extend(arr1[i:])\n    result.extend(arr2[j:])\n    return result"
                }
            ]
        
        data = []
        for problem in problems:
            if max_samples and len(data) >= max_samples:
                break
            
            query = f"Complete the following function:\n\n{problem['prompt']}\n\nSolution:"
            
            data.append({
                "query": query,
                "reference_answer": problem['solution'],
                "type": "code",
                "difficulty": difficulty
            })
        
        return data
    
    def create_mixed_dataset(
        self,
        math_ratio: float = 0.4,
        code_ratio: float = 0.4,
        reasoning_ratio: float = 0.2,
        medical_ratio: float = 0.0,
        total_samples: int = 100
    ) -> List[Dict[str, str]]:
        """Create a mixed dataset from different sources"""
        
        math_samples = int(total_samples * math_ratio)
        code_samples = int(total_samples * code_ratio)
        reasoning_samples = int(total_samples * reasoning_ratio)
        medical_samples = int(total_samples * medical_ratio)
        
        data = []
        
        # Add math problems
        data.extend(self.load_math_problems("easy", math_samples // 2))
        data.extend(self.load_math_problems("medium", math_samples // 2))
        
        # Add code problems
        data.extend(self.load_code_problems("easy", code_samples // 2))
        data.extend(self.load_code_problems("medium", code_samples // 2))
        
        # Add reasoning problems
        try:
            data.extend(self.load_hellaswag(max_samples=reasoning_samples))
        except:
            # If HuggingFace datasets not available, skip reasoning
            print("Warning: Could not load HellaSwag dataset, skipping reasoning problems")
        
        # Add medical problems
        if medical_samples > 0:
            data.extend(self.load_medical_qa("easy", medical_samples // 2))
            data.extend(self.load_medical_qa("medium", medical_samples - medical_samples // 2))
        
        # Shuffle the combined dataset
        random.shuffle(data)
        
        return data[:total_samples]

def create_reward_evaluator():
    """Create reward functions for different problem types"""
    
    def evaluate_math_response(query: str, response: str, reference: str) -> float:
        """Evaluate math response quality"""
        response_lower = response.lower().strip()
        reference_lower = reference.lower().strip()
        
        # Extract numbers from both responses
        response_numbers = re.findall(r'-?\d+\.?\d*', response)
        reference_numbers = re.findall(r'-?\d+\.?\d*', reference)
        
        base_reward = 0.0
        
        # Check for exact match
        if response_lower == reference_lower:
            return 1.0
        
        # Check if response contains the reference answer
        if reference_lower in response_lower:
            base_reward = 0.8
        
        # Check if numbers match
        elif response_numbers and reference_numbers:
            if any(rnum in reference_numbers for rnum in response_numbers):
                base_reward = 0.6
        
        # Check for mathematical keywords
        math_keywords = ['solve', 'answer', 'equals', '=', 'result', 'solution']
        if any(keyword in response_lower for keyword in math_keywords):
            base_reward = max(base_reward, 0.3)
        
        # Check if response has reasonable length
        if len(response.strip()) > 5:
            base_reward = max(base_reward, 0.2)
        
        # Add small random noise for training stability
        noise = random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, base_reward + noise))
    
    def evaluate_code_response(query: str, response: str, reference: str) -> float:
        """Evaluate code response quality"""
        response_lower = response.lower().strip()
        
        base_reward = 0.0
        
        # Check for code structure
        code_patterns = ['def ', 'return', 'if', 'for', 'while', 'class']
        code_score = sum(1 for pattern in code_patterns if pattern in response_lower)
        
        if code_score >= 2:
            base_reward = 0.7
        elif code_score >= 1:
            base_reward = 0.4
        
        # Check for proper indentation (rough proxy for code quality)
        lines = response.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        if indented_lines > 0:
            base_reward = max(base_reward, 0.3)
        
        # Check if response is not empty
        if len(response.strip()) > 10:
            base_reward = max(base_reward, 0.2)
        
        noise = random.uniform(-0.05, 0.05)
        return max(0.0, min(1.0, base_reward + noise))
    
    def evaluate_reasoning_response(query: str, response: str, reference: str) -> float:
        """Evaluate reasoning response quality"""
        response_clean = response.strip().upper()
        reference_clean = reference.strip().upper()
        
        # For multiple choice, check if the letter matches
        if len(response_clean) == 1 and response_clean.isalpha():
            if response_clean == reference_clean:
                return 1.0
            else:
                return 0.0
        
        # Check if response contains the reference
        if reference_clean in response_clean:
            return 0.8
        
        # Basic quality check
        if len(response.strip()) > 5:
            return 0.3
        
        return 0.1

    def evaluate_medical_response(query: str, response: str, reference: str) -> float:
        """Evaluate medical QA response quality.

        Medical questions are formatted as multiple-choice (A/B/C/D). The reward is
        determined primarily by whether the model selects the correct option letter,
        with a bonus for including clinical reasoning.
        """
        response_clean = response.strip()
        reference_clean = reference.strip().upper()

        # Scan for a selected option letter at the start of the response or after
        # common answer prefixes (e.g. "Answer: B", "The answer is C", "B)")
        # Try explicit answer patterns first
        match = re.search(
            r'\b(?:answer[:\s]+|the answer is[:\s]+|correct(?:\s+answer)?[:\s]+)?([A-D])[)\.]?\b',
            response_clean,
            re.IGNORECASE,
        )
        selected_letter = match.group(1).upper() if match else None

        base_reward = 0.0
        if selected_letter is not None:
            if selected_letter == reference_clean:
                base_reward = 1.0
            else:
                base_reward = 0.0
        elif reference_clean in response_clean.upper():
            # Reference letter appears somewhere in the free-form response
            base_reward = 0.7
        elif len(response_clean) > 5:
            base_reward = 0.2

        # Bonus for clinical reasoning keywords
        reasoning_keywords = [
            "because", "therefore", "due to", "caused by", "mechanism",
            "diagnosis", "treatment", "pathophysiology", "symptom", "sign",
        ]
        reasoning_bonus = min(
            sum(0.02 for kw in reasoning_keywords if kw in response_clean.lower()), 0.1
        )

        noise = random.uniform(-0.03, 0.03)
        return max(0.0, min(1.0, base_reward + reasoning_bonus + noise))

    def unified_reward_function(query: str, response: str, data_item: Dict[str, str]) -> float:
        """Unified reward function that routes to appropriate evaluator"""
        problem_type = data_item.get('type', 'math')
        reference = data_item.get('reference_answer', '')
        
        if problem_type == 'math':
            return evaluate_math_response(query, response, reference)
        elif problem_type == 'code':
            return evaluate_code_response(query, response, reference)
        elif problem_type == 'reasoning':
            return evaluate_reasoning_response(query, response, reference)
        elif problem_type == 'medical':
            return evaluate_medical_response(query, response, reference)
        else:
            # Default to math evaluation
            return evaluate_math_response(query, response, reference)
    
    return unified_reward_function

# Example usage
if __name__ == "__main__":
    loader = DatasetLoader()
    
    # Test different datasets
    print("Loading GSM8K...")
    gsm8k_data = loader.load_gsm8k(max_samples=5)
    for item in gsm8k_data[:2]:
        print(f"Query: {item['query'][:100]}...")
        print(f"Answer: {item['reference_answer'][:100]}...")
        print()
    
    print("Loading Math Problems...")
    math_data = loader.load_math_problems("easy", max_samples=3)
    for item in math_data:
        print(f"Query: {item['query']}")
        print(f"Answer: {item['reference_answer']}")
        print()

    print("Loading Medical QA Problems...")
    medical_data = loader.load_medical_qa("easy", max_samples=3)
    for item in medical_data:
        print(f"Query: {item['query'][:120]}...")
        print(f"Answer: {item['reference_answer']}")
        print()
    
    print("Creating Mixed Dataset...")
    mixed_data = loader.create_mixed_dataset(total_samples=10)
    print(f"Created {len(mixed_data)} samples")
    
    # Test reward function
    reward_fn = create_reward_evaluator()
    test_item = mixed_data[0]
    test_response = "The answer is 42"
    reward = reward_fn(test_item['query'], test_response, test_item)
    print(f"Test reward: {reward}")

    print("\nCreating Medical-only Dataset...")
    medical_only = loader.create_mixed_dataset(
        math_ratio=0.0, code_ratio=0.0, reasoning_ratio=0.0,
        medical_ratio=1.0, total_samples=10
    )
    print(f"Created {len(medical_only)} medical samples")
    med_reward_fn = create_reward_evaluator()
    med_item = medical_only[0]
    print(f"Sample query: {med_item['query'][:120]}...")
    reward = med_reward_fn(med_item['query'], med_item['reference_answer'], med_item)
    print(f"Reward for correct answer: {reward:.3f}") 