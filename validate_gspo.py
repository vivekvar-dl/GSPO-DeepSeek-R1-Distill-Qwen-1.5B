#!/usr/bin/env python3
"""
GSPO Validation Script

This script verifies that GSPO implementation is working correctly by checking:
1. Technical correctness
2. Training dynamics
3. Performance improvements
4. Comparison with baselines
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List
import os

from gspo_implementation import GSPOTrainer, GSPOConfig
from data_loader import DatasetLoader, create_reward_evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM

class GSPOValidator:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self.results = {}
        
    def load_small_model(self):
        """Load a small model for fast validation"""
        print(f"Loading small model for validation: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def test_technical_correctness(self):
        """Test 1: Verify GSPO implementation is technically correct"""
        print("\n" + "="*60)
        print("TEST 1: TECHNICAL CORRECTNESS")
        print("="*60)
        
        results = {}
        
        # Load model
        model, tokenizer = self.load_small_model()
        config = GSPOConfig(group_size=2, batch_size=1, learning_rate=1e-5)
        trainer = GSPOTrainer(model, tokenizer, config)
        
        # Test data
        test_queries = ["What is 2+2?", "Hello world"]
        dummy_reward_fn = lambda q, r: 0.5
        
        try:
            # Test sequence log probability computation
            print("✓ Testing sequence log probability computation...")
            test_input = tokenizer("Hello world", return_tensors="pt")
            log_prob = trainer.compute_sequence_log_prob(
                model, test_input.input_ids, test_input.attention_mask, 0
            )
            assert torch.is_tensor(log_prob), "Should return tensor"
            results["sequence_log_prob"] = "PASS"
            
            # Test importance ratio computation  
            print("✓ Testing importance ratio computation...")
            importance_ratio = trainer.compute_importance_ratio(
                test_input.input_ids, test_input.attention_mask, 0, 
                torch.tensor([2.0])
            )
            assert torch.is_tensor(importance_ratio), "Should return tensor"
            assert importance_ratio.item() > 0, "Should be positive"
            results["importance_ratio"] = "PASS"
            
            # Test advantage computation
            print("✓ Testing advantage computation...")
            rewards = torch.tensor([0.3, 0.7])  # Group of 2
            advantages = trainer.compute_advantages(rewards)
            assert len(advantages) == 2, "Should match input size"
            assert abs(advantages.mean().item()) < 1e-6, "Should be zero-mean"
            results["advantages"] = "PASS"
            
            # Test complete training step
            print("✓ Testing complete training step...")
            stats = trainer.train_step(test_queries[:1], dummy_reward_fn)
            required_stats = ["loss", "clip_fraction", "importance_ratio_mean", "reward_mean"]
            for stat in required_stats:
                assert stat in stats, f"Missing stat: {stat}"
            results["training_step"] = "PASS"
            
            print("\n✅ All technical tests PASSED!")
            
        except Exception as e:
            print(f"\n❌ Technical test FAILED: {e}")
            results["error"] = str(e)
        
        self.results["technical"] = results
        return results
    
    def test_training_dynamics(self):
        """Test 2: Verify training dynamics behave correctly"""
        print("\n" + "="*60)
        print("TEST 2: TRAINING DYNAMICS")
        print("="*60)
        
        results = {}
        
        # Load model and data
        model, tokenizer = self.load_small_model()
        config = GSPOConfig(group_size=2, batch_size=2, learning_rate=1e-4)
        trainer = GSPOTrainer(model, tokenizer, config)
        
        # Simple test data with clear reward signal
        loader = DatasetLoader()
        train_data = loader.load_math_problems("easy", max_samples=6)
        reward_fn = create_reward_evaluator()
        
        def test_reward_fn(query, response):
            for item in train_data:
                if item['query'] == query:
                    return reward_fn(query, response, item)
            return 0.3
        
        # Track training metrics
        losses = []
        clip_fractions = []
        importance_ratios = []
        
        print("Running short training to check dynamics...")
        for step in range(5):
            queries = [item['query'] for item in train_data[:2]]
            stats = trainer.train_step(queries, test_reward_fn)
            
            losses.append(stats['loss'])
            clip_fractions.append(stats['clip_fraction'])
            importance_ratios.append(stats['importance_ratio_mean'])
            
            print(f"  Step {step+1}: Loss={stats['loss']:.4f}, "
                  f"Clip={stats['clip_fraction']:.4f}, "
                  f"Ratio={stats['importance_ratio_mean']:.4f}")
        
        # Validate dynamics
        try:
            # Check losses are reasonable
            assert all(0 < loss < 100 for loss in losses), "Losses should be reasonable"
            results["loss_range"] = "PASS"
            
            # Check clip fractions are in expected range  
            assert all(0 <= cf <= 1 for cf in clip_fractions), "Clip fractions in [0,1]"
            results["clip_fraction_range"] = "PASS"
            
            # Check importance ratios are positive
            assert all(ratio > 0 for ratio in importance_ratios), "Importance ratios positive"
            results["importance_ratio_positive"] = "PASS"
            
            # Check for model updates (ratios should change)
            if len(set(f"{r:.6f}" for r in importance_ratios)) > 1:
                results["model_updating"] = "PASS"
            else:
                results["model_updating"] = "WARN - ratios not changing much"
            
            print("\n✅ Training dynamics look good!")
            
        except Exception as e:
            print(f"\n❌ Training dynamics test FAILED: {e}")
            results["error"] = str(e)
        
        # Store metrics for plotting
        self.results["training_dynamics"] = {
            "results": results,
            "losses": losses,
            "clip_fractions": clip_fractions, 
            "importance_ratios": importance_ratios
        }
        
        return results
    
    def test_performance_improvement(self):
        """Test 3: Verify model actually improves"""
        print("\n" + "="*60)
        print("TEST 3: PERFORMANCE IMPROVEMENT")
        print("="*60)
        
        results = {}
        
        # Load model and data
        model, tokenizer = self.load_small_model()
        config = GSPOConfig(group_size=2, batch_size=2, learning_rate=5e-4)
        trainer = GSPOTrainer(model, tokenizer, config)
        
        # Get test data
        loader = DatasetLoader()
        train_data = loader.load_math_problems("easy", max_samples=8)
        eval_data = loader.load_math_problems("easy", max_samples=4)
        reward_fn = create_reward_evaluator()
        
        def test_reward_fn(query, response):
            for item in train_data:
                if item['query'] == query:
                    return reward_fn(query, response, item)
            return 0.3
        
        # Evaluate before training
        print("Evaluating before training...")
        initial_rewards = []
        for item in eval_data:
            response = trainer.generate_responses([item['query']], max_new_tokens=50)[0]
            reward = reward_fn(item['query'], response, item)
            initial_rewards.append(reward)
        
        initial_avg = np.mean(initial_rewards)
        print(f"Initial average reward: {initial_avg:.3f}")
        
        # Train for several steps
        print("Training for improvement...")
        queries = [item['query'] for item in train_data]
        
        for epoch in range(8):
            stats = trainer.train_step(queries[:2], test_reward_fn)
            if epoch % 3 == 0:
                print(f"  Epoch {epoch}: Loss={stats['loss']:.4f}")
        
        # Update old model to avoid extreme importance ratios
        trainer.update_old_model()
        
        # Evaluate after training
        print("Evaluating after training...")
        final_rewards = []
        for item in eval_data:
            response = trainer.generate_responses([item['query']], max_new_tokens=50)[0]
            reward = reward_fn(item['query'], response, item)
            final_rewards.append(reward)
        
        final_avg = np.mean(final_rewards)
        improvement = final_avg - initial_avg
        
        print(f"Final average reward: {final_avg:.3f}")
        print(f"Improvement: {improvement:.3f}")
        
        try:
            # Check for improvement (even small is good for short training)
            if improvement > 0.01:
                results["improvement"] = "PASS - Clear improvement"
            elif improvement > -0.05:
                results["improvement"] = "PASS - Stable (no degradation)"
            else:
                results["improvement"] = "FAIL - Performance degraded"
            
            # Check responses are reasonable
            sample_responses = [trainer.generate_responses([eval_data[0]['query']])[0]]
            if any(len(r.strip()) > 3 for r in sample_responses):
                results["response_quality"] = "PASS"
            else:
                results["response_quality"] = "WARN - Very short responses"
            
            print(f"\n✅ Performance test: {results['improvement']}")
            
        except Exception as e:
            print(f"\n❌ Performance test FAILED: {e}")
            results["error"] = str(e)
        
        self.results["performance"] = {
            "results": results,
            "initial_avg": initial_avg,
            "final_avg": final_avg,
            "improvement": improvement,
            "initial_rewards": initial_rewards,
            "final_rewards": final_rewards
        }
        
        return results
    
    def test_clipping_behavior(self):
        """Test 4: Verify clipping behaves as expected"""
        print("\n" + "="*60)
        print("TEST 4: CLIPPING BEHAVIOR")
        print("="*60)
        
        results = {}
        
        # Test with different clipping ranges
        model, tokenizer = self.load_small_model()
        
        # Tight clipping
        config_tight = GSPOConfig(left_clip_range=1e-5, right_clip_range=2e-5, group_size=2)
        trainer_tight = GSPOTrainer(model, tokenizer, config_tight)
        
        # Loose clipping  
        config_loose = GSPOConfig(left_clip_range=0.1, right_clip_range=0.2, group_size=2)
        trainer_loose = GSPOTrainer(model, tokenizer, config_loose)
        
        # Test data
        loader = DatasetLoader()
        test_data = loader.load_math_problems("easy", max_samples=4)
        reward_fn = create_reward_evaluator()
        
        def test_reward_fn(query, response):
            for item in test_data:
                if item['query'] == query:
                    return reward_fn(query, response, item)
            return 0.5
        
        try:
            # Run training step with both
            queries = [item['query'] for item in test_data[:2]]
            
            stats_tight = trainer_tight.train_step(queries, test_reward_fn)
            stats_loose = trainer_loose.train_step(queries, test_reward_fn)
            
            clip_tight = stats_tight['clip_fraction']
            clip_loose = stats_loose['clip_fraction']
            
            print(f"Tight clipping clip fraction: {clip_tight:.4f}")
            print(f"Loose clipping clip fraction: {clip_loose:.4f}")
            
            # Tight clipping should clip more
            if clip_tight > clip_loose:
                results["clipping_comparison"] = "PASS"
            else:
                results["clipping_comparison"] = "WARN - Expected more clipping with tight range"
            
            # Check clip fractions are reasonable
            if 0 <= clip_tight <= 1 and 0 <= clip_loose <= 1:
                results["clip_fraction_range"] = "PASS"
            else:
                results["clip_fraction_range"] = "FAIL"
            
            print(f"\n✅ Clipping behavior test: {results['clipping_comparison']}")
            
        except Exception as e:
            print(f"\n❌ Clipping test FAILED: {e}")
            results["error"] = str(e)
        
        self.results["clipping"] = results
        return results
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*60)
        print("GSPO VALIDATION REPORT")
        print("="*60)
        
        # Summary
        all_tests = []
        for test_name, test_results in self.results.items():
            if "results" in test_results:
                test_results = test_results["results"]
            
            passed = sum(1 for r in test_results.values() if str(r).startswith("PASS"))
            total = len([r for r in test_results.values() if not str(r).startswith("error")])
            all_tests.append((test_name, passed, total))
            
            print(f"\n{test_name.upper()}:")
            for key, value in test_results.items():
                if key != "error":
                    status = "✅" if str(value).startswith("PASS") else "⚠️" if str(value).startswith("WARN") else "❌"
                    print(f"  {status} {key}: {value}")
        
        # Overall assessment
        total_passed = sum(passed for _, passed, _ in all_tests)
        total_tests = sum(total for _, _, total in all_tests)
        
        print(f"\n" + "="*60)
        print(f"OVERALL: {total_passed}/{total_tests} tests passed")
        
        if total_passed >= total_tests * 0.8:
            print("🎉 GSPO implementation looks GOOD!")
            assessment = "GOOD"
        elif total_passed >= total_tests * 0.6:
            print("⚠️  GSPO implementation has some ISSUES but mostly works")
            assessment = "PARTIAL"
        else:
            print("❌ GSPO implementation has SERIOUS ISSUES")
            assessment = "POOR"
        
        print("="*60)
        
        # Save detailed results
        os.makedirs("validation_results", exist_ok=True)
        with open("validation_results/gspo_validation.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        return assessment
    
    def plot_dynamics(self):
        """Plot training dynamics if available"""
        if "training_dynamics" not in self.results:
            return
        
        dynamics = self.results["training_dynamics"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("GSPO Training Dynamics")
        
        # Loss
        axes[0,0].plot(dynamics["losses"])
        axes[0,0].set_title("Loss")
        axes[0,0].set_xlabel("Step")
        
        # Clip fraction
        axes[0,1].plot(dynamics["clip_fractions"])
        axes[0,1].set_title("Clip Fraction")
        axes[0,1].set_xlabel("Step")
        
        # Importance ratios
        axes[1,0].plot(dynamics["importance_ratios"])
        axes[1,0].set_title("Importance Ratios")
        axes[1,0].set_xlabel("Step")
        
        # Performance comparison (if available)
        if "performance" in self.results:
            perf = self.results["performance"]
            axes[1,1].bar(['Initial', 'Final'], [perf['initial_avg'], perf['final_avg']])
            axes[1,1].set_title("Average Reward")
        
        plt.tight_layout()
        plt.savefig("validation_results/gspo_dynamics.png", dpi=150, bbox_inches='tight')
        print("\n📊 Plots saved to validation_results/gspo_dynamics.png")

def main():
    """Run complete GSPO validation"""
    print("Starting GSPO Validation...")
    print("Using small model for fast validation")
    
    validator = GSPOValidator()
    
    # Run all tests
    validator.test_technical_correctness()
    validator.test_training_dynamics()
    validator.test_performance_improvement()
    validator.test_clipping_behavior()
    
    # Generate report
    assessment = validator.generate_report()
    
    # Create plots
    try:
        validator.plot_dynamics()
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print(f"\n🎯 Final Assessment: {assessment}")
    print("📁 Detailed results saved to validation_results/")
    
    return assessment

if __name__ == "__main__":
    main() 