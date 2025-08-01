#!/usr/bin/env python3
"""
Quick GSPO Verification Script
Tests if GSPO is actually working before running full GSM8K training
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gspo_implementation import GSPOTrainer, GSPOConfig
from data_loader import DatasetLoader, create_reward_evaluator

def quick_gspo_verification():
    """Run a quick verification test of GSPO functionality"""
    
    print("🔍 GSPO Quick Verification Test")
    print("=" * 60)
    
    # Use a very small model for CPU testing
    model_name = "distilgpt2"  # Much smaller, faster on CPU
    print(f"Loading {model_name} for quick verification...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None  # Don't use device_map for small models
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = "cpu"  # Force CPU for compatibility
        model = model.to(device)
        
        print(f"✓ Model loaded: {model.num_parameters() / 1e6:.1f}M parameters")
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        # Ultimate fallback - use the smallest possible model
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        device = "cpu"
        model = model.to(device)
    
    # Create GSPO configuration for quick test
    config = GSPOConfig(
        group_size=2,
        batch_size=1,
        learning_rate=1e-5,
        max_length=128,
        gradient_accumulation_steps=1
    )
    
    # Initialize GSPO trainer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer = GSPOTrainer(model, tokenizer, config, device=device)
    
    print("\n✅ GSPO Trainer initialized successfully")
    
    # Run initial verification
    print("\n🔍 Running comprehensive GSPO verification...")
    verification_results = trainer.verify_gspo_working()
    clipping_results = trainer.verify_clipping_bounds()
    
    # Test with simple math problems  
    test_queries = [
        "What is 2 + 3?",
        "What is 5 × 4?"
    ]
    
    # Create reward function with proper data items
    def create_test_reward_function():
        """Create a simple reward function for testing with diverse rewards"""
        def reward_fn(query, response, data_item=None):
            # More diverse reward function to avoid identical rewards
            base_reward = 0.3
            
            # Check for numbers (math-related)
            if any(char.isdigit() for char in response):
                base_reward += 0.4
            
            # Check for math keywords
            math_keywords = ['plus', 'minus', 'multiply', 'divide', 'equals', '=', '+', '-', '*', '/']
            if any(keyword in response.lower() for keyword in math_keywords):
                base_reward += 0.2
            
            # Length bonus (scaled)
            length_bonus = min(len(response) / 20.0, 0.3)
            base_reward += length_bonus
            
            # Add small random component to ensure diversity
            import random
            noise = random.uniform(-0.1, 0.1)
            final_reward = max(0.1, min(1.0, base_reward + noise))
            
            print(f"  Reward for '{response[:30]}...': {final_reward:.3f}")
            return final_reward
        return reward_fn
    
    reward_function = create_test_reward_function()
    
    print("\n🧮 Testing GSPO training step...")
    
    # Run one training step
    try:
        # Use the custom reward function directly
        def wrapper_reward_fn(query, response):
            return reward_function(query, response, {"type": "math"})
        
        stats = trainer.train_step(test_queries, wrapper_reward_fn)
        print("✅ Training step completed successfully")
        print(f"📊 Training stats: {stats}")
        
        # Check if GSPO components are working
        success_indicators = []
        
        # Check if we have different models
        if verification_results.get("params_different", False):
            success_indicators.append("✅ Current and old models are different")
        else:
            success_indicators.append("⚠️ Models might be identical (check this)")
        
        # Check if clipping is happening
        if stats.get("clip_fraction", 0) > 0:
            success_indicators.append("✅ Clipping is occurring")
        else:
            success_indicators.append("⚠️ No clipping detected (might be normal for first steps)")
        
        # Check if importance ratios are reasonable
        importance_mean = stats.get("importance_ratio_mean", 1.0)
        if 0.5 <= importance_mean <= 2.0:
            success_indicators.append("✅ Importance ratios are in reasonable range")
        else:
            success_indicators.append(f"⚠️ Importance ratios seem extreme: {importance_mean}")
        
        # Check if loss is reasonable
        loss_value = stats.get("loss", 0)
        if loss_value != 0 and not (torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value))):
            success_indicators.append("✅ Loss is computed and finite")
        else:
            success_indicators.append("❌ Loss is problematic (zero, NaN, or Inf)")
        
        print("\n📋 GSPO Verification Results:")
        for indicator in success_indicators:
            print(f"  {indicator}")
        
        # Overall assessment
        success_count = sum(1 for ind in success_indicators if ind.startswith("✅"))
        warning_count = sum(1 for ind in success_indicators if ind.startswith("⚠️"))
        error_count = sum(1 for ind in success_indicators if ind.startswith("❌"))
        
        print(f"\n🎯 Overall Assessment:")
        print(f"  ✅ Successful checks: {success_count}")
        print(f"  ⚠️ Warnings: {warning_count}")
        print(f"  ❌ Errors: {error_count}")
        
        # Specific issue analysis
        print(f"\n🔍 Specific Issue Analysis:")
        print(f"  → Training loss: {loss_value}")
        print(f"  → Importance ratio mean: {importance_mean}")
        print(f"  → Clip fraction: {stats.get('clip_fraction', 0)}")
        print(f"  → Trainer step: {trainer.step}")
        
        if error_count == 0 and success_count >= 3:
            print("\n🎉 GSPO VERIFICATION PASSED!")
            print("Your GSPO implementation appears to be working correctly.")
            print("Ready for full GSM8K training!")
            return True
        elif error_count == 0:
            print("\n⚠️ GSPO VERIFICATION PARTIAL")
            print("GSPO seems to be working but with some warnings.")
            print("Consider investigating warnings before full training.")
            return True
        else:
            print("\n❌ GSPO VERIFICATION FAILED")
            print("There are issues that need to be fixed before training.")
            print("\n🔧 Common fixes:")
            if loss_value == 0:
                print("  - Loss is 0.0: Check numerical stability in GSPO loss computation")
            if importance_mean == 1.0:
                print("  - Importance ratios = 1.0: Models might be identical")
            return False
        
    except Exception as e:
        print(f"❌ Error during training step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_gspo_verification()
    
    if success:
        print("\n🚀 Ready to run full GSM8K training:")
        print("python train_gspo.py --dataset gsm8k --num_train_samples 1000 --num_epochs 5 --use_wandb")
    else:
        print("\n🔧 Fix the issues above before proceeding to full training.")
        print("\nMost likely issues:")
        print("1. Loss = 0.0 → Check numerical stability in GSPO loss computation")
        print("2. Models identical → Ensure old_model is properly updated")
        print("3. No clipping → Verify importance ratios are computed correctly") 