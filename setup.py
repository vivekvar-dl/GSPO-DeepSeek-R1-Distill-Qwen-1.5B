#!/usr/bin/env python3
"""
Setup script for GSPO implementation
"""

import subprocess
import sys
import torch

def check_gpu():
    """Check if CUDA is available and GPU info"""
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✓ CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        return True
    else:
        print("✗ CUDA not available")
        return False

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install requirements")
        return False

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    try:
        import torch
        import transformers
        import datasets
        print("✓ Core libraries imported successfully")
        
        # Test our modules
        from gspo_implementation import GSPOTrainer, GSPOConfig
        from data_loader import DatasetLoader, create_reward_evaluator
        print("✓ GSPO modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def check_model_availability():
    """Check if we can load a small model for testing"""
    print("Testing model loading...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Try to load a small model
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Successfully loaded tokenizer for {model_name}")
        
        # Don't actually load the model to save time/memory in setup
        return True
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False

def main():
    print("="*60)
    print("GSPO Setup Script")
    print("="*60)
    
    all_good = True
    
    # Check GPU
    if not check_gpu():
        print("Warning: No GPU detected. Training will be very slow.")
        all_good = False
    
    print()
    
    # Install requirements
    if not install_requirements():
        all_good = False
    
    print()
    
    # Test imports
    if not test_imports():
        all_good = False
    
    print()
    
    # Test model loading
    if not check_model_availability():
        all_good = False
    
    print()
    print("="*60)
    if all_good:
        print("✓ Setup completed successfully!")
        print("You can now run: python train_gspo.py --help")
    else:
        print("✗ Setup encountered issues. Please check the errors above.")
    print("="*60)

if __name__ == "__main__":
    main() 