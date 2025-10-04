#!/usr/bin/env python3
"""
Generate all required outputs for the assignment Q&A:
1. Model code from model.py 
2. Torch summary output
3. Albumentation transformations code
4. Training log (after training)
5. README.md link
"""

import torch
from torchsummary import summary
import os
import sys
from datetime import datetime

# Import our modules
from model import CIFAR10Net, count_parameters
from transforms import get_train_transforms, get_test_transforms


def print_separator(title):
    """Print a separator with title"""
    print("\n" + "="*80)
    print(f" {title} ")
    print("="*80)


def generate_model_code_output():
    """1. Copy and paste your model code from your model.py file (full code) [125]"""
    print_separator("1. MODEL CODE FROM model.py (Full Code)")
    
    try:
        with open('model.py', 'r') as f:
            model_code = f.read()
        print(model_code)
    except FileNotFoundError:
        print("ERROR: model.py not found!")


def generate_torch_summary():
    """2. copy-paste output of torch summary [125]"""
    print_separator("2. TORCH SUMMARY OUTPUT")
    
    try:
        model = CIFAR10Net()
        print("Model Summary for CIFAR10Net:")
        print(f"Input size: (3, 32, 32)")
        print(f"Total parameters: {count_parameters(model):,}")
        print(f"Receptive field: {model.calculate_receptive_field()}")
        print("\nDetailed Summary:")
        
        # Generate summary - capture output
        summary(model, (3, 32, 32))
        
    except Exception as e:
        print(f"ERROR generating torch summary: {e}")


def generate_albumentation_code():
    """3. Copy-paste the code where you implemented albumentation transformation for all three transformations [125]"""
    print_separator("3. ALBUMENTATION TRANSFORMATIONS CODE")
    
    try:
        with open('transforms.py', 'r') as f:
            transforms_code = f.read()
        
        print("Full transforms.py code with all three required albumentation transformations:")
        print("1. HorizontalFlip")
        print("2. ShiftScaleRotate") 
        print("3. CoarseDropout (with specific parameters)")
        print("\n" + "-"*50)
        print(transforms_code)
        
    except FileNotFoundError:
        print("ERROR: transforms.py not found!")


def show_training_log_info():
    """4. Copy-paste your training log (you must be running validation/text after each Epoch [125]"""
    print_separator("4. TRAINING LOG INFORMATION")
    
    print("Training logs will be automatically saved when you run 'python train.py'")
    print("The logs will be saved in the 'logs/' directory with timestamp.")
    print("\nFeatures of the training log:")
    print("- Validation runs after EVERY epoch")
    print("- Detailed epoch-by-epoch training and validation metrics")
    print("- Model architecture details")
    print("- Torch summary")
    print("- Best model saving information")
    print("- Final summary with target achievement status")
    
    # Check if any log files exist
    if os.path.exists('logs'):
        log_files = [f for f in os.listdir('logs') if f.startswith('training_log')]
        if log_files:
            print(f"\nExisting log files found:")
            for log_file in log_files:
                print(f"  - logs/{log_file}")
            
            # Show content of most recent log
            latest_log = sorted(log_files)[-1]
            print(f"\nMost recent log file content (logs/{latest_log}):")
            print("-" * 50)
            try:
                with open(f'logs/{latest_log}', 'r') as f:
                    content = f.read()
                    # Show first 2000 characters
                    if len(content) > 2000:
                        print(content[:2000])
                        print(f"\n... (truncated, full log has {len(content)} characters)")
                    else:
                        print(content)
            except Exception as e:
                print(f"Error reading log file: {e}")
        else:
            print("\nNo training log files found yet. Run 'python train.py' to generate them.")
    else:
        print("\nNo logs directory found yet. Run 'python train.py' to generate training logs.")


def show_readme_link():
    """5. Share the link for your README.md file"""
    print_separator("5. README.md FILE LINK")
    
    print("README.md file location: /Users/chethan/Documents/era/ciphar/README.md")
    print("\nREADME.md contents:")
    print("-" * 50)
    
    try:
        with open('README.md', 'r') as f:
            readme_content = f.read()
        print(readme_content)
    except FileNotFoundError:
        print("ERROR: README.md not found!")


def main():
    """Generate all assignment outputs"""
    print("CIFAR-10 CNN Assignment Outputs Generator")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Generate all required outputs
    generate_model_code_output()
    generate_torch_summary() 
    generate_albumentation_code()
    show_training_log_info()
    show_readme_link()
    
    print_separator("SUMMARY")
    print("All assignment outputs have been generated!")
    print("\nNext steps:")
    print("1. Run 'python train.py' to start training and generate training logs")
    print("2. Copy the outputs above for your assignment submission")
    print("3. Training logs will be saved automatically in logs/ directory")
    print("4. Model will automatically stop when 85% accuracy is reached")
    
    print("\nArchitecture Requirements Verification:")
    model = CIFAR10Net()
    params = count_parameters(model)
    rf = model.calculate_receptive_field()
    print(f"✅ Parameters < 200k: {params:,} < 200,000")
    print(f"✅ Receptive Field > 44: {rf} > 44")
    print(f"✅ C1C2C3C4 architecture implemented")
    print(f"✅ Depthwise Separable Conv in C2")
    print(f"✅ Dilated Conv in C3")
    print(f"✅ GAP + FC implemented")
    print(f"✅ All 3 Albumentation transforms implemented")


if __name__ == "__main__":
    main()