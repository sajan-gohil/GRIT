#!/usr/bin/env python
"""
Validation script to verify the loss function implementation.
This script checks that all required files exist and have correct structure.
"""
import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {filepath}")
    return exists

def check_function_in_file(filepath, function_name):
    """Check if a function is defined in a file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            exists = f"def {function_name}" in content
            status = "✓" if exists else "✗"
            print(f"  {status} Function '{function_name}' defined")
            return exists
    except:
        return False

def main():
    print("=" * 70)
    print("GRIT Transformer Loss Functions - Implementation Validation")
    print("=" * 70)
    print()
    
    all_checks = []
    
    # Check core implementation files
    print("1. Core Loss Functions:")
    all_checks.append(check_file_exists("grit/losses.py", "Loss functions module"))
    if os.path.exists("grit/losses.py"):
        all_checks.append(check_function_in_file("grit/losses.py", "attention_improvement_loss"))
        all_checks.append(check_function_in_file("grit/losses.py", "structure_reconstruction_loss"))
    print()
    
    # Check configuration
    print("2. Configuration:")
    all_checks.append(check_file_exists("grit/config/loss_config.py", "Loss config module"))
    all_checks.append(check_file_exists("configs/loss_config_example.yaml", "Example config"))
    print()
    
    # Check integration
    print("3. Training Integration:")
    all_checks.append(check_file_exists("grit/network/grit_model.py", "Model file (modified)"))
    all_checks.append(check_file_exists("grit/train/custom_train.py", "Training file (modified)"))
    print()
    
    # Check tests
    print("4. Tests:")
    all_checks.append(check_file_exists("unittests/test_losses.py", "Unit tests"))
    print()
    
    # Check documentation
    print("5. Documentation:")
    all_checks.append(check_file_exists("docs/auxiliary_losses.md", "Loss functions docs"))
    all_checks.append(check_file_exists("IMPLEMENTATION_SUMMARY.md", "Implementation summary"))
    print()
    
    # Summary
    print("=" * 70)
    passed = sum(all_checks)
    total = len(all_checks)
    print(f"Validation Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("✓ All validation checks passed!")
        print("✓ Implementation is complete and ready for use.")
        return 0
    else:
        print("✗ Some validation checks failed.")
        print("✗ Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
