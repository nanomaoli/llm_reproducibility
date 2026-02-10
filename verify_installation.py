#!/usr/bin/env python3
"""
Verification script for TBIK installation.
Run this script to verify that all core dependencies are installed correctly.
"""

import sys
import importlib.util

def check_package(package_name, import_name=None, optional=False):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            print(f"✓ {package_name:20s} - installed")
            return True
        else:
            if optional:
                print(f"⊘ {package_name:20s} - not installed (optional)")
            else:
                print(f"✗ {package_name:20s} - NOT FOUND")
            return False
    except (ImportError, ModuleNotFoundError, ValueError):
        if optional:
            print(f"⊘ {package_name:20s} - not installed (optional)")
        else:
            print(f"✗ {package_name:20s} - NOT FOUND")
        return False

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA                - available ({device_count} GPU(s), {device_name})")
            return True
        else:
            print("✗ CUDA                - NOT AVAILABLE")
            return False
    except Exception as e:
        print(f"✗ CUDA                - ERROR: {e}")
        return False

def check_mini_allreduce():
    """Check if mini_allreduce is installed."""
    try:
        from mini_allreduce import CustomTreeAllreduce
        print("✓ mini_allreduce      - installed")
        return True
    except ImportError as e:
        print(f"✗ mini_allreduce      - NOT INSTALLED")
        print(f"  Error: {e}")
        print(f"  Install with: uv pip install --no-build-isolation -v ./mini_allreduce")
        return False

def check_src_package():
    """Check if the src package is properly installed."""
    try:
        from src.patch import apply_patches
        from src.utils import batch_invariant_is_enabled
        from src.tbik.tree_based_matmul import matmul_tp_persistent
        print("✓ src package         - installed")
        return True
    except ImportError as e:
        print(f"✗ src package         - NOT PROPERLY INSTALLED")
        print(f"  Error: {e}")
        print(f"  Install with: uv pip install -e .")
        return False

def main():
    print("=" * 60)
    print("TBIK Installation Verification")
    print("=" * 60)
    print()

    print("Core Dependencies:")
    print("-" * 60)
    core_ok = True
    core_ok &= check_package("torch")
    core_ok &= check_package("vllm")
    core_ok &= check_package("transformers")
    core_ok &= check_package("huggingface_hub", "huggingface_hub")
    core_ok &= check_package("safetensors")
    core_ok &= check_package("triton")
    print()

    print("CUDA:")
    print("-" * 60)
    cuda_ok = check_cuda()
    print()

    print("TBIK Packages:")
    print("-" * 60)
    tbik_ok = True
    tbik_ok &= check_src_package()
    tbik_ok &= check_mini_allreduce()
    print()

    print("Optional Dependencies (for RL experiments):")
    print("-" * 60)
    check_package("flash_attn", "flash_attn", optional=True)
    check_package("torchtitan", "torchtitan", optional=True)
    check_package("tensorboard", "torch.utils.tensorboard", optional=True)
    print()

    print("Optional Dependencies (for evaluation):")
    print("-" * 60)
    check_package("datasets", optional=True)
    check_package("latex2sympy2", optional=True)
    check_package("word2number", optional=True)
    check_package("immutabledict", optional=True)
    check_package("nltk", optional=True)
    check_package("langdetect", optional=True)
    check_package("tqdm", optional=True)
    check_package("numpy", optional=True)
    check_package("pandas", optional=True)
    print()

    print("=" * 60)
    print("Summary:")
    print("=" * 60)

    if core_ok and cuda_ok and tbik_ok:
        print("✓ All core components are installed correctly!")
        print("\nYou can now run:")
        print("  - python simple_matmul.py")
        print("  - python simple_inference.py")
        print("  - python tests/test_custom_tree_allreduce.py (requires multiple GPUs)")
        return 0
    else:
        print("✗ Some core components are missing.")
        print("\nPlease follow the installation instructions in README.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
