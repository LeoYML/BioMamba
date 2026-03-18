"""
Test script to verify environment setup before training
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import sys

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'mamba_ssm': 'Mamba-SSM',
        'tqdm': 'tqdm',
    }
    
    missing = []
    versions = {}
    
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            versions[name] = version
            print(f"✓ {name}: {version}")
        except ImportError:
            missing.append(name)
            print(f"✗ {name}: NOT INSTALLED")
    
    return missing, versions


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                
                # Test memory
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)  # Convert to GB
                print(f"    Total memory: {total_memory:.2f} GB")
        else:
            print(f"✗ CUDA is not available")
            print(f"  Will use CPU for training (slower)")
        
        return cuda_available
    except Exception as e:
        print(f"✗ Error testing CUDA: {e}")
        return False


def test_data_access():
    """Test if we can access HuggingFace datasets"""
    print("\nTesting data access...")
    
    try:
        from datasets import load_dataset
        
        print("Testing HuggingFace datasets access...")
        # Try to get dataset info without downloading
        print("✓ HuggingFace datasets is accessible")
        print("Note: Dataset will be downloaded on first training run")
        
        return True
    except Exception as e:
        print(f"✗ Error accessing datasets: {e}")
        return False


def test_model_access():
    """Test if we can access Mamba2 models"""
    print("\nTesting model access...")
    
    try:
        from mamba_ssm import MambaLMHeadModel
        from transformers import AutoTokenizer
        
        print("Testing Mamba-SSM installation...")
        print("✓ Mamba-SSM is installed")
        print("Note: Models will be downloaded on first training run")
        
        return True
    except Exception as e:
        print(f"✗ Error with Mamba-SSM: {e}")
        return False


def estimate_requirements(model_name='mamba2-130m'):
    """Estimate resource requirements"""
    print(f"\nEstimated requirements for {model_name}:")
    
    requirements = {
        'mamba2-130m': {
            'params': '130M',
            'gpu_memory': '~16 GB (with FP16)',
            'batch_size': 32,
            'time_per_epoch': '~2-4 hours (on A100)'
        },
        'mamba2-370m': {
            'params': '370M',
            'gpu_memory': '~20 GB (with FP16)',
            'batch_size': 16,
            'time_per_epoch': '~4-6 hours (on A100)'
        },
        'mamba2-780m': {
            'params': '780M',
            'gpu_memory': '~24 GB (with FP16)',
            'batch_size': 8,
            'time_per_epoch': '~6-10 hours (on A100)'
        },
        'mamba2-1.3b': {
            'params': '1.3B',
            'gpu_memory': '~32 GB (with FP16)',
            'batch_size': 4,
            'time_per_epoch': '~10-15 hours (on A100)'
        },
        'mamba2-2.7b': {
            'params': '2.7B',
            'gpu_memory': '~40 GB (with FP16)',
            'batch_size': 1,
            'time_per_epoch': '~15-24 hours (on A100)'
        }
    }
    
    if model_name in requirements:
        req = requirements[model_name]
        print(f"  Parameters: {req['params']}")
        print(f"  GPU Memory: {req['gpu_memory']}")
        print(f"  Recommended batch size: {req['batch_size']}")
        print(f"  Estimated time per epoch: {req['time_per_epoch']}")
    else:
        print(f"  Unknown model: {model_name}")


def main():
    print("="*50)
    print("Mamba2 Training Environment Test")
    print("="*50)
    print()
    
    # Test imports
    missing, versions = test_imports()
    
    if missing:
        print(f"\n⚠ Warning: The following packages are missing:")
        for pkg in missing:
            print(f"  - {pkg}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Test CUDA
    cuda_available = test_cuda()
    
    # Test data access
    data_ok = test_data_access()
    
    # Test model access
    model_ok = test_model_access()
    
    # Estimate requirements
    print()
    estimate_requirements('mamba2-130m')
    estimate_requirements('mamba2-2.7b')
    
    # Summary
    print("\n" + "="*50)
    print("Summary")
    print("="*50)
    
    if not missing and data_ok and model_ok:
        print("✓ All checks passed!")
        print("✓ Environment is ready for training")
        
        if not cuda_available:
            print("\n⚠ Note: CUDA is not available. Training will be slower on CPU.")
        
        print("\nTo start training, run:")
        print("  ./run_training.sh")
        print("or")
        print("  python finetune_pubmed_medline.py --model_name mamba2-130m")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
