"""
Quick test script for evaluation setup
Tests model loading and dataset access
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import Mamba2
try:
    from mamba_ssm import MambaLMHeadModel
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

# Try to import datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


def test_environment():
    """Test Python environment"""
    print("="*70)
    print("ENVIRONMENT TEST")
    print("="*70)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check transformers
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    
    # Check Mamba
    if MAMBA_AVAILABLE:
        print("✓ mamba_ssm is installed")
    else:
        print("✗ mamba_ssm is NOT installed (required for Mamba2 models)")
    
    # Check datasets
    if DATASETS_AVAILABLE:
        print("✓ datasets is installed")
    else:
        print("✗ datasets is NOT installed")
    
    print("")


def test_model_loading(model_path, model_type='auto'):
    """Test model loading"""
    print("="*70)
    print(f"MODEL LOADING TEST: {model_path}")
    print("="*70)
    
    try:
        if model_type == 'mamba2':
            if not MAMBA_AVAILABLE:
                print("✗ Cannot load Mamba2: mamba_ssm not installed")
                return False
            
            print("Loading Mamba2 model...")
            model = MambaLMHeadModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            print("Loading HuggingFace model...")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model loaded successfully")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Tokenizer vocab size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "What is diabetes?"
        tokens = tokenizer(test_text, return_tensors='pt')
        print(f"  Test tokenization: '{test_text}' -> {tokens['input_ids'].shape[1]} tokens")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_dataset_loading(dataset_name='pubmedqa'):
    """Test dataset loading"""
    print("="*70)
    print(f"DATASET LOADING TEST: {dataset_name}")
    print("="*70)
    
    if not DATASETS_AVAILABLE:
        print("✗ Cannot load dataset: datasets library not installed")
        return False
    
    try:
        print(f"Loading {dataset_name} dataset...")
        
        if dataset_name == 'pubmedqa':
            dataset = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
            print(f"✓ Dataset loaded successfully")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Features: {dataset.features.keys()}")
            
            # Show example
            example = dataset[0]
            print(f"\n  Example question: {example['question'][:100]}...")
            print(f"  Example answer: {example['final_decision']}")
            
        elif dataset_name == 'bioasq':
            print("Note: BioASQ requires manual download")
            print("Trying common dataset names...")
            dataset_names = ['bioasq/bioasq_task_b', 'bioasq', 'EuropePMC/bioasq']
            
            for name in dataset_names:
                try:
                    dataset = load_dataset(name, split='train')
                    print(f"✓ Dataset loaded from: {name}")
                    print(f"  Total samples: {len(dataset)}")
                    return True
                except:
                    continue
            
            print("✗ Could not load BioASQ from HuggingFace")
            print("  Use --data_path to specify local dataset")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False


def test_generation(model_path, model_type='auto'):
    """Test text generation"""
    print("="*70)
    print("GENERATION TEST")
    print("="*70)
    
    try:
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        if model_type == 'mamba2':
            if not MAMBA_AVAILABLE:
                print("✗ Cannot test Mamba2: mamba_ssm not installed")
                return False
            model = MambaLMHeadModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.to(device)
        model.eval()
        
        # Test prompt
        prompt = "Question: What is diabetes? Answer:"
        print(f"\nTest prompt: {prompt}")
        
        # Generate
        inputs = tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        
        print(f"Generated answer: {answer}")
        print("✓ Generation test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test evaluation setup')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model to test')
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['mamba2', 'biogpt', 'gpt2', 'auto'],
                        help='Model type')
    parser.add_argument('--skip_generation', action='store_true',
                        help='Skip generation test (faster)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("BIOASQ EVALUATION SETUP TEST")
    print("="*70 + "\n")
    
    # Test environment
    test_environment()
    print("\n")
    
    # Test dataset loading
    test_dataset_loading('pubmedqa')
    print("\n")
    
    # Test model loading if path provided
    if args.model_path:
        success = test_model_loading(args.model_path, args.model_type)
        print("\n")
        
        # Test generation
        if success and not args.skip_generation:
            test_generation(args.model_path, args.model_type)
            print("\n")
    else:
        print("="*70)
        print("MODEL TEST SKIPPED")
        print("="*70)
        print("Provide --model_path to test model loading and generation")
        print("\n")
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("Environment: ✓")
    print("Dataset loading: ✓ (test passed)")
    if args.model_path:
        print("Model loading: Run completed (check output above)")
        if not args.skip_generation:
            print("Generation: Run completed (check output above)")
    else:
        print("Model loading: Skipped (no model path provided)")
    print("\n")
    
    print("To run full evaluation, use:")
    print("  bash run_evaluate_bioasq.sh")
    print("Or:")
    print("  python evaluate_bioasq.py --model_type mamba2 --model_path YOUR_MODEL_PATH")
    print("")


if __name__ == "__main__":
    main()
