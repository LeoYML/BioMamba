"""
Check problem samples that caused CUDA errors
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

from datasets import load_from_disk
from transformers import AutoTokenizer

# Load dataset
print("Loading dataset...")
dataset = load_from_disk(_os.path.join(_PROJECT_ROOT, 'data/bioasq_test'))['test']
print(f"Total samples: {len(dataset)}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    _os.path.join(_PROJECT_ROOT, "checkpoints/biomamba2_sft_mamba2-130m_full_20260203_092105/final_model")
)
vocab_size = len(tokenizer)
print(f"Vocab size: {vocab_size}")

# Check problem samples (78-81, 0-indexed)
problem_indices = [78, 79, 80, 81]

print("\n" + "="*70)
print("Checking Problem Samples")
print("="*70)

for idx in problem_indices:
    if idx >= len(dataset):
        print(f"\nSample {idx}: Out of range")
        continue
    
    example = dataset[idx]
    question = example['question']
    answer = example['answer']
    context = example['context']
    
    print(f"\n{'='*70}")
    print(f"Sample {idx} (0-indexed)")
    print(f"{'='*70}")
    print(f"Question: {question[:100]}...")
    print(f"Answer: {answer}")
    print(f"Context length: {len(context)} chars")
    print(f"Num snippets: {example['num_snippets']}")
    
    # Try tokenizing
    prompt = f"Answer the following biomedical question with yes or no.\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:"
    
    try:
        tokens = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        input_ids = tokens['input_ids'][0]
        
        max_token_id = input_ids.max().item()
        min_token_id = input_ids.min().item()
        
        print(f"Tokenization successful")
        print(f"  Num tokens: {len(input_ids)}")
        print(f"  Token ID range: {min_token_id} to {max_token_id}")
        
        if max_token_id >= vocab_size:
            print(f"  ⚠️  WARNING: Max token ID {max_token_id} >= vocab size {vocab_size}")
            print(f"  This will cause CUDA errors!")
            
            # Find problematic tokens
            bad_tokens = (input_ids >= vocab_size).nonzero(as_tuple=True)[0]
            if len(bad_tokens) > 0:
                print(f"  Problem token positions: {bad_tokens.tolist()}")
                print(f"  Problem token IDs: {input_ids[bad_tokens].tolist()}")
        
        if min_token_id < 0:
            print(f"  ⚠️  WARNING: Min token ID {min_token_id} < 0")
            
    except Exception as e:
        print(f"  ❌ Tokenization failed: {e}")

print("\n" + "="*70)
print("Analysis Complete")
print("="*70)
print("\nRecommendations:")
print("1. If token IDs are out of range, the tokenizer may not match the model")
print("2. Check if the model was trained with a different tokenizer")
print("3. Consider skipping problematic samples or fixing token ID mapping")
