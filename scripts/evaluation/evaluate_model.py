"""
Evaluation script for trained Mamba2 models on PubMed-MEDLINE test set
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import torch
import argparse
import math
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk
from mamba_ssm import MambaLMHeadModel


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Mamba2 on PubMed-MEDLINE')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory with processed data')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (None for all)')
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load trained model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    model = MambaLMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully")
    print(f"Total parameters: {total_params:,}")
    
    return model, tokenizer


def evaluate(model, dataloader, device):
    """Evaluate model on dataset"""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    print("Evaluating...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # Load model
    model, tokenizer = load_model(args.model_path, device)
    
    # Load test data
    data_path = f"{args.data_dir}/pubmed_medline_tokenized_{args.max_length}"
    print(f"Loading test data from {data_path}...")
    
    try:
        dataset = load_from_disk(data_path)
        test_dataset = dataset['test']
    except:
        print(f"Error: Could not load data from {data_path}")
        print("Please run training script first to process the data")
        return
    
    # Subsample if requested
    if args.num_samples and args.num_samples < len(test_dataset):
        import random
        random.seed(42)
        indices = random.sample(range(len(test_dataset)), args.num_samples)
        test_dataset = test_dataset.select(indices)
        print(f"Evaluating on {args.num_samples} samples")
    else:
        print(f"Evaluating on {len(test_dataset)} samples")
    
    # Create dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")
    
    avg_loss, perplexity = evaluate(model, test_dataloader, device)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {perplexity:.4f}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
