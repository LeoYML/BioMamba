"""
Universal BioASQ Evaluation Script
Supports multiple HuggingFace models including Mamba2, BioGPT, GPT2, etc.

Dataset: BioASQ Task B (Biomedical Semantic QA)
Metrics: Accuracy, F1, Precision, Recall, Exact Match
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
import torch
import argparse
import json
import numpy as np
import re
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BioGptTokenizer,
    BioGptForCausalLM
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Try to import Mamba2
try:
    from mamba_ssm import MambaLMHeadModel
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba_ssm not installed. Mamba2 models will not be available.")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate models on BioASQ dataset')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['mamba2', 'biogpt', 'gpt2', 't5', 'auto'],
                        help='Model type to use')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint or HuggingFace model name')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to tokenizer (default: same as model_path)')
    
    # Data arguments
    parser.add_argument('--dataset_name', type=str, default='bioasq',
                        choices=['bioasq', 'pubmedqa'],
                        help='Dataset to evaluate on')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to local dataset (if not using HuggingFace)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum input sequence length')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (None for all)')
    
    # Generation arguments
    parser.add_argument('--max_new_tokens', type=int, default=10,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Top-p sampling parameter')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use sampling instead of greedy decoding')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for evaluation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save individual predictions to JSON file')
    
    return parser.parse_args()


def load_model_and_tokenizer(args, device):
    """Load model and tokenizer based on model type"""
    tokenizer_path = args.tokenizer_path or args.model_path
    
    print(f"Loading {args.model_type} model from: {args.model_path}")
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    if args.model_type == 'mamba2':
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm not installed. Install with: pip install mamba-ssm")
        if device.type != "cuda":
            raise RuntimeError(
                "Mamba2 evaluation requires CUDA. Current run is on CPU. "
                "Please use a CUDA-enabled PyTorch environment."
            )
        
        # Load model
        try:
            print(f"Loading model weights...")
            # Force explicit device mapping so CPU-only environments can load GPU-saved checkpoints.
            model = MambaLMHeadModel.from_pretrained(args.model_path, device=str(device))
            print(f"Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Handle tokenizer loading for Mamba models
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"Tokenizer loaded from {tokenizer_path}")
        except (OSError, ValueError, Exception) as e:
            # HuggingFace Mamba models don't include tokenizer
            # Use the same tokenizer as used in pre-training
            if 'state-spaces' in args.model_path.lower():
                print(f"Note: {args.model_path} doesn't include a tokenizer.")
                print("Using state-spaces/mamba-2.8b-hf tokenizer")
                tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
            else:
                print(f"Error loading tokenizer: {e}")
                raise e
        
    elif args.model_type == 'biogpt':
        try:
            # Try to load with BioGPT-specific classes
            tokenizer = BioGptTokenizer.from_pretrained(tokenizer_path)
            model = BioGptForCausalLM.from_pretrained(args.model_path)
        except:
            # Fallback to Auto classes
            print("Note: Using AutoModel/AutoTokenizer for BioGPT")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    elif args.model_type == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    elif args.model_type == 't5':
        # T5 is a seq2seq model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
        print("Note: T5 is a seq2seq model")
    
    elif args.model_type == 'auto':
        # Try to automatically determine model type
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Try different model classes
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_path)
        except:
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
            except:
                raise ValueError(f"Could not load model from {args.model_path}")
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Setup padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device
    model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully")
    print(f"Total parameters: {total_params:,}")
    
    return model, tokenizer


def format_bioasq_prompt(
    question: str,
    context: str = "",
    snippets: List[str] = None,
    max_context_chars: int = 1200,
    model_type: str = "default",
    question_type: str = "yesno",
) -> str:
    """Format BioASQ question into prompt with context truncation"""
    question_type = (question_type or "").lower()
    is_factoid_like = question_type in {"factoid", "list", "summary"}
    
    # Different prompt formats for different model types
    if model_type == "t5":
        # T5-specific format (instruction-following style)
        # T5 models are trained to follow instructions
        
        # Truncate context
        context_text = ""
        if snippets and len(snippets) > 0:
            snippet = snippets[0]
            snippet_text = snippet.get('text', snippet) if isinstance(snippet, dict) else snippet
            context_text = snippet_text[:800] + "..." if len(snippet_text) > 800 else snippet_text
        elif context:
            context_text = context[:800] + "..." if len(context) > 800 else context
        
        # T5 instruction format
        if is_factoid_like:
            instruction = "Answer this biomedical question with a short factual phrase."
        else:
            instruction = "Answer this biomedical question with yes, no, or maybe."
        if context_text:
            prompt = f"{instruction}\n\nContext: {context_text}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"{instruction}\n\nQuestion: {question}\n\nAnswer:"
        return prompt
    
    elif model_type == "biogpt":
        # BioGPT is a pre-trained LM (not instruction-tuned)
        # Best approach: Use natural language that encourages yes/no generation
        # Format: "Based on research, [question]? Yes or No:"
        
        # Truncate context - keep it short for BioGPT
        context_text = ""
        if snippets and len(snippets) > 0:
            # Only use first snippet, keep it brief
            snippet = snippets[0]
            snippet_text = snippet.get('text', snippet) if isinstance(snippet, dict) else snippet
            context_text = snippet_text[:400] + "..." if len(snippet_text) > 400 else snippet_text
        elif context:
            context_text = context[:400] + "..." if len(context) > 400 else context
        
        # Very simple prompt that directly asks for yes/no
        if context_text:
            prompt = f"Background: {context_text}\n\n"
        else:
            prompt = ""
        
        # Convert question to statement form and ask for confirmation
        # This is more natural for language models
        prompt += f"Question: {question}\n"
        if is_factoid_like:
            prompt += "Answer:"
        else:
            prompt += "Answer (yes/no):"
        return prompt
    
    else:
        # Default format for instruction-following models (Mamba, etc.)
        if is_factoid_like:
            prompt = "Answer biomedical questions with a short factual phrase only.\n\n"
            prompt += "Example 1:\nQuestion: Which vitamin is deficient in scurvy?\nAnswer: vitamin C\n\n"
            prompt += "Example 2:\nQuestion: Which bacterium causes cholera?\nAnswer: Vibrio cholerae\n\n"
            prompt += "Now answer this question:\n\n"
        else:
            prompt = "Answer biomedical questions with ONLY one word: yes, no, or maybe.\n\n"
            prompt += "Example 1:\nQuestion: Is aspirin used to treat headaches?\nAnswer: yes\n\n"
            prompt += "Example 2:\nQuestion: Does vitamin C cure cancer?\nAnswer: no\n\n"
            prompt += "Now answer this question:\n\n"
        
        # Truncate context to prevent exceeding model max length
        if snippets and len(snippets) > 0:
            prompt += "Context:\n"
            context_chars = 0
            for i, snippet in enumerate(snippets):
                snippet_text = snippet.get('text', snippet) if isinstance(snippet, dict) else snippet
                if context_chars + len(snippet_text) > max_context_chars:
                    remaining = max_context_chars - context_chars
                    if remaining > 100:
                        prompt += f"{snippet_text[:remaining]}...\n\n"
                    break
                prompt += f"{snippet_text}\n\n"
                context_chars += len(snippet_text) + 2
        elif context:
            truncated_context = context[:max_context_chars]
            if len(context) > max_context_chars:
                truncated_context += "..."
            prompt += f"Context:\n{truncated_context}\n\n"
        
        prompt += f"Question: {question}\n"
        prompt += "Answer:"
        return prompt


def format_pubmedqa_prompt(question: str, context: str, max_context_chars: int = 1200, model_type: str = "default") -> str:
    """Format PubMedQA question into prompt with context truncation"""
    
    # Truncate context if too long
    truncated_context = context[:max_context_chars]
    if len(context) > max_context_chars:
        truncated_context += "..."
    
    if model_type == "t5":
        # T5 instruction format
        prompt = f"Answer this biomedical question with yes, no, or maybe.\n\nContext: {truncated_context}\n\nQuestion: {question}\n\nAnswer:"
        return prompt
    
    elif model_type == "biogpt":
        # BioGPT-specific format - simple yes/no prompt
        prompt = f"Background: {truncated_context}\n\n"
        prompt += f"Question: {question}\n"
        prompt += "Answer (yes/no):"
        return prompt
    else:
        # Default format for instruction-following models
        prompt = "Answer biomedical questions with ONLY one word: yes, no, or maybe.\n\n"
        prompt += "Example 1:\nQuestion: Is aspirin used to treat headaches?\nAnswer: yes\n\n"
        prompt += "Example 2:\nQuestion: Does vitamin C cure cancer?\nAnswer: no\n\n"
        prompt += "Now answer this question:\n\n"
        prompt += f"Context:\n{truncated_context}\n\n"
        prompt += f"Question: {question}\n"
        prompt += "Answer:"
        return prompt


def load_bioasq_dataset(args):
    """Load BioASQ dataset"""
    if args.data_path:
        print(f"Loading dataset from local path: {args.data_path}")
        try:
            dataset = load_from_disk(args.data_path)
            if args.split in dataset:
                return dataset[args.split]
            else:
                return dataset
        except:
            raise ValueError(f"Could not load dataset from {args.data_path}")
    else:
        print("Loading BioASQ dataset from HuggingFace...")
        # Try different BioASQ dataset names
        dataset_names = [
            'bioasq/bioasq_task_b',
            'bioasq',
            'EuropePMC/bioasq'
        ]
        
        for name in dataset_names:
            try:
                dataset = load_dataset(name, split=args.split)
                print(f"Successfully loaded dataset: {name}")
                return dataset
            except:
                continue
        
        raise ValueError(
            "Could not load BioASQ dataset from HuggingFace. "
            "Please provide a local dataset path with --data_path"
        )


def load_pubmedqa_dataset(args):
    """Load PubMedQA dataset"""
    if args.data_path:
        print(f"Loading PubMedQA from local path: {args.data_path}")
        dataset = load_from_disk(args.data_path)
        
        # Check if this is a tokenized dataset
        if args.split in dataset:
            loaded_dataset = dataset[args.split]
        else:
            loaded_dataset = dataset
        
        # Check if dataset has the required fields
        if 'question' not in loaded_dataset.features:
            print("\n" + "="*70)
            print("WARNING: This appears to be a tokenized dataset!")
            print("="*70)
            print("Available columns:", loaded_dataset.column_names)
            print("\nFor evaluation, please use the original (non-tokenized) dataset.")
            print("Download it from HuggingFace by NOT specifying --data_path,")
            print("or use a non-tokenized local dataset.")
            print("="*70)
            raise ValueError("Dataset is tokenized. Need original dataset with 'question' field.")
        
        return loaded_dataset
    else:
        print("Loading PubMedQA dataset from HuggingFace...")
        dataset = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
        # Split for validation
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        if args.split == 'train':
            return split_dataset['train']
        else:
            return split_dataset['test']


def extract_answer(generated_text: str, prompt: str = "", question_type: str = "yesno") -> str:
    """Extract answer from generated text"""
    # Remove prompt from generated text when full text is decoded
    if prompt and generated_text.startswith(prompt):
        answer = generated_text[len(prompt):].strip()
    else:
        answer = generated_text.strip()
    
    # Extract first line only (answer should be short)
    answer = answer.split('\n')[0].strip()
    answer = answer.lower()
    
    # Remove common prefixes and patterns
    answer = answer.replace('answer:', '').strip()
    answer = answer.replace('the answer is', '').strip()
    answer = answer.replace('a:', '').strip()
    answer = answer.replace('(yes/no):', '').strip()
    answer = answer.replace('yes/no:', '').strip()
    
    # Remove leading whitespace and punctuation
    answer = answer.lstrip(' \t:;,-.')
    question_type = (question_type or "").lower()

    if question_type in {"factoid", "list", "summary"}:
        # Keep phrase-level outputs for factoid-style QA.
        answer = re.sub(r"\s+", " ", answer).strip(" \t:;,-.\"'()[]{}")
        if answer.startswith("answer is "):
            answer = answer[len("answer is "):].strip()
        return answer

    # Remove leading articles and common words (only at start)
    while True:
        old_answer = answer
        for prefix in ['answer', 'the', 'a', 'an', 'is', 'are', 'it', 'that']:
            if answer.lower().startswith(f'{prefix} '):
                answer = answer[len(prefix)+1:].strip()
                break
        if answer == old_answer:
            break
    
    # Special handling for common BioGPT outputs
    # Sometimes it generates full sentences like "yes, ..." or "no, because..."
    if answer.startswith('yes'):
        return 'yes'
    if answer.startswith('no'):
        return 'no'
    if answer.startswith('maybe'):
        return 'maybe'
    
    # Try to extract yes/no/maybe with flexible matching
    answer_words = answer.split()
    for word in answer_words[:5]:  # Check first 5 words (increased from 3)
        word_clean = word.strip('.,;:!?()[]{}"\' ')
        if word_clean == 'yes':
            return 'yes'
        elif word_clean == 'no':
            return 'no'
        elif word_clean == 'maybe' or word_clean == 'uncertain' or word_clean == 'unclear':
            return 'maybe'
    
    # Check for yes/no in first 30 chars (increased from 20)
    answer_start = answer[:30]
    if ' yes' in answer_start or answer_start.startswith('yes'):
        # Make sure it's not "yes but" or "yes, but" which might indicate no
        if 'but' not in answer_start[:15] and 'however' not in answer_start[:15]:
            return 'yes'
    elif ' no' in answer_start or answer_start.startswith('no'):
        # Make sure it's not "not yes" or similar double negative
        if 'not yes' not in answer_start and 'no yes' not in answer_start:
            return 'no'
    elif 'maybe' in answer_start or 'unclear' in answer_start or 'uncertain' in answer_start:
        return 'maybe'
    
    # Last-resort cleanup for cases where model echoed instructions
    if "answer:" in answer:
        answer = answer.split("answer:")[-1].strip()
        if answer.startswith("yes"):
            return "yes"
        if answer.startswith("no"):
            return "no"
        if answer.startswith("maybe"):
            return "maybe"

    # Return first word if can't determine
    return answer_words[0] if answer_words else answer


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison"""
    answer = answer.lower().strip()
    
    # Map common variations
    if answer in ['yes', 'y', 'true', '1']:
        return 'yes'
    elif answer in ['no', 'n', 'false', '0']:
        return 'no'
    elif answer in ['maybe', 'uncertain', 'unclear', 'unknown']:
        return 'maybe'
    
    return answer


def generate_answer(model, tokenizer, prompt: str, args, device, question_type: str = "yesno") -> str:
    """Generate answer for a given prompt"""
    
    # Check if this is a seq2seq model (T5)
    is_seq2seq = args.model_type == 't5' or 't5' in args.model_path.lower()
    
    # Tokenize
    inputs = tokenizer(
        prompt, 
        return_tensors='pt', 
        truncation=True, 
        max_length=args.max_length
    )
    input_ids = inputs['input_ids']
    
    # Validate token IDs to prevent CUDA errors
    vocab_size = len(tokenizer)
    max_token_id = input_ids.max().item()
    min_token_id = input_ids.min().item()
    
    if max_token_id >= vocab_size or min_token_id < 0:
        print(f"Warning: Invalid token IDs detected (range: {min_token_id} to {max_token_id}, vocab size: {vocab_size})")
        # Clamp token IDs to valid range
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        print(f"Clamped to valid range: 0 to {vocab_size - 1}")
    
    # Move to device after validation
    try:
        input_ids = input_ids.to(device)
    except Exception as e:
        print(f"Warning: Failed to move input to device: {e}")
        return "error"
    
    input_length = input_ids.shape[1]
    
    # Calculate max_length for generation
    max_gen_length = input_length + args.max_new_tokens
    
    # CPU fallback for Mamba when mamba_ssm generate() hits CUDA-only paths.
    def mamba_cpu_autoregressive_generate(input_tokens):
        generated = input_tokens
        for _ in range(args.max_new_tokens):
            outputs_local = model(generated)
            next_token_logits = outputs_local.logits[:, -1, :]
            if args.do_sample and args.temperature > 0:
                probs = torch.softmax(next_token_logits / args.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
        return generated

    # Generate
    with torch.no_grad():
        try:
            # Check model type
            model_class_name = model.__class__.__name__
            is_mamba = 'Mamba' in model_class_name
            
            if is_seq2seq:
                # T5 and other seq2seq models
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.do_sample else 1.0,
                    top_p=args.top_p if args.do_sample else 1.0,
                    do_sample=args.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            elif is_mamba:
                # Mamba2 generation parameters
                gen_kwargs = {
                    'max_length': max_gen_length,
                    'cg': device.type == 'cuda',  # CUDA graphs only on CUDA
                    'return_dict_in_generate': False,
                    'output_scores': False,
                }
                
                # Add sampling parameters only if supported
                if args.do_sample and args.temperature > 0:
                    gen_kwargs['temperature'] = args.temperature
                    gen_kwargs['top_p'] = args.top_p
                
                outputs = model.generate(input_ids, **gen_kwargs)
                
            else:
                # Standard HuggingFace model
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature if args.do_sample else 1.0,
                    top_p=args.top_p if args.do_sample else 1.0,
                    do_sample=args.do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
        except Exception as e:
            error_str = str(e)
            
            # Try alternative generation methods
            if "max_length" in error_str or "max_new_tokens" in error_str or "do_sample" in error_str:
                try:
                    # Minimal generation call
                    outputs = model.generate(
                        input_ids,
                        max_length=max_gen_length,
                    )
                except Exception as e2:
                    print(f"Warning: Generation failed with error: {e2}")
                    return "error"
            elif is_mamba and device.type != 'cuda':
                try:
                    outputs = mamba_cpu_autoregressive_generate(input_ids)
                except Exception as e3:
                    print(f"Warning: CPU fallback generation failed with error: {e3}")
                    return "error"
            else:
                print(f"Warning: Generation failed with error: {e}")
                return "error"
    
    # Decode only newly generated tokens for causal models.
    # This avoids prompt-echo artifacts being parsed as answers.
    try:
        if is_seq2seq:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            generated_ids = outputs[0][input_length:]
            if generated_ids.numel() == 0:
                generated_ids = outputs[0]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    except Exception as e:
        print(f"Warning: Decode failed with error: {e}")
        return "error"
    
    # Extract answer
    answer = extract_answer(generated_text, prompt, question_type=question_type)
    
    return answer


def normalize_factoid_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"^(answer|the answer is)\s*[:\-]?\s*", "", text)
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    return text


def is_factoid_correct(prediction: str, answer_candidates: List[str]) -> bool:
    pred = normalize_factoid_text(prediction)
    if not pred:
        return False

    refs = [normalize_factoid_text(a) for a in answer_candidates if normalize_factoid_text(a)]
    if not refs:
        return False

    if pred in refs:
        return True

    for ref in refs:
        if len(ref) >= 3 and re.search(rf"\b{re.escape(ref)}\b", pred):
            return True
    return False


def evaluate_bioasq(model, tokenizer, dataset, args, device) -> Tuple[Dict, List]:
    """Evaluate model on BioASQ dataset"""
    predictions = []
    references = []
    exact_match_overrides = []
    all_results = []
    
    # Subsample if requested
    if args.max_samples and args.max_samples < len(dataset):
        import random
        random.seed(42)
        indices = random.sample(range(len(dataset)), args.max_samples)
        dataset = dataset.select(indices)
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        try:
            # Get question and answer
            if args.dataset_name == 'bioasq':
                # BioASQ format
                question = example.get('question', example.get('body', ''))
                question_type = str(example.get('question_type', example.get('type', 'yesno'))).lower()
                true_answer = example.get('exact_answer', example.get('answer', ''))
                answer_aliases = example.get('answer_aliases', [])
                if not isinstance(answer_aliases, list):
                    answer_aliases = [str(answer_aliases)]
                answer_aliases = [a for a in answer_aliases if isinstance(a, str) and a.strip()]
                if isinstance(true_answer, list):
                    for item in true_answer:
                        if isinstance(item, str) and item.strip():
                            answer_aliases.append(item)
                if not answer_aliases and isinstance(true_answer, str) and true_answer.strip():
                    answer_aliases = [true_answer]
                snippets = example.get('snippets', [])
                context = '\n'.join([s.get('text', s) if isinstance(s, dict) else s for s in snippets])
                
                # Format prompt (pass model_type for BioGPT-specific formatting)
                prompt = format_bioasq_prompt(
                    question,
                    context=context,
                    model_type=args.model_type,
                    question_type=question_type,
                )
                
            elif args.dataset_name == 'pubmedqa':
                # PubMedQA format
                question = example['question']
                true_answer = example.get('final_decision', example.get('answer', ''))
                answer_aliases = [str(true_answer)] if true_answer is not None else []
                question_type = "yesno"
                
                # Get context
                context_dict = example.get('context', {})
                if isinstance(context_dict, dict):
                    contexts = context_dict.get('contexts', [])
                    context = '\n\n'.join(contexts)
                else:
                    context = str(context_dict)
                
                # Format prompt (pass model_type for BioGPT-specific formatting)
                prompt = format_pubmedqa_prompt(question, context, model_type=args.model_type)
            
            # Debug: print first example
            if idx == 0:
                print(f"\n[DEBUG] First example:")
                print(f"Question: {question[:100]}...")
                print(f"True answer: {true_answer}")
                print(f"Prompt length: {len(prompt)} chars")
                print(f"Prompt (last 200 chars): ...{prompt[-200:]}")
            
            # Generate answer
            predicted_answer = generate_answer(model, tokenizer, prompt, args, device, question_type=question_type)
            
            # Normalize answers
            pred_normalized = normalize_answer(predicted_answer)
            true_normalized = normalize_answer(str(true_answer)) if not isinstance(true_answer, list) else normalize_answer(str(true_answer[0]) if true_answer else "")
            if question_type in {"factoid", "list", "summary"}:
                normalized_aliases = [normalize_factoid_text(a) for a in answer_aliases]
                normalized_aliases = [a for a in normalized_aliases if a]
                if normalized_aliases:
                    true_normalized = normalized_aliases[0]
                is_correct = is_factoid_correct(predicted_answer, normalized_aliases)
            else:
                normalized_aliases = []
                is_correct = pred_normalized == true_normalized
            
            # Debug: print first few predictions
            if idx < 3:
                print(f"\n[Example {idx}]")
                print(f"Predicted (raw): {predicted_answer}")
                print(f"Predicted (normalized): {pred_normalized}")
                print(f"True (normalized): {true_normalized}")
                print(f"Correct: {is_correct}")
            
            predictions.append(pred_normalized)
            references.append(true_normalized)
            exact_match_overrides.append(is_correct)
            
            # Save individual result
            if args.save_predictions:
                all_results.append({
                    'index': idx,
                    'question': question,
                    'true_answer': true_answer,
                    'answer_aliases': normalized_aliases,
                    'question_type': question_type,
                    'predicted_answer': predicted_answer,
                    'normalized_prediction': pred_normalized,
                    'normalized_reference': true_normalized,
                    'correct': is_correct
                })
        
        except Exception as e:
            error_msg = str(e)
            print(f"\nError processing example {idx}: {error_msg}")
            
            # If CUDA error, try to recover
            if "CUDA" in error_msg or "cuda" in error_msg:
                print("CUDA error detected. Attempting to recover...")
                try:
                    # Clear CUDA cache and reset device
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # Move model back to device to reset CUDA context
                        model.to(device)
                    print("CUDA recovery attempted. Continuing with next sample...")
                except Exception as recovery_error:
                    print(f"Recovery failed: {recovery_error}")
                    pass
            
            # Print traceback for debugging
            if idx < 5 or idx % 20 == 0:  # Only print detailed trace for first few and every 20th error
                import traceback
                traceback.print_exc()
            
            predictions.append("error")
            references.append("unknown")
            exact_match_overrides.append(None)
            continue
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references, exact_match_overrides)
    
    return metrics, all_results


def calculate_metrics(predictions: List[str], references: List[str], exact_match_overrides: Optional[List[Optional[bool]]] = None) -> Dict:
    """Calculate evaluation metrics"""
    # Filter out errors
    valid_indices = [i for i, (p, r) in enumerate(zip(predictions, references))
                     if p != "error" and r != "unknown"]
    valid_pairs = [(predictions[i], references[i]) for i in valid_indices]
    
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'exact_match': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'total_samples': len(predictions),
            'valid_samples': 0,
            'error_samples': len(predictions),
            'per_class': {}
        }
    
    preds, refs = zip(*valid_pairs)
    
    # Exact match / Accuracy
    exact_matches = []
    for valid_idx, (p, r) in zip(valid_indices, valid_pairs):
        if (
            exact_match_overrides is not None
            and len(exact_match_overrides) == len(predictions)
            and exact_match_overrides[valid_idx] is not None
        ):
            exact_matches.append(bool(exact_match_overrides[valid_idx]))
        else:
            exact_matches.append(p == r)
    accuracy = sum(exact_matches) / len(exact_matches)
    
    # Per-class metrics
    unique_labels = list(set(refs + preds))
    
    # Convert to numerical labels for sklearn
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    pred_ids = [label_to_id.get(p, -1) for p in preds]
    ref_ids = [label_to_id.get(r, -1) for r in refs]
    
    # Calculate precision, recall, F1
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            ref_ids, pred_ids, average='macro', zero_division=0
        )
    except:
        precision, recall, f1 = 0.0, 0.0, 0.0
    
    # Calculate per-class metrics
    per_class_metrics = {}
    for label in unique_labels:
        label_preds = [1 if p == label else 0 for p in preds]
        label_refs = [1 if r == label else 0 for r in refs]
        
        if sum(label_refs) > 0:
            try:
                p, r, f, _ = precision_recall_fscore_support(
                    label_refs, label_preds, average='binary', zero_division=0
                )
                per_class_metrics[label] = {
                    'precision': float(p),
                    'recall': float(r),
                    'f1': float(f),
                    'support': sum(label_refs)
                }
            except:
                per_class_metrics[label] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'support': sum(label_refs)
                }
    
    metrics = {
        'accuracy': float(accuracy),
        'exact_match': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'total_samples': len(predictions),
        'valid_samples': len(valid_pairs),
        'error_samples': len(predictions) - len(valid_pairs),
        'per_class': per_class_metrics
    }
    
    return metrics


def print_metrics(metrics: Dict, args):
    """Print evaluation metrics in a formatted way"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nModel: {args.model_path}")
    print(f"Model Type: {args.model_type}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Split: {args.split}")
    print("\n" + "-"*70)
    print("OVERALL METRICS")
    print("-"*70)
    print(f"Total Samples:     {metrics['total_samples']}")
    print(f"Valid Samples:     {metrics['valid_samples']}")
    print(f"Error Samples:     {metrics['error_samples']}")
    print(f"\nAccuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Exact Match:       {metrics['exact_match']:.4f}")
    print(f"Precision (macro): {metrics['precision']:.4f}")
    print(f"Recall (macro):    {metrics['recall']:.4f}")
    print(f"F1 Score (macro):  {metrics['f1']:.4f}")
    
    # Per-class metrics
    if 'per_class' in metrics and metrics['per_class']:
        print("\n" + "-"*70)
        print("PER-CLASS METRICS")
        print("-"*70)
        for label, class_metrics in metrics['per_class'].items():
            print(f"\nClass: {label}")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall:    {class_metrics['recall']:.4f}")
            print(f"  F1 Score:  {class_metrics['f1']:.4f}")
            print(f"  Support:   {class_metrics['support']}")
    
    print("\n" + "="*70 + "\n")


def save_results(metrics: Dict, predictions: List, args):
    """Save evaluation results to file"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create filename
    model_name = os.path.basename(args.model_path)
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.dataset_name}_{args.model_type}_{model_name}_{timestamp}"
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"{filename}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'model_path': args.model_path,
            'model_type': args.model_type,
            'dataset': args.dataset_name,
            'split': args.split,
            'metrics': metrics,
            'args': vars(args)
        }, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")
    
    # Save predictions if requested
    if args.save_predictions and predictions:
        predictions_path = os.path.join(args.output_dir, f"{filename}_predictions.json")
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Predictions saved to: {predictions_path}")


def main():
    args = parse_args()
    
    # Set device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args, device)
    
    # Load dataset
    print("\n" + "="*70)
    print("Loading dataset...")
    print("="*70)
    
    if args.dataset_name == 'bioasq':
        dataset = load_bioasq_dataset(args)
    elif args.dataset_name == 'pubmedqa':
        dataset = load_pubmedqa_dataset(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    print(f"Loaded {len(dataset)} samples from {args.dataset_name}")
    
    # Evaluate
    print("\n" + "="*70)
    print("Starting evaluation...")
    print("="*70 + "\n")
    
    metrics, predictions = evaluate_bioasq(model, tokenizer, dataset, args, device)
    
    # Print results
    print_metrics(metrics, args)
    
    # Save results
    save_results(metrics, predictions, args)
    
    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
