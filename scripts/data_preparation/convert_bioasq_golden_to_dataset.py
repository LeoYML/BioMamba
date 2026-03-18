"""
Convert BioASQ Golden JSON files to HuggingFace Dataset format for evaluation
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
import json
import argparse
from datasets import Dataset, DatasetDict
from typing import List, Dict
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Convert BioASQ Golden files to HuggingFace format')
    parser.add_argument('--input_files', type=str, nargs='+', required=True,
                        help='List of BioASQ golden JSON files')
    parser.add_argument('--output_dir', type=str, default='./data/bioasq_test',
                        help='Output directory for HuggingFace dataset')
    parser.add_argument('--question_type', type=str, default='yesno',
                        choices=['yesno', 'factoid', 'list', 'summary', 'all'],
                        help='Type of questions to extract')
    return parser.parse_args()


def load_bioasq_golden(file_path: str) -> List[Dict]:
    """Load BioASQ golden JSON file"""
    print(f"Loading: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'questions' in data:
        questions = data['questions']
    elif isinstance(data, list):
        questions = data
    else:
        raise ValueError(f"Unknown format in {file_path}")
    
    print(f"  Found {len(questions)} questions")
    return questions


def flatten_answer_candidates(answer_obj) -> List[str]:
    """Flatten BioASQ exact_answer field into a list of string candidates."""
    candidates: List[str] = []

    def _walk(node):
        if node is None:
            return
        if isinstance(node, str):
            text = node.strip()
            if text:
                candidates.append(text)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)
            return

        text = str(node).strip()
        if text:
            candidates.append(text)

    _walk(answer_obj)

    # Preserve order while removing duplicates (case-insensitive)
    deduped: List[str] = []
    seen = set()
    for c in candidates:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def convert_bioasq_question(question: Dict) -> Dict:
    """Convert a single BioASQ question to evaluation format"""
    
    # Extract question text
    body = question.get('body', '')
    
    # Extract exact answers (factoid/list can contain nested lists)
    exact_answer = question.get('exact_answer', '')
    answer_candidates = flatten_answer_candidates(exact_answer)
    answer = answer_candidates[0] if answer_candidates else ''
    
    # Extract ideal answer
    ideal_answer = question.get('ideal_answer', [])
    if isinstance(ideal_answer, list):
        ideal_answer_text = ' '.join(ideal_answer) if ideal_answer else ''
    else:
        ideal_answer_text = str(ideal_answer) if ideal_answer else ''
    
    # Extract snippets (context)
    snippets = question.get('snippets', [])
    snippet_texts = []
    snippet_documents = []
    
    for snippet in snippets:
        if isinstance(snippet, dict):
            text = snippet.get('text', '').strip()
            if text:
                snippet_texts.append(text)
                snippet_documents.append(snippet.get('document', ''))
        elif isinstance(snippet, str):
            snippet_texts.append(snippet.strip())
    
    # Combine snippets into context
    context = '\n\n'.join(snippet_texts)
    
    # Extract documents
    documents = question.get('documents', [])
    
    return {
        'id': question.get('id', ''),
        'question': body,
        'answer': answer.lower() if answer else '',
        'answer_aliases': [a.lower() for a in answer_candidates],
        'ideal_answer': ideal_answer_text,
        'question_type': question.get('type', ''),
        'snippets': snippet_texts,
        'context': context,
        'documents': documents,
        'num_snippets': len(snippet_texts)
    }


def process_bioasq_files(input_files: List[str], question_type: str = 'yesno') -> List[Dict]:
    """Process multiple BioASQ golden files"""
    
    all_questions = []
    stats = {
        'total': 0,
        'yesno': 0,
        'factoid': 0,
        'list': 0,
        'summary': 0,
        'other': 0
    }
    
    for file_path in input_files:
        questions = load_bioasq_golden(file_path)
        
        for q in questions:
            q_type = q.get('type', '').lower()
            stats['total'] += 1
            
            if q_type in stats:
                stats[q_type] += 1
            else:
                stats['other'] += 1
            
            # Filter by question type
            if question_type == 'all' or q_type == question_type:
                converted = convert_bioasq_question(q)
                all_questions.append(converted)
    
    return all_questions, stats


def main():
    args = parse_args()
    
    print("="*70)
    print("Converting BioASQ Golden Files to Test Dataset")
    print("="*70)
    print(f"Input files: {len(args.input_files)}")
    for f in args.input_files:
        print(f"  - {f}")
    print(f"Output directory: {args.output_dir}")
    print(f"Question type filter: {args.question_type}")
    print("")
    
    # Check if files exist
    for file_path in args.input_files:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return
    
    # Process all files
    print("Processing files...")
    questions, stats = process_bioasq_files(args.input_files, args.question_type)
    
    print("\n" + "="*70)
    print("Statistics")
    print("="*70)
    print(f"Total questions in files: {stats['total']}")
    print(f"  - Yes/No questions: {stats['yesno']}")
    print(f"  - Factoid questions: {stats['factoid']}")
    print(f"  - List questions: {stats['list']}")
    print(f"  - Summary questions: {stats['summary']}")
    print(f"  - Other: {stats['other']}")
    print(f"\nQuestions after filtering ({args.question_type}): {len(questions)}")
    
    if len(questions) == 0:
        print("\nError: No questions found after filtering!")
        return
    
    # Create HuggingFace Dataset
    print("\nCreating HuggingFace dataset...")
    dataset = Dataset.from_list(questions)
    
    print(f"✓ Created dataset with {len(dataset)} samples")
    print(f"  Features: {list(dataset.features.keys())}")
    
    # Show statistics about the dataset
    print("\nDataset Statistics:")
    if 'num_snippets' in dataset.features:
        num_snippets = [q['num_snippets'] for q in questions]
        print(f"  Average snippets per question: {sum(num_snippets)/len(num_snippets):.2f}")
        print(f"  Min snippets: {min(num_snippets)}")
        print(f"  Max snippets: {max(num_snippets)}")
    
    if args.question_type == 'yesno':
        # Count answer distribution
        answers = [q['answer'] for q in questions]
        yes_count = sum(1 for a in answers if a == 'yes')
        no_count = sum(1 for a in answers if a == 'no')
        other_count = len(answers) - yes_count - no_count
        print(f"\n  Answer distribution:")
        print(f"    Yes: {yes_count} ({yes_count/len(answers)*100:.1f}%)")
        print(f"    No: {no_count} ({no_count/len(answers)*100:.1f}%)")
        if other_count > 0:
            print(f"    Other: {other_count} ({other_count/len(answers)*100:.1f}%)")
    
    # Show examples
    print("\n" + "="*70)
    print("Example Samples")
    print("="*70)
    
    for i in range(min(3, len(questions))):
        example = questions[i]
        print(f"\nExample {i+1}:")
        print(f"  ID: {example['id']}")
        print(f"  Question: {example['question'][:100]}...")
        print(f"  Answer: {example['answer']}")
        print(f"  Type: {example['question_type']}")
        print(f"  Snippets: {example['num_snippets']}")
        if example['context']:
            print(f"  Context preview: {example['context'][:100]}...")
    
    # Create dataset dict with test split
    dataset_dict = DatasetDict({
        'test': dataset
    })
    
    # Save to disk
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving dataset to: {args.output_dir}")
    dataset_dict.save_to_disk(args.output_dir)
    
    print(f"✓ Dataset saved successfully")
    
    # Save metadata
    metadata = {
        'dataset_name': 'BioASQ Task B',
        'source_files': [os.path.basename(f) for f in args.input_files],
        'question_type': args.question_type,
        'total_samples': len(dataset),
        'statistics': stats,
        'features': list(dataset.features.keys())
    }
    
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Metadata saved to: {metadata_path}")
    
    # Print usage instructions
    print("\n" + "="*70)
    print("Conversion Complete!")
    print("="*70)
    print(f"\nDataset location: {args.output_dir}")
    print(f"Total test samples: {len(dataset)}")
    print("\nTo use with evaluation script:")
    print("")
    print("  python evaluate_bioasq.py \\")
    print("      --model_type mamba2 \\")
    print("      --model_path ./checkpoints/your_model \\")
    print("      --dataset_name bioasq \\")
    print(f"      --data_path {args.output_dir} \\")
    print("      --split test \\")
    print("      --save_predictions")
    print("")


if __name__ == "__main__":
    main()
