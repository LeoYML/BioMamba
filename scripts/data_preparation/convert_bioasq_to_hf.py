"""
Convert BioASQ JSON format to HuggingFace Dataset format
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


def parse_args():
    parser = argparse.ArgumentParser(description='Convert BioASQ to HuggingFace format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing BioASQ JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for HuggingFace dataset')
    parser.add_argument('--split_name', type=str, default='test',
                        help='Name of the split (train/validation/test)')
    return parser.parse_args()


def load_bioasq_json(file_path: str) -> List[Dict]:
    """Load BioASQ JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # BioASQ format has a 'questions' key
    if isinstance(data, dict) and 'questions' in data:
        return data['questions']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown BioASQ format in {file_path}")


def convert_bioasq_question(question: Dict) -> Dict:
    """Convert a single BioASQ question to standard format"""
    
    # Extract question text
    body = question.get('body', '')
    
    # Extract answer (if available)
    exact_answer = question.get('exact_answer', '')
    ideal_answer = question.get('ideal_answer', '')
    
    # For yes/no questions, exact_answer is typically 'yes' or 'no'
    if isinstance(exact_answer, list):
        if len(exact_answer) > 0:
            answer = exact_answer[0] if isinstance(exact_answer[0], str) else str(exact_answer[0])
        else:
            answer = ''
    else:
        answer = str(exact_answer) if exact_answer else ''
    
    # Extract snippets (context)
    snippets = question.get('snippets', [])
    snippet_texts = []
    for snippet in snippets:
        if isinstance(snippet, dict):
            snippet_texts.append(snippet.get('text', ''))
        else:
            snippet_texts.append(str(snippet))
    
    return {
        'id': question.get('id', ''),
        'question': body,
        'answer': answer.lower() if answer else '',
        'ideal_answer': ideal_answer,
        'question_type': question.get('type', ''),
        'snippets': snippet_texts,
        'context': '\n\n'.join(snippet_texts)
    }


def process_bioasq_dataset(input_dir: str, split_name: str) -> Dataset:
    """Process BioASQ dataset from directory"""
    
    # Find JSON files
    json_files = []
    for file in os.listdir(input_dir):
        if file.endswith('.json'):
            json_files.append(os.path.join(input_dir, file))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")
    
    print(f"Found {len(json_files)} JSON files")
    
    # Load and convert all questions
    all_questions = []
    for json_file in json_files:
        print(f"Processing: {json_file}")
        questions = load_bioasq_json(json_file)
        print(f"  Loaded {len(questions)} questions")
        
        for q in questions:
            converted = convert_bioasq_question(q)
            all_questions.append(converted)
    
    print(f"\nTotal questions: {len(all_questions)}")
    
    # Filter for yes/no questions (for QA evaluation)
    yesno_questions = [q for q in all_questions if q['question_type'].lower() == 'yesno']
    print(f"Yes/No questions: {len(yesno_questions)}")
    
    if len(yesno_questions) > 0:
        print("\nUsing only yes/no questions for evaluation")
        dataset_questions = yesno_questions
    else:
        print("\nUsing all questions")
        dataset_questions = all_questions
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(dataset_questions)
    
    return dataset


def main():
    args = parse_args()
    
    print("="*70)
    print("Converting BioASQ to HuggingFace Format")
    print("="*70)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split name: {args.split_name}")
    print("")
    
    # Process dataset
    dataset = process_bioasq_dataset(args.input_dir, args.split_name)
    
    print("\nDataset info:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Features: {list(dataset.features.keys())}")
    
    # Show example
    if len(dataset) > 0:
        print("\nExample sample:")
        example = dataset[0]
        print(f"  Question: {example['question'][:100]}...")
        print(f"  Answer: {example['answer']}")
        print(f"  Type: {example['question_type']}")
    
    # Create dataset dict with split
    dataset_dict = DatasetDict({
        args.split_name: dataset
    })
    
    # Save to disk
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)
    
    print(f"\n✓ Dataset saved to: {args.output_dir}")
    
    # Save metadata
    metadata = {
        'dataset_name': 'BioASQ',
        'source': args.input_dir,
        'split': args.split_name,
        'total_samples': len(dataset),
        'features': list(dataset.features.keys())
    }
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Conversion complete!")
    print(f"\nTo use with evaluation:")
    print(f"  python evaluate_bioasq.py \\")
    print(f"      --dataset_name bioasq \\")
    print(f"      --data_path {args.output_dir} \\")
    print(f"      --split {args.split_name}")


if __name__ == "__main__":
    main()
