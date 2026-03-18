"""
Download and prepare evaluation datasets locally
"""

# --- Project root resolution (auto-generated) ---
import os as _os, sys as _sys
_PROJECT_ROOT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..')
_sys.path.insert(0, _PROJECT_ROOT)
_os.chdir(_PROJECT_ROOT)
# --- End project root resolution ---

import os
import argparse
from datasets import load_dataset
import json


def parse_args():
    parser = argparse.ArgumentParser(description='Download evaluation datasets')
    parser.add_argument('--output_dir', type=str, default='./data/evaluation_datasets',
                        help='Directory to save datasets')
    parser.add_argument('--dataset', type=str, choices=['pubmedqa', 'all'], default='all',
                        help='Which dataset to download')
    return parser.parse_args()


def download_pubmedqa(output_dir):
    """Download PubMedQA dataset"""
    print("="*70)
    print("Downloading PubMedQA Dataset")
    print("="*70)
    
    output_path = os.path.join(output_dir, 'pubmedqa_pqa_labeled')
    
    if os.path.exists(output_path):
        print(f"Dataset already exists at: {output_path}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Skipping...")
            return
    
    print("\nDownloading from HuggingFace...")
    try:
        # Load the pqa_labeled subset (1000 labeled examples)
        dataset = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')
        
        print(f"✓ Downloaded {len(dataset)} samples")
        print(f"Features: {list(dataset.features.keys())}")
        
        # Save to disk
        os.makedirs(output_path, exist_ok=True)
        dataset.save_to_disk(output_path)
        
        print(f"✓ Saved to: {output_path}")
        
        # Show example
        print("\nExample sample:")
        example = dataset[0]
        print(f"  Question: {example['question'][:100]}...")
        print(f"  Answer: {example['final_decision']}")
        
        # Save metadata
        metadata = {
            'dataset_name': 'PubMedQA',
            'subset': 'pqa_labeled',
            'source': 'qiaojin/PubMedQA',
            'total_samples': len(dataset),
            'features': list(dataset.features.keys())
        }
        
        with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("\n✓ PubMedQA download complete!")
        
    except Exception as e:
        print(f"✗ Error downloading PubMedQA: {e}")
        import traceback
        traceback.print_exc()


def print_bioasq_instructions():
    """Print instructions for BioASQ dataset"""
    print("\n" + "="*70)
    print("BioASQ Dataset Instructions")
    print("="*70)
    print("""
BioASQ requires registration and manual download from the official website.

Steps to download BioASQ:

1. Register at: http://participants-area.bioasq.org/

2. Download the dataset files for Task B (Biomedical Semantic QA)

3. Extract the data files

4. Convert to HuggingFace format using the provided script:
   python convert_bioasq_to_hf.py --input_dir /path/to/bioasq --output_dir ./data/evaluation_datasets/bioasq

5. Use with evaluation script:
   python evaluate_bioasq.py \\
       --dataset_name bioasq \\
       --data_path ./data/evaluation_datasets/bioasq

Alternative: Use PubMedQA for evaluation
PubMedQA is similar to BioASQ and is publicly available (already downloaded).
""")
    print("="*70)


def main():
    args = parse_args()
    
    print("\n" + "="*70)
    print("Dataset Downloader for BioASQ Evaluation")
    print("="*70)
    print(f"Output directory: {args.output_dir}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download datasets
    if args.dataset in ['pubmedqa', 'all']:
        download_pubmedqa(args.output_dir)
        print("\n")
    
    # BioASQ instructions
    if args.dataset in ['all']:
        print_bioasq_instructions()
    
    print("\n" + "="*70)
    print("Download Complete!")
    print("="*70)
    print(f"\nDatasets saved to: {args.output_dir}")
    print("\nTo use with evaluation:")
    print(f"  python evaluate_bioasq.py \\")
    print(f"      --dataset_name pubmedqa \\")
    print(f"      --data_path {os.path.join(args.output_dir, 'pubmedqa_pqa_labeled')}")
    print("")


if __name__ == "__main__":
    main()
