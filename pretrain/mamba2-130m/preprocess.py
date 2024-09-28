from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer, AutoConfig, AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from datasets import load_dataset, load_from_disk, DatasetDict
from mamba_ssm import MambaLMHeadModel
from torch.cuda.amp import autocast, GradScaler
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
import random


def data_preprocess_pubmed():
    
    model_name = "state-spaces/mamba-2.8b-hf"
    tokenizer_name = "state-spaces/mamba-2.8b-hf"
    #model = MambaForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    config = AutoConfig.from_pretrained(model_name)
    print(config)


    # Generate Pubmed Abstracts
    from datasets import load_dataset, load_from_disk, DatasetDict

    dataset_loaded = load_dataset('ncbi/pubmed', split='train')

    # 36555430
    print(f"Length of PubMed: {len(dataset_loaded)}")

    def extract_abstract(example):
        return {'abstract': example['MedlineCitation']['Article']['Abstract']['AbstractText']}

    processed_dataset = dataset_loaded.map(extract_abstract, num_proc=16)
    processed_dataset = processed_dataset.filter(lambda x: x['abstract'] is not None, num_proc=16)
    processed_dataset = processed_dataset.filter(lambda x: len(x['abstract']) > 0, num_proc=16)
    processed_dataset = processed_dataset.remove_columns("MedlineCitation")
    processed_dataset = processed_dataset.remove_columns("PubmedData")

    # 25277041
    print(f"Length of PubMed with abstracts: {len(processed_dataset)}")

    # Split the dataset into train and test sets
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=0)

    # Create a DatasetDict with the train and test sets
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'test': split_dataset['test']
    })

    # Print the lengths of the train and test sets
    print(f"Train dataset length: {len(dataset_dict['train'])}")
    print(f"Test dataset length: {len(dataset_dict['test'])}")

    dataset_dict.save_to_disk('pubmed_abstract')
    
    
    # Load Dataset
    dataset = load_from_disk('pubmed_abstract')

    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['abstract'], padding='max_length', truncation=True, max_length=512)

    train_tokenized = train_dataset.map(tokenize_function, batched=True, num_proc=32)
    train_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    test_tokenized = test_dataset.map(tokenize_function, batched=True, num_proc=32)
    test_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    tokenized_datasets = DatasetDict({
        'train': train_tokenized,
        'test': test_tokenized
    })

    tokenized_datasets.save_to_disk("tokenized_pubmed_abstract_512")
    
    

def data_preprocess_pile():
    
    model_name = "state-spaces/mamba-2.8b-hf"
    tokenizer_name = "state-spaces/mamba-2.8b-hf"
    #model = MambaForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    config = AutoConfig.from_pretrained(model_name)
    print(config)
    
    test_dataset = load_dataset("ola13/small-the_pile", split="train")
    test_dataset = test_dataset.filter(lambda example: example['meta']['pile_set_name'] == 'Wikipedia (en)')
    test_dataset = test_dataset.remove_columns('meta')
    
    print(f"Length of Pile dataset: {len(test_dataset)}")
    print(test_dataset[0])
    
    num_samples = 5000
    random.seed(0)

    # Randomly sample the indices
    sampled_indices = random.sample(range(len(test_dataset)), num_samples)
    sampled_dataset = test_dataset.select(sampled_indices)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

    test_tokenized = sampled_dataset.map(tokenize_function, batched=True, num_proc=32)
    test_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    test_tokenized = test_tokenized.remove_columns('text')
    
    # save the tokenized dataset
    test_tokenized.save_to_disk("tokenized_pile_512")
    

    
    
    
    
    
    
if __name__ == "__main__":

    data_preprocess_pubmed()
    data_preprocess_pile()