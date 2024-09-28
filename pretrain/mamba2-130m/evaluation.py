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



def eval_general(device = torch.device("cpu"), model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")):
    
    
    model.to(device)
    test_tokenized = load_from_disk("tokenized_pile_512")

    batch_size = 32
    test_dataloader = DataLoader(test_tokenized, shuffle=False, batch_size=batch_size)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)
            
            outputs = model(inputs)
            
            # Get logits
            logits = outputs.logits  # Assuming model outputs a named tuple with logits
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute the loss using CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Accumulate the loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    print(f"Perplexity: {perplexity:.4f}")
    print(f"avg_loss: {avg_loss:.4f}")
    
    

    
    
def eval_pubmed(device = torch.device("cpu"), model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")):
    
    
    model.to(device)
    
    test_dataset = load_from_disk('pubmed_abstract')['test']
    test_dataset = test_dataset.rename_column('abstract', 'text')
    
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
    
    batch_size = 32
    test_dataloader = DataLoader(test_tokenized, shuffle=False, batch_size=batch_size)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)  # Same as inputs, since we predict next token
            
            # Forward pass (no labels in the forward call)
            outputs = model(inputs)
            
            # Get logits
            logits = outputs.logits  # Assuming model outputs a named tuple with logits
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # Compute the loss using CrossEntropyLoss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Accumulate the loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    print(f"Perplexity: {perplexity:.4f}")
    print(f"avg_loss: {avg_loss:.4f}")
    
    return perplexity




if __name__ == "__main__":
    
    device = torch.device("cuda")
    torch.cuda.set_device(4)
    
    model = MambaLMHeadModel.from_pretrained("../checkpoints/biomamba2-130m")
    tokenizer = AutoTokenizer.from_pretrained("../checkpoints/biomamba2-130m")
    
    print("Evaluating biomamba2-130m on Pubmed dataset")
    eval_pubmed(device, model)
    
    print("Evaluating biomamba2-130m on Pile dataset")
    eval_general(device, model)
    
    model = MambaLMHeadModel.from_pretrained(pretrained_model_name="state-spaces/mamba2-130m")
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    
    print("Evaluating mamba2-130m on Pubmed dataset")
    eval_pubmed(device, model)
    
    print("Evaluating mamba2-130m on Pile dataset")
    eval_general(device, model)
    

    
    
    
    