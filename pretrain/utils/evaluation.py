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



def eval_general(model_name = "mamba2-130m",
                 model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf"),
                 device = torch.device("cpu"), ):
    
    print(f"Evaluating {model_name} on Pile dataset")
    
    
    model.to(device)
    
    if model_name == "biomamba2-130m" or model_name == "biomamba2-2.7b" or model_name == "mamba2-130m" or model_name == "mamba2-2.7b":
        test_tokenized = load_from_disk("data/mamba2_tokenized_pile_512")
    elif model_name == "BioGPT":
        test_tokenized = load_from_disk("data/BioGPT_tokenized_pile_512")
    elif model_name == "BioGPT-Large":
        test_tokenized = load_from_disk("data/BioGPT-Large_tokenized_pile_512")
    else:
        print("Invalid model name {model_name}")
        return
    
    
    batch_size = 32
    test_dataloader = DataLoader(test_tokenized, shuffle=False, batch_size=batch_size)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)
            
            if model_name == "biomamba2-130m" or model_name == "biomamba2-2.7b" or model_name == "mamba2-130m" or model_name == "mamba2-2.7b":
                
                outputs = model(inputs)
                
                # Get logits
                logits = outputs.logits  # Assuming model outputs a named tuple with logits
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                # Compute the loss using CrossEntropyLoss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            elif model_name == "BioGPT" or model_name == "BioGPT-Large":
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss
            else:
                print("Invalid model name {model_name}")
                return None, None
            
            # Accumulate the loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    print(f"Perplexity: {perplexity:.4f}")
    print(f"avg_loss: {avg_loss:.4f}")
    
    

    
    
def eval_pubmed(model_name = "mamba2-130m",
                model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf"),
                device = torch.device("cpu")):
    
    print(f"Evaluating {model_name} on Pubmed dataset")
    
    model.to(device)
    
    if model_name == "biomamba2-130m" or model_name == "biomamba2-2.7b" or model_name == "mamba2-130m" or model_name == "mamba2-2.7b":
        test_tokenized = load_from_disk("data/mamba2_tokenized_pubmed_abstract_512")
    elif model_name == "BioGPT":
        test_tokenized = load_from_disk("data/BioGPT_tokenized_pubmed_abstract_512")
    elif model_name == "BioGPT-Large":
        test_tokenized = load_from_disk("data/BioGPT-Large_tokenized_pubmed_abstract_512")
    else:
        print("Invalid model name {model_name}")
        return
    
    test_tokenized = test_tokenized['test']
    
    num_samples = 5000 ### 5000
    random.seed(0)
    # get a subset of the dataset
    test_tokenized = test_tokenized.select(range(num_samples))

    
    #test_tokenized = test_tokenized.select(num_samples)
    
    batch_size = 32
    test_dataloader = DataLoader(test_tokenized, shuffle=False, batch_size=batch_size)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            inputs = batch['input_ids'].to(device)
            labels = batch['input_ids'].to(device)  # Same as inputs, since we predict next token
            
            if model_name == "biomamba2-130m" or model_name == "biomamba2-2.7b" or model_name == "mamba2-130m" or model_name == "mamba2-2.7b":
                
                outputs = model(inputs)
                
                # Get logits
                logits = outputs.logits  # Assuming model outputs a named tuple with logits
                
                # Shift logits and labels for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                
                # Compute the loss using CrossEntropyLoss
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
            elif model_name == "BioGPT" or model_name == "BioGPT-Large":
                outputs = model(input_ids=inputs, labels=labels)
                loss = outputs.loss
            else:
                print("Invalid model name {model_name}")
                return None, None
            
            # Accumulate the loss
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)

    print(f"Perplexity: {perplexity:.4f}")
    print(f"avg_loss: {avg_loss:.4f}")
    
    return perplexity




# if __name__ == "__main__":
    
#     device = torch.device("cuda")
#     torch.cuda.set_device(4)
    
#     model = MambaLMHeadModel.from_pretrained("../checkpoints/biomamba2-130m")
#     tokenizer = AutoTokenizer.from_pretrained("../checkpoints/biomamba2-130m")
    
#     print("Evaluating biomamba2-130m on Pubmed dataset")
#     eval_pubmed(device, model)
    
#     print("Evaluating biomamba2-130m on Pile dataset")
#     eval_general(device, model)
    
#     model = MambaLMHeadModel.from_pretrained(pretrained_model_name="state-spaces/mamba2-130m")
#     tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    
#     print("Evaluating mamba2-130m on Pubmed dataset")
#     eval_pubmed(device, model)
    
#     print("Evaluating mamba2-130m on Pile dataset")
#     eval_general(device, model)
    

    
    
    
    