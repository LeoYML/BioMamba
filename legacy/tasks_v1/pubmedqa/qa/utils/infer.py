import torch
import torch.nn.functional as F


def generate_text(model, tokenizer, device, input_str, max_length=16):
    
    input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"].to(device)
    output_ids = model.generate(input_ids, max_length=max_length, temperature=0.01)
    generated_tokens = output_ids[0, input_ids.shape[1]:]
    output_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return output_str


def generate_token_probs(model, tokenizer, device, input_str, max_length=3, top_k=5):
    
    model.to(device)
    input_ids = tokenizer(input_str, return_tensors="pt")["input_ids"].to(device)
    stepwise_token_probs = {}
    
    for step in range(1, max_length + 1):
        
        # Get the model outputs 
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_token_ids = torch.topk(probs, top_k, dim=-1)
        
        # Decode the tokens and pair them with their probabilities
        token_prob_pairs = []
        for i in range(top_k):
            token_str = tokenizer.decode(top_k_token_ids[0, i].cpu())
            token_prob = top_k_probs[0, i].item()
            token_prob_pairs.append((token_str, token_prob))
        
        stepwise_token_probs[step] = token_prob_pairs
        
        # Get the next token ID (greedy selection)
        next_token_id = top_k_token_ids[:, 0].unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    
    return stepwise_token_probs