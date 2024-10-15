from utils.infer import generate_token_probs, generate_text
from utils.load_data import load_train_texts, load_test_texts, load_test_answer_texts
from utils.finetune import model_finetune
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
from transformers import AutoModel
import os
import torch
import tqdm
import json

if not os.path.exists("results"):
    os.makedirs("results")
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")
    
output_dir = "results"
    
model = MambaLMHeadModel.from_pretrained ("state-spaces/mamba2-130m")
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(7)

checkpoint_path = "checkpoints/finetuned_mamba2-130m"

model_finetune(model = model, tokenizer = tokenizer, checkpoint_path=checkpoint_path, device = device)


model = MambaLMHeadModel.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)



test_texts = load_test_texts()
test_result_token_probs = {}

# loop in tqdm
for id, text in tqdm.tqdm(test_texts.items()):

    test_result_token_probs[id] = generate_token_probs(model, tokenizer, device, text, max_length=1, top_k=5)

# save the results
json.dump(test_result_token_probs, open(f"{output_dir}/test_result_token_probs.json", "w"), indent=4)


