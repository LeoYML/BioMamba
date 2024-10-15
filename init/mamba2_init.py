# from transformers import Mamba2Config, Mamba2ForCausalLM, AutoTokenizer
# import torch
# model_id = 'mistralai/Mamba-Codestral-7B-v0.1'
# tokenizer = AutoTokenizer.from_pretrained(model_id, revision='refs/pr/9', from_slow=True, legacy=False)
# model = Mamba2ForCausalLM.from_pretrained(model_id, revision='refs/pr/9')
# input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

# out = model.generate(input_ids, max_new_tokens=10)
# print(tokenizer.batch_decode(out))




from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
import os
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)

model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m")
tokenizer = AutoTokenizer.from_pretrained('state-spaces/mamba2-130m-hf')
model.to(device)
input_ids = tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"].to(device)
out = model.generate(input_ids, max_length=16)
print(tokenizer.batch_decode(out))
