from utils.utils import model_training
from transformers import AutoTokenizer
from mamba_ssm import MambaLMHeadModel
import torch


models_names = [
    "mamba2-130m",
    "mamba2-2.7b"
]

models = {
    "mamba2-130m": MambaLMHeadModel.from_pretrained ("state-spaces/mamba2-130m"),
    "mamba2-2.7b": MambaLMHeadModel.from_pretrained ("state-spaces/mamba2-2.7b")
}

tokenizers = {
    "mamba2-130m": AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf"),
    "mamba2-2.7b": AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
}

config = {
    "mamba2-130m": {'lr': 2.9309052235844563e-05, 'num_training_steps': 30, 'weight_decay': 0.1},
    "mamba2-2.7b": {'lr': 2.8433115362145255e-05, 'num_training_steps': 30, 'weight_decay': 0.1}
}

if __name__ == "__main__":

    for model_name in models_names:
        model = models[model_name]
        tokenizer = tokenizers[model_name]
        params = config[model_name]
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(0)
        model_training(model_name, model, tokenizer, params, device)
    
    # model = MambaLMHeadModel.from_pretrained ("state-spaces/mamba2-130m")
    # tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
    # params = {'lr': 2.9309052235844563e-05, 'num_training_steps': 30, 'weight_decay': 0.1}
    # device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_device(7)
    # model_training("mamba2-130m", model, tokenizer, params, device)