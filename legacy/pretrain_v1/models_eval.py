from utils.evaluation import eval_general, eval_pubmed
from transformers import AutoModelForCausalLM, BioGptForCausalLM
from mamba_ssm import MambaLMHeadModel
import torch



models_names = [
    "BioGPT",
    "BioGPT-Large",
    "mamba2-130m",
    "mamba2-2.7b",
    "biomamba2-130m",
    "biomamba2-2.7b" 
]

models = {
    "BioGPT": BioGptForCausalLM.from_pretrained("microsoft/biogpt"),
    "BioGPT-Large": AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large"),
    "mamba2-130m": MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m"),
    "mamba2-2.7b": MambaLMHeadModel.from_pretrained("state-spaces/mamba2-2.7b"),
    "biomamba2-130m": MambaLMHeadModel.from_pretrained("checkpoints/biomamba2-130m"),
    "biomamba2-2.7b": MambaLMHeadModel.from_pretrained("checkpoints/biomamba2-2.7b")
}







if __name__ == "__main__":
    
   
    
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    
    for models_name in models_names:
        model = models[models_name]
        eval_general(models_name, model, device)
        eval_pubmed(models_name, model, device)
    
    
    # models_name = "BioGPT"
    # model = models[models_name]
    # eval_general(models_name, model, device)
    # eval_pubmed(models_name, model, device)
        
    

    
    
    
    