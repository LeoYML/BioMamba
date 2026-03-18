from utils.preprocess import data_preprocess_pubmed, data_preprocess_pile
from transformers import AutoTokenizer, AutoModelForCausalLM, BioGptTokenizer, BioGptForCausalLM


models_names = [
    "BioGPT",
    "BioGPT-Large",
    "mamba2-130m"
    #"mamba2-2.7b",  # both mamba models share the same tokenizer
]


tokenizers = {
    "BioGPT": BioGptTokenizer.from_pretrained("microsoft/biogpt"),
    "BioGPT-Large": AutoTokenizer.from_pretrained("microsoft/BioGPT-Large"),
    "mamba2-130m": AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf"),
    "mamba2-2.7b": AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
}


# models = {
#     "BioGPT": AutoModelForCausalLM.from_pretrained("microsoft/biogpt"),
#     "BioGPT-Large": AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large"),
#     "mamba2-130m": MambaLMHeadModel.from_pretrained ("state-spaces/mamba2-130m"),
#     "mamba2-2.7b": MambaLMHeadModel.from_pretrained ("state-spaces/mamba2-2.7b")
# }

if __name__ == "__main__":
    
    for model_name in models_names:
        tokenizer = tokenizers[model_name]
        data_preprocess_pubmed(model_name, tokenizer)
        
    for model_name in models_names:
        tokenizer = tokenizers[model_name]
        data_preprocess_pile(model_name, tokenizer)
        
    
    # # # example usage

    # model = "BioGPT"
    # tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    # data_preprocess_pubmed(model, tokenizer)
    # data_preprocess_pile(model, tokenizer)
    

    
