#!/bin/bash

# Define models and corresponding pretrained paths
declare -A models
models["BioGPT"]="microsoft/biogpt"
models["BioGPT-Large"]="microsoft/BioGPT-Large"
models["mamba2-130m"]="state-spaces/mamba2-130m"
models["mamba2-2.7b"]="state-spaces/mamba2-2.7b"
models["biomamba2-130m"]="checkpoints/biomamba2-130m"
models["biomamba2-2.7b"]="checkpoints/biomamba2-2.7b"

# Define the GPUs to be used (in this case, 2 to 7)
gpus=(2 3 4 5 6 7)

# Counter to track the GPU allocation
gpu_counter=0

# Loop through each model and run evaluation concurrently on different GPUs
for model_name in "${!models[@]}"; do
  # Get the model path
  model_path=${models[$model_name]}

  # Get the next GPU in the list (wrap around if necessary)
  gpu_id=${gpus[$gpu_counter]}

  # Log file for the current model
  log_file="${model_name}_eval.log"

  # Run the evaluation in the background using nohup and redirect output to the log file
  echo "Running evaluation for $model_name on GPU $gpu_id... Logs: $log_file"
  nohup python3 - <<END > "$log_file" 2>&1 &
import torch
from transformers import AutoModelForCausalLM, BioGptForCausalLM
from mamba_ssm import MambaLMHeadModel
from utils.evaluation import eval_general, eval_pubmed

# Load the model
if "$model_name" == "BioGPT":
    model = BioGptForCausalLM.from_pretrained("$model_path")
elif "$model_name" == "BioGPT-Large":
    model = AutoModelForCausalLM.from_pretrained("$model_path")
else:
    model = MambaLMHeadModel.from_pretrained("$model_path")

# Set the device to the specified GPU
device = torch.device(f"cuda:$gpu_id" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device($gpu_id)

# Run evaluations
eval_general("$model_name", model, device)
eval_pubmed("$model_name", model, device)
END

  # Increment the GPU counter, wrap around if it exceeds the length of the GPU list
  gpu_counter=$(( (gpu_counter + 1) % ${#gpus[@]} ))
done

echo "All models are running in the background. Check log files for progress."
