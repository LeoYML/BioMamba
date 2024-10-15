from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, rand
import numpy as np
import argparse
import torch



# Load pre-trained Mamba model and tokenizer
from utils import model_training, eval


def hyperparameter_tune(search_space = {"lr": (1e-4, 1e-6), "num_training_steps": 100, "weight_decay": 0.1}, device = torch.device("cpu")):
    
    lr_max, lr_min = search_space["lr"]
    num_training_steps = search_space["num_training_steps"]
    weight_decay = search_space["weight_decay"]
    
    if search_space:
        space = {
            "lr": hp.loguniform("lr", np.log(lr_min), np.log(lr_max)),
            "num_training_steps": hp.choice("num_training_steps", [50, 100, 200, 500]),
            "weight_decay": weight_decay
        }
    else:
        # Default hyperparameter space
        # space = {
        #     "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-6)),
        #     "num_training_steps": hp.choice("num_training_steps", [2000, 1000, 500]),
        #     "weight_decay": 0.1
        # }
        
        space = {
            "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-5)),
            "num_training_steps": 100,
            "weight_decay": 0.1
        }

    def objective(params):
        model, tokenizer = model_training(params, device)
        perplexity = eval(model, tokenizer, device)
        return perplexity

    # Perform hyperparameter optimization
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        max_queue_len=16
    )

    print("Best hyperparameters:", space_eval(space, best))



if __name__ == "__main__":
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Script")
    parser.add_argument("--lr_max", type=float, default=1e-4, help="Learning rate upper bound")
    parser.add_argument("--lr_min", type=float, default=1e-6, help="Learning rate lower bound")
    parser.add_argument("--num_training_steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay value")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device to use")  # Add CUDA device argument
    args = parser.parse_args()

    # Create the search space from the arguments
    search_space = {
        "lr": (args.lr_max, args.lr_min),
        "num_training_steps": args.num_training_steps,
        "weight_decay": args.weight_decay
    }

    print(f"Using CUDA device: {args.cuda_device}")
    
    # 
    
    
    # Set the CUDA device
    num_devices = torch.cuda.device_count()
    print(f"Number of available CUDA devices: {num_devices}")
    device = torch.device("cuda")
    torch.cuda.set_device(args.cuda_device)

    # Run hyperparameter tuning
    hyperparameter_tune(search_space, device)
    

    # device = torch.device("cuda")
    # torch.cuda.set_device(2)

    # hyperparameter_tune(
    #     {"lr": (1e-4, 1e-5), "num_training_steps": 10, "weight_decay": 0.1},
    #     device
    # )