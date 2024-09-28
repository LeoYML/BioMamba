from utils import model_training, eval
import torch

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(7)

params = {'lr': 2.9309052235844563e-05, 'num_training_steps': 30, 'weight_decay': 0.1}
model_training(params, device)
#eval(device)