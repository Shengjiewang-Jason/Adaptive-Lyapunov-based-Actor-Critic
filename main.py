import imp
import os
from config import *
from eval import eval
from train import train
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # count the device according to nvidia-smi command
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print(torch.__version__);
print(torch.cuda.is_available())

if __name__ == '__main__':

    # log path
    root_dir = CONFIG['log_path']
    # device
    CONFIG['device'] = device

    if CONFIG['train']:
        for i in range(CONFIG['start_of_trial'], CONFIG['start_of_trial']+CONFIG['num_of_trials']):
            CONFIG['log_path'] = root_dir +'/'+ str(i)
            print('logging to ' + CONFIG['log_path'])
            train(CONFIG)
    else:
        # eval = get_eval(CONFIG['algorithm_name'])
        eval(CONFIG)

