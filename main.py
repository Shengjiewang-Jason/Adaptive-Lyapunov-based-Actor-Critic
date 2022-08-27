import imp
import os
from config import CONFIG, get_env_from_name,  get_train
from eval import eval
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    root_dir = CONFIG['log_path']
    CONFIG['device'] = device
    if CONFIG['train']:
        for i in range(CONFIG['start_of_trial'], CONFIG['start_of_trial']+CONFIG['num_of_trials']):
            CONFIG['log_path'] = root_dir +'/'+ str(i)
            print('logging to ' + CONFIG['log_path'])
            train = get_train(CONFIG['algorithm_name'])
            train(CONFIG)
    else:
        # eval = get_eval(CONFIG['algorithm_name'])
        eval(CONFIG)

