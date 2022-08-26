import imp
import tensorflow as tf
import os
from config import CONFIG, get_env_from_name,  get_train
from eval import eval
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

tf.device('/gpu:1')


if __name__ == '__main__':
    root_dir = CONFIG['log_path']
    if CONFIG['train']:
        for i in range(CONFIG['start_of_trial'], CONFIG['start_of_trial']+CONFIG['num_of_trials']):
            CONFIG['log_path'] = root_dir +'/'+ str(i)
            print('logging to ' + CONFIG['log_path'])
            train = get_train(CONFIG['algorithm_name'])
            train(CONFIG)

            tf.reset_default_graph()
    else:
        # eval = get_eval(CONFIG['algorithm_name'])
        eval(CONFIG)

