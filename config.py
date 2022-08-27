import imp
import gym
import datetime
import numpy as np
import ENV.env
SEED = None

CONFIG = {
    'env_name': 'Cartpole-cost', 
    # 'env_name': 'Pointcirclecost-v0',    
    # 'env_name': 'Minitaurcost-v0',  
    # 'env_name': 'Swimmercost-v0',   
    # 'env_name': 'HalfCheetahcost-v0',  
    # 'env_name': 'Antcost-v0', 
    # 'env_name': 'Humanoidcost-v0', 
    # 'env_name': 'Spacereachcost-v0', 
    # 'env_name': 'Spacerandomcost-v0',
    # 'env_name': 'Spacedualarmcost-v0',
    
    
    # training prams
    'algorithm_name': 'ALAC',  # ALAC
    'additional_description': '-trial',  # record the results
    
    # if true training, false evaluation
    'train': True,
    # 'train': False,

    'num_of_trials': 5,   # number of random seeds 5
    'num_of_evaluation_paths': 5,  # number of rollouts for evaluation 
    'num_of_training_paths': 5,  # number of training rollouts stored for analysis 
    'start_of_trial': 0,

    #evaluation params
    'evaluation_form': 'constant_impulse',
    'eval_list': [
        'ALAC-trial',
    ],
    'trials_for_eval': [str(i) for i in range(0, 5)],

    'evaluation_frequency': 2048,
}

CONFIG['log_path']='/'.join(['./log', CONFIG['env_name'], CONFIG['algorithm_name'] + CONFIG['additional_description']])

ENV_PARAMS = {
    'Cartpole-cost': {
        'max_ep_steps': 250,
        'max_global_steps': int(3e5),
        'max_episodes': int(3e5),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64,64],
             },
    },
    'Swimmer-cost': {
        'max_ep_steps': 250,
        'max_global_steps': int(3e5),
        'max_episodes': int(3e5),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'Minitaur-cost': {
        'max_ep_steps': 500,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 1,
        'eval_render': False,
        'network_structure':
            {'critic': [256, 256, 16],
             'actor': [64,64],
             },
    },
    'HalfCheetahcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(5e5),
        'max_episodes': int(5e5),
        'disturbance dim': 6,
        'eval_render': False,
        'network_structure':
            {'critic': [256, 256, 16], 
             'actor': [64, 64],
             },
    },
    'Pointcirclecost-v0': {
        'max_ep_steps': 65,
        'max_global_steps': int(3e5),
        'max_episodes': int(3e5),
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
                'actor': [64, 64],
                },
    },  
    'Antcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 8,
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
             'actor': [64, 64],
             },
    },
    'Humanoidcost-v0': {
        'max_ep_steps': 500,
        'max_global_steps': int(1e6),
        'max_episodes': int(1e6),
        'disturbance dim': 17,
        'eval_render': False,
        'network_structure':
            {'critic': [256, 256, 128],
             'actor': [256, 256],
             },
    },
    'Spacereachcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(3e5),
        'max_episodes': int(3e5),
        'disturbance dim': 17,
        'eval_render': False,
        'network_structure':
            {'critic': [256, 256, 128],
             'actor': [256, 256],
             },
    },      
    'Spacerandomcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(5e5),
        'max_episodes': int(5e5),
        'disturbance dim': 17,
        'eval_render': False,
        'network_structure':
            {'critic': [256, 256, 128],
             'actor': [16, 16],
             },
    },
    'Spacedualarmcost-v0': {
        'max_ep_steps': 200,
        'max_global_steps': int(5e5),
        'max_episodes': int(5e5),
        'disturbance dim': 17,
        'eval_render': False,
        'network_structure':
            {'critic': [512, 512, 256],
             'actor': [512, 512],
             },
    },      
}
ALG_PARAMS = {

    'ALAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'lambda_l': 1.,
        'lambda_e': 2.,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        'steps_per_cycle': 100,
        'train_per_cycle': 80,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'approx_value': True,
        'value_horizon': 2,
        'finite_horizon': False,
        'soft_predict_horizon': False,
        'target_entropy': None,
        'history_horizon': 0, 
    }, 
}


EVAL_PARAMS = {
    'constant_impulse': {
        'magnitude_range': np.arange(0.0, 0.6, .1),
        'num_of_paths': 20,   # number of path for evaluation
        'impulse_instant': 20,
    },
}
CONFIG['env_params']=ENV_PARAMS[CONFIG['env_name']]
CONFIG['eval_params']=EVAL_PARAMS[CONFIG['evaluation_form']]
CONFIG['alg_params']=ALG_PARAMS[CONFIG['algorithm_name']]

RENDER = False

def get_env_from_name(name):
    if name == 'Cartpole-cost':
        from ENV.CartPole_env import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'Minitaur-cost':
        from ENV.minitaur_env import minitaur_env as env
        env = env(render=CONFIG['env_params']['eval_render'])
        env = env.unwrapped
    elif name == 'Swimmer-cost':
        from ENV.swimmer import swimmer_env as env
        env = env()
        env = env.unwrapped
    else:
        env = gym.make(name)
        env = env.unwrapped
    env.seed(SEED)
    return env

def get_train(name):
    if 'ALAC' in name:
        from algorithm.ALAC.ALAC import train 
    return train

def get_policy(name):
    if 'ALAC' in name:
        from algorithm.ALAC.ALAC import ALAC as build_func
    return build_func



