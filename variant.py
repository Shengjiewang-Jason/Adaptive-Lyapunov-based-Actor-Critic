import imp
import gym
import datetime
import numpy as np
import ENV.env
SEED = None

VARIANT = {
    'env_name': 'Cartpole-cost', 
    # 'env_name': 'Pointcircle-cost',    
    # 'env_name': 'Minitaur-cost',  
    # 'env_name': 'Swimmer-cost',   
    # 'env_name': 'HalfCheetah-cost',  
    # 'env_name': 'Ant-cost', 
    # 'env_name': 'Humanoid-cost', 
    # 'env_name': 'Spacereach-cost', 
    # 'env_name': 'Spacerandom-cost',
    # 'env_name': 'Spacedualarm-cost',
    
    
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

VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])

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
    'HalfCheetah-cost': {
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
    'Pointcircle-cost': {
        'max_ep_steps': 65,
        'max_global_steps': int(3e5),
        'max_episodes': int(3e5),
        'eval_render': False,
        'network_structure':
            {'critic': [64, 64, 16],
                'actor': [64, 64],
                },
    },  
    'Ant-cost': {
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
    'Humanoid-cost': {
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
    'Spacereach-cost': {
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
    'Spacerandom-cost': {
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
    'Spacedualarm-cost': {
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
        'labda': 1.,
        'alpha': 2.,
        'alpha3': .1,
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
VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]
VARIANT['eval_params']=EVAL_PARAMS[VARIANT['evaluation_form']]
VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]

RENDER = False

def get_env_from_name(name):
    if name == 'Cartpole-cost':
        from ENV.CartPole_env import CartPoleEnv_adv as dreamer
        env = dreamer()
        env = env.unwrapped
    elif name == 'Minitaur-cost':
        from ENV.minitaur_env import minitaur_env as env
        env = env(render=VARIANT['env_params']['eval_render'])
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
        from ALAC.ALAC import train 
    return train

def get_policy(name):
    if 'ALAC' in name:
        from ALAC.ALAC import ALAC as build_func
    return build_func



