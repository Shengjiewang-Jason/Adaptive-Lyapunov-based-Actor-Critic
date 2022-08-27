import os
from config import *
import numpy as np
import time
import logger
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def get_distrubance_function(env_name):
    if 'Cartpole' in env_name:
        disturbance_step = cartpole_disturber
    elif 'Pointcircle' in env_name:
        disturbance_step = point_disturber
    elif 'HalfCheetah' in env_name:
        disturbance_step = halfcheetah_disturber
    elif 'Space' in env_name:
        disturbance_step = space_disturber
    elif 'Ant' in env_name:
        disturbance_step = ant_disturber
    elif 'Humanoid' in env_name:
        disturbance_step = humanoid_disturber
    elif 'Minitaur' in env_name:
        disturbance_step = minitaur_disturber
    elif 'Swimmer' in env_name:
        disturbance_step = swimmer_disturber
    else:
        print('no disturber designed for ' + env_name)
        raise NameError
        # disturbance_step = None

    return disturbance_step


def cartpole_disturber(time, s, action, env, eval_params, form_of_eval, disturber=None):
    if form_of_eval=='constant_impulse':
        if time % eval_params['impulse_instant']==0:
            d = eval_params['magnitude'] * np.sign(s[0])
        else:
            d = 0
    else:
        d = np.zeros_like(action)
    s_, r, done, info = env.step(action, impulse=d)
    return s_, r, done, info


def halfcheetah_disturber(time, s, action, env, eval_params, form_of_eval, disturber=None):
    if form_of_eval == 'constant_impulse':
        if time % eval_params['impulse_instant'] == 0:
            d = eval_params['magnitude'] * (-np.sign(action))
        else:
            d = np.zeros_like(action)
    else:
        d = np.zeros_like(action)    
    s_, r, done, info = env.step(action+d)
    return s_, r, done, info

def minitaur_disturber(time, s, action, env, eval_params, form_of_eval, disturber=None):
    if form_of_eval == 'constant_impulse':
        if time % eval_params['impulse_instant'] == 0:
            d = eval_params['magnitude'] * (-np.sign(action))
        else:
            d = np.zeros_like(action)
    else:
        d = np.zeros_like(action)
    s_, r, done, info = env.step(action+d)
    return s_, r, done, info

def ant_disturber(time, s, action, env, eval_params, form_of_eval, disturber=None):
    if form_of_eval == 'constant_impulse':
        if time % eval_params['impulse_instant'] == 0:
            d = eval_params['magnitude'] * (-np.sign(action))
        else:
            d = np.zeros_like(action)
    else:
        d = np.zeros_like(action)
    s_, r, done, info = env.step(action+d)
    return s_, r, done, info

def swimmer_disturber(time, s, action, env, eval_params, form_of_eval, disturber=None):
    if form_of_eval == 'constant_impulse':
        if time % eval_params['impulse_instant'] == 0:
            d = eval_params['magnitude'] * (-np.sign(action))
        else:
            d = np.zeros_like(action)
    else:
        d = np.zeros_like(action)
    s_, r, done, info = env.step(action+d)
    return s_, r, done, info


def space_disturber(time, s, action, env, eval_params, form_of_eval, disturber=None):
    if form_of_eval == 'constant_impulse':
        if time % eval_params['impulse_instant'] == 0:
            d = eval_params['magnitude'] * (-np.sign(action))
        else:
            d = np.zeros_like(action)
    else:
        d = np.zeros_like(action)
    s_, r, done, info = env.step(action+d)
    done = False
    return s_, r, done, info

def point_disturber(time, s, action, env, eval_params, form_of_eval, disturber=None):
    if form_of_eval == 'constant_impulse':
        if time % eval_params['impulse_instant'] == 0:
            d = eval_params['magnitude'] * (-np.sign(action))
        else:
            d = np.zeros_like(action)
    else:
        d = np.zeros_like(action)
    s_, r, done, info = env.step(action+d)
    done = False
    return s_, r, done, info

def humanoid_disturber(time, s, action, env, eval_params, form_of_eval, disturber=None):
    if form_of_eval == 'constant_impulse':
        if time % eval_params['impulse_instant'] == 0:
            d = eval_params['magnitude'] * (-np.sign(action))
        else:
            d = np.zeros_like(action)
    else:
        d = np.zeros_like(action)
    s_, r, done, info = env.step(action+d)
    return s_, r, done, info



def constant_impulse(CONFIG):
    env_name = CONFIG['env_name']
    env = get_env_from_name(env_name)
    env_params = CONFIG['env_params']

    eval_params = CONFIG['eval_params']
    policy_params = CONFIG['alg_params']
    policy_params['network_structure'] = env_params['network_structure']

    build_func = get_policy(CONFIG['algorithm_name'])

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    policy = build_func(a_dim, s_dim, policy_params)
    # disturber = Disturber(d_dim, s_dim, disturber_params)

    log_path = CONFIG['log_path'] + '/eval/constant_impulse'
    CONFIG['eval_params'].update({'magnitude': 0})
    logger.configure(dir=log_path, format_strs=['csv'])
    for magnitude in eval_params['magnitude_range']:
        CONFIG['eval_params']['magnitude'] = magnitude
        
        npy_path = log_path + '/magnitude_{}'.format(magnitude)
        diagnostic_dict, _ = evaluation(CONFIG, env, policy,npy_path)

        string_to_print = ['magnitude', ':', str(magnitude), '|']
        [string_to_print.extend([key, ':', str(round(diagnostic_dict[key], 2)), '|'])
         for key in diagnostic_dict.keys()]
        print(''.join(string_to_print))

        logger.logkv('magnitude', magnitude)
        [logger.logkv(key, diagnostic_dict[key]) for key in diagnostic_dict.keys()]
        logger.dumpkvs()


def evaluation(CONFIG, env, policy, npy_path,disturber= None):
    env_name = CONFIG['env_name']

    env_params = CONFIG['env_params']
    disturbance_step = get_distrubance_function(env_name)
    max_ep_steps = env_params['max_ep_steps']

    eval_params = CONFIG['eval_params']
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    # For analyse
    Render = env_params['eval_render']


    # Training setting

    total_cost = []
    death_rates = []
    form_of_eval = CONFIG['evaluation_form']
    trial_list = os.listdir(CONFIG['log_path'])
    episode_length = []
    cost_paths = []
    value_paths = []
    state_paths = []
    ref_paths = []
    for trial in trial_list:
        if trial == 'eval':
            continue
        if trial not in CONFIG['trials_for_eval']:
            continue
        success_load = policy.restore(os.path.join(CONFIG['log_path'], trial)+'/policy')
        if not success_load:
            continue
        die_count = 0
        seed_average_cost = []
        for i in range(int(np.ceil(eval_params['num_of_paths']/(len(trial_list)-1)))):
            path = []
            state_path = []
            value_path = []
            ref_path = []
            cost = 0
            s = env.reset()
            global initial_pos
            initial_pos = np.random.uniform(0., np.pi, size=[a_dim])
            for j in range(max_ep_steps):


                if Render:
                    env.render()
                a = policy.choose_action(s, True)

                action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2


                s_, r, done, info = disturbance_step(j, s, action, env, eval_params, form_of_eval)

                value_path.append(policy.evaluate_value(s,a))
                state_path.append(s.tolist())
                path.append(r)
                cost += r

                if j == max_ep_steps - 1:
                    done = True
                s = s_
                if done:
                    seed_average_cost.append(cost)
                    episode_length.append(j)
                    if j < max_ep_steps-1:
                        die_count += 1
                    break
            
            cost_paths.append(path)
            value_paths.append(value_path)
            state_paths.append(state_path)
            ref_paths.append(ref_path)
            
        
        death_rates.append(die_count/(i+1)*100)
        total_cost.append(np.mean(seed_average_cost))
        
        # convert to np.array and save 
        states_arr = np.array(state_paths)
        values_arr = np.array(value_paths)
        costs_arr = np.array(cost_paths)
        
        # # mkdir npy_path
        # os.makedirs(npy_path+'/{}'.format(trial), exist_ok=True)
        # # save npy file
        # np.save(npy_path+'/{}/states.npy'.format(trial),states_arr)
        # np.save(npy_path+'/{}/values.npy'.format(trial),values_arr)
        # np.save(npy_path+'/{}/costs.npy'.format(trial),costs_arr)

    total_cost_std = np.std(total_cost, axis=0)
    total_cost_mean = np.average(total_cost)
    death_rate = np.mean(death_rates)
    death_rate_std = np.std(death_rates, axis=0)
    average_length = np.average(episode_length)

    diagnostic = {'return': total_cost_mean,
                  'return_std': total_cost_std,
                  'death_rate': death_rate,
                  'death_rate_std': death_rate_std,
                  'average_length': average_length}

    path_dict = {'c': cost_paths, 'v':value_paths}


    return diagnostic, path_dict

def training_evaluation(CONFIG, env, policy, disturber= None):
    env_name = CONFIG['env_name']

    env_params = CONFIG['env_params']

    max_ep_steps = env_params['max_ep_steps']

    eval_params = CONFIG['eval_params']

    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    # For analyse
    Render = env_params['eval_render']

    # Training setting

    total_cost = []
    death_rates = []
    form_of_eval = CONFIG['evaluation_form']
    trial_list = os.listdir(CONFIG['log_path'])
    episode_length = []

    die_count = 0
    seed_average_cost = []
    for i in range(CONFIG['num_of_evaluation_paths']):

        cost = 0
        s = env.reset()
        for j in range(max_ep_steps):
            if Render:
                env.render()
            a = policy.choose_action(s, True)

            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2

            s_, r, done, info = env.step(action)
            # done = False
            cost += r

            if j == max_ep_steps - 1:
                done = True
            s = s_
            if done:
                seed_average_cost.append(cost)
                episode_length.append(j)
                if j < max_ep_steps-1:
                    die_count += 1
                break
    death_rates.append(die_count/(i+1)*100)
    total_cost.append(np.mean(seed_average_cost))

    total_cost_std = np.std(total_cost, axis=0)
    total_cost_mean = np.average(total_cost)
    death_rate = np.mean(death_rates)
    death_rate_std = np.std(death_rates, axis=0)
    average_length = np.average(episode_length)

    diagnostic = {'return': total_cost_mean,
                  'average_length': average_length}
    return diagnostic



def eval(CONFIG):
    for name in CONFIG['eval_list']:
        CONFIG['log_path'] = '/'.join(['./log', CONFIG['env_name'], name])

        if 'ALAC' in name:
            CONFIG['alg_params'] = ALG_PARAMS['ALAC']
            CONFIG['algorithm_name'] = 'ALAC'
        print('evaluating '+name)
        if EVAL_PARAMS['evaluation_form'] == 'constant_impulse':
            constant_impulse(CONFIG)


