import numpy as np
from pool.memory import ReplayBuffer 
from variant import get_env_from_name
from logger.logger import logger

def train(variant):
    env_name = variant["env_name"]
   
    env = get_env_from_name(env_name)
    env_params = variant['env_params']
    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['num_of_training_paths']
    evaluation_frequency = variant['evaluation_frequency']

    policy_params = variant['alg_params']
    policy_params['network_structure'] = env_params['network_structure']

    memory_capacity = policy_params['memory_capacity'],

    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle']
    train_per_cycle = policy_params['train_per_cycle']
    batch_size = policy_params['batch_size']

    if 'Fetch' in env_name or 'Hand' in env_name:
        s_dim = env.observation_space.spaces['observation'].shape[0]\
                + env.observation_space.spaces['achieved_goal'].shape[0]+ \
                env.observation_space.spaces['desired_goal'].shape[0]
    else:
        s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]

    policy = eval(variant['algorithm_name'])(variant=variant)


    should_render = env_params['eval_render']

    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=store_last_n_paths)
    training_started = False

    replay_buffer = ReplayBuffer(obs_dim = s_dim,
                                 act_dim = a_dim, 
                                 size = memory_capacity)


    for i in range(max_episodes):
        current_path = {'cost_reward': [],
                        'a_loss': [],

                        'lambda_e': [],

                        'lambda_l': [],
                        'lyapunov_error': [],
                        'entropy': [],
                        }
        if global_step > max_global_steps:
            break


        state = env.reset()
        if 'Fetch' in env_name or 'Hand' in env_name:
            state = np.concatenate([state[key] for key in state.keys()])

        for j in range(max_ep_steps):
            if should_render:
                env.render()

            action = policy.choose_action(state)

            #not sure if i should impliment the bound thing here as i already 
            #multiply action by the bound

            new_state , cost_reward, done , info = env.step(action)

            if 'Fetch' in env_name or 'Hand' in env_name:
                new_state = np.concatenate([new_state[key] for key in new_state.keys()])
            if info['done'] > 0:
                    done = True
            
            if training_started:
                global_step+=1

            if j == max_ep_steps - 1:
                done = True
            
            replay_buffer.store(state, action, cost_reward, new_state, done)

            state = new_state

            if replay_buffer.size > min_memory_size and global_step % steps_per_cycle == 0:
                training_started = True

                for _ in range(train_per_cycle):
                    batch = replay_buffer.sample_batch(batch_size)
                    lamda_l, lamda_e, loss_L, entropy, a_loss = policy.update(batch)

            if training_started:
                current_path['cost_reward'].append(cost_reward)
                current_path['lyapunov_error'].append(l_loss)
                current_path['alpha'].append(alpha)
                current_path['lambda'].append(labda)
                current_path['entropy'].append(entropy)
                current_path['a_loss'].append(a_loss)

            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:
    
                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                if training_diagnotic is not None:
                    if variant['num_of_evaluation_paths'] > 0:
                        eval_diagnotic = training_evaluation(variant, env, policy)
                        [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
                        training_diagnotic.pop('return')
                    [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)
                    logger.logkv('lr_l', lr_l_now)

                    string_to_print = ['time_step:', str(global_step), '|']
                    if variant['num_of_evaluation_paths'] > 0:
                        [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
                         for key in eval_diagnotic.keys()]
                    [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)) , '|'])
                     for key in training_diagnotic.keys()]
                    print(''.join(string_to_print))

                logger.dumpkvs()
                
            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)

                frac = 1.0 - (global_step - 1.0) / max_global_steps
                if variant['algorithm_name'] is in ['ALAC']:  # can be expanded
                    policy.lr_actor_network = lr_a_now = lr_a * frac  # learning rate for actor
                    policy.lr_criric_network = lr_c_now = lr_c * frac  # learning rate for critic
                    policy.lr_langrangian_multipliter = lr_l_now = lr_l * frac  # learning rate for critic
                break                   
    policy.save_result(log_path)

    print('Running time: ', time.time() - t1)

    return

def evaluate_training_rollouts(paths):
    data = copy.deepcopy(paths)
    if len(data) < 1:
        return None
    try:
        diagnostics = OrderedDict((
            ('return', np.mean([np.sum(path['cost_reward']) for path in data])),
            ('length', np.mean([len(p['cost_reward']) for p in data])),
        ))
    except KeyError:
        return
    [path.pop('cost_reward') for path in data]
    for key in data[0].keys():
        result = [np.mean(path[key]) for path in data]
        diagnostics.update({key: np.mean(result)})

    return diagnostics