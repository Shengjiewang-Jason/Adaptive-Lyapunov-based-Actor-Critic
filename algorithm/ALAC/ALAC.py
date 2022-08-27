from cProfile import label
from webbrowser import get
import torch as T
import numpy as np
import core
from copy import deepcopy
from torch.optim import Adam
import torch.optim as optim
import time
from collections import OrderedDict, deque
from common import logger
class ALAC():
    def __init__(self, a_dim, s_dim, policy_params,
                action_prior = "uniform", lambda_optimizer: str = 'Adam') -> None:
        self.SCALE_lambda_MIN_MAX = (0, 1)
        self.gamma = policy_params['gamma']
        self.tau = policy_params['tau'] # value for plymac update
        self.action_prior = action_prior
        self.a_dim = a_dim
        self.target_entropy = policy_params['target_entropy']

        
        # learning rate declerations
        self.lr_actor_network, self.lr_criric_network, self.lr_langrangian_multipliter = \
            policy_params['lr_a'], policy_params['lr_c'], policy_params['lr_l']


        # declare networks and target networks
        actor_critic_Lyapunov = core.MLPActorCritic
        # Create actor-critic module and target networks
        self.actor_critic_agent = actor_critic_Lyapunov(s_dim, action_space=a_dim).to(CONFIG)
        self.actor_critic_agent_target = deepcopy(self.actor_critic_agent)

        #actor and critic optimizers
        self.pi_optimizer = Adam(self.actor_critic_agent.actor.parameters(), lr=self.lr_actor_network)
        self.critic_lyapunov_optimizer = Adam(self.actor_critic_agent.critic_lyapunov.parameters(), lr=self.lr_criric_network)


        #declare lagrange multipliers and optimizers lambda_l and lambda_e
       
        lambda_l = policy_params['lambda_l']
        lambda_e = policy_params['lambda_e']

        self.log_lambda_l = T.nn.Parameter(
                        T.as_tensor(T.log(lambda_l)),
                        requires_grad=True)

        torch_opt = getattr(optim, lambda_optimizer)

        # optimiser for lambda_l
        self.lambda_l_optimizer = torch_opt([self.log_lambda_l, ],
                                        lr=self.lr_langrangian_multipliter)


        self.log_lambda_e =  T.nn.Parameter(
                            T.as_tensor(T.log(lambda_e)),
                            requires_grad=True)
        # optimiser for lambda_e
        self.lambda_e_optimizer = torch_opt([self.log_lambda_e, ],
                                        lr=self.lr_langrangian_multipliter)

    def lambda_e(self):
        '''
            return : lambda_e  using learned lambda parameter
        '''
        return T.exp(self.log_lambda_e)

    def lambda_l(self):
        '''
            return : lambda_l  using learned lambda parameter
        '''
        return T.clamp(T.exp(self.log_lambda_l), *self.SCALE_lambda_MIN_MAX)

    def compute_L_delta(self, l_, l):
        '''
        page 7 to compute delta L
        value of k_l :page 7 is chosen as (1 - lambda_bar)
        '''
        lambda_l = lambda_l()
        k_l = 1 - lambda_l
        l_delta_1 = (l_ - l + (k_l) *(l - 0)).mean()
        l_delta_2 = (l_ - l + (k_l) *(l - l_)).mean()

        l_delta = (k_l) * l_delta_1 + lambda_l * l_delta_2
        
        return l_delta

    def compute_prior_policy_log_probs(self, action):
        if self.action_prior == 'normal':
            loc = T.zeros(self.a_dim),
            scale_diag = T.ones(self.a_dim)
            # policy_prior = T.distributions.MultivariateNormal(loc, scale_tril=torch.diag(scale))
            policy_prior = T.distributions.MultivariateNormal(loc, scale_tril=T.diag(scale_diag))
            policy_prior_log_probs = policy_prior.log_prob(action)
        elif self.action_prior == 'uniform':
            policy_prior_log_probs = 0.0
        return policy_prior_log_probs

    def choose_action(self, state, deterministic = False):
        '''
        Function to select action with torch turned off
        used for evaluation and action take during simulation
        '''
        return self.actor_critic_agent.act(T.as_tensor(state, dtype=T.float32), 
                      deterministic)

    def compute_critic_loss(self, data):
        state, action, cost_reward, new_state, terminal = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # get action from policy from new state
        actor_policy, _ = self.actor_critic_agent_target.actor(new_state) 

        lyapunov_value_target = self.actor_critic_agent_target.critic_lyapunov(new_state, actor_policy)

        lyapunov_value = self.actor_critic_agent.critic_lyapunov(state, action)

        # eq(16) l_prime
        l_target = cost_reward + ( self.gamma * (1- terminal) * lyapunov_value_target )
        l_error = ((l_target -  lyapunov_value)**2).mean()
        return l_error

    def compute_loss_pi(self, data):
        '''
         returns: actor loss and entropy
        '''
        state, action, new_state = data['obs'], data['act'], data['obs2']

        actor_policy, probalility_dist = self.actor_critic_agent.actor(new_state)
        ent = actor_policy.entropy().mean().item()
        log_pi = probalility_dist.log_prob(actor_policy)

        lambda_e = self.lambda_e()
        lambda_l = self.lambda_l()


        # actor_policy, _ = self.actor_critic_agent.actor(new_state) 

        lyapunov_value_2 = self.actor_critic_agent.critic_lyapunov(new_state, actor_policy)
        
        lyapunov_value = self.actor_critic_agent.critic_lyapunov(state, action)
        

        actor_loss = lambda_l * self.compute_L_delta(l_ = lyapunov_value_2, l = lyapunov_value)  +  lambda_e * log_pi.mean() - self.compute_prior_policy_log_probs(action)
        return actor_loss , ent

    def compute_lambda_l_loss(self, data):
        state, action,  new_state = data['obs'], data['act'], data['obs2']

        actor_policy, _ = self.actor_critic_agent.actor(new_state) 

        lyapunov_value_2 = self.actor_critic_agent.critic_lyapunov(new_state, actor_policy)
        
        lyapunov_value = self.actor_critic_agent.critic_lyapunov(state, action)

        return -(self.log_lambda_l * self.compute_L_delta(l_ = lyapunov_value_2, l = lyapunov_value)).mean()

    def compute_lambda_e_loss(self, data):
        new_state =  data['obs2']
        actor_policy, probalility_dist = self.actor_critic_agent.actor(new_state)
        log_pi = probalility_dist.log_prob(actor_policy)
        
        # record entropy
        self.entropy = - log_pi.mean()

        return -(self.log_lambda_e * (log_pi + self.target_entropy)).mean()
  
    def update_lagrange_multiplier_l(self, data):
        """ Update Lagrange multiplier (lambda_e)
            .
        """
       
        self.lambda_l_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_l_loss(data)
        lambda_loss.backward()
        self.lambda_l_optimizer.step()
        # do i need to clamp multiplier
        # self.lagrangian_multiplier.data.clamp_(0)  # enforce: lambda in [0, inf]
        return lambda_loss

    def update_lagrange_multiplier_e(self, data):
        """ Update Lagrange multiplier (lambda_e)
        """
        self.lambda_e_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_e_loss(data)
        lambda_loss.backward()
        self.lambda_e_optimizer.step()
        # do i need to clamp multiplier
        # self.lagrangian_multiplier.data.clamp_(0)  # enforce: lambda in [0, inf]
        return lambda_loss.item()

    def update_critic_lyapunov_net(self, data):
        self.critic_lyapunov_optimizer.zero_grad()
        loss_L = self.compute_critic_loss(data)
        loss_L.backward()
        self.critic_lyapunov_optimizer.step()
        return loss_L.item()

    def update_policy_net(self, data):
          # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, ent = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()
        return loss_pi.item(), ent.item()

    def update_target_net(self):
         # Finally, update target networks by polyak averaging.
        with T.no_grad():
            for p, p_targ in zip(self.actor_critic_agent.parameters(),self.actor_critic_agent_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.tau)
                p_targ.data.add_((1 - self.tau) * p.data)

    def learning_rate_decay(self):
        # impliment this to decay the lr 
        pass

    def update(self, data):
    

        critic_loss =  self.update_critic_lyapunov_net(data)
       
        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.actor_critic_agent.critic_lyapunov.parameters():
            p.requires_grad = False
        
        pi_loss, ent = self.update_policy_net(data)

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.actor_critic_agent.critic_lyapunov.parameters():
            p.requires_grad = True

        # Record things
        # writer.add_scalar("Loss_Pi", loss_pi.item(), n_update_step)
        # logger.store(LossPi=loss_pi.item(), **pi_info)
        self.update_target_net()
       
        lambda_l_loss = self.update_lagrange_multiplier_l(data)
        lambda_e_loss = self.update_lagrange_multiplier_e(data)

        self.learning_rate_decay()

        return self.lambda_l().data().cpu().numpy(), 
            self.lambda_e().data().cpu().numpy(), 
            loss_L.data().cpu().numpy(), 
            self.entropy.data().cpu().numpy(), 
            loss_pi.data().cpu().numpy(),
            lambda_l_loss.data().cpu().numpy(),
            lambda_e_loss.data().cpu().numpy()



    


         


        

            