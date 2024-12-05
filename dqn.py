import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from joblib import Parallel, delayed

import wandb

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
import optuna

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
max_loss = 0.0

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_layers:int=1, hidden_dim:int=128, rho:float=0.2):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.layer3 = nn.Linear(hidden_dim, n_actions)

        self.rho = rho #residual weight

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        y = nn.functional.scaled_dot_product_attention(x, x, x)
        y = F.relu(self.layer2(y))
        return self.layer3(y+self.rho*x)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
LR = 1e-3

class epsDQN:
    def __init__(self, n_observations, n_actions, batch_size=128, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, tau=0.005, lr=1e-4):
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts
        # batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss:torch.Tensor = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def train(self, num_episodes):
        """Train the epsDQN agent for num_episodes episodes"""

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break

        print('Complete')
        plot_durations(show_result=True)
        plt.ioff()
        plt.show()

class thompDQNv1: #TODO implement our new thompDQN agent
    def __init__(self, n_observations, n_actions, batch_size=128, gamma=0.99, tau=0.005, lr=1e-4):
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = torch.tensor([1.0 for _ in range(n_actions)], device=device, dtype=torch.float)
        self.beta = torch.tensor([1.0 for _ in range(n_actions)], device=device, dtype=torch.float)
        self.tau = tau
        self.lr = lr

    def select_action(self, state):
        # sample = random.random()
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        #     math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        with torch.no_grad():
            actions:torch.Tensor = self.policy_net(state)
        
        #sample the beta distribution
        probs = torch.distributions.beta.Beta(self.alpha, self.beta).sample()

        #choose the action with the highest probability and chosen value by the policy network
        prob_actions = actions + probs
        #norm the probabilities
        prob_actions = F.softmax(prob_actions, dim=1)

        return prob_actions.max(1).indices.view(1, 1)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts
        # batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                            if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss:torch.Tensor = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes):
        """Train the thompQDNv1 agent for num_episodes episodes"""

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # update the alpha and beta values, according to thompson, +1 alpha for reward, +1 beta for no reward
                if reward > 0:
                    self.alpha[action] += 1
                else:
                    self.beta[action] += 1

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break

        print('Complete')
        plot_durations(show_result=True)
        plt.ioff()
        plt.show()
    

TransitionV2 = namedtuple('Transition',
                        ('state', 'alpha','beta', 'action', 'next_state', 'reward'))
class ReplayMemoryV2(object):
    
        def __init__(self, capacity):
            self.memory = deque([], maxlen=capacity)
    
        def push(self, *args):
            """Save a transition"""
            self.memory.append(TransitionV2(*args))
    
        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
    
        def __len__(self):
            return len(self.memory)
        
class thompDQNv2:
    def __init__(self, num_episodes=500, memory_size:int=10000,batch_size=128, gamma=0.99, tau=0.005, rho=0.2, lr=1e-4, hidden_layers:int=1, hidden_dim:int=128, report_to_optuna:bool=False, trial=None, report_to_wandb:bool=False):
        self.env = gym.make("CartPole-v1")
        n_observations = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.policy_net = DQN(n_observations+2*n_actions, n_actions, hidden_layers=hidden_layers, hidden_dim=hidden_dim, rho=rho).to(device)
        self.target_net = DQN(n_observations+2*n_actions, n_actions, hidden_layers=hidden_layers, hidden_dim=hidden_dim, rho=rho).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.optimizer.param_groups[0]['initial_lr'] = lr
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_episodes, eta_min=lr/100, last_epoch=num_episodes)
        self.memory = ReplayMemoryV2(memory_size)
        self.steps_done = 0

        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = torch.tensor([1.0 for _ in range(n_actions)], device=device, dtype=torch.float)
        self.beta = torch.tensor([1.0 for _ in range(n_actions)], device=device, dtype=torch.float)
        self.tau = tau
        self.lr = lr
        self.num_episodes = num_episodes

        self.report_to_optuna = report_to_optuna
        self.trial:optuna.trial.Trial = trial
        self.report_to_wandb = report_to_wandb
        

    def select_action(self, state):
        # sample = random.random()
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
        #     math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        with torch.no_grad():
            # print(state, self.alpha, self.beta)
            ab_size = self.alpha.sum() + self.beta.sum()
            actions:torch.Tensor = self.policy_net(torch.cat([state, self.alpha.unsqueeze(0)/ab_size, self.beta.unsqueeze(0)/ab_size], dim=1))

        return actions.max(1).indices.view(1, 1)
    
    def optimize_model(self, return_loss=False):
        #if we want to return the loss, we do prediction, so just take all the memory for computing the loss
        if return_loss:
            transitions = [self.memory.memory.pop() for _ in range(len(self.memory.memory))]

        else: #otherwise in training wait until enough memories, then sample them.
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts
        # batch-array of Transitions to Transition of batch-arrays.
        batch = TransitionV2(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_alpha = torch.cat([a for i,a in enumerate(batch.alpha) if batch.next_state[i] is not None])
        non_final_beta = torch.cat([b for i,b in enumerate(batch.beta) if batch.next_state[i] is not None])

        state_batch = torch.cat(batch.state)
        alpha_batch = torch.cat(batch.alpha)
        beta_batch = torch.cat(batch.beta)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # print(alpha_batch)
        # print(ab_size)
        state_action_values = self.policy_net(torch.cat((state_batch, alpha_batch/alpha_batch.sum(dim=1).unsqueeze(1), beta_batch/beta_batch.sum(dim=1).unsqueeze(1)), dim=1)).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad(): #just use the same alpha beta, as similar enough
            next_state_values[non_final_mask] = self.target_net(torch.cat((non_final_next_states, non_final_alpha/non_final_alpha.sum(dim=1).unsqueeze(1), non_final_beta/non_final_alpha.sum(dim=1).unsqueeze(1)), dim=1)).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss:torch.Tensor = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        #if we want to return the loss, do so and not update
        if return_loss:
            return loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        global max_loss
        if loss.item() > max_loss:
            max_loss = float(loss.item())
            print(max_loss)

        #apply the gradfilte
        # grads = gradfilter_ma(self.policy_net, grads=grads, lamb=5.0, trigger=duration<100) 
        # grads = gradfilter_ema(self.policy_net, grads=grads, lamb=0.5, alpha=0.8)
        #alpha...momentum
        #lamb...amplication factor

        #print the gradient magnitude
        # print("Gradient magnitude:", self.policy_net.layer1.weight.grad.norm().item(), self.policy_net.layer2.weight.grad.norm().item(), self.policy_net.layer3.weight.grad.norm().item())
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.lr_scheduler.step()


    def train(self):
        """Train the thompQDNv1 agent for num_episodes episodes"""

        grads = None
        for i_episode in range(self.num_episodes):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            in_done = False
            for t in count():
                if in_done:
                    break
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                in_done = done

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(
                    state, 
                    torch.zeros_like(self.alpha.unsqueeze(0)).copy_(self.alpha.unsqueeze(0)), 
                    torch.zeros_like(self.beta.unsqueeze(0)).copy_(self.beta.unsqueeze(0)), action, next_state, reward)

                # update the alpha and beta values, according to thompson, +1 alpha for reward, +1 beta for no reward
                # print("This action / reward:", action, reward)
                if reward > 0:
                    self.alpha[action] += 1
                else:
                    self.beta[action] += 1

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    break
            
            
            if self.report_to_optuna:
                self.trial.report(np.mean(episode_durations), i_episode)

            if self.report_to_wandb:
                wandb.log({'avg_duration': np.mean(episode_durations)})
            elif self.trial is None:
                plot_durations()

            if self.report_to_optuna and self.trial.should_prune():
                wandb.finish()
                raise optuna.TrialPruned()

        if self.trial!=None:
            return sum(episode_durations)/self.num_episodes
        else:
            print('Complete')
            plot_durations(show_result=True)
            plt.ioff()
            plt.show()

    def predict(self, num_episodes)->list[float]:
        """Predict the thompQDNv1 agent for num_episodes episodes until death. Returns the losses of the episodes"""
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        losses:list[torch.Tensor] = []
        for i in range(num_episodes):
            self.memory.memory.clear()
            in_done = False
            for t in count():
                if in_done:
                    break
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated
                in_done = done

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(
                    state, 
                    torch.zeros_like(self.alpha.unsqueeze(0)).copy_(self.alpha.unsqueeze(0)), 
                    torch.zeros_like(self.beta.unsqueeze(0)).copy_(self.beta.unsqueeze(0)), action, next_state, reward)

                # update the alpha and beta values, according to thompson, +1 alpha for reward, +1 beta for no reward
                # print("This action / reward:", action, reward)
                if reward > 0:
                    self.alpha[action] += 1
                else:
                    self.beta[action] += 1

                # Move to the next state
                state = next_state

            # compute the loss for the entire memory
            loss = self.optimize_model(return_loss=True)
            losses.append(loss)

        return losses
    
class ThresholdController:#loss variant
    def __init__(self, eps=0.2, alpha=0.2, initial_loss=20.0):
        """
        Args:
            eps: The prefactor to the loss-ema for the threshold
            alpha: The smoothing factor for the ema
            initial_loss: The initial loss value, may be set to None, then the first loss will be used for initialization.
        """
        self.eps = eps
        self.alpha = alpha
        self.smoothed_loss = initial_loss

        self.threshold = 1.0
    
    def update_threshold(self, loss):
        # Update smoothed loss using EWMA in the log domain
        if self.smoothed_loss is None:
            self.smoothed_loss = np.log(loss)
        else:
            self.smoothed_loss = self.alpha * np.log(loss) + (1 - self.alpha) * self.smoothed_loss

        self.threshold = self.eps * min(1,self.smoothed_loss)
    def __call__(self):
        return self.threshold
    
class thompDQNv3(thompDQNv2):
    def __init__(self, eps=0.6, eps_moment:float=0.1, threshold_init_loss:float=20.0, **kwargs):

        super().__init__(**kwargs)

        self.threshold_controller = ThresholdController(alpha=eps_moment, eps=eps, initial_loss=threshold_init_loss)
    
    def optimize_model(self, return_loss=False):
        #includes the entropy_controller update after loss computation
        #if we want to return the loss, we do prediction, so just take all the memory for computing the loss
        if return_loss:
            transitions = [self.memory.memory.pop() for _ in range(len(self.memory.memory))]

        else: #otherwise in training wait until enough memories, then sample them.
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts
        # batch-array of Transitions to Transition of batch-arrays.
        batch = TransitionV2(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_alpha = torch.cat([a for i,a in enumerate(batch.alpha) if batch.next_state[i] is not None])
        non_final_beta = torch.cat([b for i,b in enumerate(batch.beta) if batch.next_state[i] is not None])

        state_batch = torch.cat(batch.state)
        alpha_batch = torch.cat(batch.alpha)
        beta_batch = torch.cat(batch.beta)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(torch.cat((state_batch, alpha_batch/alpha_batch.sum(dim=1).unsqueeze(1), beta_batch/beta_batch.sum(dim=1).unsqueeze(1)), dim=1)).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad(): #just use the same alpha beta, as similar enough
            next_state_values[non_final_mask] = self.target_net(torch.cat((non_final_next_states, non_final_alpha/non_final_alpha.sum(dim=1).unsqueeze(1), non_final_beta/non_final_alpha.sum(dim=1).unsqueeze(1)), dim=1)).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss:torch.Tensor = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        #if we want to return the loss, do so and not update
        if return_loss:
            return loss
        
        self.threshold_controller.update_threshold(loss.item())
        # print(self.steps_done, loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.lr_scheduler.step()

    def select_action(self, state):
        # Update threshold based on current loss
        z:float = random.random()#random number
        self.steps_done += 1
        with torch.no_grad():
            # print(state, self.alpha, self.beta)
            ab_size = self.alpha.sum() + self.beta.sum()
            actions:torch.Tensor = self.policy_net(torch.cat([state, self.alpha.unsqueeze(0)/ab_size, self.beta.unsqueeze(0)/ab_size], dim=1))

        if z > self.threshold_controller():
            # print("actions", actions)
            return actions.max(1).indices.view(1, 1)
        
        else:#return based on thompson sampling
            #sample the beta distribution
            probs = torch.distributions.beta.Beta(self.alpha, self.beta).sample()
            # print("beta", probs)
            return probs.max(0).indices.view(1, 1)


def v2_CV_train(fold:int, **kwargs):
    num_episodes = kwargs['num_episodes']
    trial = kwargs.get('trial', None)
    if trial is None:
        raise ValueError("Trial is required for optuna study")
    class thompDQNv2_Parallel(thompDQNv2):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.grads = None
            self.i_episode = 0
            
        def train(self):#parallel train, does one episode then waits for the next call
            """Train the thompQDNv1 agent for num_episodes episodes"""

            if self.i_episode < self.num_episodes:
                self.i_episode += 1
                # Initialize the environment and get its state
                state, info = self.env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                in_done = False
                for t in count():
                    if in_done:
                        break
                    action = self.select_action(state)
                    observation, reward, terminated, truncated, _ = self.env.step(action.item())
                    reward = torch.tensor([reward], device=device)
                    done = terminated or truncated
                    in_done = done

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                    # Store the transition in memory
                    self.memory.push(
                        state, 
                        torch.zeros_like(self.alpha.unsqueeze(0)).copy_(self.alpha.unsqueeze(0)), 
                        torch.zeros_like(self.beta.unsqueeze(0)).copy_(self.beta.unsqueeze(0)), action, next_state, reward)

                    # update the alpha and beta values, according to thompson, +1 alpha for reward, +1 beta for no reward
                    # print("This action / reward:", action, reward)
                    if reward > 0:
                        self.alpha[action] += 1
                    else:
                        self.beta[action] += 1

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the policy network)
                    self.optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                    self.target_net.load_state_dict(target_net_state_dict)

                    if done:
                        episode_durations.append(t + 1)
                        if self.report_to_optuna:
                            return(np.mean(episode_durations))
                        else:
                            plot_durations()

                        break

            else:
                if self.report_to_optuna:
                    return np.mean(episode_durations)
                else:
                    print('Complete')
                    plot_durations(show_result=True)
                    plt.ioff()
                    plt.show()
        
        def optimize_model(self):
            if len(self.memory) < self.batch_size:
                return
            transitions = self.memory.sample(self.batch_size)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts
            # batch-array of Transitions to Transition of batch-arrays.
            batch = TransitionV2(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            non_final_alpha = torch.cat([a for i,a in enumerate(batch.alpha) if batch.next_state[i] is not None])
            non_final_beta = torch.cat([b for i,b in enumerate(batch.beta) if batch.next_state[i] is not None])

            state_batch = torch.cat(batch.state)
            alpha_batch = torch.cat(batch.alpha)
            beta_batch = torch.cat(batch.beta)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            # print(alpha_batch)
            # print(ab_size)
            state_action_values = self.policy_net(torch.cat((state_batch, alpha_batch/alpha_batch.sum(dim=1).unsqueeze(1), beta_batch/beta_batch.sum(dim=1).unsqueeze(1)), dim=1)).gather(1, action_batch)
            
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.batch_size, device=device)
            with torch.no_grad(): #just use the same alpha beta, as similar enough
                next_state_values[non_final_mask] = self.target_net(torch.cat((non_final_next_states, non_final_alpha/non_final_alpha.sum(dim=1).unsqueeze(1), non_final_beta/non_final_alpha.sum(dim=1).unsqueeze(1)), dim=1)).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.gamma) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss:torch.Tensor = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            # print(loss.item())
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()

            #apply the gradfilte
            # self.grads = gradfilter_ma(self.policy_net, grads=self.grads, lamb=5.0, trigger=duration<100) 
            # self.grads = gradfilter_ema(self.policy_net, grads=self.grads, lamb=0.5, alpha=0.8)
            #alpha...momentum
            #lamb...amplication factor

            #print the gradient magnitude
            # print("Gradient magnitude:", self.policy_net.layer1.weight.grad.norm().item(), self.policy_net.layer2.weight.grad.norm().item(), self.policy_net.layer3.weight.grad.norm().item())
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()
            self.lr_scheduler.step()
    
    agents = [thompDQNv2_Parallel(**kwargs) for _ in range(fold)]
    vals = [0 for _ in range(fold)]
    for i in range(num_episodes):
        vals = Parallel(n_jobs=fold, backend="threading")(delayed(agent.train)() for agent in agents)
        # trial.report(np.mean(vals), i)

        # if trial.should_prune():
        #     raise optuna.TrialPruned()
    
    return np.mean(vals)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            display.clear_output(wait=True)


NUM_EPISODES = 500

# #now do a optuna hpo
def objectivev2(trial):
    hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    tau = trial.suggest_float('tau', 0.0001, 0.1, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999, log=True)
    memory_size = trial.suggest_int("memory_size", 1000, 10000, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256, log=True)
    rho = trial.suggest_float('rho', 0.1, 0.5)
    num_episodes = NUM_EPISODES

    #do a 5-fold CV
    # agent = thompDQNv2(hidden_layers=hidden_layers, hidden_dim=hidden_dim, num_episodes=num_episodes, lr=lr, tau=tau, gamma=gamma, batch_size=batch_size, memory_size=memory_size, rho=rho, report_to_optuna=True, trial=trial)
    # score = agent.train()
    score = v2_CV_train(5, hidden_layers=hidden_layers, hidden_dim=hidden_dim, num_episodes=num_episodes, lr=lr, tau=tau, gamma=gamma, batch_size=batch_size, memory_size=memory_size, rho=rho, report_to_optuna=True, trial=trial)

    return score


def objectivev3(trial:optuna.trial.Trial):    

    hidden_layers = trial.suggest_int('hidden_layers', 1, 3)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    tau = trial.suggest_float('tau', 0.0001, 0.1, log=True)
    gamma = trial.suggest_float('gamma', 0.9, 0.999, log=True)
    memory_size = trial.suggest_int("memory_size", 1000, 10000, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256, log=True)
    rho = trial.suggest_float('rho', 0.1, 0.5)
    num_episodes = NUM_EPISODES
    eps = trial.suggest_float('eps', 0.1, 1.0)
    eps_moment = trial.suggest_float('eps_moment', 0.01, 0.9, log=True)
    threshold_init_loss = trial.suggest_float('threshold_init_loss', 1.0, 100.0)

    wandb.init(project="thompDQNv3", config=trial.params)

    agent = thompDQNv3(hidden_layers=hidden_layers, hidden_dim=hidden_dim, num_episodes=num_episodes, lr=lr, tau=tau, gamma=gamma, batch_size=batch_size, memory_size=memory_size, rho=rho, eps=eps, eps_moment=eps_moment, threshold_init_loss=threshold_init_loss, report_to_optuna=True, trial=trial, report_to_wandb=True)
    score = agent.train()

    wandb.finish()

    return score

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner(max_resource=NUM_EPISODES))
study.optimize(objectivev3, n_trials=500)

# print('Best trial:')
# trial = study.best_trial

# print('Value: ', trial.value)
# print('Params: ')
# for key, value in trial.params.items():
#     print(f'    {key}: {value}')


#v2
# agent = thompDQNv2(hidden_layers=1, hidden_dim=240, num_episodes=500, lr=0.007535, tau=0.08124, gamma=0.9923, memory_size=4728, batch_size=128, rho=0.11745)
# agent.train()

#v3
agent = thompDQNv3(**study.best_params, num_episodes=500)
agent.train()