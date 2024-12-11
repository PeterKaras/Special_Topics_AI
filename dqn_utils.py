import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import getpass
from typing import List, Tuple, Any, Optional, Callable, overload, Literal, Generator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
from joblib import Parallel, delayed

import wandb
import ray

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

from collections import namedtuple, deque
import random

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state"])

TransitionV2 = namedtuple("TransitionV2", ["state", 'alpha','beta', "action", "reward", "next_state"])

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

        def reset(self):
            self.memory = deque([], maxlen=self.memory.maxlen)

class RLTask:
    def __init__(self, env:gym.Env, episodes:int, portion_test:float):
        self.env:gym.Env = env
        self.size:int = episodes
        self.portion_test:float = portion_test
        self.train_set:Generator[int] = range(math.ceil(episodes*(1-portion_test)))
        self.test_set:Generator[int] = range(math.ceil(episodes*(1-portion_test)), episodes)
    
    def train_sample(self)->Generator[Tuple[gym.Env, Any], None, None]:
        """Samples the environment of the task.
            For that it resets the environment with a new seed, and returns the environment and the initial state.
        """
        for task in self.train_set:
            yield self.env

    def test_sample(self)->Generator[Tuple[gym.Env, Any], None, None]:
        """Samples the environment of the task.
            For that it resets the environment with a new seed, and returns the environment and the initial state.
        """
        for task in self.test_set:
            yield self.env
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_layers:int=1, hidden_dim:int=128, rho:float=0.2, xi:float=0.2):
        super(DQN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_observations, hidden_dim), 
            # nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )

        hidden_layers_list = []
        hidden_layers_list.append(nn.Linear(hidden_dim, hidden_dim))
        # hidden_layers_list.append(nn.BatchNorm1d(hidden_dim))
        hidden_layers_list.append(nn.ReLU())
        self.layer2 = nn.Sequential(*[nn.Sequential(*hidden_layers_list) for _ in range(hidden_layers)])

        self.layer3 = nn.Linear(hidden_dim, n_actions)
        # self.bn3 = nn.BatchNorm1d(n_actions) 
        self.rho = rho #residual weight

        self.skip_proj = nn.Linear(n_observations, n_actions, bias=False) #skip connection projection 
        self.xi = xi #skip connection weight
        # self.skip_bn = nn.BatchNorm1d(n_actions)
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, inp:torch.Tensor)->torch.Tensor:
        # was_single=False
        # if inp.size(0) == 1: #if we do not have a batch, make it a batch so the batch norm works
        #     was_single=True
        #     inp = inp.repeat(2, 1)

        x:torch.Tensor = self.layer1(inp)
            
        batch_size = inp.size(0)
        seq_length = x.size(1)
        x = x.view(batch_size, 1, seq_length)  # Reshape to [batch, heads, features] for attention
        y = nn.functional.scaled_dot_product_attention(x, x, x).squeeze(1)  # Self-Attention
        y = self.layer2(y)

        # main_out = self.bn3(self.layer3(y+self.rho*x.squeeze(1))) #self.bn3(self.layer3(y+self.rho*x.squeeze(1)))
        # skip_out = self.skip_bn(self.skip_proj(inp))
        main_out = self.layer3(y+self.rho*x.squeeze(1))
        skip_out = self.skip_proj(inp)
        out:torch.Tensor = (1-self.xi)*main_out * self.xi+skip_out
        # if was_single:
        #     out = out[:1] #return the original input
        return out
    
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
    
class thompDQNv3():
    def __init__(self, n_observations:int, n_actions:int, num_episodes=500, memory_size:int=10000, batch_size=128, gamma=0.99, tau=0.005, rho=0.2, xi=0.2, lr=1e-4, loss_fn:Callable=torch.nn.SmoothL1Loss(), optimizer:type[optim.Optimizer]=optim.AdamW, optim_kwargs:dict={"amsgrad":True}, grad_clip:float=100., update_rate:int=1, hidden_layers:int=1, hidden_dim:int=128, eps=0.6, eps_moment:float=0.1, threshold_init_loss:float=20.0):
        """DQN with thompson sampling and threshold controller
        Args:
            n_observations: The number of observations in the environment
            n_actions: The number of actions in the environment
            num_episodes: Number of episodes to train the agent
            memory_size: The size of the replay memory
            batch_size: The size of the batch to sample from the replay memory
            gamma: The discount factor
            tau: The soft update factor for the target network
            rho: The residual weight for the attention residual connection
            xi: The skip connection weight
            lr: The learning rate
            loss_fn: The loss function to use. Must be initialised already. E.g. torch.nn.SmoothL1Loss()
            optimizer: The optimizer to use. Should not be initialised.
            grad_clip: The gradient clipping value
            update_rate: The rate at which to update the networks. Should normally be 1, but for meta-learning higher values may be used.
            hidden_layers: The number of hidden layers
            hidden_dim: The dimension of the hidden layers
            eps: The exploration factor, higher means exploration for longer
            eps_moment: The smoothing factor for the threshold controller. It is ema loss based.
            threshold_init_loss: The initial loss for the threshold controller
        """

        self.policy_net = DQN(n_observations+2*n_actions, n_actions, hidden_layers=hidden_layers, hidden_dim=hidden_dim, rho=rho).to(device)
        self.target_net = DQN(n_observations+2*n_actions, n_actions, hidden_layers=hidden_layers, hidden_dim=hidden_dim, rho=rho).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optimizer(self.policy_net.parameters(),**optim_kwargs|{"lr":lr})
        self.optimizer.param_groups[0]['initial_lr'] = lr
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_episodes, eta_min=lr/100, last_epoch=num_episodes)
        self.memory = ReplayMemoryV2(memory_size)
        self.steps_done = 0

        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = torch.tensor([1.0 for _ in range(n_actions)], device=device, dtype=torch.float)
        self.beta = torch.tensor([1.0 for _ in range(n_actions)], device=device, dtype=torch.float)
        self.thomp_sampling = True #whether to use random_sampling or thompson sampling for exploration
        self.tau = tau
        self.lr = lr
        self.loss_fn = loss_fn
        self.grad_clip = grad_clip 
        self.update_rate = update_rate
        self.num_episodes = num_episodes
        self.eps_done = 0
        self.threshold_controller = ThresholdController(alpha=eps_moment, eps=eps, initial_loss=threshold_init_loss)

        self.episode_durations = []

        print("Running the ThompDQN on", str(device))
    
    @overload
    def optimize_model(self, return_loss:Literal[True]=True)->torch.Tensor:
        pass

    @overload
    def optimize_model(self, return_loss:Literal[False]=False)->None:
        pass
        
    def optimize_model(self, return_loss=False)->None|torch.Tensor:
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
        next_state_values = torch.zeros_like(non_final_mask, device=device, dtype=torch.float)
        with torch.no_grad(): #just use the same alpha beta, as similar enough
            next_state_values[non_final_mask] = self.target_net(torch.cat((non_final_next_states, non_final_alpha/non_final_alpha.sum(dim=1).unsqueeze(1), non_final_beta/non_final_alpha.sum(dim=1).unsqueeze(1)), dim=1)).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss:torch.Tensor = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
        #if we want to return the loss, do so and not update
        if return_loss:
            return loss
        
        self.threshold_controller.update_threshold(loss.item())
        # print(self.steps_done, loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()
        self.lr_scheduler.step()

    def select_action(self, state):
        # Update threshold based on current loss
        z:float = random.random()#random number
        self.steps_done += 1
        with torch.no_grad():
            # print(state, self.alpha, self.beta)
            # self.policy_net.eval()
            actions:torch.Tensor = self.policy_net(torch.cat([state, self.alpha.unsqueeze(0)/self.alpha.sum(), self.beta.unsqueeze(0)/self.beta.sum()], dim=1))
            # self.policy_net.train()

        if z > self.threshold_controller():
            # print("actions", actions)
            return actions.max(1).indices.view(1, 1)
        
        else:#return based on thompson sampling
            #sample the beta distribution
            if self.thomp_sampling:
                probs = torch.distributions.beta.Beta(self.alpha, self.beta).sample()
            else:
                #random sample
                probs = torch.rand_like(self.alpha)
            # print("beta", probs)
            return probs.max(0).indices.view(1, 1)

    @overload
    def train(self, env, num_eps:Optional[int]=None, return_indiv_losses:Literal[False]=False, ignore_episodes:bool=False, return_average_score:bool=False)->float:
        pass

    @overload
    def train(self, env, num_eps:Optional[int]=None, return_indiv_losses:Literal[True]=True, ignore_episodes:bool=False, return_average_score:bool=False)->Tuple[float, List[torch.FloatTensor]]:        
        pass

    def train(self, env, num_eps:Optional[int]=None, return_indiv_losses:bool=False, ignore_episodes:bool=False, return_average_score:bool=False)->float | Tuple[float, List[torch.FloatTensor]]:
        """Train the thompQDNv3 agent for num_episodes episodes"""
        if not ignore_episodes and self.eps_done == self.num_episodes: #maybe we ignore the given number of episodes for the model
            return float(np.mean(self.episode_durations))
        losses:list[torch.FloatTensor] = []
        for i_episode in range(self.eps_done, self.num_episodes if num_eps is None else min(self.num_episodes, self.eps_done+num_eps)):
            self.eps_done += 0 if ignore_episodes else 1
            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            in_done = False
            for t in count():
                if in_done:
                    break
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
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
                    torch.zeros_like(self.beta.unsqueeze(0)).copy_(self.beta.unsqueeze(0)), action, reward, next_state)

                # update the alpha and beta values, according to thompson, +1 alpha for reward, +1 beta for no reward
                # print("This action / reward:", action, reward)
                if reward > 0:
                    self.alpha[action] += 1
                else:
                    self.beta[action] += 1

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if t % self.update_rate == 0:
                    loss = self.optimize_model()
                    if return_indiv_losses:
                        losses.append(loss)

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
                    self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    break

        if return_indiv_losses:
            # print("in meta training")
            return float(np.mean(self.episode_durations) if return_average_score else self.episode_durations[-1]), losses
        else:
            # print("not in meta training")
            return float(np.mean(self.episode_durations) if return_average_score else self.episode_durations[-1])

    def predict(self, env)->Tuple[int, torch.Tensor]:
        """Predict the thompQDNv2 agent and return the loss tensor"""
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        in_done = False
        score = 0
        for t in count():
            if in_done:
                break
            action = self.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
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
                torch.zeros_like(self.beta.unsqueeze(0)).copy_(self.beta.unsqueeze(0)), action, reward, next_state)

            # Move to the next state
            state = next_state
            if done:
                score = t+1

        loss:torch.Tensor = self.optimize_model(return_loss=True) #just computes the loss without updating the params
        return score, loss



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
    print("Episode:", len(durations_t), "Duration:", durations_t[-1])
    print("Average duration:", np.mean(episode_durations))
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            display.clear_output(wait=True)