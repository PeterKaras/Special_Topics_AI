from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium.core import Env
from collections import deque, OrderedDict, namedtuple
from itertools import count
from tqdm import tqdm
import random
from typing import Dict, Optional, Callable, List, Generator, Tuple, Literal, overload, Any
import math

from env_sets import EnvSet, possible_envs
from scipy.stats import norm, uniform

import wandb

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
TransitionV2 = namedtuple('Transition',
                        ('state', 'alpha','beta', 'action', 'next_state', 'reward'))
class QNetwork(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_layers:int=1, hidden_dim:int=128, rho:float=0.2, xi:float=0.2):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_layers)])
        self.layer3 = nn.Linear(hidden_dim, n_actions)

        self.rho = rho #attention residual weight

        self.xi = xi #input residual weight

        self.skip = nn.Linear(n_observations, n_actions, bias=False)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, inp):
        x = F.relu(self.layer1(inp))
        y = nn.functional.scaled_dot_product_attention(x, x, x)
        y = F.relu(self.layer2(y))
        bn_inp = F.batch_norm(self.skip(inp))
        return self.layer3(y + self.rho*x + self.xi*bn_inp)
    
class RLTask:
    def __init__(self, env:gym.Env, episodes:int, portion_test:float):
        self.env:gym.Env = env
        self.size:int = episodes
        self.portion_test:float = portion_test
        self.train_set:Generator[int] = range(math.ceil(episodes*(1-portion_test)))
        self.test_set:Generator[int] = range(math.ceil(episodes*(1-portion_test)), episodes)

    
    def train_sample(self)->Generator[Tuple[Env, Any], None, None]:
        """Samples the environment of the task.
            For that it resets the environment with a new seed, and returns the environment and the initial state.
        """
        for task in self.train_set:
            state, _ = self.env.reset(seed=task)
            yield self.env, state

    def test_sample(self)->Generator[Tuple[Env, Any], None, None]:
        """Samples the environment of the task.
            For that it resets the environment with a new seed, and returns the environment and the initial state.
        """
        for task in self.test_set:
            state, _ = self.env.reset(seed=task)
            yield self.env, state

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
class EntropyThresholdController:
    def __init__(self, tau_min=0.1, tau_max=1.0, alpha=0.1, loss_max=1.0):
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.alpha = alpha
        self.loss_max = loss_max
        self.smoothed_loss = None

    def update_threshold(self, current_loss):
        # Update smoothed loss using EWMA
        if self.smoothed_loss is None:
            self.smoothed_loss = current_loss
        else:
            self.smoothed_loss = self.alpha * current_loss + (1 - self.alpha) * self.smoothed_loss

        # Scale threshold based on smoothed loss
        scaled_loss = min(self.smoothed_loss / self.loss_max, 1.0)  # Ensure it's within [0, 1]
        threshold = self.tau_min + (self.tau_max - self.tau_min) * scaled_loss
        return threshold
    
class thompDQNv2:
    def __init__(self, task:RLTask, memory_size:int=10000,batch_size=128, gamma=0.99, tau=0.005, rho=0.2, xi=0.2, lr=1e-4, hidden_layers:int=1, hidden_dim:int=128, report_to_optuna:bool=False, trial=None):
        self.env = gym.make("CartPole-v1")
        n_observations = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.policy_net = QNetwork(n_observations+2*n_actions, n_actions, hidden_layers=hidden_layers, hidden_dim=hidden_dim, rho=rho, xi=xi).to(device)
        self.target_net = QNetwork(n_observations+2*n_actions, n_actions, hidden_layers=hidden_layers, hidden_dim=hidden_dim, rho=rho, xi=xi).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.optimizer.param_groups[0]['initial_lr'] = lr
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=task.size, eta_min=lr/100, last_epoch=task.size)
        self.memory = ReplayMemory(memory_size)
        self.steps_done = 0

        self.batch_size = batch_size
        self.gamma = gamma
        self.alpha = torch.tensor([1.0 for _ in range(n_actions)], device=self.device, dtype=torch.float)
        self.beta = torch.tensor([1.0 for _ in range(n_actions)], device=self.device, dtype=torch.float)
        self.tau = tau
        self.lr = lr

    def select_action(self, state):
        self.steps_done += 1
        with torch.no_grad():
            # print(state, self.alpha, self.beta)
            ab_size = self.alpha.sum() + self.beta.sum()
            actions:torch.Tensor = self.policy_net(torch.cat([state, self.alpha.unsqueeze(0)/ab_size, self.beta.unsqueeze(0)/ab_size], dim=1))

        #if the network is sure about the action, take it, otherwise sample from thompson directly
        if nn.Softmax(actions).

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
                                                batch.next_state)), device=self.device, dtype=torch.bool)
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
        next_state_values = torch.zeros(self.batch_size, device=self.device)
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
        self.entropy_controller.update_threshold(loss.item())
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
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            in_done = False
            for t in count():
                if in_done:
                    break
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                in_done = done

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

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
                    break


    def predict(self, num_episodes)->list[float]:
        """Predict the thompQDNv1 agent for num_episodes episodes until death. Returns the losses of the episodes"""
        state, info = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        losses:list[torch.Tensor] = []
        for i in range(num_episodes):
            self.memory.memory.clear()
            in_done = False
            for t in count():
                if in_done:
                    break
                action = self.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                in_done = done

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

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

def generate_task(env:gym.Env, episodes:int=10, portion_test:float=0.5)->RLTask:
    return RLTask(env, episodes, portion_test)

MetaUpdateMethod = Literal["reptile", "maml", "mix"]
class MetaNet():
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=64, layers:int=3, 
        meta_lr:float=0.01, meta_min_lr:float=0.0001, num_task_batches:int=1000, 
        meta_transition_frame:Tuple[int, int, int]=(50, 500, 800), transition_sigma:float=10.0, reptile_decay:float=0.2, reptile_factor:float=2.0,#TODO: how to mix reptile and maml properly
        gamma:float=0.99, lossFn:Callable=nn.SmoothL1Loss, inner_lr:float=0.0001, heuristic_update_inner_lr:bool=False, 
        tau:float=0.001, num_steps:int=500, update_rate:int=8, batch_size:int=64, memory_size:int=10000,
        initial_alpha=1.0, initial_beta=1.0, useMetaSampling:bool=False,
        inner_grad_clip:float=100.0, meta_grad_clip:float=15.0,
        device="auto"
    ):
        """A class for organizing all the MetaNet learning functionalities.
        Args:
        state_dim: int
            The dimension of the state space.
        action_dim: int 
            The dimension of the action space.
        hidden_dim: int = 256
            The dimension of the hidden layers.
        layers: int = 3
            The number of layers in the Q-NN. 
        meta_lr: float = 0.01
            The learning rate of the meta-net. 
        meta_min_lr: float = 0.0001
            The minimum learning rate of the meta-net, used for the CosineAnnealingLR scheduler.
        num_task_batches: int = 1000
            The total number of task batches that are learned. "Just like an episode".
        meta_transition_frame: Tuple[int, int, int] = (10, 50, 80)
            Given (m,a,b): The number of inner updates `u` done for the meta update to behave like MAML (u<a), like reptile (u>=b), or a smooth transition between the two (a<=u<b). The smooth transition is a sigmoid. #TODO: maybe with linear end pieces. If u<m, we don't use the task for an update to not have a negative influence on the meta-net.
        transition_sigma: float = 10.0
            The sigma parameter for the sigmoid transition between MAML and Reptile.
        reptile_decay:float = 0.2
            The rate of the exponential decay applied to reptile updates based on parameter l2-distance.
        reptile_factor: float = 2.0
            The factor by which the reptile update is multiplied. This is to balance reptile relative to MAML. The overall update strength should be guided by the meta_lr.
        gamma: float = 0.99
            The discount factor for the q-net rewards.
        lossFn: Callable = nn.SmoothL1Loss
            The loss function to be used for the inner-net update.
        inner_lr: float = 0.005
            The learning rate of the inner-net. 
        heuristic_update_inner_lr: bool = False
            Whether to use the heuristic to update the inner-net learning rate or not.
            The heuristic is that the inner lr should depend on the outer lr by some factor. This makes sure that while the outer net progresses, the inner net does not overshoot with too high learning rates, but still learns fast in the beginning. #TODO implement inner lr heuristic
        tau: float = 0.001
            The tau parameter for the soft update of the target net.
        num_steps: int = 500
            The number of steps to take in each environment pass. How many environment passes are done is determined by how many are given in the RLTask.
        update_rate: int = 8
            The number of steps before the inner-net is updated. Should be <= batch_size so that each batch can be used multiple times for the inner-net update. Batch contents are drawn randomly from the memory.
        batch_size: int = 32
            The size of the batch for the inner-net update.
        memory_size: int = 10000
            The size of the memory buffer.
        initial_alpha: float = 1.0 
            The initial value of the alpha parameter for the sampling.
        initial_beta: float = 1.0
            The initial value of the beta parameter for the sampling.
        useMetaSampling: bool = False
            Whether to use the meta-assisted sampling or not. When true a meta-trained net will be used to sample the actions from the q-values instead of directly using the thompson sampling.
        inner_grad_clip: float = 100.0
            The gradient clipping value for the inner-net.
        meta_grad_clip: float = 15.0
            The gradient clipping value for the meta-net.
        
        device: str = "auto"
            The torch device to be used. If "auto" is given, the device will be set to "cuda" if available, else to "cpu".
        """
        #general Q-NN params
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        

        #meta learning params
        self.meta_lr = meta_lr
        self.meta_min_lr = meta_min_lr

        self.meta_grad_clip = meta_grad_clip
        self.curr_task_batch = 0
        self.num_task_batches = num_task_batches
        
        #meta-update related params
        self.meta_transition_frame = meta_transition_frame
        self.transition_sigma = transition_sigma
        self.reptile_decay = reptile_decay
        self.reptile_factor = reptile_factor
        

        #inner learning params
        self.useMetaSampling = useMetaSampling
        self.lossfn = lossFn
        self.inner_lr = inner_lr
        self.tau = tau

        self.num_steps = num_steps
        self.inner_update_rate = update_rate
        self.heuristic_update_inner_lr = heuristic_update_inner_lr
        self.batch_size = batch_size   

        self.memory_size = memory_size
        self.memory = MemoryBuffer(memory_size)
        self.gamma = gamma

        self.penalty_factor = 0.1

        #why should we use thompson sampling when already using NNs? because this is feature engineering, and we want to keep the NNs as simple as possible, to avoid overfitting and to make the meta-learning easier.
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta

        self.inner_grad_clip = inner_grad_clip

        
        #device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.useMetaSampling:
            self.inner_target_net = QNetwork(state_dim + 2*action_dim, action_dim, hidden_dim=self.hidden_dim, layers=layers).to(self.device)
            self.inner_policy_net = QNetwork(state_dim+2*action_dim, action_dim, self.hidden_dim, layers).to(self.device)
            
            self.meta_net = QNetwork(state_dim + 2*action_dim, action_dim, hidden_dim=self.hidden_dim, layers=layers).to(self.device)
            
        else:
            self.inner_target_net = QNetwork(state_dim, action_dim, self.hidden_dim, layers).to(self.device)
            self.inner_policy_net = QNetwork(state_dim, action_dim, self.hidden_dim, layers).to(self.device)

            self.meta_net = QNetwork(state_dim, action_dim, self.hidden_dim, layers).to(self.device)
            
        self.inner_policy_net.load_state_dict(self.inner_target_net.state_dict())
        self.meta_net.load_state_dict(self.inner_target_net.state_dict())
        
        self.meta_optimizer = optim.Adam(self.meta_net.parameters(), lr=self.meta_lr)
        self.meta_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.meta_optimizer, T_max=self.num_task_batches, eta_min=self.meta_min_lr)

        #initialize the sampling
        self.alpha = [self.initial_alpha for _ in range(action_dim)]
        self.beta = [self.initial_beta for _ in range(action_dim)]

    def reset_sampling(self):
        self.alpha = [self.initial_alpha for _ in range(self.action_dim)]
        self.beta = [self.initial_beta for _ in range(self.action_dim)]

    def reset_memory(self):
        self.memory.buffer = list() #new list()

    def inner_lr_heuristic(self)->float:
        """Returns the heuristic learning rate for the inner-net based on the current epoch and the total epochs."""

        #TODO implement smart heuristic, otherwise use dumb heuristic
        if self.heuristic_update_inner_lr:
            raise NotImplementedError("The heuristic update for the inner learning rate is not implemented yet.")
        else:
            factor = self.meta_lr / self.inner_lr
        return self.meta_lr_scheduler.get_lr()[0] * factor

    def inner_update(self, pop_buffer:bool, ret_gradient_ptr:Optional[List[torch.Tensor]]=None, apply_update:bool=True)->torch.Tensor:
        """Updates the inner-net based on the memory buffer.
        
        This function combines the loss calculation and the inner update"""
        batch = self.memory.sample(self.batch_size, pop=pop_buffer)
        states, ab, actions, rewards, next_states, dones = (torch.stack(el).to(self.device) for el in zip(*batch))

        #==q_values
        if self.useMetaSampling:
            state_action_values = self.inner_policy_net(torch.cat([
                torch.tensor([states]).to(self.device), 
                torch.tensor([*list(ab.flatten())]).to(self.device)
            ])).gather(1, actions)
        else:
            state_action_values = self.inner_policy_net(states).gather(1, actions)

        #==target_q_values
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.stack([s for s in next_states if s is not None]).to(self.device)
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        with torch.no_grad():
            if self.useMetaSampling:
                # we should pass enhanced alpha, beta values to the inner-net, 
                # but we just pass the same, as this most likely does not change much
                next_state_values[non_final_mask] = self.inner_target_net(torch.cat([
                    torch.tensor([non_final_next_states]).to(self.device), 
                    torch.tensor([*list(ab.flatten())]).to(self.device)
                ])).max(1).values
            else:
                next_state_values[non_final_mask] = self.inner_target_net(non_final_next_states).max(1).values 

        expected_state_action_values = rewards + ((~dones) * self.gamma * next_state_values) 

        # compute loss
        loss = self.lossfn()(state_action_values, expected_state_action_values.unsqueeze(1))

        #from this part on it's basically https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py#L83
        self.inner_policy_net.zero_grad()
        grads = torch.autograd.grad(loss, self.inner_policy_net.parameters(), create_graph=True, allow_unused=True)
        weights = self.inner_policy_net.state_dict()

        grads = tuple(grad.clamp(-self.inner_grad_clip, self.inner_grad_clip) for grad in grads)
        #apply the new weights with an SGD step to the inner-net
        if apply_update:
            self.inner_policy_net.load_state_dict({name: param - self.inner_lr * grad for ((name, param), grad) in zip(weights.items(), grads)})

            #also update the target net
            policy_state_dict = self.inner_policy_net.state_dict()
            target_state_dict = self.inner_target_net.state_dict()

            for key in target_state_dict.keys():
                target_state_dict[key] = policy_state_dict[key]*self.tau + target_state_dict[key]*(1-self.tau)

        if ret_gradient_ptr !=None:
            ret_gradient_ptr.append(grads) #just append the gradients to the list, we can access the list like a pointer to the next empty element in the buffer

        return loss
    

    def compute_importances(self, task_losses:int, curr_epoch:int=-1)->torch.FloatTensor:
        """Computes the importances of the individual losses computed in the current epoch.

        Args:
        tasks: int
            The number of tasks.
        task_losses: int
            the number of losses in the task
        curr_epoch: int = -1
            The current epoch. If not given, the current epoch of the MetaNet is used.
        
        Returns:
        torch.FloatTensor: The importance of the individual losses.

        See the MAML++ paper with the code at https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py for the implementation of the importance computation. This function is the equivalent for get_per_step_loss_importance_vector().
        """
        current_epoch = (self.curr_task_batch if curr_epoch==-1 else curr_epoch)

        #initialize the loss weights, we do it as a matrix, not vector because prettier
        loss_weights:np.ndarray = np.ones((task_losses,)) # size=n_tasks x n_losses
        n_losses = task_losses
        loss_weights /= n_losses #set to uniform distribution

        #decay rate for the loss weights (later epochs have less importance)
        decay_rate = 1.0 / (n_losses * self.num_task_batches)
        min_value_for_non_final_losses = 0.03 / n_losses

        #just the weird iteration in one step
        loss_weights = np.maximum(loss_weights - (current_epoch * decay_rate), min_value_for_non_final_losses)
    
        #compute last element
        curr_value = np.minimum(
            loss_weights[-1] + (current_epoch * (n_losses - 1) * decay_rate),
            1.0 - ((n_losses - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        #retransform into list
        return torch.tensor(loss_weights).to(self.device)


    def select_action(self, state:torch.Tensor)->torch.Tensor:
        """Selects an action based on the state. The action is selected by sampling from the q-values. And then selecting the action with the highest value. If the meta-sampling is used, the inner net also gets the alpha and beta values, otherwise we use thompson sampling and multiply the q-values with the sampled probabilities."""
        if self.useMetaSampling:
            #how do we regularize? 
            # -> for large alpha and beta, it should be close to the thompson sampling, as this is the optimal policy
            # -> for low alpha and beta, it does not need to be close as we want the meta-learning to have an effect
            sampled_values = self.inner_policy_net(torch.cat([#softmax already appied
                torch.tensor([state]).to(self.device), 
                torch.tensor(self.alpha, dtype=torch.float32).to(self.device), 
                torch.tensor(self.beta, dtype=torch.float32).to(self.device)
            ])) 
            #TODO possible use some other thing as softmax if it is unstable?
            
        else:
            #thompson sampling and q-value prediction
            q_values = self.inner_policy_net(state) 
            sampled_values = torch.tensor([np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.action_dim)]).to(self.device) * q_values.detach()
        
        return torch.argmax(sampled_values)
    
    def penalize_termination(self, step:int)->torch.Tensor:
        """Penalize the early termination of the episode, this is necessary to get the meta network to have a consistent loss available. Minimizing the penalty and the actual loss go hand in hand
        
        Args:
        step: int = s
            The step at which the episode was terminated.
        
        Returns:
            torch.Tensor: The penalty for the termination."""

        #based on the last action, we penalize the termination
        #probably the 2nd best action should have been taken, so we penalize, by the square difference of the 2nd best and the best action

        state, ab, action, _, _, _ = self.memory.buffer[-1]
        if self.useMetaSampling:
            q_values:torch.Tensor = self.inner_policy_net(torch.cat([
                torch.tensor([state]).to(self.device), 
                torch.tensor([*list(ab.flatten())]).to(self.device)
            ])).squeeze(0)
        else:
            q_values:torch.Tensor = self.inner_policy_net(state)
        #normalize q_values to with min-max
        q_values = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-8)

        #3 variations: either 2nd best / average of all other actions / ema of loss over all actions

        # 2nd best action
        # scnd_best_action = torch.argsort(q_values, descending=True)[1]
        # penalty = (q_values[action] - q_values[scnd_best_action]).pow(2) + 1e-2 #add a small value to have smoother gradients

        # average of all other actions
        average_penalty = (q_values.sum() - q_values[action]) / (self.action_dim - 1)
        penalty = (q_values[action] - average_penalty).pow(2) + 1e-2 #add a small value to have smoother gradients

        # ema of loss over all actions
        #get all the losses for all actions until step s, then ema them

        return (self.num_steps -step)/self.num_steps * penalty * self.penalty_factor



    @overload
    def inner_loop(self, task: RLTask, train:Literal[True]=True, ret_memory_ptr:Optional[MemoryBuffer]=None, ret_gradient_ptr:Optional[List[torch.Tensor]]=None)->Tuple[float, torch.Tensor]:
        pass

    @overload
    def inner_loop(self, task: RLTask, train:Literal[False]=False, ret_memory_ptr:Optional[MemoryBuffer]=None, ret_gradient_ptr:Optional[List[torch.Tensor]]=None)->Tuple[float, torch.Tensor, torch.FloatTensor]:
        pass
    
    def inner_loop(self, task: RLTask, train:bool=True, ret_memory_ptr:Optional[MemoryBuffer]=None, ret_gradient_ptr:Optional[List[torch.Tensor]]=None)->Tuple[float, torch.Tensor]|Tuple[float, torch.Tensor, torch.FloatTensor]:
        """This function should be used to learn, predict the inner-net. It should return the average score of the tasks, and the average loss of the tasks.
        
        Args:
        task: RLTask
            Consists of an environment which is used a couple of times.
        train: bool = True
            Whether the inner-net should be trained or not.
        ret_memory_ptr: Optional[MemoryBuffer] = None
            If given the memories will be returned in the given memory buffer.
        ret_gradient_ptr: Optional[List[torch.Tensor]] = None
            If given the gradients will be returned into this tensor.
        Returns:
        - Tuple[float, torch.Tensor, List[torch.FloatTensor]|None]: the score and loss of the task, and if train==True the individual losses of the tasks batches.
        """
        total_score:float = 0.0

        indiv_losses:List[torch.Tensor] = []

        if train:
            self.reset_sampling()
        self.reset_memory()

        env_scores = []
        env_losses = []
        env_step = 0
        for env, init_state in (task.train_sample() if train else task.test_sample()):
            score = 0
            duration = 0
            losses = []
            state = torch.tensor(init_state, dtype=torch.float32).to(self.device)
            for s in range(self.num_steps):
                env_step += 1
                duration += 1
                #select action and do it
                action = self.select_action(state) #uses the inner-net to select the action
                next_state, reward, terminated, truncated, _ = env.step(action.numpy()) #take the action in the environment
                next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

                #only do the updating if we are in the training phase
                if train:
                    if reward > 0: # Update Thompson Sampling parameters, each round, unlike the net-update which is sparse
                        self.alpha[action] += 1  # Rewarded actions increase alpha
                    else:
                        self.beta[action] += 1  # Unsuccessful actions increase beta

                #store the experience and update sampling and the net if necessary
                self.memory.append((state, [e for e in self.alpha+self.beta], action, reward, next_state, terminated)) #store the experience

                #TODO: there should be a heuristic for the inner_update_rate to lower it if the episodes die to fast, to enhance learning.
                #when enough in the memory buffer, and in an update round, make the gradients and possibly update the inner-net 
                # or during testing, just get the gradients each round.
                if env_step>=self.batch_size and ((env_step % self.inner_update_rate == 0) or not train):
                    #update the inner-net, and pop the selected indices from the buffer if we have enough samples: 3x the batch size in the current buffer
                    batch_loss = self.inner_update(
                        pop_buffer = (
                            env_step - (self.batch_size*env_step/self.inner_update_rate) 
                            > 3*self.batch_size
                        ), #maybe we should not pop the buffer?
                        ret_gradient_ptr=ret_gradient_ptr,
                        apply_update=train
                    )
                    losses.append(batch_loss*self.inner_update_rate)
                    indiv_losses.append(batch_loss*self.inner_update_rate)

                #prepare for the next step
                state = next_state
                score += reward
                if terminated:#penalize the agent for terminating the episode
                    penalty = self.penalize_termination(step=s)
                    break
                if truncated:
                    break
            
            #show current performance in the environment: the duration of the epsiode
            print(f"\rCurrent Episode Duration: {duration} / {self.num_steps}; Num Updates: {len(indiv_losses)}", end="")
                

            env_scores.append(score)
            env_losses.append(torch.stack(losses).sum() if len(losses) > 0 else torch.tensor(500).to(self.device))

            if ret_memory_ptr is not None:
                ret_memory_ptr.buffer = deepcopy(self.memory.buffer)

        total_score = sum(env_scores)
        if len(indiv_losses) == 0 and not train:#if no losses and test, we have a problem
            raise RuntimeError("No losses were computed. Increase the number of episodes per environment.")
        return (total_score, torch.stack(env_losses).mean()) if train else (score, torch.stack(env_losses).mean(), torch.stack(indiv_losses))
    
    

        
    def meta_learn(self, task:RLTask|List[RLTask])->Tuple[float, float]:
        """This function should be used to learn the meta-net. It should return the average score of the tasks, and the average loss of the tasks.
        Args:
        task: RLTask | List[RLTask]
            The task to be learned. Consists of an environment which is reset when the task is sampled.
            If a list of tasks is given, the function will learn from all of them. And apply the meta update after all tasks have been seen.

        Returns:
        Tuple[float, float]: The average score of the tasks, and the average loss of the tasks.
        """
        def compute_soft_weight(U:int, R:int, M:int, sigma:float=10.0):
            """
            Compute the soft weight for Reptile based on the number of inner updates.

            Args:
                U (int): The actual number of inner updates performed.
                R (int): Minimum number of updates required for Reptile to be used.
                M (int): Maximum number of updates for MAML to be used.
                sigma (float): Controls the steepness of the blending. Higher values make the transition sharper.

            Returns:
                float: The weight for Reptile (0 to 1).
            """
            #TODO: theoretically we don't want jumps so we could use these tangents to make it smooth:
            # d(1/(1+exp(-5*(x-0.5))))/(dx) *x = 1/(1+exp(-5*(x-0.5))) (for x->0), symmetric for x->1
            if U < R:
                return 0.0
            elif U > M:
                return 1.0
            else:
                # Smooth transition between R and M
                scale = (U - R) / (M - R)  # Normalize U to range [0, 1]
                return 1.0 / (1.0 + torch.exp(-sigma * (scale - 0.5)))  # Sigmoid-based blend
        
        tasks:List[RLTask] = []
        if isinstance(task, RLTask):
            tasks = [task]
        else:
            tasks = task

        #first make some new memories on the test data
        # test_memories = [MemoryBuffer(self.memory_size) for _ in range(len(tasks))]
        # indiv_test_losses:List[torch.FloatTensor] = [None for _ in range(len(tasks))]
        train_gradients = [] #empty list, we append to it in the inner-loop, use it like a pointer
        test_reward = 0
        test_loss = torch.tensor(0.0).to(self.device)

        task_meta_gradients = []

        #see here on how/why we do the batching like this: https://stackoverflow.com/questions/62067400/understanding-accumulated-gradients-in-pytorch
        self.meta_optimizer.zero_grad(set_to_none=False)#we want it to be 0
        for t, __task in tqdm(enumerate(tasks), "Task i from Current Task Batch", total=len(tasks)):
            #reset the inner-net to the meta-net
            self.inner_policy_net.load_state_dict(self.meta_net.state_dict())
            self.inner_target_net.load_state_dict(self.meta_net.state_dict())

            #this is basically support phase (== adaption during meta-training)
            task_train_score, task_train_loss  = self.inner_loop(__task, train=True, ret_gradient_ptr=train_gradients) #train the inner-net
            
            #now we do the query phase
            task_test_score, task_test_loss, indiv_test_losses = self.inner_loop(__task, train=False)
            test_reward += task_test_score / len(tasks)
            test_loss += task_test_loss / len(indiv_test_losses) / len(tasks)

            #and now we do a really special meta-update that blends reptile and maml.
            #if the number of inner updates is less than b of the transition frame, we do a maml update
            #if it is more than a we do a reptile update,
            #if it is in between, we do a sigmoid blend of the two.
            if len(indiv_test_losses) < self.meta_transition_frame[0]:
                continue #we don't use the task for the update, as it would probably have a negative effect on the meta-net #TODO we should not need this all tasks should be learnable. (in reality some tasks might also not be learnable and the algorithm should ignore them, but that is a whole different problem. How and when to ignore learning tasks.)

            if len(indiv_test_losses) < self.meta_transition_frame[2]:
                loss_weights = self.compute_importances(indiv_test_losses.size(0))
                weighted_losses = torch.sum(loss_weights * indiv_test_losses) / len(tasks)
                weighted_losses.backward()
                
            if len(indiv_test_losses) >= self.meta_transition_frame[1]:
                #reptile decay
                l2_distance = torch.tensor(0.0)
                for meta_param, inner_param in zip(self.meta_net.parameters(), self.inner_policy_net.parameters()):
                    l2_distance += torch.norm(inner_param.data - meta_param.data).pow(2)
                l2_distance = torch.sqrt(l2_distance)
                decay_factor = torch.exp(-self.reptile_decay * l2_distance)
                
                #compute the soft reptile factor. Is ==1 if only reptile, 0<factor<1 if maml and reptile are blended
                w_reptile = compute_soft_weight(
                    len(indiv_test_losses), 
                    self.meta_transition_frame[1], 
                    self.meta_transition_frame[2], 
                    sigma=self.transition_sigma
                )

                for param, meta_param in zip(self.inner_policy_net.parameters(), self.meta_net.parameters()):
                    #if only reptile this makes the meta_param.grad 0 and only the reptile update remains
                    #if both this blends the two updates
                    if meta_param.grad is None:
                        meta_param.grad = torch.zeros_like(meta_param.data, device=self.device)
                    meta_param.grad = (1-w_reptile) * meta_param.grad\
                                    +    w_reptile  * decay_factor * (meta_param.data - param.data / len(tasks))

            self.meta_optimizer.step()

            # if not self.use_reptile: #if using maml, we need to backpropagate the loss through the inner updates
            #     #this is the query phase
                

            #     loss_weights = self.compute_importances(indiv_test_losses.size(0))
            #     weighted_losses = torch.sum(loss_weights * indiv_test_losses) / len(tasks)

            #     weighted_losses.backward()
            
            # else: #use reptile, just compute the difference of the inner-net and the meta-net

            #     for param, meta_param in zip(self.inner_policy_net.parameters(), self.meta_net.parameters()):
            #         meta_param.grad = meta_param.data - param.data / len(tasks)

            __task.env.close()

        self.meta_optimizer.step()

        return test_reward, test_loss


    def meta_test(self, task:RLTask)->Tuple[float, float]:

        """This function should be used to test the meta-net. It should return the average score of the tasks, and the average loss of the tasks.
        Args:
        task: RLTask
            The task to be learned. Consists of an environment which is reset when the task is sampled.
            
        Returns:
        Tuple[float, float]: The average score of the tasks, and the average loss of the tasks.
        """
        #reset the inner-net to the meta-net
        self.inner_policy_net.load_state_dict(self.meta_net.state_dict())
        self.inner_target_net.load_state_dict(self.meta_net.state_dict())

        #this is basically adaption phase
        train_values = self.inner_loop(task, train=True) #train the inner-net

        test_values = self.inner_loop(task, train=False) #test the inner-net

        return test_values[0], test_values[2]


def episode_curriculum(max_episodes:int, min_episodes:int, T_max:int, t:int, less:bool=True)->int:
    """The episode curriculum function, which returns the number of episodes to be done in the current task.
    Args:
    max_episodes: int
        The maximum number of episodes to be done in a task.
    min_episodes: int
        The minimum number of episodes to be done in a task.
    T_max: int
        The maximum number of time steps.
    t: int
        the current time step.
    less: bool = True
        Whether the number of episodes should decrease or increase over time. If True, the number of episodes will decrease over time, if False, the number of episodes will increase over time.
    Returns:
    int: The number of episodes to be done in the current task.
    """
    if t >= 2*T_max/3:# after 2/3 of the time, we should have reached the goal number of episodes and train on them some time.
        if less:
            return min_episodes
        else:
            return max_episodes
    else:
        if less:
            return math.floor(max_episodes - ((max_episodes - min_episodes) * t / (2*T_max/3)))
        else:
            return math.floor(min_episodes + ((max_episodes - min_episodes) * t+1 / (2*T_max/3)))

def learn_meta_net(env_name:str="RandomCartPole", n_tasks:int=1000, task_batch_size:int=10, episodes:int=30, n_steps:int=500, **kwargs):
    """
    Learns a meta and inner RL net for the given environment.
    Args:
    env_name: str
        The name of the environment to be used. The environment should be registered in gym.
    n_tasks: int
        The number of tasks to be generated.
    task_batch_size: int
        The size of the task batches.
    episodes: int
        The number of episodes per task task. This is normally how many times the agent gets to revisit the environment from the ground up. In training, we start with a much higher number of episodes, and then reduce it to a lower number of episodes.
    n_steps: int
        The number of steps per episode. This is how long the agent may interact with the environment, before reset. Should be less equal to the max steps of the environment.
    kwargs: dict
        The keyword arguments to be passed to the MetaNet constructor. 
    """
    gym_env:Env
    if env_name not in possible_envs.keys():
        print(f"Environment {env_name} not found in the meta environments. Assuming it is a normal gym environment.")
        gym_env = gym.make(env_name)
    else:
        gym_env = gym.make(possible_envs[env_name])
    action_dim:int = gym_env.action_space.n
    state_dim:int = gym_env.observation_space.shape[0]
    gym_env.close()
    del gym_env

    meta_net:MetaNet = MetaNet(state_dim, action_dim, num_task_batches=n_tasks, num_steps=n_steps, **kwargs) #total_epochs is task_iterations

    envset = EnvSet(env_name, norm)
    train_task_set:List[RLTask]|List[List[RLTask]]
    if task_batch_size == 1:
        train_task_set:List[RLTask] = [
            generate_task(
                envset.sample(), 
                episodes=episode_curriculum(10*episodes, episodes, n_tasks, t), 
                portion_test=0.5
            ) for t in range(n_tasks)
        ]
    else:
        train_task_set:List[List[RLTask]] = [
            [generate_task(
                envset.sample(), 
                episodes=episode_curriculum(10*episodes, episodes, n_tasks, t), 
                portion_test=0.5
            ) for _ in range(task_batch_size)] for t in range(n_tasks)
        ]
    test_task_set:List[RLTask] = [generate_task(envset.sample(), episodes=episodes, portion_test=0.5) for _ in range(n_tasks)]

    train_scores = []
    test_scores = []

    #wandb init
    
    wandb.init(project="meta-rl-thoml", config=kwargs|{"env_name": env_name, "n_tasks": n_tasks, "episodes": episodes, "n_steps": n_steps})

    #TODO find out whether it helps to pass multiple tasks per learning step: doing a task_batch size > 1?

    #TODO find out whether we should see each task batch multiple times before going to the next task batch. See MAML++ code: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py#L232. 
    # Our current analog is that we sample from the environment many times. But we could also think of this as seeing each task batch multiple times before seeing the next one.
    for t, tasks in tqdm(enumerate(train_task_set), desc="Training MetaNet", total=n_tasks):
        avg_score, avg_loss = meta_net.meta_learn(tasks) #training epoch
        train_scores.append(avg_score)

        #log to wandb
        wandb.log({"meta_train_score": avg_score, "meta_train_loss": avg_loss})

        #simulate one episode
        meta_net.curr_task_batch += 1
        meta_net.meta_lr_scheduler.step()

    #here it does not makes sense to choose multiple tasks at the same time, as we want to see the generalization of the meta-net.
    for t, task in tqdm(enumerate(test_task_set), desc="Testing MetaNet"):
        avg_score, avg_loss = meta_net.meta_test(task) #testing
        test_scores.append(avg_score)

        #log to wandb
        wandb.log({"meta_test_score": avg_score, "meta_test_loss": avg_loss})
    
    return meta_net, train_scores, test_scores
    

def run_experiment():
    #get the cmd line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run the meta-learning experiment.")
    parser.add_argument("--env", type=str, default="RandomCartPole", help="The environment to be used.")
    parser.add_argument("--n_tasks", type=int, default=1000, help="The number of tasks to be generated.")
    parser.add_argument("--use_task_batches", type=bool, default=False, help="Whether to use task batches or not.")
    parser.add_argument("--task_batch_size", type=int, default=10, help="The size of the task batches, only used if use_task_batches is True.")
    parser.add_argument("--update_method", type=str, default="reptile", help="Whether to use second order gradients or not.")
    parser.add_argument("--num_episodes", type=int, default=100, help="The number of episodes to run each environment for.")
    parser.add_argument("--num_steps", type=int, default=500, help="The number of steps to be done in each environment episode.")
    parser.add_argument("--hpt", type=bool, default=False, help="Whether to use hyperparameter tuning or not. All not given, hyperparameters are searched for.")
    parser.add_argument("--config", type=str, default="", help="The json-config file to be used for hyperparameter tuning. See the documentation for the format.")

    args = parser.parse_args()

    if args.hpt:
        #TODO implement hyperparameter tuning
        pass
    
    else:
        if args.config != "":
            #TODO implement config file reading
            pass
        #run the experiment
        meta_net, train_scores, test_scores = learn_meta_net(
            env_name="CartPole-v1", 
            # n_tasks=args.n_tasks, 
            # episodes=args.num_episodes, 
            # n_steps=args.num_steps, 
            # meta_update_method=args.update_method
        )

        # plot the train and test scores
        import matplotlib.pyplot as plt
        plt.plot(train_scores, label="Train Scores")
        plt.plot(test_scores, label="Test Scores")
        plt.legend()
        plt.show()


if __name__ == "__main__":

    run_experiment()
