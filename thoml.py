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
from typing import Dict, Optional, Callable, List, Generator, Tuple, Literal, overload, Any, Iterator
import math, os, pickle

from env_sets import EnvSet, possible_envs
from scipy.stats import norm, uniform
import optuna 
from optuna.pruners import HyperbandPruner
import wandb
import ray
import ray.tune as tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.hyperopt import HyperOptSearch
import ray.train as train#
from ray.air.integrations.wandb import WandbLoggerCallback
# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

from dqn_utils import thompDQNv3 as QNetwork, DQN, ReplayMemoryV2 as MemoryBuffer, RLTask, ThresholdController

def generate_task(env:gym.Env, episodes:int=10, portion_test:float=0.5)->RLTask:
    return RLTask(env, episodes, portion_test)

good_config={
    "batch_size":74,
    "eps":4.919879003318238,
    "eps_moment":0.12401566282338826,
    "gamma":0.96,
    "hidden_dim":177,
    "hidden_layers":2,
    "inner_grad_clip":85.60393678702776,
    "inner_lr":0.0021453528290143317,
    "inner_update_rate":4,
    "memory_size":9600,
    "meta_grad_clip":66.38519245246144,
    "meta_lr":0.013444219134471564,
    "meta_min_lr":0.00004303480102558296,
    "meta_transition_frame1":55,
    "meta_transition_frame2":335,
    "n_tasks":100,
    "reptile_decay":0.26197925796271426,
    "reptile_factor":5.3099177713732795,
    "rho":0.41849685437857065,
    "task_batch_size":1,
    "tau":0.11572959485321996,
    "threshold_init_loss":2,
    "transition_sigma":55.3158334648015,
    "xi":0.14888519417506646,
}

class MetaNet():
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=64, hidden_layers:int=1, 
        meta_lr:float=0.01, meta_min_lr:float=0.0001, meta_grad_clip:float=15.0, num_task_batches:int=1000, 
        meta_transition_frame:Tuple[int, int]=(80, 400), transition_sigma:float=10.0, reptile_decay:float=0.2, reptile_factor:float=2.0,#TODO: how to mix reptile and maml properly
        memory_size:int=10000, batch_size:int=128, gamma:float=0.99, tau:float=0.005, rho:float=0.2, xi:float=0.2, inner_lr:float=1e-4, inner_loss_fn:Callable=torch.nn.SmoothL1Loss(), inner_optimizer:type[optim.Optimizer]=optim.AdamW, inner_optim_kwargs:dict={"amsgrad":True}, inner_grad_clip:float=100., inner_update_rate:int=8, eps:float=0.6, eps_moment:float=0.1, threshold_init_loss:float=20.0, heuristic_update_inner_lr:bool=False,
        device="auto", **kwargs
    ):
        #num_episodes=500, memory_size:int=10000, batch_size=128, gamma=0.99, tau=0.005, rho=0.2, xi=0.2, lr=1e-4, loss_fn:Callable=torch.nn.SmoothL1Loss(), optimizer:type[optim.Optimizer]=optim.AdamW, optim_kwargs:dict={"amsgrad":True}, grad_clip:float=100., update_rate:int=1, hidden_layers:int=1, hidden_dim:int=128, eps=0.6, eps_moment:float=0.1, threshold_init_loss:float=20.0
        """A class for organizing all the MetaNet learning functionalities.
        Args:
        state_dim: int
            The dimension of the state space.
        action_dim: int 
            The dimension of the action space.
        hidden_dim: int = 256
            The dimension of the hidden layers.
        hidden_layers: int = 1
            The number of hidden layers in the Q-NN. 

        meta_lr: float = 0.01
            The learning rate of the meta-net. 
        meta_min_lr: float = 0.0001
            The minimum learning rate of the meta-net, used for the CosineAnnealingLR scheduler.
        num_task_batches: int = 1000
            The total number of task batches that are learned. "Just like an episode".
        meta_transition_frame: Tuple[int, int] = (80, 400)
            Given (a,b): The number of inner updates `u` done for the meta update to behave like MAML (u<a), like reptile (u>=b), or a smooth transition between the two (a<=u<b). The smooth transition is a sigmoid. #TODO: maybe with linear end pieces.
        transition_sigma: float = 10.0
            The sigma parameter for the sigmoid transition between MAML and Reptile.
        reptile_decay:float = 0.2
            The rate of the exponential decay applied to reptile updates based on parameter l2-distance.
        reptile_factor: float = 2.0
            The factor by which the reptile update is multiplied. This is to balance reptile relative to MAML. The overall update strength should be guided by the meta_lr.

        memory_size: int = 10000 
            The size of the memory buffer.
        batch_size: int = 128
            The size of the batch used for the inner-net updates.
        gamma: float = 0.99
            The discount factor.
        tau: float = 0.005
            The factor for the soft update of the target net.
        rho: float = 0.2
            The factor for the residual skip connection around the self-attention.
        xi: float = 0.2
            The factor for the projectional skip connection. 
        inner_lr: float = 1e-4
            The learning rate of the inner-net. If heuristic_update_inner_lr is True, the learning rate is updated based on the meta lr
        inner_loss_fn: Callable = torch.nn.SmoothL1Loss()
            The loss function used for the inner-net.
        inner_optimizer: type[optim.Optimizer] = optim.AdamW
            The optimizer used for the inner-net.
        inner_optim_kwargs: dict = {}
            The keyword arguments for the optimizer.
        inner_grad_clip: float = 100.
            The gradient clipping value for the inner-net.
        inner_update_rate: int = 8
            The number of step in the environment before the inner-net is updated. Can reduce the number of necessary updates to learn. Useful to allow maml to work.
        inner_hidden_layers: int = 1
            The number of hidden layers in the inner-net.
        eps: float = 0.6
            The epsilon value for the inner exploration threshold controller 
        eps_moment: float = 0.1
            The momentum for the threshold controller.
        threshold_init_loss: float = 20.0
            The initial loss value for the threshold controller. The threshold controller uses this as initial value to the EMA. If not given, the ema starts with the actual loss only.
        heuristic_update_inner_lr: bool = False
            Whether the inner learning rate should be updated based on the meta learning rate. If set, the inner_lr is ignored.
  
        device: str = "auto"
            The torch device to be used. If "auto" is given, the device will be set to "cuda" if available, else to "cpu".
        """
        #general Q-NN params
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        

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
        self.lossfn = inner_loss_fn
        self.inner_lr = inner_lr
        self.tau = tau
        self.rho = rho
        self.xi = xi

        self.inner_update_rate = inner_update_rate
        self.heuristic_update_inner_lr = heuristic_update_inner_lr
        self.batch_size = batch_size   

        self.eps = eps
        self.eps_moment = eps_moment
        self.threshold_init_loss = threshold_init_loss

        self.memory_size = memory_size
        self.memory = MemoryBuffer(memory_size)
        self.gamma = gamma

        self.penalty_factor = 0.1

        #why should we use thompson sampling when already using NNs? because this is feature engineering, and we want to keep the NNs as simple as possible, to avoid overfitting and to make the meta-learning easier.
        self.initial_alpha = 1.0
        self.initial_beta = 1.0

        self.inner_grad_clip = inner_grad_clip

        
        #device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # self.inner_net = QNetwork() will get initialised each time in the inner loop
        self.inner_net = QNetwork(state_dim, action_dim, num_episodes=500, memory_size=self.memory_size, batch_size=self.batch_size, gamma=self.gamma, tau=self.tau, rho=self.rho, xi=self.xi, lr=self.inner_lr, loss_fn=self.lossfn, optimizer=inner_optimizer, optim_kwargs=inner_optim_kwargs, grad_clip=self.inner_grad_clip, update_rate=self.inner_update_rate, hidden_layers=self.hidden_layers, hidden_dim=self.hidden_dim, eps=self.eps, eps_moment=self.eps_moment, threshold_init_loss=self.threshold_init_loss)#
        self.inner_optim_kwargs = inner_optim_kwargs
        self.inner_optimizer = inner_optimizer
        self.inner_net.policy_net.to(self.device)
        self.inner_net.target_net.to(self.device)
        self.meta_net = DQN(state_dim + 2*action_dim, action_dim, hidden_dim=self.hidden_dim, hidden_layers=self.hidden_layers).to(self.device)

        self.inner_net.policy_net.load_state_dict(self.meta_net.state_dict())
        self.inner_net.target_net.load_state_dict(self.meta_net.state_dict())
        
        self.meta_optimizer = optim.Adam(self.meta_net.parameters(), lr=self.meta_lr)
        self.meta_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.meta_optimizer, T_max=self.num_task_batches, eta_min=self.meta_min_lr)

        self.progress:float = 0.0

    def reset_model(self, num_episodes:int=500):
        #reset model weights to the meta net'S
        self.inner_net.policy_net.load_state_dict(self.meta_net.state_dict())
        self.inner_net.target_net.load_state_dict(self.meta_net.state_dict())
        #reset thompson
        self.inner_net.alpha = torch.tensor([1.0 for _ in range(self.action_dim)], device=device, dtype=torch.float)
        self.inner_net.beta = torch.tensor([1.0 for _ in range(self.action_dim)], device=device, dtype=torch.float)
        #reset optimizer and lr-scheduler
        self.inner_net.optimizer = self.inner_optimizer(self.inner_net.policy_net.parameters(), **self.inner_optim_kwargs|{"lr":self.inner_lr_heuristic()})
        self.inner_net.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.inner_net.optimizer, T_max=num_episodes, eta_min=self.inner_lr_heuristic()/100)

        #reset the threshold controller
        self.inner_net.threshold_controller = ThresholdController(self.eps, self.eps_moment, max(1, self.threshold_init_loss*(1-self.progress)))


    def reset_memory(self):
        self.memory.reset()

    def inner_lr_heuristic(self)->float:
        """Returns the heuristic learning rate for the inner-net based on the current epoch and the total epochs."""

        #TODO implement smart heuristic, otherwise use dumb heuristic
        if self.heuristic_update_inner_lr:
            raise NotImplementedError("The heuristic update for the inner learning rate is not implemented yet.")
        else:
            factor = self.inner_lr / self.meta_lr
        return self.meta_lr_scheduler.get_lr()[0] * factor
    

    def compute_importance(self, curr_epoch:int=-1)->torch.FloatTensor:
        """Computes the importances of the individual losses computed in the current epoch.

        Args:
        curr_epoch: int = -1
            The current epoch. If not given, the current epoch of the MetaNet is used.
        
        Returns:
        torch.FloatTensor: The importance of the current loss. It is based on the episode / task batch number

        See the MAML++ paper with the code at https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py for the implementation of the importance computation. This function is the equivalent for get_per_step_loss_importance_vector().
        """
        current_epoch = (self.curr_task_batch if curr_epoch==-1 else curr_epoch)

        #initialize the loss weight
        loss_weight = 1.0 

        #decay rate for the loss weights (later epochs have less importance)
        decay_rate = 1.0 / self.num_task_batches
        min_value_for_non_final_losses = 0.03

        #just the weird iteration in one step
        loss_weight = np.maximum(loss_weight - (current_epoch * decay_rate), min_value_for_non_final_losses)
    
        #compute last element
        # curr_value = np.minimum(
        #     loss_weights[-1] + (current_epoch * (n_losses - 1) * decay_rate),
        #     1.0 - ((n_losses - 1) * min_value_for_non_final_losses))
        # loss_weights[-1] = curr_value
        #retransform into list
        return torch.tensor(loss_weight).to(self.device)
    
    # def penalize_termination(self, step:int)->torch.Tensor:
    #     """Penalize the early termination of the episode, this is necessary to get the meta network to have a consistent loss available. Minimizing the penalty and the actual loss go hand in hand
        
    #     Args:
    #     step: int = s
    #         The step at which the episode was terminated.
        
    #     Returns:
    #         torch.Tensor: The penalty for the termination."""

    #     #based on the last action, we penalize the termination
    #     #probably the 2nd best action should have been taken, so we penalize, by the square difference of the 2nd best and the best action

    #     state, ab, action, _, _, _ = self.memory.buffer[-1]
    #     if self.useMetaSampling:
    #         q_values:torch.Tensor = self.inner_policy_net(torch.cat([
    #             torch.tensor([state]).to(self.device), 
    #             torch.tensor([*list(ab.flatten())]).to(self.device)
    #         ])).squeeze(0)
    #     else:
    #         q_values:torch.Tensor = self.inner_policy_net(state)
    #     #normalize q_values to with min-max
    #     q_values = (q_values - q_values.min()) / (q_values.max() - q_values.min() + 1e-8)

    #     #3 variations: either 2nd best / average of all other actions / ema of loss over all actions

    #     # 2nd best action
    #     # scnd_best_action = torch.argsort(q_values, descending=True)[1]
    #     # penalty = (q_values[action] - q_values[scnd_best_action]).pow(2) + 1e-2 #add a small value to have smoother gradients

    #     # average of all other actions
    #     average_penalty = (q_values.sum() - q_values[action]) / (self.action_dim - 1)
    #     penalty = (q_values[action] - average_penalty).pow(2) + 1e-2 #add a small value to have smoother gradients

    #     # ema of loss over all actions
    #     #get all the losses for all actions until step s, then ema them

    #     return (self.num_steps -step)/self.num_steps * penalty * self.penalty_factor

    @overload
    def inner_loop(self, task: RLTask, train:Literal[True]=True)->Tuple[float, List[torch.FloatTensor]]:
        pass

    @overload
    def inner_loop(self, task: RLTask, train:Literal[False]=False)->Tuple[float, torch.FloatTensor]:
        pass
    
    def inner_loop(self, task: RLTask, progress:float=0.0, train:bool=True)->Tuple[float, List[torch.FloatTensor]]|Tuple[float, torch.FloatTensor]:
        """This function should be used to learn, predict the inner-net. It should return the average score of the tasks, and the average loss of the tasks.
        Args:
        task: RLTask
            Consists of an environment which is used a couple of times.
        train: bool = True
            Whether the inner-net should be trained or predicted.
        Returns:
        - Tuple[float, torch.Tensor, List[torch.FloatTensor]|None]: the score and loss of the task, and if train==True the individual losses of the tasks batches.
        """
        total_score:float = 0.0

        indiv_losses:List[torch.Tensor] = []

        if train:
            self.reset_model(num_episodes=task.size)
        self.reset_memory()

        env_scores = []
        env_losses = []
        indiv_losses = []
        for env in (task.train_sample() if train else task.test_sample()):
            
            if train:
                score, losses = self.inner_net.train(env, num_eps=1, return_indiv_losses=True, ignore_episodes=True)
                env_scores.append(score)
                indiv_losses.extend(losses)
            else:
                score, loss = self.inner_net.predict(env)
                env_scores.append(score)
                env_losses.append(loss)

        total_score = sum(env_scores) / len(env_scores)
        return (total_score, indiv_losses) if train else (total_score, torch.sum(torch.stack(env_losses)))
        
    def meta_learn(self, task:RLTask|List[RLTask])->Tuple[float, torch.Tensor]:
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
                return 1.0 / (1.0 + np.exp(-sigma * (scale - 0.5)))  # Sigmoid-based blend
        
        tasks:List[RLTask] = []
        if isinstance(task, RLTask):
            tasks = [task]
        else:
            tasks = task

        print("This task has", tasks[0].size, "episodes!")

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
            
            #this is basically support phase (== adaption during meta-training) 
            #also reset the inner-net to the meta-net
            task_train_score, indiv_losses = self.inner_loop(__task, train=True) #train the inner-net
            
            #now we do the query phase
            task_test_score, task_test_loss = self.inner_loop(__task, train=False)
            test_reward += task_test_score / len(tasks)
            test_loss += task_test_loss / len(tasks)

            #and now we do a really special meta-update that blends reptile and maml.
            #if the number of inner updates is less than b of the transition frame, we do a maml update
            #if it is more than a we do a reptile update,
            #if it is in between, we do a sigmoid blend of the two.
            print(f"Task {t}/{len(tasks)} - Num. inner updates was:", len(indiv_losses))
            if len(indiv_losses) < self.meta_transition_frame[1]:
                loss_weight = self.compute_importance()
                weighted_losses = loss_weight * task_test_loss / len(tasks)
                weighted_losses.backward()
                
            if len(indiv_losses) >= self.meta_transition_frame[0]:
                #reptile decay
                l2_distance = torch.tensor(0.0).to(self.device)
                for meta_param, inner_param in zip(self.meta_net.parameters(), self.inner_net.policy_net.parameters()):
                    l2_distance += torch.norm(inner_param.data - meta_param.data).pow(2)
                l2_distance = torch.sqrt(l2_distance)
                decay_factor = torch.exp(-self.reptile_decay * l2_distance)
                
                #compute the soft reptile factor. Is ==1 if only reptile, 0<factor<1 if maml and reptile are blended
                w_reptile = compute_soft_weight(
                    len(indiv_losses), 
                    self.meta_transition_frame[0], 
                    self.meta_transition_frame[1], 
                    sigma=self.transition_sigma
                )

                for param, meta_param in zip(self.inner_net.policy_net.parameters(), self.meta_net.parameters()):
                    #if only reptile this makes the meta_param.grad 0 and only the reptile update remains
                    #if both this blends the two updates
                    if meta_param.grad is None:
                        meta_param.grad = torch.zeros_like(meta_param.data, device=self.device)
                    meta_param.grad = (1-w_reptile) * meta_param.grad\
                                    +    w_reptile  * decay_factor * (meta_param.data - param.data / len(tasks))

            self.meta_optimizer.step()

            __task.env.close()

        self.meta_optimizer.step()

        return test_reward, test_loss


    def meta_test(self, task:RLTask)->Tuple[float, torch.FloatTensor]:

        """This function should be used to test the meta-net. It should return the average score of the tasks, and the average loss of the tasks.
        Args:
        task: RLTask
            The task to be learned. Consists of an environment which is reset when the task is sampled.
            
        Returns:
        Tuple[float, float]: The average score of the tasks, and the average loss of the tasks.
        """
        #reset the inner-net to the meta-net
        self.inner_net.policy_net.load_state_dict(self.meta_net.state_dict())
        self.inner_net.target_net.load_state_dict(self.meta_net.state_dict())

        #this is basically adaption phase
        train_values = self.inner_loop(task, train=True) #train the inner-net

        test_values = self.inner_loop(task, train=False) #test the inner-net

        return test_values[0], test_values[1]


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

def learn_meta_net(env_name:str="RandomCartPole", n_tasks:int=1000, task_batch_size:int=10, episodes:int=50, n_steps:int=500, **kwargs):
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
    if env_name not in possible_envs:
        print(f"Environment {env_name} not found in the meta environments. Assuming it is a normal gym environment.")
        gym_env = gym.make(env_name)
    else:
        gym_env = gym.make(possible_envs[env_name])
    action_dim:int = gym_env.action_space.n
    state_dim:int = gym_env.observation_space.shape[0]
    gym_env.close()
    del gym_env

    meta_net:MetaNet = MetaNet(state_dim, action_dim, num_task_batches=kwargs.get("num_task_batches", n_tasks), **{k:v for k,v in kwargs.items() if not k=="num_task_batches"}) #total_epochs is task_iterations

    envset = EnvSet(env_name, norm)
    train_task_set:List[RLTask]|List[List[RLTask]]
    if task_batch_size == 1:
        train_task_set:List[RLTask] = [
            generate_task(
                envset.sample(), 
                episodes=episode_curriculum(3*2*episodes, 2*episodes, n_tasks, t), 
                portion_test=0.5
            ) for t in range(n_tasks)
        ]
    else:
        train_task_set:List[List[RLTask]] = [
            [generate_task(
                envset.sample(), 
                episodes=episode_curriculum(3*2*episodes, 2*episodes, n_tasks, t), 
                portion_test=0.5
            ) for _ in range(task_batch_size)] for t in range(n_tasks)
        ]
    test_task_set:List[RLTask] = [generate_task(envset.sample(), episodes=2*episodes, portion_test=0.5) for _ in range(n_tasks)]
    # test_task_set:List[List[RLTask]] = [[generate_task(envset.sample(), episodes=2*episodes, portion_test=0.5) for _ in range(5)] for _ in range(n_tasks)]
    train_scores = []
    test_scores = []

    #wandb init
    
    wandb.init(project="reptile-thomp-dqn", config=kwargs|{"env_name": env_name, "n_tasks": n_tasks, "episodes": episodes, "n_steps": n_steps})

    #TODO find out whether we should see each task batch multiple times before going to the next task batch. See MAML++ code: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py#L232. 
    # Our current analog is that we sample from the environment many times. But we could also think of this as seeing each task batch multiple times before seeing the next one.
    for t, (train_tasks, test_task) in tqdm(enumerate(zip(train_task_set, test_task_set)), desc="Training&Testing MetaNet", total=n_tasks):


        avg_score, avg_loss = meta_net.meta_learn(train_tasks) #training epoch
        train_scores.append(avg_score)

        #log to wandb
        wandb.log({"train_score": avg_score, "meta_train_loss": avg_loss}, step=t)
        #simulate one episode
        # meta_net.curr_task_batch += 1
        # meta_net.meta_lr_scheduler.step()

        #testing
        # test_score = 0
        # test_loss = 0
        # for i, tt in enumerate(test_batch):
        #     avg_score, avg_loss = meta_net.meta_test(tt)
        #     test_scores.append(avg_score)
        #     test_score += avg_score
        #     test_loss += avg_loss
        
        # test_score /= len(test_batch)
        # test_loss /= len(test_batch)

        test_score, test_loss = meta_net.meta_test(test_task) #testing
        test_scores.append(test_score)

        #log to wandb
        wandb.log({"test_score": test_score, "meta_test_loss": test_loss}, step=t)

        meta_net.progress = t / n_tasks
    
    return meta_net, train_scores, test_scores

@ray.remote(num_cpus=1)
def learn_meta_net_remote(env_name:str="RandomCartPole", n_tasks:int=1000, task_batch_size:int=10, episodes:int=50, n_steps:int=500, **kwargs):
    return learn_meta_net(env_name, n_tasks, task_batch_size, episodes, n_steps, **kwargs)

ray.init(num_cpus=11)

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
        print("Please use thoml_opt.py for hyperparameter tuning.\nThis functionality is deprecated.")
    
    else:
        if args.config != "":
            #TODO implement config file reading
            pass
        #run the experiment
        # meta_net, train_scores, test_scores = learn_meta_net(
        #     env_name="CartPole-v1", 
        #     **good_config
        # )

        learners = [learn_meta_net_remote.remote(env_name="CartPole-v1", **good_config) for _ in range(10)]
        results = ray.get(learners)
        train_scores = [r[1] for r in results]
        test_scores = [r[2] for r in results]

        #transpose the double lists
        train_scores = list(map(list, zip(*train_scores)))
        test_scores = list(map(list, zip(*test_scores)))

        os.makedirs("out", exist_ok=True)

        #write to csv, the learners side by side, the values in the columns
        with open("out/train_scores.csv", "w+") as f:
            for i in range(len(train_scores)):
                f.write(",".join([str(s) for s in train_scores[i]]) + "\n")

        with open("out/test_scores.csv", "w+") as f:
            for i in range(len(test_scores)):
                f.write(",".join([str(s) for s in test_scores[i]]) + "\n")

        # plot the train and test scores
        import matplotlib.pyplot as plt
        plt.plot(train_scores, label="Train Scores")
        plt.plot(test_scores, label="Test Scores")
        plt.legend()
        plt.show()


if __name__ == "__main__":

    run_experiment()
