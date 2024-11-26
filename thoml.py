import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque, OrderedDict
from tqdm import tqdm

from typing import Dict, Optional, Callable, List, Generator
import math

from env_sets import EnvSet, possible_envs
from scipy.stats import norm, uniform

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, layers=3):
        super(QNetwork, self).__init__()
        if layers < 2:
            raise ValueError("Number of layers should be at least 2")
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if layers == 2:
            self.fc2 = nn.Identity(hidden_dim, hidden_dim)
        else:
            self.fc2 = nn.Sequential(*[nn.Linear(hidden_dim, hidden_dim) for _ in range(layers-2)])

        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Outputs Q-values for each action

class RLTask:
    def __init__(self, env:gym.Env, episodes:int, portion_test:float):
        self.env:gym.Env = env
        self.size:int = episodes
        self.portion_test:float = portion_test
        self.train_set:List[int] = [i for i in range(math.ceil(episodes*portion_test))]
        self.test_set:List[int] = [i for i in range(math.ceil(episodes*portion_test), episodes)]

    
    def sample_train(self)->Generator[gym.Env, None, None]:
        for task in self.train_set:
            self.env.reset(seed=task)
            yield self.env

    def sample_test(self)->Generator[gym.Env, None, None]:
        for task in self.test_set:
            self.env.reset(seed=task)
            yield self.env

def generate_task(env:gym.Env, episodes:int=10, portion_test:float=0.5)->RLTask:
    return RLTask(env, episodes, portion_test)

class MetaNet():
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=256, layers:int=3, 
        meta_lr:float=0.001, gamma:float=0.99, target_update_freq=10,
        inner_lr:float=0.0001, memory_size=10000, batch_size=64,
        initial_alpha=1.0, initial_beta=1.0, device="auto"
    ):
        #general Q-NN params
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta

        #meta learning params
        self.meta_lr = meta_lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        #inner learning params
        self.inner_lr = inner_lr
        self.memory = deque(maxlen=memory_size)

        #device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.meta_q_net = QNetwork(state_dim, action_dim, hidden_dim, layers).to(device)
        self.meta_optimizer = optim.Adam(self.meta_q_net.parameters(), lr=self.meta_lr)

        self.inner_q_net = QNetwork(state_dim, action_dim, hidden_dim, layers).to(device)
        self.inner_optimizer = optim.Adam(self.inner_q_net.parameters(), lr=self.inner_lr)

        #initialize the sampling
        self.alpha = [initial_alpha for _ in range(action_dim)]
        self.beta = [initial_beta for _ in range(action_dim)]

    def learn(self, task:RLTask|List[RLTask], n_steps:int)->Tuple[float, float]:
        """This function should be used to learn the meta-net. It should return the average score of the tasks, and the average loss of the tasks.
        Args:
        task: RLTask | List[RLTask]
            The task to be learned. Consists of an environment which is reset when the task is sampled.
            If a list of tasks is given, the function will learn from all of them. And apply the meta update after all tasks have been seen.
        n_steps: int
            The number of steps to be taken in the environment sample == in each episode.
        """
        total_score = 0

        if isinstance(task, RLTask):
            tasks = [task]
        else:
            tasks = task

        for task_ in tasks:
            for env in task_.sample_train():
                score = 0
                for step in range(n_steps):
                    action = self.sample_action(state)
                    next_state, reward, done, _ = env.step(action)
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                    self.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    score += reward
                    if done:
                        break
                total_score += score
        
        self.update()

        return total_score/task.size, 0




def learn_meta_net(env_name:str="RandomCartPole", n_tasks:int=100, episodes:int=10, n_steps:int=500, **kwargs):
    """
    Learns a meta and inner RL net for the given environment.
    Args:
    env_name: str
        The name of the environment to be used. The environment should be registered in gym.
    n_tasks: int
        The number of tasks to be generated.
    episodes: int
        The number of episodes per task. This is normally how many times the agent gets to revisit the environment from the ground up.
    n_steps: int
        The number of steps per episode. This is how long the agent may interact with the environment, before reset.
    kwargs: dict
        The keyword arguments to be passed to the MetaNet constructor. 
    """
    gym_env = gym.make(possible_envs[env_name])
    action_dim:int = gym_env.action_space.n
    state_dim:int = gym_env.observation_space.shape[0]

    meta_net:MetaNet = MetaNet(state_dim, action_dim, **kwargs)

    envset = EnvSet(env_name, norm)
    train_task_set:List[RLTask] = [generate_task(envset.sample(), episodes=episodes, portion_test=0.5) for _ in range(n_tasks)]
    test_task_set:List[RLTask] = [generate_task(envset.sample(), episodes=episodes, portion_test=0.5) for _ in range(n_tasks)]

    train_scores = []
    test_scores = []

    for t, task in tqdm(enumerate(train_task_set), desc="Training MetaNet"):
        avg_score, _ = meta_net.learn(task, n_steps) #training
        train_scores.append(avg_score)

    for t, task in tqdm(enumerate(test_task_set), desc="Testing MetaNet"):
        avg_score, _ = meta_net.search(task, n_steps) #testing
        test_scores.append(avg_score)
    
    return meta_net, train_scores, test_scores
    