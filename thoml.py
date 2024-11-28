from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque, OrderedDict
from tqdm import tqdm

from typing import Dict, Optional, Callable, List, Generator, Tuple
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

        #we introduce a transform skip connection to allow for gradients to flow easier to the maml network during it's update.
        self.skip_transform = nn.Linear(state_dim, action_dim) 

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x) + self.skip_transform(state) # Outputs Q-values for each action
    
class MemoryBuffer:
    # A simple memory buffer to store experiences
    def __init__(self, max_size:int):
        self.max_size = max_size
        self.buffer = list() #new list()
    
    def append(self, experience:Tuple[torch.Tensor, int, float, torch.Tensor, bool]):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size:int, pop:bool=False)->List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer.pop(i) for i in indices] if pop else [self.buffer[i] for i in indices]

class RLTask:
    def __init__(self, env:gym.Env, episodes:int, portion_test:float):
        self.env:gym.Env = env
        self.size:int = episodes
        self.portion_test:float = portion_test
        self.train_set:List[int] = [i for i in range(math.ceil(episodes*portion_test))]
        self.test_set:List[int] = [i for i in range(math.ceil(episodes*portion_test), episodes)]

    
    def train_sample(self)->Generator[Tuple[gym.Env, gym.ObsState], None, None]:
        """Samples the environment of the task.
            For that it resets the environment with a new seed, and returns the environment and the initial state.
        """
        for task in self.train_set:
            state, _ = self.env.reset(seed=task)
            yield self.env, state

    def test_sample(self)->Generator[Tuple[gym.Env, gym.ObsState], None, None]:
        """Samples the environment of the task.
            For that it resets the environment with a new seed, and returns the environment and the initial state.
        """
        for task in self.test_set:
            state, _ = self.env.reset(seed=task)
            yield self.env, state

def generate_task(env:gym.Env, episodes:int=10, portion_test:float=0.5)->RLTask:
    return RLTask(env, episodes, portion_test)

class MetaNet():
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=256, layers:int=3, 
        meta_lr:float=0.01, meta_min_lr:float=0.0001, total_epochs:int=100, use_second_order:bool=False,
        gamma:float=0.99, lossFn:Callable=nn.SmoothL1Loss, inner_lr:float=0.0001, heuristic_update_inner_lr:bool=False, 
        update_rate:int=50, batch_size:int=64, memory_size:int=10000,
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
        total_epochs: int = 100
            The total number of epochs for the meta-net.
        use_second_order: bool = False
        gamma: float = 0.99
            The discount factor.
        lossFn: Callable = nn.SmoothL1Loss
        inner_lr: float = 0.0001
            The learning rate of the inner-net. 
        heuristic_update_inner_lr: bool = False
            Whether to use the heuristic to update the inner-net learning rate or not.
            The heuristic is that the inner lr should depend on the outer lr by some factor. This makes sure that while the outer net progresses, the inner net does not overshoot with too high learning rates, but still learns fast in the beginning. #TODO implement inner lr heuristic
        update_rate: int = 50
            The number of steps before the inner-net is updated.
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
        self.current_epoch = 0
        self.total_epochs = total_epochs
        
        self.use_second_order = use_second_order #whether to allow second order gradients to be build up

        #inner learning params
        self.useMetaSampling = useMetaSampling
        self.lossfn = lossFn
        self.inner_lr = inner_lr
        self.inner_update_rate = update_rate
        self.heuristic_update_inner_lr = heuristic_update_inner_lr
        self.batch_size = batch_size   
        self.memory_size = memory_size
        self.memory = MemoryBuffer(memory_size)
        self.gamma = gamma
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta

        self.inner_grad_clip = inner_grad_clip

        
        #device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.useMetaSampling:
            self.inner_q_net = QNetwork(state_dim + 2*action_dim, action_dim).to(self.device)
            
            self.meta_q_net = QNetwork(state_dim + 2*action_dim, action_dim).to(self.device)
            self.meta_optimizer = optim.Adam(self.meta_q_net.parameters(), lr=self.meta_lr)
            
        else:
            self.meta_q_net = QNetwork(state_dim, action_dim, hidden_dim, layers).to(self.device)
            self.meta_optimizer = optim.Adam(self.meta_q_net.parameters(), lr=self.meta_lr)

            self.inner_q_net = QNetwork(state_dim, action_dim, hidden_dim, layers).to(self.device)

        self.meta_q_net.load_state_dict(self.inner_q_net.state_dict())

        self.meta_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.meta_optimizer, T_max=self.total_epochs, eta_min=self.meta_min_lr)


        
        
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

    def inner_update(self, pop_buffer:bool):
        """Updates the inner-net based on the memory buffer.
        
        This function combines the loss calculation and the inner update"""
        batch = self.memory.sample(self.batch_size, pop=pop_buffer)

        states, actions, rewards, next_states, dones = (torch.stack(el).to(self.device) for el in zip(*batch))
        state_action_values = self.inner_q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.stack([s for s in next_states if s is not None]).to(self.device)

        next_state_values = torch.zeros(self.batch_size).to(self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.meta_q_net(non_final_next_states).max(1).values

        expected_state_action_values = rewards + (self.gamma * next_state_values)

        loss = self.lossfn()(state_action_values, expected_state_action_values)


        #from this part on it's basically https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py#L83
        self.inner_q_net.zero_grad()
        grads = torch.autograd.grad(loss, self.inner_q_net.parameters(), create_graph=self.use_second_order, allow_unused=True)
        name_weights = {name :param for name, param in self.inner_q_net.named_parameters()} # make a copy of the weights
        name_grads = {name: grad for name, grad in zip(name_weights, grads) if grad is not None}

        for key, grad in name_grads.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            name_grads[key] = name_grads[key].sum(dim=0)

        #clip the gradients
        name_grads = {name: grad.clamp(-self.inner_grad_clip, self.inner_grad_clip) for name, grad in name_grads.items()}

        #do the gradient descent step with the learning rate (possibly heuristic) and clip the gradients
        named_weights = {name: weight-self.inner_lr_heuristic()*name_grads[name] for name, weight in name_weights.items()}

        #apply the new weights to the inner-net
        for name, param in self.inner_q_net.named_parameters():
            param.data = named_weights[name]

    def compute_importances(self, individual_losses:List[torch.FloatTensor], curr_epoch:int=-1)->List[torch.FloatTensor]:
        """Computes the importances of the individual losses computed in the current epoch.

        Args:
        individual_losses: List[torch.FloatTensor]
            The losses of the individual tasks. The first dimension holds the losses per task.
        curr_epoch: int = -1
            The current epoch. If not given, the current epoch of the MetaNet is used.
        
        Returns:
        List[torch.FloatTensor]: The importance of the individual losses. The list holds the tasks. The second dimension holds the importance of the losses of the task.

        See the MAML++ paper with the code at https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py for the implementation of the importance computation. This function is the equivalent for get_per_step_loss_importance_vector().

        #TODO this is a bit weird as the weights are same among the whole epoch. So we could just do this in one line. But we keep it like this for now. See the bottom comments of the function
        """
        current_epoch = (self.current_epoch if curr_epoch==-1 else curr_epoch)

        #initialize the loss weights, we do it as a matrix, not vector because prettier
        loss_weights:np.ndarray = np.ones((len(individual_losses), max(individual.size(0) for individual in individual_losses)))
        n_losses = sum(individual.size(0) for individual in individual_losses)
        loss_weights /= n_losses #set to uniform distribution

        #decay rate for the loss weights (later epochs have less importance)
        decay_rate = 1.0 / (n_losses * self.total_epochs)
        min_value_for_non_final_losses = 0.03 / n_losses

        #just the weird iteration in one step
        loss_weights = np.maximum(loss_weights - (current_epoch * decay_rate), min_value_for_non_final_losses)
    
        #compute last element
        curr_value = np.minimum(
            loss_weights[-1, -1] + (current_epoch * (n_losses - 1) * decay_rate),
            1.0 - ((n_losses - 1) * min_value_for_non_final_losses))
        loss_weights[-1, -1] = curr_value
        #retransform into list
        loss_weights_list = [loss_weights[i, :individual_losses[i].size(0)] for i in range(len(individual_losses))]
        return loss_weights_list
    
        # doing the above in a two-liner
        # loss_weights = np.ones((len(individual_losses), max(individual.size(0) for individual in individual_losses))) *( 
        #   (1/sum(individual.size(0) for individual in individual_losses)) -
        #   (current_epoch * 1.0 / (sum(individual.size(0) for individual in individual_losses) * self.total_epochs))
        # ) #basically this, idk why the last one is different


    def meta_update(self, tasks:List[RLTask], individual_losses:List[torch.FloatTensor], n_steps:int)->float:
        """Updates the meta-net based on the rewards of the inner net. The loss is calculated as the mean squared error between the q-values of the meta-net and the inner-net. The q-values are calculated by the inner-net and the meta-net is updated to predict the q-values of the inner-net."""
        #first make some new memories on the test data
        memories = [MemoryBuffer(self.memory_size) for _ in range(len(tasks))]
        test_values = self.inner_loop(tasks, n_steps, train=False, ret_memory_ptr=memories)
        total_test_loss = test_values[2]

        #get the importances of all the train losses to the total test loss
        loss_weights = self.compute_importances(individual_losses, total_test_loss)


        #Our goal is that our inner net can adapt fast from the meta-net weights. So we encourage the inner-net to perform well early on. (via the list of rewards for the environment rounds.)

        #we first replay the memory
        pass


    def select_action(self, state:torch.Tensor)->np.intp:
        """Selects an action based on the state. The action is selected by sampling from the q-values. And then selecting the action with the highest value. If the meta-sampling is used, the inner net also gets the alpha and beta values, otherwise we use thompson sampling and multiply the q-values with the sampled probabilities."""
        if self.useMetaSampling:
            #how do we regularize? 
            # -> for large alpha and beta, it should be close to the thompson sampling, as this is the optimal policy
            # -> for low alpha and beta, it does not need to be close as we want the meta-learning to have an effect
            sampled_values = self.inner_q_net(torch.cat([
                state, 
                torch.tensor(self.alpha, dtype=torch.float32).to(self.device), 
                torch.tensor(self.beta, dtype=torch.float32).to(self.device)
            ])) 
            sampled_values = torch.softmax(sampled_values, dim=0) #softmax to get probabilities
            #we use the cross-entropy loss during loss computation which has the softmax included,
            #but for prediction we also need the probabilities, so we apply it here as well.
            
        else:
            #thompson sampling and q-value prediction
            q_values = self.inner_q_net(state) 
            sampled_values = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.action_dim)]
            sampled_values *= q_values.detach().cpu().numpy()
        
        return np.argmax(sampled_values)
    
    def inner_loop(self, tasks:List[RLTask], n_steps:int, train:bool=True, ret_memory_ptr:Optional[List[MemoryBuffer]]=None)->Tuple[float, List[float], torch.Tensor, List[torch.Tensor]]:
        """This function should be used to learn, predict the inner-net. It should return the average score of the tasks, and the average loss of the tasks.
        
        Args:
        tasks: List[RLTask]
            The tasks to be learned. Consists of an environment which is reset when the task is sampled.
        n_steps: int
            The number of steps to be taken in the environment sample == in each episode.
        train: bool = True
            Whether the inner-net should be trained or not.
        ret_memory_ptr: Optional[List[MemoryBuffer]] = None
            If set the memories for each task will be returned in the given memory buffers. Same order as the tasks.
        
        Returns:
            Tuple[float, List[float], torch.Tensor, List[torch.Tensor], List[torch.Tensor]]: The average score of the tasks, the scores of the tasks, the total loss of the tasks, and the losses of the tasks, all the individual losses of each task stacked in the 0th dimension. The losses are all 0 if train is False.
        """
        task_scores = []
        task_losses = [torch.tensor(0.0).to(self.device) for _ in range(len(tasks))]
        for t, __task in enumerate(tasks):
            #reset the sampling and iterate over the task by getting similar environments each time
            self.reset_sampling()
            self.reset_memory()
            env_scores = []
            for env, init_state in (__task.train_sample() if train else __task.test_sample()):
                score = 0
                env_step = 0
                state = torch.tensor(init_state, dtype=torch.float32).to(self.device)
                for s in range(n_steps):
                    env_step += 1
                    #select action and do it
                    action = self.select_action(state) #uses the inner-net to select the action
                    next_state, reward, terminated, truncated = env.step(action) #take the action in the environment
                    next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

                    #store the experience and update sampling and the net if necessary
                    self.memory.append((state, action, reward, next_state, terminated)) #store the experience

                    #only do the updating if we are in the training phase
                    if train:
                        if reward > 0: # Update Thompson Sampling parameters, each round, unlike the net-update which is sparse
                            self.alpha[action] += 1  # Rewarded actions increase alpha
                        else:
                            self.beta[action] += 1  # Unsuccessful actions increase beta

                        if env_step>self.batch_size and s % self.inner_update_rate == 0:
                            #update the inner-net, and pop the selected indices from the buffer if we have enough samples: 3x the batch size in the current buffer
                            task_losses[t] += self.inner_update(pop_buffer=(env_step - (self.batch_size*env_step/self.inner_update_rate) > 3*self.batch_size)) 

                    #prepare for the next step
                    state = next_state
                    score += reward
                    if terminated or truncated:
                        break

                env_scores.append(score)
            task_scores.append(sum(env_scores)/len(env_scores))

            if ret_memory_ptr is not None:
                ret_memory_ptr[t].buffer = deepcopy(self.memory.buffer)

        return (
            sum(task_scores), task_scores,
            torch.sum(torch.tensor(task_losses)), task_losses
        )
        
    def learn(self, task:RLTask|List[RLTask], n_steps:int)->Tuple[float, float]:
        """This function should be used to learn the meta-net. It should return the average score of the tasks, and the average loss of the tasks.
        Args:
        task: RLTask | List[RLTask]
            The task to be learned. Consists of an environment which is reset when the task is sampled.
            If a list of tasks is given, the function will learn from all of them. And apply the meta update after all tasks have been seen.
        n_steps: int
            The number of steps to be taken in the environment sample == in each episode.

        Returns:
        Tuple[float, float]: The average score of the tasks, and the average loss of the tasks.
        """
        
        tasks:List[RLTask] = []
        if isinstance(task, RLTask):
            tasks = [task]
        else:
            tasks = task

        #reset the inner-net to the meta-net
        self.inner_q_net.load_state_dict(self.meta_q_net.state_dict())

        #this is basically adaption phase
        train_values = self.inner_loop(tasks, n_steps, train=True)
            
        #and now we learn the meta-net, for this we bascially do the same as above but this time don't update the alpha/beta and the inner-net
        #and then we update the meta-net based on the rewards of the inner-net given the prior weights it got from the meta net
        test_values = self.meta_update(tasks, n_steps) #update the meta-net

        return ...




def learn_meta_net(env_name:str="RandomCartPole", n_tasks:int=1000, episodes:int=10, n_steps:int=500, **kwargs):
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

    meta_net:MetaNet = MetaNet(state_dim, action_dim, **kwargs, total_epochs=)

    envset = EnvSet(env_name, norm)
    train_task_set:List[RLTask] = [generate_task(envset.sample(), episodes=episodes, portion_test=0.5) for _ in range(n_tasks)]
    test_task_set:List[RLTask] = [generate_task(envset.sample(), episodes=episodes, portion_test=0.5) for _ in range(n_tasks)]

    train_scores = []
    test_scores = []

    #TODO find out whether it helps to pass multiple tasks per learning step

    #TODO find out whether we should see each task batch multiple times before going to the next task batch. See MAML++ code: https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py#L232. 
    # Our current analog is that we sample from the environment many times. 

    #TODO find out how an epoch should be defined. it normally is all the data again, but we have infinite data (environments),
    for t, task in tqdm(enumerate(train_task_set), desc="Training MetaNet"):
        avg_score, _ = meta_net.learn(task, n_steps) #training epoch
        train_scores.append(avg_score)


    #here it does not makes sense to choose multiple tasks at the same time, as we want to see the generalization of the meta-net.
    for t, task in tqdm(enumerate(test_task_set), desc="Testing MetaNet"):
        avg_score, _ = meta_net.search(task, n_steps) #testing
        test_scores.append(avg_score)
    
    return meta_net, train_scores, test_scores
    