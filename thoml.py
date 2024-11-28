from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque, OrderedDict
from tqdm import tqdm

from typing import Dict, Optional, Callable, List, Generator, Tuple, Literal, overload
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
    
    def append(self, experience:Tuple[torch.Tensor, List[int], int, float, torch.Tensor, bool]):
        """Appends an experience to the memory buffer. The experience is a tuple of the 
            - state, 
            - the alpha-beta values
            - the action, 
            - the reward,
            - the next state, 
            - and whether the episode is done.
        """
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
    
    def sample(self, batch_size:int, pop:bool=False)->List[Tuple[torch.Tensor, List[int], int, float, torch.Tensor, bool]]:
        """Samples a batch of experiences from the memory buffer. The experience is a tuple of the
            - state, 
            - the alpha-beta values
            - the action, 
            - the reward,
            - the next state, 
            - and whether the episode is done.
        """
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
        meta_lr:float=0.01, meta_min_lr:float=0.0001, num_task_batches:int=1000, use_second_order:bool=False,
        gamma:float=0.99, lossFn:Callable=nn.SmoothL1Loss, inner_lr:float=0.0001, heuristic_update_inner_lr:bool=False, 
        num_steps:int=500, update_rate:int=32, batch_size:int=64, memory_size:int=10000,
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
        num_task_batches: int = 100
            The total number of task batches that are learned. "Just like an episode".
        use_second_order: bool = False
        gamma: float = 0.99
            The discount factor.
        lossFn: Callable = nn.SmoothL1Loss
        inner_lr: float = 0.0001
            The learning rate of the inner-net. 
        heuristic_update_inner_lr: bool = False
            Whether to use the heuristic to update the inner-net learning rate or not.
            The heuristic is that the inner lr should depend on the outer lr by some factor. This makes sure that while the outer net progresses, the inner net does not overshoot with too high learning rates, but still learns fast in the beginning. #TODO implement inner lr heuristic
        num_steps: int = 500
            The number of steps to take in each environment pass. How many environment passes are done is determined by how many are given in the RLTask.
        update_rate: int = 32
            The number of steps before the inner-net is updated. Should be <= batch_size so that each batch can be used multiple times for the inner-net update. Batch contents are drawn randomly from the memory.
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
        
        self.use_second_order = use_second_order #whether to allow second order gradients to be build up

        #inner learning params
        self.useMetaSampling = useMetaSampling
        self.lossfn = lossFn
        self.inner_lr = inner_lr

        self.num_steps = num_steps
        self.inner_update_rate = update_rate
        self.heuristic_update_inner_lr = heuristic_update_inner_lr
        self.batch_size = batch_size   

        self.memory_size = memory_size
        self.memory = MemoryBuffer(memory_size)
        self.gamma = gamma

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
            self.inner_q_net = QNetwork(state_dim + 2*action_dim, action_dim).to(self.device)
            
            self.meta_q_net = QNetwork(state_dim + 2*action_dim, action_dim).to(self.device)
            self.meta_optimizer = optim.Adam(self.meta_q_net.parameters(), lr=self.meta_lr)
            
        else:
            self.inner_q_net = QNetwork(state_dim, action_dim, hidden_dim, layers).to(self.device)

            self.meta_q_net = QNetwork(state_dim, action_dim, hidden_dim, layers).to(self.device)
            self.meta_optimizer = optim.Adam(self.meta_q_net.parameters(), lr=self.meta_lr)

        self.meta_q_net.load_state_dict(self.inner_q_net.state_dict())

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

    def inner_update(self, pop_buffer:bool, ret_gradient_ptr:Optional[List[torch.Tensor]]=None):
        """Updates the inner-net based on the memory buffer.
        
        This function combines the loss calculation and the inner update"""
        batch = self.memory.sample(self.batch_size, pop=pop_buffer)
        states, ab, actions, rewards, next_states, dones = (torch.stack(el).to(self.device) for el in zip(*batch))

        #==q_values
        if self.useMetaSampling:
            state_action_values = self.inner_q_net(torch.cat([
                torch.tensor([states]).to(self.device), 
                torch.tensor([*list(ab.flatten())]).to(self.device)
            ])).gather(1, actions.unsqueeze(1)).squeeze(1) 
        else:
            state_action_values = self.inner_q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        #==target_q_values
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_states)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.stack([s for s in next_states if s is not None]).to(self.device)
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        with torch.no_grad():
            if self.useMetaSampling:
                # we should pass enhanced alpha, beta values to the inner-net, 
                # but we just pass the same, as this most likely does not change much
                next_state_values[non_final_mask] = self.inner_q_net(torch.cat([
                    torch.tensor([non_final_next_states]).to(self.device), 
                    torch.tensor([*list(ab.flatten())]).to(self.device)
                ])).max(1).values
            else:
                next_state_values[non_final_mask] = self.meta_q_net(non_final_next_states).max(1).values 

        expected_state_action_values = rewards + ((1-dones) * self.gamma * next_state_values) 

        # compute loss
        loss = self.lossfn()(state_action_values, expected_state_action_values)

        #from this part on it's basically https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py#L83
        self.inner_q_net.zero_grad()
        grads = torch.autograd.grad(loss, self.inner_q_net.parameters(), create_graph=self.use_second_order, allow_unused=True)
        weights = self.inner_q_net.state_dict()

        grads = tuple(grad.clamp(-self.inner_grad_clip, self.inner_grad_clip) for grad in grads)
        #apply the new weights with an SGD step to the inner-net
        self.inner_q_net.load_state_dict({name: param - self.inner_lr * grad for ((name, param), grad) in zip(weights.items(), grads)})

        if ret_gradient_ptr !=None:
            ret_gradient_ptr.append(grads) #just append the gradients to the list, we can access the list like a pointer to the next empty element in the buffer

        return loss
    

    def compute_importances(self, task_losses:List[int], curr_epoch:int=-1)->List[torch.FloatTensor]:
        """Computes the importances of the individual losses computed in the current epoch.

        Args:
        tasks: int
            The number of tasks.
        task_losses: List[int]
            A list over the tasks: the number of losses per task.
        curr_epoch: int = -1
            The current epoch. If not given, the current epoch of the MetaNet is used.
        
        Returns:
        List[torch.FloatTensor]: The importance of the individual losses. The list holds the tasks. The second dimension holds the importance of the losses of the task.

        See the MAML++ paper with the code at https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py for the implementation of the importance computation. This function is the equivalent for get_per_step_loss_importance_vector().
        """
        current_epoch = (self.curr_task_batch if curr_epoch==-1 else curr_epoch)

        #initialize the loss weights, we do it as a matrix, not vector because prettier
        loss_weights:np.ndarray = np.ones((len(task_losses), max(losses for losses in task_losses))) # size=n_tasks x n_losses
        n_losses = sum(losses for losses in task_losses)
        loss_weights /= n_losses #set to uniform distribution

        #decay rate for the loss weights (later epochs have less importance)
        decay_rate = 1.0 / (n_losses * self.num_task_batches)
        min_value_for_non_final_losses = 0.03 / n_losses

        #just the weird iteration in one step
        loss_weights = np.maximum(loss_weights - (current_epoch * decay_rate), min_value_for_non_final_losses)
    
        #compute last element
        curr_value = np.minimum(
            loss_weights[-1, -1] + (current_epoch * (n_losses - 1) * decay_rate),
            1.0 - ((n_losses - 1) * min_value_for_non_final_losses))
        loss_weights[-1, -1] = curr_value
        #retransform into list
        loss_weights_list = [loss_weights[i, :task_losses[i]] for i in range(len(task_losses))]
        return loss_weights_list


    def meta_update(self, tasks:List[RLTask], test_task_losses:List[torch.Tensor], train_task_gradients:List[torch.Tensor])->float:
        """Updates the meta-net based on the rewards of the inner net. The loss is calculated as the mean squared error between the q-values of the meta-net and the inner-net. The q-values are calculated by the inner-net and the meta-net is updated to predict the q-values of the inner-net."""
        #get the importances of all the train losses to the total test loss
        loss_weights = self.compute_importances([len(task.train_set)*(self.num_steps/self.inner_update_rate) for task in tasks])

        meta_gradient = torch.zeros_like(self.meta_q_net.parameters())

        total_test_loss = torch.tensor(0.0).to(self.device)
        for task_loss, task_weight in zip(test_task_losses, loss_weights):
            total_test_loss += torch.sum(task_weight * task_loss)
            
        task_loss.backward() #this should be the 2nd order gradient for the meta-net

        #compute the task meta gradients
        for meta_param, task_grad in zip(self.meta_q_net.parameters(), train_task_gradients):
            meta_gradient += meta_param.grad * task_grad

        meta_gradient /= len(tasks)

        self.meta_optimizer.zero_grad()
        meta_gradient.backward()
        self.meta_optimizer.step()


    def select_action(self, state:torch.Tensor)->np.intp:
        """Selects an action based on the state. The action is selected by sampling from the q-values. And then selecting the action with the highest value. If the meta-sampling is used, the inner net also gets the alpha and beta values, otherwise we use thompson sampling and multiply the q-values with the sampled probabilities."""
        if self.useMetaSampling:
            #how do we regularize? 
            # -> for large alpha and beta, it should be close to the thompson sampling, as this is the optimal policy
            # -> for low alpha and beta, it does not need to be close as we want the meta-learning to have an effect
            sampled_values = self.inner_q_net(torch.cat([
                torch.tensor([state]).to(self.device), 
                torch.tensor(self.alpha, dtype=torch.float32).to(self.device), 
                torch.tensor(self.beta, dtype=torch.float32).to(self.device)
            ])) 
            sampled_values = torch.softmax(sampled_values, dim=0) #softmax to get probabilities
            #we use the cross-entropy loss during loss computation which has the softmax included,
            #but for prediction we also need the probabilities, so we apply it here as well.
            #TODO possible use some other thing as softmax if it is unstable?
            
        else:
            #thompson sampling and q-value prediction
            q_values = self.inner_q_net(state) 
            sampled_values = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.action_dim)]
            sampled_values *= q_values.detach().cpu().numpy()
        
        return np.argmax(sampled_values)
    
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
        score:float = 0.0
        total_loss:torch.Tensor = torch.tensor(0.0).to(self.device)

        losses:List[torch.Tensor] = []

        if train:
            self.reset_sampling()
        self.reset_memory()

        env_scores = []
        for env, init_state in (task.train_sample() if train else task.test_sample()):
            score = 0
            env_step = 0
            state = torch.tensor(init_state, dtype=torch.float32).to(self.device)
            for s in range(self.num_steps):
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

                if train:
                    if env_step>self.batch_size and s % self.inner_update_rate == 0:
                        #update the inner-net, and pop the selected indices from the buffer if we have enough samples: 3x the batch size in the current buffer
                        batch_loss = self.inner_update(
                            pop_buffer = (
                                env_step - (self.batch_size*env_step/self.inner_update_rate) 
                                > 3*self.batch_size
                            ), 
                            ret_gradient_ptr=ret_gradient_ptr
                        )
                        total_loss += batch_loss
                        losses.append(batch_loss)

                #prepare for the next step
                state = next_state
                score += reward
                if terminated or truncated:
                    break

            env_scores.append(score)

            if ret_memory_ptr is not None:
                ret_memory_ptr.buffer = deepcopy(self.memory.buffer)

        return (score, total_loss) if train else (score, total_loss, torch.tensor(losses).to(self.device))
        
    def meta_learn(self, task:RLTask|List[RLTask])->Tuple[float, float]:
        """This function should be used to learn the meta-net. It should return the average score of the tasks, and the average loss of the tasks.
        Args:
        task: RLTask | List[RLTask]
            The task to be learned. Consists of an environment which is reset when the task is sampled.
            If a list of tasks is given, the function will learn from all of them. And apply the meta update after all tasks have been seen.

        Returns:
        Tuple[float, float]: The average score of the tasks, and the average loss of the tasks.
        """
        
        tasks:List[RLTask] = []
        if isinstance(task, RLTask):
            tasks = [task]
        else:
            tasks = task

        #first make some new memories on the test data
        test_memories = [MemoryBuffer(self.memory_size) for _ in range(len(tasks))]
        indiv_test_losses:List[torch.FloatTensor] = [None for _ in range(len(tasks))]
        train_gradients = [] #empty list, we append to it in the inner-loop, use it like a pointer

        for t, __task in enumerate(tasks):
            #reset the inner-net to the meta-net
            self.inner_q_net.load_state_dict(self.meta_q_net.state_dict())

            #this is basically support phase (== adaption during meta-training)
            task_train_score, task_train_loss  = self.inner_loop(__task, train=True, ret_gradient_ptr=train_gradients) #train the inner-net
            
            #this is the query phase
            task_test_score, task_test_loss, indiv_test_losses[t] = self.inner_loop(__task, train=False, ret_memory_ptr=test_memories[t])
            test_reward += task_test_score
            test_loss += task_test_loss

            __task.env.close()

        #and now we learn the meta-net, for this we bascially do the same as above but this time don't update the alpha/beta and the inner-net
        #and then we update the meta-net based on the rewards of the inner-net given the prior weights it got from the meta net
        test_reward, test_loss = self.meta_update(tasks, test_task_losses=indiv_test_losses, train_task_gradients=train_gradients) #update the meta-net

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
        self.inner_q_net.load_state_dict(self.meta_q_net.state_dict())

        #this is basically adaption phase
        train_values = self.inner_loop(task, train=True) #train the inner-net

        test_values = self.inner_loop(task, train=False) #test the inner-net

        return test_values[0], test_values[2]




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
        The number of steps per episode. This is how long the agent may interact with the environment, before reset. Should be less equal to the max steps of the environment.
    kwargs: dict
        The keyword arguments to be passed to the MetaNet constructor. 
    """
    gym_env = gym.make(possible_envs[env_name])
    action_dim:int = gym_env.action_space.n
    state_dim:int = gym_env.observation_space.shape[0]
    gym_env.close()
    del gym_env

    meta_net:MetaNet = MetaNet(state_dim, action_dim, num_tasks=1000, num_steps=n_steps, **kwargs) #total_epochs is task_iterations

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
        avg_score, _ = meta_net.meta_learn(task) #training epoch
        train_scores.append(avg_score)

        #simulate one episode
        meta_net.curr_task_batch += 1
        meta_net.meta_lr_scheduler.step()

    #here it does not makes sense to choose multiple tasks at the same time, as we want to see the generalization of the meta-net.
    for t, task in tqdm(enumerate(test_task_set), desc="Testing MetaNet"):
        avg_score, _ = meta_net.meta_test(task) #testing
        test_scores.append(avg_score)
    
    return meta_net, train_scores, test_scores
    