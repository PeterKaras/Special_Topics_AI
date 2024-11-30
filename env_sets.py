'''We take the classical gym environments, and amend them by allowing some more random variation between the tasks.
We introduce at least one new random variable per environment'''

import gymnasium as gym

from collections import deque, OrderedDict
from typing import Generator, Tuple, Any, Optional, Type
from scipy.stats import rv_continuous, uniform
import numpy as np

"""You can use the following environments: (the keys of the dict)
The equivalent gym environments are the values of the dictionary."""
possible_envs:dict[str, str] = {
    "RandomCartPole" : "CartPole-v1",
    # "RandomMountainCar" : "MountainCar-v0",
    # "RandomAcrobot" : "Acrobot-v1",
    # "RandomPendulum" : "Pendulum-v0",
    # "RandomLunarLander" : "LunarLander-v2",
}


gym.register(
    id='RandomCartPole',
    entry_point='environments:RandomCartPoleEnv',
    max_episode_steps=500,
    reward_threshold=500,
)

class EnvSet(Generator):
    def __init__(self, id:str, *random_vars:rv_continuous, **kwargs):
        """The class for generating a set of environments with the same id, but with random variables that generate the seeds for the environments. kwargs to the gym may be passed as well.
        
        Please only parse continuous [0,1] random variables, as the seed is meant to be a float in [0,1]. If you do not parse any, the environment will use its default random variable. Make sure to parse the correct amount of random variables for the environment. Normally that is just 1, but some may accept multiple.
        """
        self.id = id
        self.random_vars = random_vars
        self.kwargs = kwargs
    
    def sample(self, seed:Optional[int]=None) -> gym.Env:#
        if seed is None:
            seed = np.random.random_integers(2^31-1)
        if self.id not in possible_envs:
            return gym.make(self.id, **self.kwargs)
        else:
            return gym.make(self.id, seed=seed, random_vars=self.random_vars, **self.kwargs)

    def send(self, value: None) -> Any:
        return super().send(value)
    

    def throw(self, typ: Optional[Type] = None, val: Optional[BaseException] = None, tb: Optional[Any] = None) -> None:
        return super().throw(typ, val, tb)

