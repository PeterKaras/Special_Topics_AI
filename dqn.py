import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import getpass
from typing import List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
from joblib import Parallel, delayed

import wandb
import ray
from ray import tune
from ray import train
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float

from dqn_utils import ReplayMemory, DQN, thompDQNv2, thompDQNv3, v2_CV_train, device

ray.init(num_cpus=48)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
import optuna

plt.ion()


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
import os
import json
class WandbTrainable(tune.Trainable):
    def setup(self, config):
        self.wandb = setup_wandb(
            config,
            trial_id=self.trial_id,
            trial_name=self.trial_name,
            project="thompDQNv3_hyperband_2",
        )

        self.agent = thompDQNv3(**config, num_episodes = NUM_EPISODES)
        self.agent.report_to_wandb = True
        self.preserve_agent = False

        self.best_result:dict[int, float] = {0:0.0} #should go up
        self.step_idx = 0

    def step(self):
        self.step_idx += 100
        avg_duration:float = self.agent.train(num_eps=100)
        self.wandb.log({"avg_duration": avg_duration})
        if avg_duration > list(self.best_result.values())[0]:
            self.best_result = {self.step_idx: avg_duration}

        return {"avg_duration": avg_duration}

    def save_checkpoint(self, checkpoint_dir:str):
        #write the config 
        #and the agent only if really wanted

        torch.save(self.agent.policy_net.state_dict(), os.path.join(checkpoint_dir, "policy.pth"))
        torch.save(self.agent.target_net.state_dict(), os.path.join(checkpoint_dir, "target.pth"))
        #now save the agent without the nets
        del self.agent.policy_net 
        del self.agent.target_net
        pickle.dump(self.agent, open(os.path.join(checkpoint_dir, "agent.pkl"), "wb"))
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            #write the best result so far to the config
            this_config = {"params":self.config.copy()}
            this_config["best_result"] = self.best_result
            json.dump(this_config, f)

    def load_checkpoint(self, checkpoint_dir:Optional[str]=None):
        if checkpoint_dir is None:
            return
        with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
            this_config = json.load(f)
            self.config = this_config["params"]
            self.best_result = this_config["best_result"]

        self.agent = pickle.load(open(os.path.join(checkpoint_dir, "agent.pkl"), "rb"))
        self.agent.policy_net = DQN(self.agent.n_observations+2*self.agent.n_actions, self.agent.n_actions, hidden_layers=self.config["hidden_layers"], hidden_dim=self.config["hidden_dim"], rho=self.config["rho"]).to(device)
        self.agent.target_net = DQN(self.agent.n_observations+2*self.agent.n_actions, self.agent.n_actions, hidden_layers=self.config["hidden_layers"], hidden_dim=self.config["hidden_dim"], rho=self.config["rho"]).to(device)
        self.agent.policy_net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "policy.pth"), weights_only=True))
        self.agent.target_net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "target.pth"), weights_only=True))
        self.agent.report_to_wandb = True


# Define search space
# search_space = {
#     "hidden_layers": tune.randint(1, 3),
#     "hidden_dim": tune.lograndint(64, 256),
#     "lr": tune.loguniform(1e-5, 1e-2),
#     "tau": tune.loguniform(0.0001, 0.1),
#     "gamma": tune.loguniform(0.9, 0.999),
#     "memory_size": tune.lograndint(1000, 10000),
#     "batch_size": tune.lograndint(32, 256),
#     "rho": tune.uniform(0.1, 0.5),
#     "eps": tune.uniform(0.1, 1.0),
#     "eps_moment": tune.loguniform(0.01, 0.9),
#     "threshold_init_loss": tune.uniform(1.0, 100.0),
# }
configspace = ConfigurationSpace()#
hidden_layers = Integer('hidden_layers', bounds=(1, 3))
hidden_dim = Integer('hidden_dim', bounds=(32, 256), log=True)
lr = Float('lr', bounds=(1e-5, 1e-1), log=True)
tau = Float('tau', bounds=(1e-4, 1e-1), log=True)
gamma = Float('gamma', bounds=(0.9, 0.999), log=True)
memory_size = Integer("memory_size", bounds=(1000, 10000), log=True)
batch_size = Integer('batch_size', bounds=(32, 256), log=True)
rho = Float('rho', bounds=(0.1, 0.5))
eps = Float('eps', bounds=(0.1, 1.0))
eps_moment = Float('eps_moment', bounds=(0.01, 0.9), log=True)
threshold_init_loss = Float('threshold_init_loss', bounds=(1.0, 100.0))

configspace.add([hidden_layers, hidden_dim, lr, tau, gamma, memory_size, batch_size, rho, eps, eps_moment, threshold_init_loss])

# Define the Hyperband scheduler and search algorithm: use BOHB (Bayesian Optimization HyperBand)
algo = TuneBOHB(configspace, metric="avg_duration", mode="max", max_concurrent=1)
hyperband = HyperBandForBOHB(
    time_attr="training_iteration",
    max_t=30,
    reduction_factor=3,
    metric="avg_duration",
    mode="max",
)
resourceTrainer = tune.with_resources(WandbTrainable, {"cpu": 48})
tuner = tune.Tuner(
    resourceTrainer,
    tune_config=tune.TuneConfig(
        num_samples=100,
        scheduler=hyperband,
        search_alg = algo,
    ),
    run_config=train.RunConfig(
        checkpoint_config = train.CheckpointConfig(
            num_to_keep=100,
            checkpoint_score_attribute="avg_duration",
            checkpoint_score_order="max",
            checkpoint_frequency=10,
        ),
        storage_path=f"/work/{getpass.getuser()}/ray_results/checkpoints"
    )
)

results = tuner.fit()
print(results)
try:
    best_result = results.get_best_result("avg_duration", "max")
    print("Best result found was: ", best_result)
    print("It had the following configuration: ", best_result.config)
except Exception as e:
    print("This did not work as intended. v2!", e)
