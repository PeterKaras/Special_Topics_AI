import matplotlib
import matplotlib.pyplot as plt
from dqn_utils import thompDQNv3
import optuna
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
import wandb
import random
import torch

import concurrent.futures as cf
import ray
import ray.tune as tune
import ray.train as train
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler as AHSAScheduler

# ray.init(num_cpus=N_RUNS)

import gymnasium as gym

N_RUNS = 4
if torch.cuda.is_available():
  ray.init(num_cpus=12, num_gpus=1)
else:
  ray.init(num_cpus=11*4)

base_config = { #this is a realllly good config lol. have to adapt for meta learning tho. but good starting point.
  "batch_size":122,
  "eps":0.8742978515539621,
  "eps_moment":0.3496265379180274,
  "gamma":0.8123068472166642,
  "hidden_dim":244,
  "hidden_layers":3,
  "lr":0.002993549689203733,
  "memory_size":1281,
  "rho":0.3527593629817173,
  "tau":0.0850198664648507,
  "xi":0.2,
  "threshold_init_loss":30.42264638096774,
}

base_config2 = {
  "batch_size":112,
  "eps":1.158886178670203,
  "eps_moment":0.33800004227833164,
  "gamma":0.8746488329186444,
  "hidden_dim":198,
  "hidden_layers":1,
  "lr":0.003909816156209336,
  "memory_size":1439,
  "rho":0.3087914402231445,
  "tau":0.08502873372537675,
  "xi":0.2,
  "threshold_init_loss": 31.35995045651688
}



#now that we found a good solution, let's enlarge the good solution space with optuna
@ray.remote(num_cpus=1, num_gpus=0.25 if torch.cuda.is_available() else 0)
class agent:
  def __init__(self, config):
    self.config = config
    n_obs = config["n_observations"]
    n_act = config["n_actions"]
    #clean from the config
    del config["n_observations"]
    del config["n_actions"]
    self.agent = thompDQNv3(n_observations=n_obs, n_actions=n_act, **config, num_episodes = 500)

  def train(self, env, num_eps):
    return self.agent.train(env=env, num_eps=num_eps, return_average_score=True)

# Define the Optuna objective function
def objective(config):
    
    # Define search space as perturbations around the base config
    
    id = random.randint(0, 10000000)
    # run = wandb.init(config=config, id=str(id), name=f"optuna_local_v12_Nr{trial._trial_id}", project="thompDQNv3_localsearch")
    env = gym.make("CartPole-v1")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    #do a cv with 10 runs simultaneously
    agents = [agent.remote(config|{"n_observations":n_observations, "n_actions":n_actions}) for _ in range(N_RUNS)]
    avg_duration = 0

    for i in range(50):
      durations = ray.get([a.train.remote(env, 10) for a in agents])
      avg_duration = sum(durations)/N_RUNS
  
      train.report({"avg_duration":avg_duration})
    # Train the agent and return a metric to maximize (e.g., total reward or test performance)
    # run.finish()
    return {"avg_duration":avg_duration}

# # Create the Optuna study
# study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=50))

# # # Optionally enqueue the base configuration as an initial trial
# # study.enqueue_trial(base_config)

# # Start the optimization
# study.optimize(objective, n_trials=81, n_jobs=1)

# # Show the best parameters and value
# print("Best value:", study.best_value)
# print("Best parameters:", study.best_params)

# # Plot optimization history
# optuna.visualization.plot_optimization_history(study).show()

config = {
  "batch_size": tune.randint(64, 256),
  "eps": tune.uniform(0.5, 5.0),
  "eps_moment": tune.uniform(0.1, 0.5),
  "gamma": tune.uniform(0.7, 0.99),
  "hidden_dim": tune.randint(128, 256),
  "hidden_layers": tune.randint(1, 3),
  "lr": tune.loguniform(0.0001, 0.01),
  "memory_size": tune.lograndint(1000, 10000),
  "rho": tune.uniform(0.1, 0.4),
  "tau": tune.loguniform(0.05, 0.5),
  "xi": tune.loguniform(0.05, 0.5),
  "threshold_init_loss": tune.uniform(10.0, 40.0),
}

tune.run(
    objective,
    config=config,
    num_samples=100,
    callbacks=[WandbLoggerCallback(
        project="thompDQNv3_localsearch",
    )],
    search_alg=HyperOptSearch(
       metric="avg_duration", 
       mode="max",
       points_to_evaluate=[base_config, base_config2]
    ),
    scheduler=AHSAScheduler(
        metric="avg_duration",
        mode="max",
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        grace_period=10
    ),
    max_concurrent_trials=4 if not torch.cuda.is_available() else 1,
    resources_per_trial=tune.PlacementGroupFactory([{'CPU': 1.0}] + [{'CPU': 1.0}] * N_RUNS) if not torch.cuda.is_available() else tune.PlacementGroupFactory([{'CPU': 2.0}] + [{'CPU': 2.0, "GPU":0.25}] * N_RUNS)
)

#just run the thompDQNv3 with the base config
# env = gym.make("CartPole-v1")
# n_observations = env.observation_space.shape[0]
# n_actions = env.action_space.n
# model = thompDQNv3(n_observations=n_observations, n_actions=n_actions, **base_config, num_episodes = 500)
# wandb.init(config=base_config, project="thompDQNv3_localsearch")
# for eps in range(0, 500, 10):
#   avg_duration = model.train(env, 10)
#   print(f"Episode {eps} done")
#   wandb.log({"avg_duration": avg_duration})
