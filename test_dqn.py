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

import concurrent.futures as cf
import ray
N_RUNS = 5
ray.init(num_cpus=N_RUNS)

plt.ion()

base_config = { #this is a realllly good config lol. have to adapt for meta learning tho. but good starting point.
  "batch_size": 116,
  "eps": 0.9689087383915,
  "eps_moment": 0.3504356490412,
  "gamma": 0.9278873298653,
  "hidden_dim": 213,
  "hidden_layers": 2,
  "lr": 0.003697352671,
  "memory_size": 1278,
  "rho": 0.296589496246,
  "tau": 0.0964432367551,
  "threshold_init_loss": 33.3465753757362
}

#now that we found a good solution, let's enlarge the good solution space with optuna

# agent = thompDQNv3(**config, num_episodes = 500)
# agent.train()
@ray.remote(num_cpus=1)
class agent:
  def __init__(self, config):
    self.config = config
    self.agent = thompDQNv3(**config, num_episodes = 500)

  def train(self, num_eps):
    return self.agent.train(num_eps=num_eps)

# Define the Optuna objective function
def objective(trial:optuna.trial.Trial):
    
    # Define search space as perturbations around the base config
    batch_size = trial.suggest_int("batch_size", base_config["batch_size"]-20, base_config["batch_size"]+20)
    eps = trial.suggest_float("eps", base_config["eps"]*0.8, base_config["eps"]*1.2)
    eps_moment = trial.suggest_float("eps_moment", base_config["eps_moment"]*0.8, base_config["eps_moment"]*1.2)
    gamma = trial.suggest_float("gamma", base_config["gamma"]*0.8, base_config["gamma"]*1.2)
    hidden_dim = trial.suggest_int("hidden_dim", base_config["hidden_dim"]-40, base_config["hidden_dim"]+40)
    hidden_layers = trial.suggest_int("hidden_layers", base_config["hidden_layers"]-1, base_config["hidden_layers"]+1)
    lr = trial.suggest_float("lr", base_config["lr"]*0.8, base_config["lr"]*1.2)
    memory_size = trial.suggest_int("memory_size", base_config["memory_size"]-200, base_config["memory_size"]+200)
    rho = trial.suggest_float("rho", base_config["rho"]*0.8, base_config["rho"]*1.2)
    tau = trial.suggest_float("tau", base_config["tau"]*0.8, base_config["tau"]*1.2)
    threshold_init_loss = trial.suggest_float("threshold_init_loss", base_config["threshold_init_loss"]*0.8, base_config["threshold_init_loss"]*1.2)

    # Use the perturbed hyperparameters to create an agent
    config = {
        "batch_size": batch_size,
        "eps": eps,
        "eps_moment": eps_moment,
        "gamma": gamma,
        "hidden_dim": hidden_dim,
        "hidden_layers": hidden_layers,
        "lr": lr,
        "memory_size": memory_size,
        "rho": rho,
        "tau": tau,
        "threshold_init_loss": threshold_init_loss,
    }
    id = random.randint(0, 10000000)
    run = wandb.init(config=config, id=str(id), name=f"optuna_local_v9_Nr{trial._trial_id}", project="thompDQNv3_localsearch")

    
    #do a cv with 10 runs simultaneously
    agents = [agent.remote(config) for _ in range(N_RUNS)]
    avg_duration = 0

    with cf.ThreadPoolExecutor(max_workers=N_RUNS) as executor:
      for i in range(50):
        durations = ray.get([a.train.remote(10) for a in agents])
        avg_duration = sum(durations)/N_RUNS
    
        trial.report(avg_duration, i)
        run.log({"avg_duration": avg_duration})

        if trial.should_prune():
            run.finish()
            raise optuna.TrialPruned()
    # Train the agent and return a metric to maximize (e.g., total reward or test performance)
    run.finish()
    return avg_duration

# Create the Optuna study
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=10, n_min_trials=20))

# Optionally enqueue the base configuration as an initial trial
study.enqueue_trial(base_config)

# Start the optimization
study.optimize(objective, n_trials=80, n_jobs=1)

# Show the best parameters and value
print("Best value:", study.best_value)
print("Best parameters:", study.best_params)

# Plot optimization history
optuna.visualization.plot_optimization_history(study).show()