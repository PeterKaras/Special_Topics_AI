from thoml import MetaNet, RLTask, EnvSet, generate_task, episode_curriculum
import ray
import ray.tune as tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import ray.train as train#
from ray.air.integrations.wandb import WandbLoggerCallback

import pickle, os, torch
import gymnasium as gym
from gymnasium import Env
from env_sets import EnvSet, possible_envs
from scipy.stats import norm

from typing import List, Iterator#
from copy import deepcopy

HAS_GPU=False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    HAS_GPU=True
    ray.init(num_cpus=24, num_gpus=1)#, num_gpus=2)
else:
    ray.init(num_cpus=48)



config1 = {
    "batch_size":131,
    "eps":3.0957435088765894,
    "eps_moment":0.4168374586968928,
    "gamma":0.9321447718272076,
    "hidden_dim":253,
    "hidden_layers":2,
    "inner_grad_clip":30.153954052534036,
    "inner_lr":0.00011224198942099849,
    "inner_update_rate":8,
    "memory_size":1374,
    "meta_grad_clip":9.428531465983925,
    "meta_lr":0.0011260393097345348,
    "meta_min_lr":0.00012535360351340996,
    "meta_transition_frame1":73,
    "meta_transition_frame2":283,
    "reptile_decay":0.08112428061027666,
    "reptile_factor":28.476303982437685,
    "rho":0.15950175591343346,
    "task_batch_size":2,
    "tau":0.08442134387727845,
    "threshold_init_loss":21.15757372516343,
    "transition_sigma":69.88251905851794,
    "xi":0.33246964857260264,
    "n_tasks":100, 
}
config2 = {
    "batch_size":170,
    "eps":4.740157852787799,
    "eps_moment":0.40573119322229023,
    "gamma":0.8732994163874841,
    "hidden_dim":161,
    "hidden_layers":2,
    "inner_grad_clip":16.68284242059916,
    "inner_lr":0.0026181080884346043,
    "inner_update_rate":24,
    "memory_size":6029,
    "meta_grad_clip":58.44634805493503,
    "meta_lr":0.020608971510684432,
    "meta_min_lr":0.00000072625410905713,
    "meta_transition_frame1":18,
    "meta_transition_frame2":329,
    "reptile_decay":0.08919838902318701,
    "reptile_factor":0.011731196570831718,
    "rho":0.13363294306623522,
    "task_batch_size":1,
    "tau":0.14002009811368604,
    "threshold_init_loss":19.41758222872672,
    "transition_sigma":2.0657262115190624,
    "xi":0.3146287547622937,
    "n_tasks":100, 
}
config3 = {
    "batch_size":170,
    "eps":4.740157852787799,
    "eps_moment":0.40573119322229023,
    "gamma":0.8732994163874841,
    "hidden_dim":161,
    "hidden_layers":2,
    "inner_grad_clip":100,
    "inner_lr":0.0026181080884346043,
    "inner_update_rate":1,
    "memory_size":6029,
    "meta_grad_clip":15,
    "meta_lr":0.05,
    "meta_min_lr":0.001,
    "meta_transition_frame1":5,
    "meta_transition_frame2":20,
    "reptile_decay":0.2,
    "reptile_factor":2,
    "rho":0.13363294306623522,
    "task_batch_size":1,
    "tau":0.14002009811368604,
    "threshold_init_loss":19.41758222872672,
    "transition_sigma":5,
    "xi":0.3146287547622937,
    "n_tasks":100, 
}
config4 = {
    "batch_size":170,
    "eps":4.740157852787799,
    "eps_moment":0.40573119322229023,
    "gamma":0.8732994163874841,
    "hidden_dim":161,
    "hidden_layers":2,
    "inner_grad_clip":10.136565956058249,
    "inner_lr":0.0026181080884346043,
    "inner_update_rate":5,
    "memory_size":6029,
    "meta_grad_clip":25.23750387370719,
    "meta_lr":0.0071445556141484625,
    "meta_min_lr":0.00000284860307511656,
    "meta_transition_frame1":13,
    "meta_transition_frame2":126,
    "reptile_decay":0.06721414558067752,
    "reptile_factor":0.029150364511625105,
    "rho":0.13363294306623522,
    "task_batch_size":1,
    "tau":0.14002009811368604,
    "threshold_init_loss":19.41758222872672,
    "transition_sigma":3.31445213848809,
    "xi":0.3146287547622937,
    "n_tasks":100, 
}
config5 = {
    "batch_size":170,
    "eps":4.740157852787799,
    "eps_moment":0.40573119322229023,
    "gamma":0.8732994163874841,
    "hidden_dim":161,
    "hidden_layers":2,
    "inner_grad_clip":14.88633164729819,
    "inner_lr":0.0026181080884346043,
    "inner_update_rate":6,
    "memory_size":6029,
    "meta_grad_clip":15.276238460096886,
    "meta_lr":0.02345454991115573,
    "meta_min_lr":0.00000167766919865385,
    "meta_transition_frame1":34,
    "meta_transition_frame2":305,
    "reptile_decay":0.15350888402849602,
    "reptile_factor":0.01733770342213516,
    "rho":0.13363294306623522,
    "task_batch_size":1,
    "tau":0.14002009811368604,
    "threshold_init_loss":19.41758222872672,
    "transition_sigma":2.1798431416616095,
    "xi":0.3146287547622937,
    "n_tasks":100, 
}




class Meta_actor(tune.Trainable):
    def setup(self, config:dict):
        env_name:str=config.get("env_name","CartPole-v1")
        
        n_tasks:int=config.get("n_tasks", 1000)
        print("Doing the environment ", env_name, n_tasks, " times over.")
        task_batch_size:int=config.get("task_batch_size", 10)
        episodes:int=config.get("episodes", 50)
        n_steps:int=config.get("n_steps", 500)
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
        config["meta_transition_frame"] = (config.get("meta_transition_frame1", 20), config.get("meta_transition_frame2", 100))
        self.meta_net:MetaNet = MetaNet(state_dim, action_dim, num_task_batches=n_tasks, **config)

        envset = EnvSet(env_name, norm)
        self.train_task_set:Iterator[RLTask |List[RLTask]]
        if task_batch_size == 1:
            self.train_task_set = (
                generate_task(
                    envset.sample(),            #150 train/test -> 50 train/test
                    episodes=episode_curriculum(2*3*episodes, 2*episodes, n_tasks, t), #we do some curriculum learning here, to ease the learning process
                    portion_test=0.5
                ) for t in range(n_tasks)
            )
        else:
            self.train_task_set = (
                [generate_task(
                    envset.sample(), 
                    episodes=episode_curriculum(2*3*episodes, 2*episodes, n_tasks, t), 
                    portion_test=0.5
                ) for _ in range(task_batch_size)] for t in range(n_tasks)
            )
        #the test set always only has minimal number of episodes to see how well our meta model actually does.
        self.test_task_set:Iterator[RLTask] = (generate_task(envset.sample(), episodes=2*episodes, portion_test=0.5) for _ in range(n_tasks))

        self.train_scores = []
        self.test_scores = []

    def learn(self):
        tasks = next(self.train_task_set)
        avg_score, avg_loss = self.meta_net.meta_learn(tasks)
        self.train_scores.append(avg_score)

    def test(self):
        task = next(self.test_task_set)
        avg_score, avg_loss = self.meta_net.meta_test(task)
        self.test_scores.append(avg_score)

    def get_results(self):
        return self.train_scores, self.test_scores

    def get_latest_result(self):
        return self.train_scores[-1], self.test_scores[-1]

    def step(self):
        self.learn()
        self.test()
        results = self.get_latest_result()

        return {"train_score": results[0], "test_score": results[1]}

    @classmethod
    def default_resource_request(cls, config):
        return tune.PlacementGroupFactory([{"CPU": 6, "GPU":0.25} if HAS_GPU else {"CPU":4}])

    def save_checkpoint(self, checkpoint_dir):
        #write the whole meta net and the task sets into storage
        with open(os.path.join(checkpoint_dir, "meta_net.pth"), "wb") as f:
            torch.save(self.meta_net, f)
        #we cannot pickle the iterators, so we save the task sets as lists and reinitialize them on load
        with open(os.path.join(checkpoint_dir, "train_task_set.pkl"), "wb") as f:
            #we need to duplicate the generator, but this is not possible, so duplicate the list and remake it into a generator
            tmp = list(self.train_task_set) 
            pickle.dump(deepcopy(tmp), f)
            self.train_task_set = (el for el in tmp)
        with open(os.path.join(checkpoint_dir, "test_task_set.pkl"), "wb") as f:
            tmp = list(self.test_task_set)
            pickle.dump(deepcopy(tmp), f)
            self.test_task_set = (el for el in tmp)
        with open(os.path.join(checkpoint_dir, "train_scores.pkl"), "wb") as f:
            pickle.dump(self.train_scores, f)
        with open(os.path.join(checkpoint_dir, "test_scores.pkl"), "wb") as f:
            pickle.dump(self.test_scores, f)
        
    def load_checkpoint(self, checkpoint_dir):
        #load the whole meta net and the task sets from storage
        with open(os.path.join(checkpoint_dir, "meta_net.pth"), "rb") as f:
            self.meta_net = torch.load(f, weights_only=False)
        with open(os.path.join(checkpoint_dir, "train_task_set.pkl"), "rb") as f:
            self.train_task_set = (el for el in pickle.load(f))
        with open(os.path.join(checkpoint_dir, "test_task_set.pkl"), "rb") as f:
            self.test_task_set = (el for el in pickle.load(f))
        with open(os.path.join(checkpoint_dir, "train_scores.pkl"), "rb") as f:
            self.train_scores = pickle.load(f)
        with open(os.path.join(checkpoint_dir, "test_scores.pkl"), "rb") as f:
            self.test_scores = pickle.load(f)


dqn_config={
    "batch_size":170,
    "eps":4.740157852787799,
    "eps_moment":0.40573119322229023,
    "gamma":0.8732994163874841,
    "hidden_dim":161,
    "hidden_layers":2,
    "inner_lr":0.0026181080884346043,
    "memory_size":6029,
    "rho":0.13363294306623522,
    "tau":0.14002009811368604,
    "threshold_init_loss":19.41758222872672,
    "xi":0.3146287547622937,
}

config = {
    "hidden_layers": tune.randint(1, 3),
    "hidden_dim": tune.qlograndint(64, 256, 1),
    "meta_lr": tune.loguniform(1e-5, 1e-1),
    "meta_min_lr": tune.loguniform(1e-7, 1e-3),
    "meta_grad_clip": tune.loguniform(1.0, 100.0),
    "task_batch_size": tune.randint(1, 4),
    "meta_transition_frame1": tune.randint(10, 80),
    "meta_transition_frame2": tune.randint(80, 400), #
    "transition_sigma": tune.loguniform(1.0, 100.0), #sigma for the sigmoid transition
    "reptile_decay": tune.loguniform(0.05, 0.5), #reptile decay for lowering larger gradients
    "reptile_factor": tune.loguniform(0.01, 100.0), #reptile factor for blending with maml gradients
    "memory_size": tune.qlograndint(1000, 10000, 100),
    "batch_size": tune.qlograndint(32, 256, 1),
    "eps": tune.loguniform(0.5, 5.0), #loss threshold factor for the exploration threshold policy
    "eps_moment": tune.loguniform(0.1, 0.5), #loss momentum for the exploration threshold policy
    "gamma": tune.quniform(0.7, 0.98, 0.02), #discount factor
    "inner_lr": tune.loguniform(0.0001, 0.01),
    "inner_grad_clip": tune.loguniform(10.0, 100.0),
    "inner_update_rate": tune.lograndint(1, 256), #inner update rate for the inner_network, probably 1 works best 
    "rho": tune.loguniform(0.05, 0.5), #attention residual weight
    "tau": tune.loguniform(0.005, 0.5), #target soft update rate
    "xi": tune.loguniform(0.05, 0.5), #skip residual weight
    "threshold_init_loss": tune.qloguniform(1.0, 40.0, 1.0), #initial value for the ema of the exploration threshold policy
    "n_tasks":100
}

#set the dqn_config values fixed, to ease the search, when some good other values found, we decrease the search space and deactivated the fiation
# for key, val in dqn_config.items():
#     config[key] = val

tuner = tune.Tuner(
    Meta_actor, 
    run_config = train.RunConfig(
        stop = {"training_iteration": 100},
        checkpoint_config=train.CheckpointConfig(
            checkpoint_frequency=0,
            checkpoint_at_end=False,
        ),
        callbacks=[WandbLoggerCallback(
            project="reptile-thomp-dqn-short",
            log_config=True
        )],
        storage_path=f"/work/sg114224/ray_results/checkpoints/thoml_opt_quick"
    ),
    tune_config = tune.TuneConfig(
        num_samples=243,
        max_concurrent_trials=4 if HAS_GPU else 12, #change to 2x2 if 2 gpu
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="test_score",
            mode="max",
            max_t=100,
            grace_period=20,
            reduction_factor=3
        ),
        search_alg=HyperOptSearch(
            metric="test_score",
            mode="max",
            points_to_evaluate=[config1, config2, config3, config4, config5]
        )
    ),
    param_space=config
)

tuner.fit()

