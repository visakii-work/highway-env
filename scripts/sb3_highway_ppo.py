import gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
from moviepy.editor import ImageSequenceClip
import imageio
import matplotlib.pyplot as plt
import time
from multiprocessing import cpu_count
from evaluation import Evaluation
import os
import argparse

# ==================================
#        Main script
# ==================================

def train_model():
    n_cpu = cpu_count()
    batch_size = 256
    env = make_vec_env("roundabout-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[512,256], vf=[512,256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=50,
                learning_rate=1e-3,
                gamma=0.99,
                verbose=2,
                seed=18734,
                tensorboard_log="highway_ppo/")
    
    # Train the agent
    model.learn(total_timesteps=int(1.5e5))
    # Save the agent
    # model.save(filename) #model_latest - 0.50 treshold, model_latest - 0.3 threshold
    #model 3 - was with distance metric
    #model 4 - was with no distance metric best until now - model_new_obs_exp1_seed_3_b256_8cars
    #model 5 - neg reward not bad model_new_obs_exp1_seed_3_b256_8cars_512t256_randomvehicles_4hz_binary

def parse_args():
    parser = argparse.ArgumentParser(description='reactive policy')
    parser.add_argument('--train', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    filename = "highway_ppo_new/model_latest_1"
    if args.train:
        model = train_model()
        model.save(filename)

    model = PPO.load(filename)
    env = gym.make("roundabout-v0")
    evaluation = Evaluation(model, env, 500, render_dir=os.path.basename(filename), render_frequencey=100)
    df = evaluation.run()
    df.to_csv(os.path.basename(filename) +'_roundabout_v0_eval.csv')



