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
# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = False
    if train:
        n_cpu = 1
        batch_size = 64
        env = gym.make("roundabout-v1")#make_vec_env("roundabout-v1", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[512, 256], vf=[512, 256])]),
                    n_steps=batch_size * 12 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=30,
                    learning_rate=1e-3,
                    gamma=0.99,
                    verbose=2,
                    tensorboard_log="highway_ppo/")
        
        # Train the agent
        model.set_parameters("highway_ppo_new/model_new_obs_crowded_m1")
        model.learn(total_timesteps=int(1e5))
        # Save the agent

        model.save("highway_ppo_new/model_new_obs_crowded_m2")
        #model 3 - was with distance metric
        #model 4 - was with no distance metric
        #model 5 - neg reward

    #model_new_obs_exp1_seed_3_b256_8cars_512t256_randomvehicles_4hz_binary2
    model = PPO.load("highway_ppo_new/model_new_obs_crowded_m2")
    env = gym.make("roundabout-v1")
    crash_count = 0
    #frames = []
    
    render = False
    for i in range(500):
        print("iter",i)
        obs = env.reset()
        done = False
        frames = []

        while not done:
            actions = []
            if type(obs) != list:
                action, _ = model.predict(obs)
                actions.append(action)
            else:
                for item in obs:
                    action, _ = model.predict(item)
                    actions.append(action)

            #print(actions)
            #input()
            obs, reward, done, info = env.step(actions)
            if render:
                img = env.render()
                frames.append(env.render(mode="rgb_array"))

            if info["crashed"]:
                crash_count+=1

        if render:
            #imageio.mimsave("scene"+str(i)+".gif",frames,duration=0.25)
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_gif('scene_roundaboutss' + str(i) + '.gif', fps=5)
            frames = []

    print("crash rate :",crash_count)


