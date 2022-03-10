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
# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 4
        batch_size = 64
        env = make_vec_env("roundabout-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[512, 512], vf=[512, 512])]),
                    n_steps=batch_size * 12 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=30,
                    learning_rate=1e-3,
                    gamma=0.99,
                    verbose=2,
                    tensorboard_log="highway_ppo/")
        
        # Train the agent
        model.learn(total_timesteps=int(2e5))
        # Save the agent
        model.save("highway_ppo_new/model_new_obs_crowdeddd")
        #model 3 - was with distance metric
        #model 4 - was with no distance metric
        #model 5 - neg reward


    model = PPO.load("highway_ppo_new/model_new_obs_crowdeddd")
    env = gym.make("roundabout-v0")
    plt.ion()
    x = range(25)
    figure,ax = plt.subplots(figsize=(5,4))
    #frames = []
    crash_count = 0
    for i in range(20):
        obs = env.reset()
        done = False
        frames = []
        #distance per bin
        #reward terms 

        ax.set_ylim(-10,10)
        ax.set_xlim(0,25)
        line_rew, = ax.plot([],[],'r',label="total reward")
        line_sp, = ax.plot([],[],'b',label="speed reward")
        line_obsrew, = ax.plot([],[],'g',label="obs rew")
        line_acc, = ax.plot([],[],'k',label="acceleration rew")
        plt.legend()
        reward_buffer = []
        vel_rew = []
        obs_rew = []
        acc_rew = []

        x = []
        frame = 0
        while not done:
            action, _ = model.predict(obs)

            obs, reward, done, info = env.step(action)

            #print("distance per bin",info['distance_per_bin'])
            print("---------------------------------")
            print("reward_terms",info['reward_terms'])
            print("obstacle distance",info["obstacle_distance"])
            print("---------------------------------")
            reward_buffer.append(info["reward"])
            vel_rew.append(info['reward_terms']['speed'])
            obs_rew.append(info['reward_terms']['obstacle'])
            acc_rew.append(info['reward_terms']['acc'])
            x.append(frame)
            line_rew.set_xdata(x)
            line_rew.set_ydata(reward_buffer)
            line_sp.set_xdata(x)
            line_sp.set_ydata(vel_rew)


            line_obsrew.set_xdata(x)
            line_obsrew.set_ydata(obs_rew)

            line_acc.set_xdata(x)
            line_acc.set_ydata(acc_rew)

            if info["crashed"]:
                crash_count+=1

            frame+=1

            figure.canvas.draw()

            figure.canvas.flush_events()
            img = env.render()
            frames.append(env.render(mode="rgb_array"))

        figure.canvas.flush_events()
        ax.clear()
        time.sleep(0.1)
        
        #imageio.mimsave("scene"+str(i)+".gif",frames,duration=0.25)
        clip = ImageSequenceClip(frames, fps=5)
        clip.write_gif('scene_roundabout_cr' + str(i) + '.gif', fps=5)
        frames = []

    print("crash rate :",crash_count)



