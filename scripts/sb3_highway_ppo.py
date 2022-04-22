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

from stable_baselines3.common.policies import ActorCriticPolicy


import highway_env
from moviepy.editor import ImageSequenceClip
import imageio
import matplotlib.pyplot as plt
import time
import pickle as pkl
# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    train = False
    if train:
        n_cpu = 16
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
        # Set the parameters to the previous policy in the graph
        #model.set_parameters("highway_ppo_new/model_PPO_continuous_new")
        model.learn(total_timesteps=int(3.5e5))
        # Save the agent
        
        # this saves new policy in the graph
        model.save("highway_ppo_new/node_1") #model_latest - 0.50 treshold, model_latest - 0.3 threshold
        
        #model_PPO_continuous_scratch and model_PPO_continous_scratch_g1 - this was with -5 penalty for collision
        
        #model_PPO_continuous_scratch_1 and model_PPO_continuous_scratch_1_g1_1 - -10 penalty for collision
        
        #node_0 , node_1 new reward function, trying something out

        #test - 25 (0.5 obstacle rew), -40
        #test1 - 27 (0.05), -40
        #test3 -  0.05 and collision rew -5

        #model 3 - was with distance metric
        #model 4 - was with no distance metric best until now - model_new_obs_exp1_seed_3_b256_8cars
        #model 5 - neg reward not bad model_new_obs_exp1_seed_3_b256_8cars_512t256_randomvehicles_4hz_binary

    render = True
    reward_plot = False
    test_graph = False

    model = PPO.load("highway_ppo_new/node_0")
    if test_graph:
        model_1 = PPO.load("highway_ppo_new/node_1")

    filename = "NNModel.sav"

    nn_classifer = pkl.load(open(filename,'rb'))
    

    env = gym.make("roundabout-v0")
    
    #graph_0 = ActorCriticPolicy()
    if reward_plot:
        plt.ion()
        x = range(25)
        figure,ax = plt.subplots(figsize=(5,4))
    
    #frames = []
    crash_count = 0
    values = []
    norm_rewards = []
    vehicle_info = []
    observations = []
    for i in range(1000):
        print("test :",i)
        obs = env.reset()
        done = False
        frames = []
        #distance per bin
        #reward terms 
        if reward_plot:
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
        switch = False
        t = 0
        while not done:
            action, _ = model.predict(obs)

            if test_graph:
                prediction = nn_classifer.predict(obs.flatten().reshape(-1,50))

            #print("prediction",prediction)
            #input()
            if test_graph:
                action_1,_ = model_1.predict(obs)
            #cuda0 = torch.device('cuda:0')
            with th.no_grad():
                vfs = model.policy.predict_values(torch.from_numpy(obs.reshape((1,50))).to('cuda')).cpu().numpy()
                #vfs_1 = model_1.policy.predict_values(torch.from_numpy(obs.reshape((1,50))).to('cuda')).cpu().numpy()

            values.append(vfs)
            observations.append(obs)

            #print("Value of the state",vfs)
            #print("Value of state node 1",vfs_1)
            #input()
            if test_graph:
                if prediction[0] == 1:
                    a = action_1
                    #input()
                    print("Policy 2")
                else:
                    a = action
                    print("Policy 1")

            if test_graph:
                obs, reward, done, info = env.step(a)
            else:
                obs, reward, done, info = env.step(action)

            norm_rewards.append(reward)
            vehicle_info.append(info['vehicle_info'])

            if t == 0:
                print("***********************")
                print(info['vehicle_info'])
                print("***********************")

            if done:
                values.append(-100.0)
                norm_rewards.append(-100.0)
                vehicle_info.append({'0':0.0})
                observations.append(obs*0.0)

            #print("distance per bin",info['distance_per_bin'])
            #print("---------------------------------")
            #print("reward_terms",info['reward_terms'])
            #print("total reward",info["reward"])

            #print("---------------------------------")
            reward_buffer.append(info["reward"])
            vel_rew.append(info['reward_terms']['speed'])
            obs_rew.append(info['reward_terms']['obstacle'])
            acc_rew.append(info['reward_terms']['acc'])
            x.append(frame)
            

            if reward_plot:
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
            t+=1
            if reward_plot:
                figure.canvas.draw()

                figure.canvas.flush_events()
            

            if render:
                env.render()
                frames.append(env.render(mode="rgb_array"))

        if reward_plot:
            figure.canvas.flush_events()
            ax.clear()
            time.sleep(0.1)
        
        #imageio.mimsave("scene"+str(i)+".gif",frames,duration=0.25)
        if render:
            clip = ImageSequenceClip(frames, fps=5)
            clip.write_gif('scene_roundabout_best' + str(i) + '.gif', fps=5)
            frames = []

    print("crash rate :",crash_count*(100/200))
    '''
    with open("TerminalValues1.txt","wb") as fp:
        np.savetxt(fp,np.asarray(values),fmt="%1.5f")

    with open("RewardsPerStep1.txt","wb") as fp:
        np.savetxt(fp,np.asarray(norm_rewards),fmt="%1.5f")


    pkl.dump(vehicle_info,open("vehicle_info1.pkl","wb"))
    pkl.dump(observations,open("observations1.pkl","wb"))
    '''
    








