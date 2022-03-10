import gym
import numpy as np

import highway_env
import time


epi_list = []
num_epi = 10
render = True
env = gym.make("roundabout-v0")

for i in range(num_epi):
    done = False
    obs = env.reset()
    count = 0
    epi_reward = 0
    while not done:
        # TODO: select action here
        action = 1#policy[count]
        # Get reward
        obs, reward, done, info = env.step(action)
        # Render
        #print("action:",action)
        if render: time.sleep(0.1)
        count += 1
        epi_reward += reward
        # print('count: ', count, 'action: ', action)

        if render: env.render()
        if done:
            print('####################################')
            assert count == 20 # episode length 10
            print('episode length:', count)
            print('episode reward:', epi_reward)
            print('####################################')
            epi_list.append(epi_reward)
print('mean rew', np.mean(np.array(epi_list)), 'var reward', np.std(np.array(epi_list).squeeze()))
env.close()