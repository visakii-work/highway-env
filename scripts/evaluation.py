from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


class Evaluation():
    """Run evaluation on trained policy in specific env"""
    def __init__(self, model, env, iterations=10):
        self.model = model
        self.env = env
        self.iterations = iterations
        # TODO: environment seed?

    def run_once(self):

        obs = self.env.reset()
        acc = defaultdict(float)
        denom = defaultdict(int)
        done = False
        t = 0

        while not done:
            t += 1
            action, _ = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)

            # keep track of accumulated rewards
            for k in ['speed', 'reward']:
                acc[k] += info[k]
                denom[k] += 1
            for k in ['collision', 'speed', 'lane', 'obstacle', 'acc']:
                acc[f'reward_{k}'] = info['reward_terms'][k]
                denom[f'reward_{k}'] += 1
            for k in ['distance_per_bin', 'obstacle_distance']:
                if info[k]:
                    acc[f'min_{k}'] += np.min(info[k])
                    denom[f'min_{k}'] += 1

        results = {f'mean_{k}': v / denom[k]
                   for k, v in acc.items()}  # averages
        results['total_reward'] = acc['reward']
        for k in [
                'reward_collision', 'reward_speed', 'reward_lane',
                'reward_obstacle', 'reward_acc'
        ]:
            results[f'total_{k}'] = acc[k]
        results['t_terminal'] = t
        results['crashed'] = info['crashed']
        if info['crashed']:
            results['t_crashed'] = t
        else:
            results['t_crashed'] = None

        return results

    def run(self):
        results = []
        for _ in tqdm(range(self.iterations)):
            res = self.run_once()
            results.append(res)

        df = pd.DataFrame(results)
        summary = df.mean().reset_index()
        summary.columns = ['metric', 'value']
        summary
        print('-' * 80)
        print(
            summary.to_string(index=False,
                              float_format=lambda x: "{:.4f}".format(x)))
        print('-' * 80)

        return df


if __name__ == '__main__':
    import sys
    sys.path.append('..')

    from stable_baselines3 import PPO
    import gym
    import highway_env

    model_path = 'highway_ppo_new/model_latest_1'
    model = PPO.load(model_path)
    env = gym.make("roundabout-v0")
    evaluation = Evaluation(model, env, 10)
    df = evaluation.run()
