import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

NUM_ENVS=32
LOOP = 15
TOTAL_TIMESTEPS = 1_000_000
vec_env = make_vec_env("Humanoid-v4", n_envs=NUM_ENVS)

model = PPO(
    "MlpPolicy",
    vec_env, 
    verbose=0, 
    n_epochs=5,
    n_steps=512,
    policy_kwargs=dict(net_arch=dict(pi=[1024, 512, 512], vf=[1024, 512, 512])),
    ent_coef=0.01,
    tensorboard_log="./ppo_humanoid_tensorboard/")

for i in range(1, LOOP+1):
    print(f"Loop: {i}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar= True)
    model.save(f"ppo_humanoid_{i}")

del model

for i in range(1, LOOP+1):
    model = PPO.load(f"ppo_humanoid_{i}")

    vec_env_tmp = make_vec_env("Humanoid-v4", n_envs=NUM_ENVS)
    obs = vec_env_tmp.reset()
    sum_rew = 0
    episodes = 10
    for episode in range(1, episodes+1):
        done = False
        score = 0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, dones, info = vec_env_tmp.step(action)
            score += reward[0]
            done = dones[0]
        # print(f'Episode: {episode}, Score: {score}')    
        sum_rew += score
    print(f'{i}M. Mean Reward: {sum_rew/episodes}')
    del vec_env_tmp
# %%
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    fig = plt.figure(title, figsize=(10, 5))
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    min_x, max_x = min(x), max(x)
    plt.scatter(x, y, s=1)
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    plt.plot(x, y, color="black")
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " (Smoothed)")
    plt.xlim(min_x, max_x)
    plt.show()

# plot_results("ppo_lunarlander_tensorboard/")