import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

NUM_ENVS=32
LOOP = 15
TOTAL_TIMESTEPS = 1_000_000
vec_env = make_vec_env("Humanoid-v4", n_envs=NUM_ENVS)

model = SAC(
    "MlpPolicy",
    vec_env, 
    verbose=0, 
    policy_kwargs=dict(net_arch=dict(pi=[1024, 512, 512], qf=[1024, 512, 512])),
    tensorboard_log="./sac_humanoid_tensorboard/")

for i in range(1, LOOP+1):
    print(f"Loop: {i}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar= True)
    model.save(f"sac_humanoid_{i}")

del model

for i in range(1, LOOP+1):
    model = SAC.load(f"sac_humanoid_{i}")

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
        sum_rew += score
    print(f'{i}M. Mean Reward: {sum_rew/episodes}')
    del vec_env_tmp

