import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

NUM_ENVS = 16
TOTAL_TIMESTEPS = 1_000_000

vec_env = make_vec_env("LunarLander-v2", n_envs=NUM_ENVS)

model = PPO(
    "MlpPolicy",
    vec_env, 
    verbose=0, 
    n_epochs=5,
    n_steps=512,
    policy_kwargs=dict(net_arch=dict(pi=[512, 128, 64], vf=[512, 128, 64])),
    ent_coef=0.01,
    tensorboard_log="./ppo_lunarlander_tensorboard/")

model.learn(total_timesteps=TOTAL_TIMESTEPS, 
            progress_bar= True)
model.save("ppo_lunarlander_512_128_64")

del model

model = PPO.load("ppo_lunarlander_512_128_64")
obs = vec_env.reset()

episodes = 10
sum_score = 0
for episode in range(1, episodes+1):
    done = False

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, dones, info = vec_env.step(action)
        sum_score += reward[0]
        done = dones[0]
print(f'MeanScore: {sum_score/episodes}')
