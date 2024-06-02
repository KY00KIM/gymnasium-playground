import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

env_id = "CartPole-v1"
NUM_ENVS = 16
TOTAL_TIMESTEPS = 1_000_000

env = make_vec_env(env_id, n_envs=NUM_ENVS)

model = A2C("MlpPolicy", 
            env, 
            verbose=0, 
            tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=TOTAL_TIMESTEPS, 
            progress_bar=True
            )

model.save("./a2c_cartpole")

del model
del env

model = A2C.load("a2c_cartpole")
vec_env = make_vec_env(env_id, n_envs=NUM_ENVS)
obs = vec_env.reset()

sum_score = 0
episodes = 10

for episode in range(1, episodes+1):
    done = False

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        sum_score += rewards[0]
        done = dones[0]
print(f'MeanScore: {sum_score/episodes}')



