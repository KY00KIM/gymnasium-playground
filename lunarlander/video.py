import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

env_id = 'LunarLander-v2'
video_folder = "./"
video_length = 500

env = make_vec_env(env_id=env_id, n_envs=16)

env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="ppo-agent-{}".format(env_id))
model = PPO.load("ppo_lunarlander_512_128_64", env=env)

obs = env.reset()
for _ in range(video_length + 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = env.step(action)
env.close()


