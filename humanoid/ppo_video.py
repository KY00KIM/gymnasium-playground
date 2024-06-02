import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

env_id = 'Humanoid-v4'
video_folder = "./"
video_length = 1000

env = make_vec_env("Humanoid-v4", n_envs=32)

env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="ppo-agent-{}".format(env_id))
model = PPO.load("ppo_humanoid_9", env=env)

obs = env.reset()
for _ in range(video_length + 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = env.step(action)
env.close()


