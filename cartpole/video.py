from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder

env_id = 'CartPole-v1'
video_folder = "./"
video_length = 500

env = make_vec_env(env_id=env_id, n_envs=16)

env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="ppo-agent-{}".format(env_id))
model = A2C.load("a2c_cartpole", env=env)

obs = env.reset()
for _ in range(video_length + 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, _, _ = env.step(action)
env.close()


