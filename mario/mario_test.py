from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import env_wrapper
from PIL import Image
import numpy as np
import time

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
frame_skip = 4
fshape = (256, 256)
env = env_wrapper.LifeLostEndEnv(env)
env = env_wrapper.SkipEnv(env, skip=frame_skip)
env = env_wrapper.BufferedObsEnv(env, n=4, skip=1, shape=fshape)


obs = env.reset()
obs, reward, done, info = env.step(1)
for i in range(20):
    if done:
        obs = env.reset()
    obs, reward, done, info = env.step(1)
    env.render()
    time.sleep(0.1)
    if done:
        obs = env.reset()
    obs, reward, done, info = env.step(2)
    if done:
        obs = env.reset()
    obs, reward, done, info = env.step(2)
    if done:
        obs = env.reset()
    obs, reward, done, info = env.step(2)
    env.render()
    time.sleep(0.1)

obs_c = np.clip(obs,0,1)
Image.fromarray((obs_c[:,:,-3:]*255).astype('uint8')).show()

# env.close()