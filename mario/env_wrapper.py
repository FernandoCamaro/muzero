"""
Author: Deepak Pathak

Acknowledgement:
    - The wrappers (BufferedObsEnv, SkipEnv) were originally written by
        Evan Shelhamer and modified by Deepak. Thanks Evan!
    - This file is derived from
        https://github.com/shelhamer/ourl/envs.py
        https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers_deprecated.py
"""
from __future__ import print_function
import numpy as np
from collections import deque
from PIL import Image
from gym.spaces.box import Box
import gym
import time, sys


class BufferedObsEnv(gym.ObservationWrapper):
    """Buffer observations and stack e.g. for frame skipping.

    n*2 is the length of the buffer. n (action,observations) pairs are stacked.
    skip is the number of steps between buffered observations (min=1, which means that no frame is skipped).

    n.b. first obs is the oldest, last obs is the newest.
         the buffer is zeroed out on reset.
         *must* call reset() for init!
         The returned observatinon has shape (shape[0], shape[1], (1+c)*n), where in the last dimension we alternatively find 
         1 bias plane that encodes the action, and then the resulting environment observation made of c channels (e.g. 3 for RGB).
         Note, in case skip is larger than 1, some frames and actions will be skipped.
    """
    def __init__(self, env=None, n=4, skip=1, shape=(84, 84)):
        super(BufferedObsEnv, self).__init__(env)
        self.obs_shape = shape
        self.n = n
        self.skip = skip
        self.buffer = deque(maxlen=self.n*2)
        self.counter = 0  # init and reset should agree on this
        shape = shape + ((1+3)*n,)
        self.observation_space = Box(0.0, 255.0, shape)
        self.scale = 1.0 / 255
        self.action_scale = 1./(self.env.action_space.n-1)
        self.observation_space.high[...] = 1.0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs, action), reward, done, info

    def _observation(self, obs, action):
        obs = self._convert(obs)
        self.counter += 1
        if self.counter % self.skip == 0:
            self.buffer.append(np.expand_dims(np.ones(self.obs_shape), axis=2)*action*self.action_scale) # action plane
            self.buffer.append(obs * self.scale) 
        obsNew = np.concatenate(self.buffer, axis=2)
        return obsNew.astype(np.float32) 

    def reset(self):
        """Clear buffer and re-fill by duplicating the first observation."""
        obs = self._convert(self.env.reset())
        self.buffer.clear()
        self.counter = 0
        for _ in range(self.n - 1):
            self.buffer.append(np.expand_dims(np.zeros(self.obs_shape),axis=2))
            self.buffer.append(np.zeros_like(obs))
        self.buffer.append(np.expand_dims(np.zeros(self.obs_shape), axis=2))
        self.buffer.append(obs * self.scale)
        obsNew = np.concatenate(self.buffer, axis=2)
        return obsNew.astype(np.float32)

    def _convert(self, obs):
        small_frame = np.array(Image.fromarray(obs).resize(
            self.obs_shape, resample=Image.BILINEAR), dtype=np.uint8)
        return small_frame


class SkipEnv(gym.Wrapper):
    """Skip timesteps: repeat action, accumulate reward, take last obs."""
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        for i in range(0, self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            info['steps'] = i + 1
            if done:
                break
        return obs, total_reward, done, info

    def reset(self):
        return self.env.reset()

class LifeLostEndEnv(gym.Wrapper):
    """If life is less than 2, it the game is considered ended"""
    def __init__(self, env=None, skip=4):
        super(LifeLostEndEnv, self).__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['life'] < 2:
            done = True
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()
