from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from util_leaf import Action

import mario.env_wrapper as env_wrapper
from util_leaf import Action


class MarioEnv():
    def __init__(self):
        self.player = 1
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        frame_skip = 4
        fshape = (96, 96)
        env = env_wrapper.LifeLostEndEnv(env)
        env = env_wrapper.SkipEnv(env, skip=frame_skip)
        env = env_wrapper.BufferedObsEnv(env, n=4, skip=1, shape=fshape)
        self.env = env
        self.observation = self.env.reset()
        self.game_ended = False

    def state(self):
        return self.observation.copy()

    def terminal(self):
        return self.game_ended

    def getLegalActions(self):
        # all actions are always legal in Mario Bros.
        legal_actions = [Action(i) for i in range(self.env.action_space.n)]
        return legal_actions

    def step(self, action: Action):
        obs, reward, done, info = self.env.step(action.index)
        self.observation = obs
        self.game_ended = done


    