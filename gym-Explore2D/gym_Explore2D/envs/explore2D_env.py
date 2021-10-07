import gym
from gym import error, spaces, utils
from gym.utils import seeding

class Explore2D_Env(gym.Env):
  metadata = {'render.modes': ['human']}
    ###logic goes here
  def __init__(self):
    ...
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    ...
  def close(self):
    ...