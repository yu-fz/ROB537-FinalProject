import gym
import gym_Explore2D
import numpy as np
import torch
from frontiersnew import FrontierPointFinder
import pdb
env = gym.make('Explore2D-v0', map = "./maps/gridWorld_easy.csv")
env.reset()
stepCounter = 0
oldcounter = -1
while (env.returnExplorationProgress() < 0.9):
  # move from the final co-ordinate obtained from dijkstra
  # randomMove = np.random.randint(low =0, high = 5) 
  # call dijkstra to calculate the path exploration cost
  fgoal, dcost, initloc = env.performDijsktra()
  # env.step(randomMove) 
  stepCounter+=1 + dcost
  if oldcounter == env.returnExplorationProgress():
    # adding new tricks to improve exploration
    break
  oldcounter = env.returnExplorationProgress()
  print(env.returnExplorationProgress())

print("agent completed {a} time steps".format(a = stepCounter))
env.render()