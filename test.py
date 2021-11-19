import gym
import gym_Explore2D
import numpy as np
import torch
from frontiersnew import FrontierPointFinder

env = gym.make('Explore2D-v0', map = "./maps/gridWorld_easy.csv")
env.reset()
totalReward = 0
stepCounter = 0
randomMove = []

while (env.returnExplorationProgress() < 0.9):
  randomMove.append(np.random.randint(low =0, high = 5)) #RL agent takes observation and selects a move. RNG in placeholder of agent 
  # generate set of moves using Dijsktra search
  randomMove = env.performDijsktra()
  for moves in randomMove:
    observation, reward, done, info = env.step(randomMove) 
    # add the observation into the frontier map is it sees any 3
    totalReward += reward
    stepCounter+=1

print("agent completed {a} time steps".format(a = stepCounter))
env.render()
  # if done:
  #   print("Episode finished after {} timesteps".format(i+1))
  #   print("total reward for episode: " + str(totalReward))
  #   break
#env.reset()
