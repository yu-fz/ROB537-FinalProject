import gym
import gym_Explore2D
import numpy as np
import torch

env = gym.make('Explore2D-v0', map = "./maps/gridWorld_hard.csv")
env.reset()
totalReward = 0
stepCounter = 0
while (env.returnExplorationProgress() < 0.9):
  randomMove = np.random.randint(low =0, high = 5) #RL agent takes observation and selects a move. RNG in placeholder of agent 
  observation, reward, done, info = env.step(randomMove) 
  totalReward += reward
  stepCounter+=1

print("agent completed {a} time steps".format(a = stepCounter))
env.render()
  # if done:
  #   print("Episode finished after {} timesteps".format(i+1))
  #   print("total reward for episode: " + str(totalReward))
  #   break
#env.reset()
