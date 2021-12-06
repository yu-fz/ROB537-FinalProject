import gym
import gym_Explore2D
import numpy as np
import torch
from frontiers import FrontierPointFinder
from dijkstra import DijsktraSearch

env = gym.make('Explore2D-v0', map = "./maps/gridWorld_easy.csv")
env.reset()
totalReward = 0
stepCounter = 0


frontierMap = env.returnFrontierMap()
shape = frontierMap.shape
agentPosition = np.where(frontierMap == 2) 
agentCoord = np.array([agentPosition[0][0],agentPosition[1][0]])
rowID = agentCoord[0]
columnID = agentCoord[1]

j = [agentCoord[1]-1, agentCoord[1]+1]
for k in range(0, len(j)):
  for i in range(agentCoord[0]-1, agentCoord[0]+1):
    if (i in range(0,shape[0]) and j[k] in range(0, shape[1])):
      if (frontierMap[i, j[k]] == 0):
        frontierMap[i, j[k]] = 4

j = [agentCoord[0]-1, agentCoord[0]+1]
for k in range(0, len(j)):
  for i in range(agentCoord[1]-1, agentCoord[1]+1):
    if (i in range(0,shape[1]) and j[k] in range(0, shape[0])):
      if (frontierMap[j[k], i] == 0):
        frontierMap[j[k], i] = 4
#if (columnID > 1):
#  frontierMap[rowID-1][columnID-2] = 4
#  frontierMap[rowID][columnID-2] = 4
#  frontierMap[rowID+1][columnID-2] = 4
#
#if (columnID < shape[1]-1):
#  frontierMap[rowID-1][columnID+2] = 4
#  frontierMap[rowID][columnID+2] = 4
#  frontierMap[rowID+1][columnID+2] = 4
#
#if (rowID > 1):
#  #frontierMap[rowID-2][columnID-2] = 4
#  frontierMap[rowID-2][columnID-1] = 4
#  frontierMap[rowID-2][columnID] = 4
#  frontierMap[rowID-2][columnID+1] = 4
#  #frontierMap[rowID-2][columnID+2] = 4
#
#if (rowID < shape[0]-1):
#  #frontierMap[rowID+2][columnID-2] = 4
#  frontierMap[rowID+2][columnID-1] = 4
#  frontierMap[rowID+2][columnID] = 4
#  frontierMap[rowID+2][columnID+1] = 4
#  #frontierMap[rowID+2][columnID+2] = 4

print(frontierMap)
# Target point
frontiers = FrontierPointFinder(frontierMap)
frontierWaypoint = frontiers.returnTargetFrontierPoint()
# Update map using new target point
env.updateMaps(agentCoord, frontierWaypoint)
frontierMap = env.returnFrontierMap()
# Update current position to 0
frontierMap[agentCoord[0]][agentCoord[1]] = 0

agentCoord = frontierWaypoint

j = [agentCoord[1]-1, agentCoord[1]+1]
for k in range(0, len(j)):
  for i in range(agentCoord[0]-1, agentCoord[0]+1):
    if (i in range(0,shape[0]) and j[k] in range(0, shape[1])):
      if (frontierMap[i, j[k]] == 0):
        frontierMap[i, j[k]] = 4

j = [agentCoord[0]-1, agentCoord[0]+1]
for k in range(0, len(j)):
  for i in range(agentCoord[1]-1, agentCoord[1]+1):
    if (i in range(0,shape[1]) and j[k] in range(0, shape[0])):
      if (frontierMap[j[k], i] == 0):
        frontierMap[j[k], i] = 4

print(frontierMap)


#while (env.returnExplorationProgress() < 0.9):
  
  #randomMove = np.random.randint(low =0, high = 5) #RL agent takes observation and selects a move. RNG in placeholder of agent 
  #observation, reward, done, info = env.step(randomMove) 
  #totalReward += reward
  #stepCounter+=1

#print("agent completed {a} time steps".format(a = stepCounter))
#env.render()

  # if done:
  #   print("Episode finished after {} timesteps".format(i+1))
  #   print("total reward for episode: " + str(totalReward))
  #   break
#env.reset()
