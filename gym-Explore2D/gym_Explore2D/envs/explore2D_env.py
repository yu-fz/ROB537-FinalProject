import gym
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from gym import error, spaces, utils
from gym.spaces import Discrete, Box
from gym.utils import seeding


pathToGroundTruthMap = "./gridWorld.csv"

class Explore2D_Env(gym.Env):
  metadata = {'render.modes': ['human']}
    ###logic for environment goes here
  def __init__(self):
    #import ground truth
    self.groundTruthMap = np.loadtxt(pathToGroundTruthMap, delimiter=",").astype(int)
    self.agentMap = None
    self.shape = self.groundTruthMap.shape
    self.objectSpawnCoords = dict()
    self.spawnObjects()
    self.generateAgentMap()
    self.agentDetect() #initial agent scan
    self.action_space = Discrete(4)
    self.observation_space = Box(low=0, high=4, shape=(30, 30), dtype=int)

  def spawnObjects(self):
    coordList = []
    for i in range(4):
      #generate 4 random numbers inside map bounds to use as spawn coordinates for agent
      #and objective
      coordList.append(np.random.randint(low = 1, high = self.shape[0]-1))
    
    self.objectSpawnCoords["agent"] = coordList[:2]
    self.objectSpawnCoords["objective"] = coordList[2:]
    print(self.objectSpawnCoords["agent"])
    agentXCoord = self.objectSpawnCoords["agent"][1]
    agentYCoord = self.objectSpawnCoords["agent"][0]    
    
    objectiveXCoord = self.objectSpawnCoords["objective"][1]
    objectiveYCoord = self.objectSpawnCoords["objective"][0]
    #check if random spawn location is obstructed
    while(self.groundTruthMap[agentYCoord, agentXCoord] == 1):
      #if grid is obstructed, spawn again 
      print("oops")  
      self.objectSpawnCoords["agent"][1] = np.random.randint(low = 1, high = self.shape[0]-1)
      self.objectSpawnCoords["agent"][0] = np.random.randint(low = 1, high = self.shape[0]-1)
      agentXCoord = self.objectSpawnCoords["agent"][1]
      agentYCoord = self.objectSpawnCoords["agent"][0]

    while(self.groundTruthMap[objectiveYCoord, objectiveXCoord] == 1):
      #if grid is obstructed, spawn again 
      print("oops")  
      self.objectSpawnCoords["objective"][1] = np.random.randint(low = 1, high = self.shape[0]-1)
      self.objectSpawnCoords["objective"][0] = np.random.randint(low = 1, high = self.shape[0]-1)
      objectiveXCoord = self.objectSpawnCoords["objective"][1]
      objectiveYCoord = self.objectSpawnCoords["objective"][0]

    self.groundTruthMap[agentYCoord, agentXCoord] = 2
    self.groundTruthMap[objectiveYCoord, objectiveXCoord] = 3
  
  def generateAgentMap(self):
    agentPosition = self.objectSpawnCoords["agent"]  
    self.agentMap = np.full(self.shape, 4, dtype=int)
    self.agentMap[agentPosition[0], agentPosition[1]] = 2
    #print(self.agentMap)

  def agentDetect(self):
    #given agents position, reveal adjacent grids
    #updates the agent map 
    agentPosition = self.objectSpawnCoords["agent"] 
    #dictionary of adjacent grids
    adjacentGridDict = dict()
    detectionRadius = 1
    print(agentPosition)
    for i in range(agentPosition[0]-detectionRadius, agentPosition[0] + detectionRadius + 1):
      for j in range(agentPosition[1]-detectionRadius, agentPosition[1] + detectionRadius + 1):
        if( i in range(0,30) and j in range(0,30)):
          self.agentMap[i,j] = self.groundTruthMap[i,j]

    #self.agentMap
    print(adjacentGridDict)
         
  def step(self, action):
    #if agent steps into unoccupied square, move agent 
      #get current agent position, set to unoccupied.
      #get coordinates of new agent position
      #set groundTruthMap[coord] to agent 
      #update agentMap accordingly
      #run agent detect
    #else terminate 



    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    plt.imshow(self.groundTruthMap)
    plt.show()
    plt.imshow(self.agentMap)
    plt.show()
  def close(self):
    ...

myEnv = Explore2D_Env()
myEnv.render()
