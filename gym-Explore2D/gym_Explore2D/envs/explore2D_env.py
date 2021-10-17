import gym
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import torch
import math
from gym import error, spaces, utils
from gym.spaces import Discrete, Box
from gym.utils import seeding


#from stable_baselines.common.env_checker import check_env
pathToGroundTruthMap = "./gridWorld.csv"

class Explore2D_Env(gym.Env):
  metadata = {'render.modes': ['human']}
    ###logic for environment goes here
  def __init__(self):
    #import ground truth
    self.groundTruthMap = np.loadtxt(pathToGroundTruthMap, delimiter=",").astype(int)
    self.agentMap = None
    self.state = None #state is agentMap array combined with number of steps remaining
    self.shape = self.groundTruthMap.shape
    self.numOfHiddenGrids = None
    self.objectCoords = dict()
    #stepLimit is the maximum episode length -> emulates agent battery limits 
    self.stepLimit = 0
    self.action_space = Discrete(3)
    self.observation_space = Box(low=0, high=9, shape=self.groundTruthMap.shape, dtype=int)

  def getEnvSize(self):
    return self.shape

  def spawnObjects(self):
    coordList = []
    for i in range(4):
      #generate 4 random numbers inside map bounds to use as spawn coordinates for agent
      #and objective
      coordList.append(np.random.randint(low = 1, high = self.shape[0]-1))
    
    self.objectCoords["agent"] = coordList[:2]
    #self.objectCoords["agent"] = [13,7]
    self.objectCoords["objective"] = coordList[2:]
    agentXCoord = self.objectCoords["agent"][1]
    agentYCoord = self.objectCoords["agent"][0]    
    
    objectiveXCoord = self.objectCoords["objective"][1]
    objectiveYCoord = self.objectCoords["objective"][0]
    #check if random spawn location is obstructed
    while(self.groundTruthMap[agentYCoord, agentXCoord] == 1):
      #if grid is obstructed, spawn again 
      #print("oops")  
      self.objectCoords["agent"][1] = np.random.randint(low = 1, high = self.shape[0]-1)
      self.objectCoords["agent"][0] = np.random.randint(low = 1, high = self.shape[0]-1)
      agentXCoord = self.objectCoords["agent"][1]
      agentYCoord = self.objectCoords["agent"][0]

    while(self.groundTruthMap[objectiveYCoord, objectiveXCoord] == 1):
      #if grid is obstructed, spawn again 
      #print("oops")  
      self.objectCoords["objective"][1] = np.random.randint(low = 1, high = self.shape[0]-1)
      self.objectCoords["objective"][0] = np.random.randint(low = 1, high = self.shape[0]-1)
      objectiveXCoord = self.objectCoords["objective"][1]
      objectiveYCoord = self.objectCoords["objective"][0]

    self.groundTruthMap[agentYCoord, agentXCoord] = 2
    #self.groundTruthMap[objectiveYCoord, objectiveXCoord] = 3
  
  def generateAgentMap(self):
    agentPosition = self.objectCoords["agent"]  
    self.agentMap = np.full(self.shape, 4, dtype=int)
    self.agentMap[agentPosition[0], agentPosition[1]] = 2
    self.numOfHiddenGrids = np.count_nonzero(self.agentMap == 4)
    #print(self.agentMap)

  def setStepLimit(self, stepLimit):
    self.stepLimit = stepLimit

  def agentDetect(self):
    #given agents position, reveal adjacent grids
    #updates the agent map 
    agentPosition = self.objectCoords["agent"] 
    #dictionary of adjacent grids

    detectionRadius = 1
    #print(agentPosition)
    for i in range(agentPosition[0]-detectionRadius, agentPosition[0] + detectionRadius + 1):
      for j in range(agentPosition[1]-detectionRadius, agentPosition[1] + detectionRadius + 1):
        if( i in range(0,self.shape[0]) and j in range(0,self.shape[0])):
          self.agentMap[i,j] = self.groundTruthMap[i,j]

  def calculateRewards(self):
    occurrences = np.count_nonzero(self.agentMap == 4)
    infoGain = self.numOfHiddenGrids - occurrences
    self.numOfHiddenGrids = occurrences
    if(infoGain == 0):
      return -5
    else:
      return infoGain
    #print(infoGain)


  def updateMaps(self, currPos, newPos):
    self.groundTruthMap[tuple(newPos)] = 2
    self.groundTruthMap[tuple(currPos)] = 0
    self.agentMap[tuple(currPos)] = 0
    self.agentMap[tuple(newPos)] = 2
    self.objectCoords["agent"] = newPos
    self.agentDetect()

  def clearAgentMap(self):
    #called upon episode termination so network knows not to predict next Q
    self.agentMap = np.full(self.shape, 9, dtype=int) 

  def numActionsAvailable(self):
    return self.action_space.n

  def getState(self):
    stepLimitVector = np.zeros(self.shape[0])
    stepLimitVector[0] = self.stepLimit
    #print(np.vstack((self.agentMap,stepLimitVector)))
    #return np.vstack((self.agentMap,stepLimitVector))
    return self.agentMap

  def step(self, action):
    # [1,2,3,4] -> [up, down, left, right]
    done = False
    reward = 0
    currAgentPos = self.objectCoords["agent"] 
    self.stepLimit -= 1
    if(self.stepLimit == 0):
      ##End of Episode
      self.clearAgentMap()
      info = {}
      done = True
      return self.getState(), reward, done, info
      
    if(action == 0):
      #move up
      newAgentPos = [currAgentPos[0]-1, currAgentPos[1]]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward = -10
        #self.clearAgentMap()
        #done = True 
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward = self.calculateRewards()
          #print(self.objectCoords["agent"])
    elif(action == 1):
      #move down
      newAgentPos = [currAgentPos[0]+1, currAgentPos[1]]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward = -10
        #self.clearAgentMap()
        #done = True 
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward = self.calculateRewards()

    elif(action == 2):
      #move left
      newAgentPos = [currAgentPos[0], currAgentPos[1]-1]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward = -10
        #self.clearAgentMap()
        #done = True 
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward = self.calculateRewards()
    elif(action == 3):
      #move right
      newAgentPos = [currAgentPos[0], currAgentPos[1]+1]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward = -10
        #self.clearAgentMap()
        #done = True
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward = self.calculateRewards()
    #if agent steps into unoccupied square, move agent 
      #get current agent position, set to unoccupied.
      #get coordinates of new agent position
      #set groundTruthMap[coord] to agent 
      #update agentMap accordingly
      #run agent detect
    #else terminate 

      #Optional additional info 
    info = {}
    #return self.getState()
    return self.getState(), reward, done, info

  def reset(self):
    self.stepLimit = 50
    objectiveCoords = np.where(self.groundTruthMap == 3)
    agentCoords = np.where(self.groundTruthMap == 2)
    self.groundTruthMap[agentCoords] = 0
    self.groundTruthMap[objectiveCoords] =0
    self.agentMap = self.groundTruthMap
    self.spawnObjects()
    self.generateAgentMap()
    self.agentDetect()
    #return self.getState()
    return self.getState()

  def render(self, mode='human'):
    
    #plt.imshow(self.groundTruthMap)
    #plt.show()
    plt.imshow(self.agentMap)
    plt.show()

  def close(self):
    pass

#myEnv = Explore2D_Env()
#check_env(myEnv, warn=True)

# for i in range(10000):
#   randomMove = np.random.randint(low =1, high = 5)
#   myEnv.step(randomMove)
#   #myEnv.render()

# myEnv.render()
# myEnv.reset()
# myEnv.render()