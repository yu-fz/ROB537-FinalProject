import gym
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import random
from gym import error, spaces, utils
from gym.spaces import Discrete, Box, Dict
from gym.utils import seeding
import time
from collections import deque


class Explore2D_Env(gym.Env):
  metadata = {'render.modes': ['human']}
    ###logic for environment goes here
  def __init__(self):
    #import ground truth
    pathToGroundTruthMap = "./maps/gridWorld_easy.csv"
    self.groundTruthMap = np.loadtxt(pathToGroundTruthMap, delimiter=",").astype(int)
    self.agentMap = None
    self.state = None #state is agentMap array combined with number of steps remaining
    self.shape = self.groundTruthMap.shape
    self.numOfRevealedGrids = 0
    self.detectionRadius = 1
    self.agentMapHeight = 2*self.detectionRadius+1 #detection radius is number of grids from agent to the up/down/left/right edges of detection area
    self.observationMap = np.zeros([self.groundTruthMap.shape[0], self.groundTruthMap.shape[1], 3], dtype=np.uint8)
    self.agentMap = np.fill([self.agentMapHeight, self.agentMapHeight])
    self.objectCoords = dict()
    #stepLimit is the maximum episode length -> emulates agent battery limits 
    self.stepLimit = 0
    self.action_space = Discrete(3)
    self.observation_space = Box(low=0, high=3, shape=self.agentMap.shape, dtype=int)

    self.visitedCoords = deque(maxlen=10)




  def getEnvSize(self):
    return self.shape

  def spawnObjects(self):
    coordList = []
    random.seed(time.time())
    for i in range(4):
      #generate 4 random numbers inside map bounds to use as spawn coordinates for agent
      #and objective
      coordList.append(random.randint(1, self.shape[0]-2))
    
    self.objectCoords["agent"] = coordList[:2]
    #self.objectCoords["agent"] = [60,5]
    self.objectCoords["objective"] = coordList[2:]
    #self.objectCoords["objective"] = [3,62]
    agentXCoord = self.objectCoords["agent"][1]
    agentYCoord = self.objectCoords["agent"][0]    
    #print("agent spawned at " + str(self.objectCoords["agent"]))
    
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

    # while(self.groundTruthMap[objectiveYCoord, objectiveXCoord] == 1 or self.calculateDistance() < 3):
    #   #if grid is obstructed, spawn again 
    #   #print("oops")  
    #   self.objectCoords["objective"][1] = np.random.randint(low = 1, high = self.shape[0]-1)
    #   self.objectCoords["objective"][0] = np.random.randint(low = 1, high = self.shape[0]-1)
    #   objectiveXCoord = self.objectCoords["objective"][1]
    #   objectiveYCoord = self.objectCoords["objective"][0]

    self.groundTruthMap[agentYCoord, agentXCoord] = 2
    #print("agent spawned at" + str(self.objectCoords["agent"]))
    #self.groundTruthMap[objectiveYCoord, objectiveXCoord] = 3
  
  def generateAgentMap(self):
    agentPosition = self.objectCoords["agent"]
    agentMapHeight = (2*self.detectionRadius+1)**2  
    self.agentMap = np.full((agentMapHeight, agentMapHeight), 4, dtype=int)
    self.agentMap[agentPosition[0], agentPosition[1]] = 2
    #print(self.agentMap)

  def setStepLimit(self, stepLimit):
    self.stepLimit = stepLimit

  def agentDetect(self):
    #given agents position, reveal adjacent grids
    #updates the agent map 
    agentPosition = self.objectCoords["agent"]
    #self.distFromObjective = self.calculateDistance() 
    #dictionary of adjacent grids

    detectionRadius = self.detectionRadius
    #print(agentPosition)
    agent_i = 0
    for i in range(agentPosition[0]-detectionRadius, agentPosition[0] + detectionRadius + 1):
      agent_j = 0
      for j in range(agentPosition[1]-detectionRadius, agentPosition[1] + detectionRadius + 1):
        if( i in range(0,self.shape[0]) and j in range(0,self.shape[0])):
          self.agentMap[agent_i,agent_j] = self.groundTruthMap[i,j]
          self.observationMap[i,j] = self.groundTruthMap[i,j]
          agent_j += 1
      
      agent_i += 1

    #print(self.agentMap)

  def calculateDistance(self):
    agentPos = np.array(self.objectCoords["agent"])
    objectivePos = np.array(self.objectCoords["objective"])
    dist = np.linalg.norm(agentPos-objectivePos)
    return dist 

  def calculateRewards(self):
    
    #distance = self.calculateDistance()

    numOfHiddenGrids = np.count_nonzero(self.observationMap == 3)
    #print(self.numOfHiddenGrids)
    #print(distance)
    #print(" ")
    #print(self.distFromObjective)
    if (numOfHiddenGrids < self.numOfHiddenGrids):
      self.numOfHiddenGrids = numOfHiddenGrids
      # if(distance <= 1):
      #   return 20
      return 0.1
    else:
        return 0


  def updateMaps(self, currPos, newPos):
    self.groundTruthMap[tuple(newPos)] = 2
    self.groundTruthMap[tuple(currPos)] = 0
    #self.agentMap[tuple(currPos)] = 0
    #self.agentMap[tuple(newPos)] = 2
    self.objectCoords["agent"] = newPos
    self.agentDetect()


  def numActionsAvailable(self):
    return self.action_space.n

  def getState(self):
    
    obsDict = {}

    obsDict["AgentMap"] = self.agentMap
    #obsDict["ObjectivePos"] = np.array(self.objectCoords["objective"])
    #obsDict["AgentPos"] = np.array(self.objectCoords["agent"])
    #obsDict["Difference"] = np.array(self.objectCoords["objective"]) - np.array(self.objectCoords["agent"])
    image = np.zeros((64,64,3))
    #print(self.observationMap.shape)
    image[:,:,0] = self.observationMap
    #obsDict["groundTruthMap"] = self.groundTruthMap
    obsDict["observationMap"] = image

    return obsDict

  def step(self, action):
    # [1,2,3,4] -> [up, down, left, right]
    done = False
    reward = 0
    currAgentPos = self.objectCoords["agent"] 
    self.stepLimit -= 1
    self.numOfHiddenGrids = np.count_nonzero(self.observationMap == 3)
    #self.distFromObjective = self.calculateDistance()
    if(self.stepLimit == 0):
      ##End of Episode
      #self.clearAgentMap()
      info = {}
      reward = 0
      done = True
      return self.getState(), reward, done, info
      
    if(action == 0):
      #move up
      newAgentPos = [currAgentPos[0]-1, currAgentPos[1]]

      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward = -1
        #self.clearAgentMap()
        #done = True 
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward = self.calculateRewards()
        # if(reward == 20):
        #   done = True
        #reward = self.calculateRewards()
          #print(self.objectCoords["agent"])
    elif(action == 1):
      #move down
      newAgentPos = [currAgentPos[0]+1, currAgentPos[1]]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward = -1
        #self.clearAgentMap()
        #done = True 
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward = self.calculateRewards()
        # if(reward == 20):
        #   done = True


    elif(action == 2):
      #move left
      newAgentPos = [currAgentPos[0], currAgentPos[1]-1]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward = -1
        #self.clearAgentMap()
        #done = True 
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward = self.calculateRewards()
        # if(reward == 20):
        #   done = True

    elif(action == 3):
      #move right
      newAgentPos = [currAgentPos[0], currAgentPos[1]+1]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward = -1
        #self.clearAgentMap()
        #done = True
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward = self.calculateRewards()
        # if(reward == 20):
        #   done = True
    
    info = {}
    #return self.getState()
    #print(self.getState())
    #print(self.objectCoords["agent"])
    return self.getState(), reward, done, info

  def reset(self):
    self.stepLimit = 100
    objectiveCoords = np.where(self.groundTruthMap == 3)
    agentCoords = np.where(self.groundTruthMap == 2)
    self.groundTruthMap[agentCoords] = 0
    self.groundTruthMap[objectiveCoords] =0
    self.spawnObjects()
    self.agentDetect()
    return self.getState()

  def render(self, mode='human'):
    
    #plt.imshow(self.groundTruthMap)
    #plt.show()
    plt.figure()
    plt.imshow(self.observationMap)
    plt.figure()
    plt.imshow(self.agentMap)
    plt.figure()
    plt.imshow(self.groundTruthMap)
    plt.show()

  def close(self):
    pass

class Explore2D_Env_Easy(Explore2D_Env):
  def __init__(self):

    #import ground truth
    pathToGroundTruthMap = "./maps/gridWorld_easy.csv"
    #pathToGroundTruthMap = "./maps/gridWorld_medium.csv"
    self.groundTruthMap = np.loadtxt(pathToGroundTruthMap, delimiter=",").astype(int)
    self.agentMap = None
    self.state = None #state is agentMap array combined with number of steps remaining
    self.shape = self.groundTruthMap.shape
    self.numOfHiddenGrids = 0
    self.detectionRadius = 1
    self.agentMapHeight = 2*self.detectionRadius+1 #detection radius is number of grids from agent to the up/down/left/right edges of detection area
    self.observationMap = np.full(self.groundTruthMap.shape,3)
    self.agentMap = np.empty([self.agentMapHeight, self.agentMapHeight])
    self.objectCoords = dict()
    #stepLimit is the maximum episode length -> emulates agent battery limits 
    self.stepLimit = 0
    self.distFromObjective = 0
    self.action_space = Discrete(4)

    self.observation_space = spaces.Dict({"AgentMap": spaces.Box(low = 0, high = 3, shape = self.agentMap.shape, dtype = np.uint8), 
                                          
                                          "ObjectivePos": spaces.Box(low = 0, high = 100, shape = np.array([1,2]).shape, dtype = float),
                                          "AgentPos": spaces.Box(low = 0, high = 100, shape = np.array([1,2]).shape, dtype = float),
                                          #"Difference": spaces.Box(low = 0, high = 255, shape = np.array([1,2]).shape, dtype = np.uint8)
                                          #"observationMap": spaces.Box(low = 0, high = 3, shape = self.groundTruthMap.shape, dtype = np.uint8),
                                          #"groundTruthMap": spaces.Box(low = 0, high = 3, shape = self.groundTruthMap.shape, dtype = np.uint8)
                                          })


class Explore2D_Env_Medium(Explore2D_Env):
  def __init__(self):
    #import ground truth
    pathToGroundTruthMap = "./maps/gridWorld_hard.csv"
    self.groundTruthMap = np.loadtxt(pathToGroundTruthMap, delimiter=",").astype(int)
    self.agentMap = None
    self.state = None #state is agentMap array combined with number of steps remaining
    self.shape = self.groundTruthMap.shape
    self.numOfHiddenGrids = 0
    self.detectionRadius = 4
    self.agentMapHeight = 2*self.detectionRadius+1 #detection radius is number of grids from agent to the up/down/left/right edges of detection area
    self.observationMap = np.full(self.groundTruthMap.shape,3)
    self.agentMap = np.empty([self.agentMapHeight, self.agentMapHeight])
    self.objectCoords = dict()
    #stepLimit is the maximum episode length -> emulates agent battery limits 
    self.stepLimit = 0
    self.distFromObjective = 0
    self.action_space = Discrete(4)

    self.observation_space = spaces.Dict({"AgentMap": spaces.Box(low = 0, high = 3, shape = self.agentMap.shape, dtype = np.uint8),                             
                                          #"AgentPos": spaces.Box(low = 0, high = 100, shape = np.array([1,2]).shape, dtype = float),
                                          "observationMap": spaces.Box(low = 0, high = 255, shape = (64,64,3), dtype = np.uint8),
                                          #"groundTruthMap": spaces.Box(low = 0, high = 3, shape = self.groundTruthMap.shape, dtype = np.uint8)
                                          })
#myEnv = Explore2D_Env()

class Explore2D_Env_Hard(Explore2D_Env):
  def __init__(self):
    #import ground truth
    pathToGroundTruthMap = "./maps/gridWorld_hard.csv"
    self.groundTruthMap = np.loadtxt(pathToGroundTruthMap, delimiter=",").astype(int)
    self.agentMap = None
    self.state = None #state is agentMap array combined with number of steps remaining
    self.shape = self.groundTruthMap.shape
    self.numOfHiddenGrids = 0
    self.detectionRadius = 1
    self.agentMapHeight = 2*self.detectionRadius+1 #detection radius is number of grids from agent to the up/down/left/right edges of detection area
    self.observationMap = np.full(self.groundTruthMap.shape,3)
    self.agentMap = np.empty([self.agentMapHeight, self.agentMapHeight])
    self.objectCoords = dict()
    #stepLimit is the maximum episode length -> emulates agent battery limits 
    self.stepLimit = 0
    self.distFromObjective = 0
    self.action_space = Discrete(4)

    self.observation_space = spaces.Dict({"AgentMap": spaces.Box(low = 0, high = 3, shape = self.agentMap.shape, dtype = np.uint8), 
                                          "observationMap": spaces.Box(low = 0, high = 255, shape = (64,64,3), dtype = np.uint8),
                                          
                                          })
#myEnv = Explore2D_Env()
#check_env(myEnv, warn=True)

# for i in range(10000):
#   randomMove = np.random.randint(low =1, high = 5)
#   myEnv.step(randomMove)
#   #myEnv.render()

# myEnv.render()
# myEnv.reset()
# myEnv.render()