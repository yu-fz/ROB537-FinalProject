import gym
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import random
from gym import error, spaces, utils
from gym.spaces import Discrete, Box, Dict
from gym.utils import seeding
import time
import pdb
from pathplanning import DijsktraSearch
from frontiersnew import FrontierPointFinder
from collections import deque

################### have to see if the loop ever ends

class Explore2D_Env(gym.Env):
  metadata = {'render.modes': ['human']}
    ###logic for environment goes here
  def __init__(self, **kwargs):
    #import ground truth

    pathToGroundTruthMap = kwargs["map"]
    self.groundTruthMap = np.loadtxt(pathToGroundTruthMap, delimiter=",").astype(int)
    self.agentMapHistory = deque(maxlen=3)
    self.agentDistanceHistory = deque(maxlen=3)
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
    self.objectiveCoord = None
    self.distFromObjective = 0
    self.action_space = Discrete(4)
    self.fgoal = (0,0)

    # self.observation_space = spaces.Dict({"AgentMap": spaces.Box(low = 0, high = 3, shape = self.agentMap.shape, dtype = np.uint8), 
                                          
    #                                       #"ObjectivePos": spaces.Box(low = 0, high = 100, shape = np.array([1,2]).shape, dtype = float),
    #                                       #"AgentPos": spaces.Box(low = 0, high = 100, shape = np.array([1,2]).shape, dtype = float),
    #                                       "Difference": spaces.Box(low = 0, high = 255, shape = np.array([1,2]).shape, dtype = float)
    #                                       #"observationMap": spaces.Box(low = 0, high = 3, shape = self.groundTruthMap.shape, dtype = np.uint8),
    #                                       #"groundTruthMap": spaces.Box(low = 0, high = 3, shape = self.groundTruthMap.shape, dtype = np.uint8)
    #                                       })

    self.observation_space = spaces.Dict({"AgentMap": spaces.Box(low = 0, high = 3, shape = (3,3,3)), 
                                          
                                          #"ObjectivePos": spaces.Box(low = 0, high = 100, shape = np.array([1,2]).shape, dtype = float),
                                          #"AgentPos": spaces.Box(low = 0, high = 100, shape = np.array([1,2]).shape, dtype = float),
                                          "Difference": spaces.Box(low = 0, high = 255, shape = (3,2))
                                          #"observationMap": spaces.Box(low = 0, high = 3, shape = self.groundTruthMap.shape, dtype = np.uint8),
                                          #"groundTruthMap": spaces.Box(low = 0, high = 3, shape = self.groundTruthMap.shape, dtype = np.uint8)
                                          })

  def getEnvSize(self):
    return self.shape
  
  def setObjectiveCoord(self, objectiveCoord):
    self.objectiveCoord = objectiveCoord
    self.spawnObjective(objectiveCoord)

  def spawnAgent(self):
    coordList = []
    random.seed(time.time())
    for i in range(2):
      #generate 4 random numbers inside map bounds to use as spawn coordinates for agent
      #and objective
      coordList.append(random.randint(1, self.shape[0]-2))
    
    self.objectCoords["agent"] = coordList
    self.objectCoords["objective"] = [0,0]


    agentXCoord = self.objectCoords["agent"][1]
    agentYCoord = self.objectCoords["agent"][0]    
    #print("agent spawned at " + str(self.objectCoords["agent"]))
    
    #check if random spawn location is obstructed
    while(self.groundTruthMap[agentYCoord, agentXCoord] == 1):
      #if grid is obstructed, spawn again 
      #print("oops")  
      
      self.objectCoords["agent"][1] = np.random.randint(low = 1, high = self.shape[1]-1)
      self.objectCoords["agent"][0] = np.random.randint(low = 1, high = self.shape[0]-1)
      agentXCoord = self.objectCoords["agent"][1]
      agentYCoord = self.objectCoords["agent"][0]
    
    self.groundTruthMap[agentYCoord, agentXCoord] = 2 #agent represented as a 2 

  def spawnObjective(self, objectiveCoord = None):


    epsilon = 20


    if(objectiveCoord is not None):

        self.objectCoords["objective"][1] = self.objectiveCoord[1]
        self.objectCoords["objective"][0] = self.objectiveCoord[0]
        objectiveXCoord = self.objectiveCoord[1]
        objectiveYCoord = self.objectiveCoord[0]
        #print(self.objectCoords["objective"])
        #self.groundTruthMap[objectiveYCoord, objectiveXCoord] = 3 

    else:
    
      self.objectCoords["objective"][1] = np.random.randint(low = max(self.objectCoords["agent"][1] - epsilon, 1), high = min(self.objectCoords["agent"][1] + epsilon, self.shape[1]-1 ))
      self.objectCoords["objective"][0] = np.random.randint(low = max(self.objectCoords["agent"][0] - epsilon, 1), high = min(self.objectCoords["agent"][0] + epsilon, self.shape[1]-1 ))

      objectiveXCoord = self.objectCoords["objective"][1]
      objectiveYCoord = self.objectCoords["objective"][0]


      while(self.groundTruthMap[objectiveYCoord, objectiveXCoord] == 1 or self.calculateDistance() < 2 ):
        #if grid is obstructed, spawn again 
  
        self.objectCoords["objective"][1] = np.random.randint(low = max(self.objectCoords["agent"][1] - epsilon, 1), high = min(self.objectCoords["agent"][1] + epsilon, self.shape[1]-1 ))
        self.objectCoords["objective"][0] = np.random.randint(low = max(self.objectCoords["agent"][0] - epsilon, 1), high = min(self.objectCoords["agent"][0] + epsilon, self.shape[1]-1 ))
        objectiveXCoord = self.objectCoords["objective"][1]
        objectiveYCoord = self.objectCoords["objective"][0]


    #print("agent spawned at" + str(self.objectCoords["agent"]))
    #self.groundTruthMap[objectiveYCoord, objectiveXCoord] = 3 
  
  # def generateAgentMap(self):
  #   agentPosition = self.objectCoords["agent"]
  #   agentMapHeight = (2*self.detectionRadius+1)**2  
  #   self.agentMap = np.full((agentMapHeight, agentMapHeight), 4, dtype=int)
  #   self.agentMap[agentPosition[0], agentPosition[1]] = 2
  #   #print(self.agentMap)

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
        if( i in range(0,self.shape[0]) and j in range(0,self.shape[1])):
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
    
    done = False

    oldDistanceFromObjective = self.distFromObjective
    newDistanceFromObjective = self.calculateDistance()

    if(newDistanceFromObjective < oldDistanceFromObjective):

      if(newDistanceFromObjective <= 2):

        reward = 20
        done = True

        return reward, done
      
      else: 

        reward = 0.1
        return reward, done

    if(self.agentDistanceHistory[2].all() == self.agentDistanceHistory[0].all()):

      reward = -5
      done = True

      return reward, done

    else:

      reward = 0

      return reward, done 
    
    pass 

  def getCollisionReward(self):
    
    reward = -5
    done = True
    
    return reward, done

  def updateMaps(self, newPos, currPos):
    self.groundTruthMap[tuple(newPos)] = 2
    self.groundTruthMap[tuple(currPos)] = 0
    #self.agentMap[tuple(currPos)] = 0
    #self.agentMap[tuple(newPos)] = 2
    self.agentDetect()
    self.agentMapHistory.append(self.agentMap)
    self.agentDistanceHistory.append(np.array(self.objectCoords["objective"]) - np.array(self.objectCoords["agent"]))

  def numActionsAvailable(self):
    return self.action_space.n

  def getState(self):
    
    obsDict = {}

    a = np.empty((3,3,3))
    a[:,:,0] = self.agentMapHistory[0]
    a[:,:,1] = self.agentMapHistory[1]
    a[:,:,2] = self.agentMapHistory[2]


    obsDict["AgentMap"] = a
    obsDict["Difference"] = [self.agentDistanceHistory[0], self.agentDistanceHistory[1], self.agentDistanceHistory[2]]
    #obsDict["Difference"] = np.array(self.objectCoords["objective"]) - np.array(self.objectCoords["agent"])

    return obsDict

  def step(self, action):
    # [1,2,3,4] -> [up, down, left, right]
    done = False
    
    reward = 0
    
    currAgentPos = self.objectCoords["agent"] 
    
    self.stepLimit -= 1

    self.distFromObjective = self.calculateDistance()
    
    if(self.stepLimit == 0):
      ##End of Episode
      info = {}
      reward = -20
      done = True
      return self.getState(), reward, done, info
      
    if(action == 0):
      #move up
      newAgentPos = [currAgentPos[0]-1, currAgentPos[1]]

      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #what to do if agent hits a wall
        #terminate, return done and give penalty or whatever
        reward, done = self.getCollisionReward()

      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward, done = self.calculateRewards()

    elif(action == 1):
      #move down
      newAgentPos = [currAgentPos[0]+1, currAgentPos[1]]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward, done = self.getCollisionReward()

      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward, done = self.calculateRewards()



    elif(action == 2):
      #move left
      newAgentPos = [currAgentPos[0], currAgentPos[1]-1]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward, done = self.getCollisionReward()
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward, done = self.calculateRewards()


    elif(action == 3):
      #move right
      newAgentPos = [currAgentPos[0], currAgentPos[1]+1]
      if(self.groundTruthMap[tuple(newAgentPos)] == 1):
        #terminate, return done and give penalty or whatever
        reward, done = self.getCollisionReward()
      else:
        self.updateMaps(currAgentPos, newAgentPos)
        reward, done = self.calculateRewards()

    info = {}

    return self.getState(), reward, done, info

  def returnFrontierMap(self):

    return self.observationMap

  def resetFrontier(self):
    self.stepLimit = self.shape[0]
    # objectiveCoords = np.where(self.groundTruthMap == 3)
    # agentCoords = np.where(self.groundTruthMap == 2)
    # self.groundTruthMap[agentCoords] = 0
    # self.groundTruthMap[objectiveCoords] =0
    # self.spawnAgent()
    self.spawnObjective(self.objectiveCoord)
    self.agentDistanceHistory.append(np.array(self.objectCoords["objective"]) - np.array(self.objectCoords["agent"]))
    # self.agentDetect()
    
    # for i in range(3):

    #   self.agentMapHistory.append(self.agentMap)
    #   self.agentDistanceHistory.append(np.array(self.objectCoords["objective"]) - np.array(self.objectCoords["agent"]))

    return self.getState()

  def returnExplorationProgress(self):

    totalNumberOfFreeGrids = len(np.where(self.groundTruthMap == 0)[0])
    numOfRevealedFreeGrids = len(np.where(self.observationMap == 0)[0])

    exploreFrac = (numOfRevealedFreeGrids/totalNumberOfFreeGrids)

    return exploreFrac

  def updateMaps2(self, fgoal):
    print(fgoal)
    # down
    if fgoal[0] + 1 < self.shape[0]:
      if self.observationMap[fgoal[0] + 1][fgoal[1]] == 3:
        self.observationMap[fgoal[0] + 1][fgoal[1]] = self.groundTruthMap[fgoal[0] + 1][fgoal[1]]
        if fgoal[1] + 1 <self.shape[1]: 
          self.observationMap[fgoal[0] + 1][fgoal[1] + 1] = self.groundTruthMap[fgoal[0] + 1][fgoal[1] + 1]
        if fgoal[1] - 1 >= 0: 
          self.observationMap[fgoal[0] + 1][fgoal[1] - 1] = self.groundTruthMap[fgoal[0] + 1][fgoal[1] - 1]
    # up
    if fgoal[0] - 1 >= 0:
      if self.observationMap[fgoal[0] - 1][fgoal[1]] == 3:
        self.observationMap[fgoal[0] - 1][fgoal[1]] = self.groundTruthMap[fgoal[0] - 1][fgoal[1]]
        if fgoal[1] + 1 <self.shape[1]: 
          self.observationMap[fgoal[0] - 1][fgoal[1] + 1] = self.groundTruthMap[fgoal[0] - 1][fgoal[1] + 1]
        if fgoal[1] - 1 >= 0: 
          self.observationMap[fgoal[0] - 1][fgoal[1] - 1] = self.groundTruthMap[fgoal[0] - 1][fgoal[1] - 1]
    # left
    if fgoal[1] + 1 < self.shape[1]:
      if self.observationMap[fgoal[0]][fgoal[1] + 1] == 3:
        self.observationMap[fgoal[0]][fgoal[1] + 1] = self.groundTruthMap[fgoal[0]][fgoal[1] + 1]
        if fgoal[0] + 1 <self.shape[0]: 
          self.observationMap[fgoal[0] + 1][fgoal[1] + 1] = self.groundTruthMap[fgoal[0] + 1][fgoal[1] + 1]
        if fgoal[0] - 1 >= 0: 
          self.observationMap[fgoal[0] - 1][fgoal[1] + 1] = self.groundTruthMap[fgoal[0] - 1][fgoal[1] + 1]
    
    # right
    if fgoal[1] - 1 >= 0:
      if self.observationMap[fgoal[0]][fgoal[1] + 1] == 3:
        self.observationMap[fgoal[0]][fgoal[1] + 1] = self.groundTruthMap[fgoal[0]][fgoal[1] + 1]
        if fgoal[0] + 1 <self.shape[0]: 
          self.observationMap[fgoal[0] + 1][fgoal[1] + 1] = self.groundTruthMap[fgoal[0] + 1][fgoal[1] + 1]
        if fgoal[0] - 1 >= 0: 
          self.observationMap[fgoal[0] - 1][fgoal[1] + 1] = self.groundTruthMap[fgoal[0] - 1][fgoal[1] + 1]
      
  def performDijsktra(self):
    # get in the frontier goal and the dijsktra map
    #pdb.set_trace()
    snode = np.where(self.groundTruthMap == 2)
    print(snode)
    fvar = FrontierPointFinder(self.returnFrontierMap())
    fgoal, dmap = fvar.findFrontierCoords()
    # update the observation map near the fgoal
    self.updateMaps2(fgoal)
    self.updateMaps(snode, fgoal)

    # save the path for analysis
    dvar = DijsktraSearch(dmap, fgoal, snode)
    return fgoal, dvar.dijkstra(), snode

  def reset(self):
    self.stepLimit = 50
    objectiveCoords = np.where(self.groundTruthMap == 3)
    agentCoords = np.where(self.groundTruthMap == 2)
    self.groundTruthMap[agentCoords] = 0
    self.groundTruthMap[objectiveCoords] =0
    self.spawnAgent()
    self.spawnObjective()
    self.agentDetect()
    
    for i in range(3):

      self.agentMapHistory.append(self.agentMap)
      self.agentDistanceHistory.append(np.array(self.objectCoords["objective"]) - np.array(self.objectCoords["agent"]))

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

