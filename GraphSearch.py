import numpy as np 
import random

from numpy.core.fromnumeric import _trace_dispatcher 
from frontiers import FrontierPointFinder
import gym_Explore2DTrial
import gym

import wandb
import torch as th
import time

class Exp_GraphSearch:
    def __init__(self, env):
#        self.traceNode = []
        self.env = env
        self.newMap = np.array([i for i in self.env.groundTruthMap])
        self.newMapWeight = [[0 for i in range (0, self.newMap.shape[0] * self.newMap.shape[1])] for j in range(0, self.newMap.shape[0] * self.newMap.shape[1])]
        self.newMapTarget = [[0 for i in range (0, self.newMap.shape[1])] for j in range(0, self.newMap.shape[0])]
        for i in range(0, self.newMap.shape[0]):
            for j in range(0, self.newMap.shape[1]):
                if (self.newMap[i, j] == 1):
                    self.newMap[i, j] = 0
        self.numVertices = self.newMap.shape[0] * self.newMap.shape[1] - 1 
        self.total_steps = 0
        self.initFlag = True
    
    def CheckBoundary(self, rowID, columnID):
        assignFrontierFlag = False

        if (self.frontierMap[rowID - 1, columnID] == 3):
            assignFrontierFlag = True
        elif (self.frontierMap[rowID + 1, columnID] == 3):
            assignFrontierFlag = True
        elif (self.frontierMap[rowID, columnID - 1] == 3):
            assignFrontierFlag = True
        elif (self.frontierMap[rowID, columnID + 1] == 3):
            assignFrontierFlag = True

        return assignFrontierFlag
    
    def FindFrontier(self):
        self.frontierMap = np.array([i for i in self.env.returnFrontierMap()])
        shape = self.frontierMap.shape
        agentPosition = np.where(self.frontierMap == 2) 
        agentCoord = np.array([agentPosition[0][0],agentPosition[1][0]])
        rowID = agentCoord[0]
        columnID = agentCoord[1]
        self.newMap[rowID, columnID] = 2

        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                if (self.frontierMap[i, j] == 0):
                    assignFrontierFlag = self.CheckBoundary(i, j)
                    self.newMap[i, j] = 1
                    if(assignFrontierFlag):
                        self.frontierMap[i, j] = 4

    
    def MinDistanceVertex(self, dist, sptSet, numVertices):
 
        min = float("Inf")
        check = False

        for v in range(numVertices):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                check = True
                min_index = v
 
        if (not check):
            return check
        return min_index

    def dijkstra_compute(self, sourceNode, numVertices, targetNode):
 
        dist = float("Inf") * np.ones(numVertices)
        dist[sourceNode] = 0
        sptSet = [False] * numVertices
 
        for cout in range(numVertices):
 
            u = self.MinDistanceVertex(dist, sptSet, numVertices)
            if ( u == False ):
                return u
 
            sptSet[u] = True
 
            setFlag = False
            for v in range(numVertices):
                if self.newMapWeight[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.newMapWeight[u][v]:
                    dist[v] = dist[u] + self.newMapWeight[u][v]
                    if (v == targetNode):
                        return dist[v]


    
    def CreateWeightMatrix(self):
        frontierNodes = np.where(self.newMap == 1)
        for i in range(0, len(frontierNodes[0])):
            frontierNode_to_vertice = frontierNodes[0][i] * self.newMap.shape[0] + frontierNodes[1][i]
            dist_frontier_source = np.linalg.norm([frontierNodes[0][i] - self.sourceNode[0][0], frontierNodes[1][i] - self.sourceNode[1][0]])
            if ( dist_frontier_source == 1 ):
                self.newMapWeight[self.sourceNode_to_vertice][frontierNode_to_vertice] = 1
                self.newMapWeight[frontierNode_to_vertice][self.sourceNode_to_vertice] = self.newMapWeight[self.sourceNode_to_vertice][frontierNode_to_vertice]

        for i in range(0, len(frontierNodes[0]) - 1):
            #j = i + 1
            for j in range(i + 1, len(frontierNodes[0])):
                if ( frontierNodes[0][i] == frontierNodes[0][j] ):
                    if (abs(frontierNodes[1][i] - frontierNodes[1][j]) == 1):
                        x = frontierNodes[0][i] * self.newMap.shape[0] + frontierNodes[1][i]
                        y = frontierNodes[0][j] * self.newMap.shape[0] + frontierNodes[1][j]
                        self.newMapWeight[x][y] = 1
                        self.newMapWeight[y][x] = 1

        for i in range(0, len(frontierNodes[1]) - 1):
            #j = i + 1
            for j in range(i + 1, len(frontierNodes[0])):
                if ( frontierNodes[1][i] == frontierNodes[1][j] ):
                    if (abs(frontierNodes[0][i] - frontierNodes[0][j]) == 1):
                        x = frontierNodes[0][i] * self.newMap.shape[0] + frontierNodes[1][i]
                        y = frontierNodes[0][j] * self.newMap.shape[0] + frontierNodes[1][j]
                        self.newMapWeight[x][y] = 1
                        self.newMapWeight[y][x] = 1     
    
    
    def FindMinFrontier(self):
        targetCoords = np.where(self.frontierMap == 4)
        
        if (self.initFlag == False):
            count = random.choice([i for i in range(0, len(targetCoords[0]))])
            targetCoord_x = targetCoords[0][count]
            targetCoord_y = targetCoords[1][count]
        else:
            min = float("Inf")

            for i in range(0, len(targetCoords[0])):
                dist_target_source = np.linalg.norm([targetCoords[0][i] - self.sourceNode[0][0], targetCoords[1][i] - self.sourceNode[1][0]])
                self.newMapTarget[targetCoords[0][i]][targetCoords[1][i]] = dist_target_source
            
                if (self.newMapTarget[targetCoords[0][i]][targetCoords[1][i]] < min):
                    min = self.newMapTarget[targetCoords[0][i]][targetCoords[1][i]]
                    targetCoord_x = targetCoords[0][i]
                    targetCoord_y = targetCoords[1][i]
        
        return targetCoord_x, targetCoord_y
    
    def ComputeTargetVertice(self):
        targetCoord_x, targetCoord_y = self.FindMinFrontier() 
        
        self.target_vertice = targetCoord_x * self.newMap.shape[0] + targetCoord_y
        return targetCoord_x, targetCoord_y
    
    def UpdateMap(self, targetCoord):
        self.env.updateMaps(self.sourceNode, targetCoord)
    
    def Exp_Dijkstra_Frontier(self):
        
        if (True):
            self.FindFrontier()       ## newMap == Frontier
            self.sourceNode = np.where(self.newMap == 2) 
            self.sourceNode_to_vertice = self.sourceNode[0][0] * self.newMap.shape[0] + self.sourceNode[1][0]
        
            self.CreateWeightMatrix()
            
            path_step = False
            self.initFlag = True
            while (path_step == False):
                (target_x, target_y) = self.ComputeTargetVertice()
                self.initFlag = False
                path_step = self.dijkstra_compute(self.sourceNode_to_vertice, self.numVertices, self.target_vertice)

            self.total_steps = self.total_steps + path_step
            targetCoord = (target_x, target_y)
            self.UpdateMap(targetCoord)


if __name__ == "__main__":
    envName = "Explore2DTrial-v0"
    path_to_map = "./maps/gridWorld_hard.csv"
    env = gym.make(envName, map = path_to_map)
    env.reset()
    stepCounter = 0

    g = Exp_GraphSearch(env)
    
    startTime = time.time()

    while (env.returnExplorationProgress() < 0.90):
        g.Exp_Dijkstra_Frontier()
        stepCounter+=1

    print("agent completed {a} time steps".format(a = stepCounter))
    print("Run Time: ------%s seconds" % (time.time() - startTime))
    print("Total Steps:", g.total_steps)
    env.render()
