from matplotlib.image import NonUniformImage
import numpy as np 
import random 
import pdb
import gym_Explore2D

class FrontierPointFinder:
     # fmap - initial the observation map
    def __init__(self, frontierMap, visited_list, fmap):
         
        self.frontierMap = frontierMap
        self.frontierMapShape = frontierMap.shape   
        self.frontierCoords = []
        self.hiddenCoords = []
        self.agentPosition = np.where(self.frontierMap == 2) 
        self.freeCoords = []
        self.tragetFrontier = []
        self.visited = visited_list
        self.old_f = fmap

    def FindFrontierMap(self):
        agentPosition = np.where(self.frontierMap == 2)
        agentCoord = np.array([agentPosition[0][0],agentPosition[1][0]])
        
        newmap = np.copy(self.frontierMap)
        shape = newmap.shape
        j = [agentCoord[1]-1, agentCoord[1]+1]

        # create a list of 4, add only if
        # 1. not on visited location
        # 2. previously 3 on observation map

        for k in range(0, len(j)):
          for i in range(agentCoord[0]-1, agentCoord[0]+2):
            if (i in range(0,shape[0]) and j[k] in range(0, shape[1])):
              if (newmap[i, j[k]] == 0):
                newmap[i, j[k]] = 4

        j = [agentCoord[0]-1, agentCoord[0]+1]

        for k in range(0, len(j)):
          for i in range(agentCoord[1]-1, agentCoord[1]+2):
            if (i in range(0,shape[1]) and j[k] in range(0, shape[0])):
              if (newmap[j[k], i] == 0):
                newmap[j[k], i] = 4
        print(newmap, "4 around old 2")
        return newmap

    def findFrontierCoordsDijsktra2(self):
        # the frontier points are on the boundary of the explored map
        # [[3 3 3 3]
        # [1 4 3 3 ]
        # [2 0 4 3]
        # [1 0 4 3]
        # [3 4 3 3]]
        # 1 - obstacle space
        # 2 - current location
        # 4 - frontier to explore
        # 3 - hidden unknown
        # 0 - open space
        
        # appending the 0,1 to the dijsktra map
        list_coordinate0 = np.where(self.frontierMap == 0)
        list_coordinate = np.where(self.frontierMap == 1)
        ###### 1. adding 4 and 1 to from the observation space
        for i in range(len(list_coordinate0[0])):
            #if self.old_f[(list_coordinate0[0][i], list_coordinate0[1][i])] == 4:
            self.old_f[(list_coordinate0[0][i], list_coordinate0[1][i])] = 4
        for i in range(len(list_coordinate[0])):
            self.old_f[(list_coordinate[0][i], list_coordinate[1][i])] = 1

        self.dijsktraMap = []
        for i in range(len(list_coordinate[0])):
            self.dijsktraMap.append((list_coordinate[0][i], list_coordinate[1][i]))
        for i in range(len(list_coordinate0[0])):
            self.dijsktraMap.append((list_coordinate0[0][i], list_coordinate0[1][i]))
        
        ###### 2. generate 4 for the initial location on the map
        agentPosition = np.where(self.frontierMap == 2)
        self.old_f[agentPosition[0][0],agentPosition[1][0]] = 0
        #newmap = self.FindFrontierMap()
        # removing 4 prev loc
        #newmap[agentPosition[0][0],agentPosition[1][0]] = 0
        #print(newmap, "removing prev 2")
        
        # get 4 from old frontier map
        # also remove from the visited list
        """get_oldf = np.where(self.old_f == 4)
        for i in range(0, len(get_oldf[0])):
            if (get_oldf[0][i], get_oldf[1][i]) in self.visited:
                print("point in visited - updating", (get_oldf[0][i], get_oldf[1][i]))
                newmap[get_oldf[0][i], get_oldf[1][i]] = 0
            else:
                newmap[get_oldf[0][i], get_oldf[1][i]] = 4

        print(newmap, "adding 4 from old map")"""
        
        list_coordinate2 = np.where(self.old_f == 4)
        self.dijsktraMap.append((agentPosition[0][0],agentPosition[1][0]))
        for i in range(len(list_coordinate2[0])):
            self.dijsktraMap.append((list_coordinate2[0][i], list_coordinate2[1][i]))
            self.frontierCoords.append((list_coordinate2[0][i], list_coordinate2[1][i]))
        # remove visited open space
        for pts in self.frontierCoords:
            if (pts in self.visited):
                self.frontierCoords.remove(pts)
        # save the updated frontier 4 to the observation space
        """self.old_f = np.copy(newmap)"""
       
    def findFrontierCoordsDijsktra(self):
        # for each non 3 vertex on the observation map, if it has a neighbour 3, make it 4
        list_coordinate = np.where(self.frontierMap == 3)
        for i in range(len(list_coordinate[0])):
            # checking up
            if list_coordinate[0][i] - 1 >=0:
                if self.frontierMap[(list_coordinate[0][i] - 1, list_coordinate[1][i])] == 0:
                    self.old_f[(list_coordinate[0][i] - 1, list_coordinate[1][i])] = 4
                if self.frontierMap[(list_coordinate[0][i] - 1, list_coordinate[1][i])] == 1:
                    self.old_f[(list_coordinate[0][i] - 1, list_coordinate[1][i])] = 1
            # checking down
            if list_coordinate[0][i] + 1 < self.frontierMap.shape[0]:
                if self.frontierMap[(list_coordinate[0][i] + 1, list_coordinate[1][i])] == 0:
                    self.old_f[(list_coordinate[0][i] + 1, list_coordinate[1][i])] = 4
                if self.frontierMap[(list_coordinate[0][i] + 1, list_coordinate[1][i])] == 1:
                    self.old_f[(list_coordinate[0][i] + 1, list_coordinate[1][i])] = 1

            # checking left
            if list_coordinate[1][i] - 1 >=0:
                if self.frontierMap[(list_coordinate[0][i], list_coordinate[1][i] - 1)] == 0:
                    self.old_f[(list_coordinate[0][i], list_coordinate[1][i] - 1)] = 4
                    if self.frontierMap[(list_coordinate[0][i], list_coordinate[1][i] - 1)] == 1:
                        self.old_f[(list_coordinate[0][i], list_coordinate[1][i] - 1)] = 1
            # checking right
            if list_coordinate[1][i] + 1 < self.frontierMap.shape[0]:
                if self.frontierMap[(list_coordinate[0][i], list_coordinate[1][i] + 1)] == 0:
                    self.old_f[(list_coordinate[0][i], list_coordinate[1][i] + 1)] = 4
                if self.frontierMap[(list_coordinate[0][i], list_coordinate[1][i] + 1)] == 1:
                    self.old_f[(list_coordinate[0][i], list_coordinate[1][i] + 1)] = 1
        
        # generate observation for the initial location on the map
        agentPosition = np.where(self.frontierMap == 2)
        self.old_f[agentPosition[0][0],agentPosition[1][0]] = 0
        

        # remove the visited 4

        # appending the 0,1 to the dijsktra map
        list_coordinate0 = np.where(self.frontierMap == 0)
        list_coordinate3 = np.where(self.frontierMap == 1)
        self.dijsktraMap = []
        for i in range(len(list_coordinate3[0])):
            self.dijsktraMap.append((list_coordinate3[0][i], list_coordinate3[1][i]))
        for i in range(len(list_coordinate0[0])):
            self.dijsktraMap.append((list_coordinate0[0][i], list_coordinate0[1][i]))
        
        
        list_coordinate2 = np.where(self.old_f == 4)
        self.dijsktraMap.append((agentPosition[0][0],agentPosition[1][0]))
        for i in range(len(list_coordinate2[0])):
            self.dijsktraMap.append((list_coordinate2[0][i], list_coordinate2[1][i]))
            self.frontierCoords.append((list_coordinate2[0][i], list_coordinate2[1][i]))
        # remove visited open space
        for pts in self.frontierCoords:
            if (pts in self.visited):
                self.frontierCoords.remove(pts)
    
    def findFrontierCoords(self):
        # the point with the minimum distance from the current location
        self.findFrontierCoordsDijsktra()
        mindist = 1000
        for pts in self.frontierCoords:
            dist = np.sqrt(np.power(self.agentPosition[0] - pts[0], 2) + np.power(self.agentPosition[1] - pts[1], 2))
            if dist <= mindist:
                mindist = dist
                self.tragetFrontier = pts
        ##### 3. add target to the map
        self.old_f[self.tragetFrontier] = 2

        return self.tragetFrontier, self.dijsktraMap, self.old_f

    def returnTargetFrontierPoint(self):
        return self.findFrontierCoords()
