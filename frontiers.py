from matplotlib.image import NonUniformImage
import numpy as np 
import random 

class FrontierPointFinder:
     
    def __init__(self, frontierMap):
         
        self.frontierMap = frontierMap
        self.frontierMapShape = frontierMap.shape   
        self.frontierCoords = []
        self.hiddenCoords = []
        self.agentPosition = np.where(self.frontierMap == 2) 
        self.freeCoords = []


    def updateFrontierMap(self, map):

        self.frontierMap = map 
        self.frontierMapShape = map.shape
    
    def findFreeCoords(self):

        freeCoords = np.where(self.frontierMap == 0)

        self.freeCoords.clear()
        for i in range(len(freeCoords[0])):

            freeCoordPair = np.array([freeCoords[0][i], freeCoords[1][i]])
            self.freeCoords.append(freeCoordPair)

        
        #print(self.frontierMap[0,0])

    def findHiddenCoords(self):

        hiddenCoords = np.where(self.frontierMap == 3)

        self.hiddenCoords.clear()

        for i in range(len(hiddenCoords[0])):

            hiddenCoordPair = np.array([hiddenCoords[0][i], hiddenCoords[1][i]])
            self.hiddenCoords.append(hiddenCoordPair)

        
        #print(self.frontierMap[0,0])


    def findFrontierCoords(self):

        pass 
    
    def returnTargetFrontierPoint(self):

        self.findHiddenCoords()
        #self.findFrontierCoords()
        return random.choice(self.hiddenCoords)

        distanceDict = {}
        nearestWaypoints = []

        agentPositionCoord = np.array([self.agentPosition[0][0],self.agentPosition[1][0]])#, self.agentPosition[1][0]])
        for element in self.frontierCoords:
            
            #print(element)
            distanceDict[tuple(element)] = np.linalg.norm(agentPositionCoord - element)
        

        epsilon = 1
        diceRoll = random.random()

        if (diceRoll <= epsilon):
            self.findHiddenCoords()
            return random.choice(self.hiddenCoords)

        else:
            return min(distanceDict, key=distanceDict.get) 


        if(len(distanceDict) > 40):
            #return min(distanceDict, key=distanceDict.get)

            for i in range(40):
                targetFrontier = min(distanceDict, key=distanceDict.get)
                nearestWaypoints.append(targetFrontier)
                del distanceDict[targetFrontier]

            return random.choice(nearestWaypoints)
            




        else: 
        
            targetFrontier = random.choice(list(distanceDict))
           # targetFrontier = min(distanceDict, key=distanceDict.get)

        return targetFrontier



        













    





