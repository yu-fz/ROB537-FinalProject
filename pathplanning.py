# https://stackabuse.com/dijkstras-algorithm-in-python/
import numpy as np
import pdb
from operator import itemgetter
from csv import writer
from queue import PriorityQueue
# generate a graph and update the weights
# find the shortest path
# save the path
class DijsktraSearch():
    def __init__(self, dmap, fgoal, sourceNode):
        self.sourceNode = sourceNode
        self.dmap = dmap
        self.fgoal = fgoal
        self.graph = []
        self.edges = []
        self.final_path = []
        self.weights = [0 for i in range(len(self.dmap))]
        self.updateGraph()
    
    def updateGraph(self):
        self.graph.append([(int(self.sourceNode[0]), int(self.sourceNode[1])), 0])
        k = 1
        for node in self.dmap:
            if node != self.sourceNode:
                self.graph.append([node, k])
                k = k + 1
        for node in self.graph:
            upnode = (node[0][0] - 1, node[0][1])
            leftnode = (node[0][0], node[0][1] - 1)
            if upnode in self.dmap:
                self.edges.append([node[1], self.graph[self.dmap.index(upnode)][1], 1])
            if leftnode in self.dmap:
                self.edges.append([node[1], self.graph[self.dmap.index(leftnode)][1], 1])
        p = 0
        for node in self.edges:
            if node[0] == node[1]:
                self.edges.pop(p)
            p = p + 1
        
        for node in self.graph:
            if node[0] == self.fgoal:
                self.fvertex = node[1]
                break
        for node in self.graph:
            if node[0] == self.sourceNode:
                self.gvertex = node[1]
                # print(self.gvertex,"init")
                break

    def dijkstra(self):
        self.numV = len(self.graph)
        D = {v:float('inf') for v in range(self.numV)}
        self.visited = []
        D[0] = 0

        pq = PriorityQueue()
        # Distance from vertex 0 to vertex 0 is 0
        pq.put((0, 0))
        while not pq.empty():
            (dist, current_vertex) = pq.get()
            self.visited.append(current_vertex)
            # print("current node", current_vertex)
            for i in range(len(self.edges)):
                # locate neighbours
                if self.edges[i][0] == current_vertex:
                    distance = self.edges[i][2]
                    # if its neighbour not visited - update weight
                    if self.edges[i][1] not in self.visited:
                        old_cost = D[self.edges[i][1]]
                        new_cost = D[current_vertex] + distance
                        if new_cost <= old_cost:
                            pq.put((new_cost, self.edges[i][1]))
                            # print("added an element in pq", self.edges[i][1])
                            D[self.edges[i][1]] = new_cost

                if self.edges[i][1] == current_vertex:
                    distance = self.edges[i][2]
                    # if its neighbour not visited - update weight
                    if self.edges[i][0] not in self.visited:
                        old_cost = D[self.edges[i][0]]
                        new_cost = D[current_vertex] + distance
                        if new_cost <= old_cost:
                            pq.put((new_cost, self.edges[i][0]))
                            # print("added an element in pq", self.edges[i][0])
                            D[self.edges[i][0]] = new_cost
        for i in D.keys():
            if D[i] != float('inf'):
                self.final_path.append([i, D[i]])
        self.savePath()
        return(D[self.fvertex])

    def savePath(self):
        with open('dijsktra_path.csv', 'a') as f_object:
            f_object.write(str(self.final_path))
            f_object.write(str(len(self.final_path)))
            f_object.close()
