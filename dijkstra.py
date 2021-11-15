import numpy

class DijsktraSearch():
    def __init__(self, vertices):
        self.numVertices = vertices
        self.weightMatrix = [[0 for column in range(vertices)] for row in range(vertices)]
        self.traceNode = []

 
    def MinDistanceVertex(self, dist, sptSet):
 
        min = float("Inf")

        for v in range(self.numVertices):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v
 
        return min_index


    def dijkstra(self, sourceNode):
 
        dist = float("Inf") * numpy.ones(self.numVertices)
        dist[sourceNode] = 0
        sptSet = [False] * self.numVertices
 
        for cout in range(self.numVertices):
 
            u = self.MinDistanceVertex(dist, sptSet)
            self.traceNode.append(u)
 
            sptSet[u] = True
 
            for v in range(self.numVertices):
                if self.weightMatrix[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.weightMatrix[u][v]:
                    dist[v] = dist[u] + self.weightMatrix[u][v]
 

if __name__ == '__main__':
    g = DijsktraSearch(9)
    # Sample Case
    g.weightMatrix = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
           [4, 0, 8, 0, 0, 0, 0, 11, 0],
           [0, 8, 0, 7, 0, 4, 0, 0, 2],
           [0, 0, 7, 0, 9, 14, 0, 0, 0],
           [0, 0, 0, 9, 0, 10, 0, 0, 0],
           [0, 0, 4, 14, 10, 0, 2, 0, 0],
           [0, 0, 0, 0, 0, 2, 0, 1, 6],
           [8, 11, 0, 0, 0, 0, 1, 0, 7],
           [0, 0, 2, 0, 0, 0, 6, 7, 0]
           ]
 
    g.dijkstra(0)
    print("Route: ", g.traceNode) #MinimumCostPath