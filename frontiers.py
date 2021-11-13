import math
import pdb
# frontier input
## sensor_input = {(x0,y0): 'occupied', (x1,y1): 'free', (x2,y2): 'free', (x2,y2): 'free'}
## frontier_list = {(x0,y0): 'visited', (x1,y1): 'not_visited', (x1,y1): 'occupied'}
## frontier_output = [x, y]
## current_location = [x, y]
class Frontier:
    def __init__(self, sensor_data, current_location):
        self.inputs = sensor_data
        self.current_location = current_location
        self.ogm = GridMap()
        self.frontiers = {(current_loc[0], current_loc[1]): 'visited'}
        self.gen_gridmap()
        self.update_frontier()
        self.goal = (current_location[0], current_location[1])
    
    def gen_gridmap(self):
        self.ogm.expand_map(self.inputs)
        # pdb.set_trace()

    def computeFrontier(self, current_location, old_frontiers):
        frontier_list = {}
        pool_list = {}
        key = (current_location[0], current_location[1])

        if key in old_frontiers:
            old_frontiers[key] == 'visited'
            self.frontiers.update({key: 'visited'})
            # pdb.set_trace()

        # Iterate over all updated cells in the grid
        for x, y in self.ogm.data:
            value = self.ogm.data[(x, y)]
            # obstacle check, old_frontiers check
            if any(old_frontiers):
                if (x, y) in old_frontiers:
                    if value == 'obstacle':
                        self.frontiers.update({(x, y): 'occupied'})
                        frontier_list.update({(x, y): 10000000})
                        # pdb.set_trace()
                    else:
                        # stays in the frontier list
                        fitness = math.sqrt(math.pow(current_location[0] - x, 2) + math.pow(current_location[1] - y, 2))
                        frontier_list.update({(x, y): fitness})
                        # pdb.set_trace()
                else:
                    # if there is at least one neighbor of the current cell which has been visited, its frontier
                    if x - 1 >= 0:
                        left_key = (x - 1, y)
                    else: 
                        left_key = None

                    right_key = (x + 1, y)

                    if y - 1 >= 0:
                        up_key = (x, y - 1)
                    else:
                        up_key = None

                    down_key = (x, y + 1)
                    a = ((left_key in old_frontiers) and old_frontiers[left_key] == 'visited')
                    b = ((right_key in old_frontiers) and old_frontiers[right_key] == 'visited')
                    c = ((up_key in old_frontiers) and old_frontiers[up_key] == 'visited')
                    d = ((down_key in old_frontiers) and old_frontiers[down_key] == 'visited')
                    if (a == True) or (b == True) or (c == True) or (d == True):
                        self.frontiers.update({(x, y): 'not visited'})
                        fitness = math.sqrt(math.pow(current_location[0] - x, 2) + math.pow(current_location[1] - y, 2))
                        frontier_list.update({(x, y): fitness})
                        # pdb.set_trace()
            else:
                self.frontiers.update({(x, y): 'not visited'})
                fitness = math.sqrt(math.pow(current_location[0] - x, 2) + math.pow(current_location[1] - y, 2))
                frontier_list.update({(x, y): fitness})
                # pdb.set_trace()
        return frontier_list

    def update_frontier(self):
        goal = []
        f_list = {}
        f_list = self.computeFrontier(self.current_location, self.frontiers)
        min = 10000
        # select the goal with the min fitness
        for key in f_list:
            if f_list[key] <= min:
                min = f_list[key]
        found_key = [key for key, value in f_list.items() if value == min]
        pdb.set_trace()

        self.goal = found_key
    
    def return_goal(self):
        return self.goal

class GridMap:
    def __init__(self):
        self.data = {}
        self.obstacle_dict = {}
        # 2D array to mark visited nodes (in the beginning, no node has been visited)

    def expand_map(self, new_data_dict):
        self.data.update(new_data_dict)
        self.dim_cells = len(self.data.keys())
        
current_loc = [0, 0]
sensor_input = {(0,1): 'occupied', (1,0): 'free'}
test_case = Frontier(sensor_input, current_loc)
print(test_case.return_goal())