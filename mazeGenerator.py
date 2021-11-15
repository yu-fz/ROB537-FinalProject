from mazelab.generators import random_shape_maze
import matplotlib.pyplot as plt
import pandas as pd

x = random_shape_maze(width = 512, height = 512, max_shapes= 160, max_size= 64, allow_overlap= True, shape= None)
pd.DataFrame(x).to_csv("./maps/gridWorld_superHard.csv", header=None,index=None)
print(x)
plt.imshow(x)
plt.show()

#pd.DataFrame(x).to_csv("./maps/gridWorld_superHard.csv", header=None,index=None)

