from mazelab.generators import random_shape_maze
import matplotlib.pyplot as plt
import pandas as pd

x = random_shape_maze(width = 16, height = 16, max_shapes= 16, max_size= 8, allow_overlap= False, shape= None)
print(x)
plt.imshow(x)
plt.show()

pd.DataFrame(x).to_csv("./gridWorld_16x16.csv", header=None,index=None)

