from mazelab.generators import random_shape_maze
import matplotlib.pyplot as plt
import pandas as pd

x = random_shape_maze(width = 64, height = 64, max_shapes= 48, max_size= 10, allow_overlap= False, shape= None)
print(x)
plt.imshow(x)
plt.show()

pd.DataFrame(x).to_csv("./gridWorld_hard.csv", header=None,index=None)

