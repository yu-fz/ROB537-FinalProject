from mazelab.generators import random_shape_maze
import matplotlib.pyplot as plt
import pandas as pd

x = random_shape_maze(width = 9, height = 9, max_shapes= 4, max_size= 2, allow_overlap= True, shape= None)
print(x)
plt.imshow(x)
plt.show()

pd.DataFrame(x).to_csv("./maps/gridWorld_easy.csv", header=None,index=None)

