import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy
from scipy import stats 



timeData = np.load("./timeDataArray.npy")
timeData_Rooms = np.load("./timeDataArray_rooms.npy")
stepData = np.load("./stepDataArray.npy")
stepData_Rooms = np.load("./stepDataArray_rooms.npy")


x_axis = 100*np.arange(0,0.91,0.1)


average_time = scipy.mean(timeData, axis = 0)

average_time_rooms = scipy.mean(timeData_Rooms,axis = 0)

average_pathCost = scipy.mean(stepData, axis = 0)
average_pathCost_rooms = scipy.mean(stepData_Rooms, axis = 0)

plt.rc("xtick", labelsize = 20)
plt.rc("ytick", labelsize = 20)

print(average_time)

plt.subplot(1, 2, 1)
for row in timeData:

    plt.scatter(x_axis, row, s = 80, marker = "s", color = "darkgreen")

for row in timeData_Rooms:
    plt.scatter(x_axis, row, s = 80, marker = "x", color = "darkmagenta")

plt.plot(x_axis, average_time,color = "steelblue", marker='P',markersize=20, linewidth=2, label = "Average Computation Time for Test Map 1")
plt.plot(x_axis, average_time_rooms,color = "orangered", marker='P',markersize=20, linewidth=2, label = "Average Computation Time for Test Map 2")
plt.xlabel("percentage of map explored", fontsize = 20)
plt.ylabel("algorithm computation time in seconds", fontsize = 20)
#plt.legend(['Test Map 1', 'Test Map 2'], fontsize = 10)
plt.legend(loc='upper center', shadow=True, fontsize='x-large', labelspacing = 2)
plt.grid()

plt.subplot(1, 2, 2)
for row in stepData:

    plt.scatter(x_axis, row, s = 80, marker = "s", color = "darkgreen")

for row in stepData_Rooms:

    plt.scatter(x_axis, row, marker = "x", s = 80, color = "darkmagenta")

plt.plot(x_axis, average_pathCost,color = "steelblue", marker='P',markersize=20, linewidth=2, label = "Average Path Cost for Test Map 1")
plt.plot(x_axis, average_pathCost_rooms,color = "orangered", marker='P',markersize=20, linewidth=2, label = "Average Path Cost for Test Map 2")
plt.ylabel("path cost", fontsize = 20)
plt.xlabel("percentage of map explored", fontsize = 20)
plt.legend(loc='upper center', shadow=True, fontsize='x-large', labelspacing = 2)
#print(timeData)
#plt.plot(x_axis, timeData)
plt.grid()
plt.show()