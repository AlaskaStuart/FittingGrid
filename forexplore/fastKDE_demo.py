import fastkde
import numpy as np


import matplotlib.pyplot as plt
from math import atan, sin, cos

points = np.genfromtxt("./soucefile/coords.csv", delimiter=",")
centriod = np.mean(points, axis=0)

i = 34
j = 45


x_i, y_i = points[i] - centriod
x_j, y_j = points[j] - centriod
print(x_i)
print(y_i)
print("")
print(x_j)
print(y_j)
print("")
theta = atan((x_j - x_i) / (y_i - y_j))

rho1 = x_i * cos(theta) + y_i * sin(theta)
rho2 = x_j * cos(theta) + y_j * sin(theta)
print(theta)
print(rho1)
print(rho2)
