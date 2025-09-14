import numpy as np

import matplotlib.pyplot as plt

points = np.genfromtxt("./soucefile/coords.csv", delimiter=",")
centriod = np.mean(points, axis=0)

x = points[:, 0].reshape([len(points), 1])

y = points[:, 1].reshape([len(points), 1])
"""
x = points[:2, 0].reshape([2, 1])
y = points[:2, 1].reshape([2, 1])
"""
x_centered = x - centriod[0]
y_centered = y - centriod[1]
theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
theta = theta.reshape([1, 100])

rho = x_centered * np.cos(theta) + y_centered * np.sin(theta)

for i in range(len(rho)):
    plt.plot(theta.T, rho[i])

plt.show()
