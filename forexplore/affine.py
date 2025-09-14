import numpy as np

# Example: Random 2D points
points = np.genfromtxt("./soucefile/coords.csv", delimiter=",")

# 1. Compute centroid (origin)
centroid = np.mean(points, axis=0)

# 2. Center the points
centered_points = points - centroid
rotated_centered_points = centroid - points
combined_points = np.vstack([centered_points, rotated_centered_points])
# 3. Compute covariance matrix
cov_matrix = np.cov(combined_points.T)

# 4. PCA: Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 5. Basis vectors (sorted by eigenvalue magnitude)
idx = np.argsort(eigenvalues)[::-1]  # Descending order
basis = eigenvectors[:, idx]

print("Origin (centroid):", centroid)
print("Basis vectors (columns):\n", basis)

transformed_points = (points - centroid) @ basis
print("转换后的坐标（前5个点）:\n", transformed_points[:5])


import matplotlib.pyplot as plt

plt.scatter(points[:, 0], points[:, 1], label="原始点")
plt.quiver(*centroid, *basis[:, 0], color="r", scale=5, label="基向量1")
plt.quiver(*centroid, *basis[:, 1], color="g", scale=5, label="基向量2")
plt.legend()
plt.axis("equal")
plt.show()
