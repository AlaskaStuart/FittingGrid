import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def find_orthogonal_basis(points):
    """
    从二维点集中找到正交基
    参数:
        points: numpy数组，形状为(n, 2)
    返回:
        正交基向量(按重要性排序)
    """
    # 方法1: 使用PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    basis = pca.components_  # 已按特征值大小排序

    # 方法2: 手动计算(验证用)
    rotated = 0 - points
    combined = np.vstack([points, rotated])
    centroid = np.mean(combined, 0)
    centered = combined - centroid
    cov_matrix = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # 按特征值降序排序
    idx = eigenvalues.argsort()[::-1]
    basis_manual = eigenvectors[:, idx].T

    return basis_manual


# 示例使用
random_points = np.genfromtxt("./soucefile/coords.csv", delimiter=",")
orthogonal_basis = find_orthogonal_basis(random_points)

print("正交基向量(按重要性排序):")
print("第一基向量(主方向):", orthogonal_basis[0])
print("第二基向量(次方向):", orthogonal_basis[1])

plt.scatter(random_points[:, 0], random_points[:, 1], alpha=0.5)
mean = np.mean(random_points, axis=0)
# 绘制基向量(缩放以便可视化)
scale = 2
plt.quiver(*mean, *orthogonal_basis[0] * scale, color="r", scale=1)
plt.quiver(*mean, *orthogonal_basis[1] * scale, color="b", scale=1)
plt.axis("equal")
plt.show()
