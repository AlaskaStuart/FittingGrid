import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def find_parallelogram_basis(
    pts, n_neighbors=6, angle_cluster_k=2, length_stat="median"
):
    """
    输入:
        pts: (N,2) numpy array
        n_neighbors: 每个点取多少个最近邻来构造差向量（包含局部结构）
        angle_cluster_k: 聚类数量（通常2）
        length_stat: 'median' or 'mean' 用于确定基向量尺度
    返回:
        mu: (2,) 重心（平移）
        B: (2,2) 基向量矩阵，列为 v1, v2
        coeffs: (N,2) 每个点的线性系数 c so that approx: x ≈ mu + B @ c
        info: 其他诊断信息（原始差向量角度、簇中心等）
    """
    N = pts.shape[0]
    mu = pts.mean(axis=0)
    Xc = pts - mu

    # 1. 为每个点找最近邻并收集差向量
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(pts)  # +1 因为包含自己
    dists, idx = nn.kneighbors(pts, return_distance=True)
    # 收集差向量（排除与自己差为0）
    vecs = []
    for i in range(N):
        for j in idx[i, 1:]:  # 跳过第一个（自己）
            v = pts[j] - pts[i]
            if np.linalg.norm(v) > 1e-8:
                vecs.append(v)
    vecs = np.array(vecs)  # M x 2

    # 2. 将差向量映射到单位向量空间（cos,sin）用于聚类角度
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    unit = vecs / norms

    # Because angle and angle+pi are equivalent for directions (v and -v same edge direction),
    # map both v and -v to the same representation: use absolute direction by forcing angle in [0, pi)
    # we can fold by taking absolute of unit or use doubling trick on angles.
    # Simpler: convert to angle in [0,pi):
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    angles = np.mod(angles, np.pi)  # fold by pi

    # cluster unit-circle representation: use (cos(2*theta), sin(2*theta)) or simply (cos theta, sin theta) after folding
    feat = np.column_stack(
        [np.cos(2 * angles), np.sin(2 * angles)]
    )  # doubles angle to distinguish opposite dirs

    kmeans = KMeans(n_clusters=angle_cluster_k, random_state=0).fit(feat)
    labels = kmeans.labels_

    # 3. 为每个簇计算平均方向向量（原始 vecs 的签名方向）
    basis_dirs = []
    lengths = []
    for k in range(angle_cluster_k):
        sel = vecs[labels == k]
        # 取簇中向量在原始空间的平均（这样不会被折半）
        mean_v = sel.mean(axis=0)
        # 方向化
        dir_u = mean_v / (np.linalg.norm(mean_v) + 1e-12)
        basis_dirs.append(dir_u)
        # 统计在这个方向上的投影长度（绝对值）
        proj_lengths = np.abs((sel @ dir_u))
        if length_stat == "median":
            lengths.append(np.median(proj_lengths))
        else:
            lengths.append(np.mean(proj_lengths))

    basis_dirs = np.array(basis_dirs)  # 2 x 2 (if k=2)
    lengths = np.array(lengths)

    # 基向量 = 方向 * 典型长度
    B = (basis_dirs.T * lengths).reshape(2, angle_cluster_k)

    # 4. 计算各点系数（最小二乘解）
    # solve B @ c = x - mu  -> c = pinv(B) @ (x - mu)
    pinvB = np.linalg.pinv(B)
    coeffs = (pinvB @ Xc.T).T  # N x 2

    info = dict(
        vecs=vecs, angles=angles, kmeans=kmeans, basis_dirs=basis_dirs, lengths=lengths
    )
    return mu, B, coeffs, info


# 使用示例
if __name__ == "__main__":
    # 假设 pts 已经是 N x 2 numpy array
    # pts = ...
    mu, B, coeffs, info = find_parallelogram_basis(pts, n_neighbors=6)
    # 恢复点的近似
    approx = mu + (B @ coeffs.T).T
    # 误差
    errs = np.linalg.norm(approx - pts, axis=1)
    print("B (columns are v1,v2):\n", B)
    print("mean reconstruction error:", errs.mean())
