import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

def cubian_dbscan_cluster(boundary_feats, class_prototypes):
    """
    - 用 kNN(k-distance) 分位数来估计 eps
    """
    if len(boundary_feats) < 2:
        raise ValueError(f"边界点数量不足（仅{len(boundary_feats)}个），无法聚类！")
    boundary_np = boundary_feats.detach().float().cpu().numpy()  # [M, D]
    if boundary_np.ndim != 2:
        raise ValueError(f"boundary_feats 维度不正确: {boundary_np.shape}，期望 [M, D]")
    n_samples = len(boundary_np)
    metric = "cosine"

    if metric == "cosine":
        norms = np.linalg.norm(boundary_np, axis=1, keepdims=True)
        boundary_np = boundary_np / np.clip(norms, 1e-12, None)
    min_samples = max(3, int(np.log(max(n_samples, 3))))  # 576 -> 6

    knn_k = min(10, max(2, n_samples - 1))  # k 至少 2，至多 10
    nn = NearestNeighbors(n_neighbors=min(n_samples, knn_k + 1), metric=metric)
    nn.fit(boundary_np)
    distances, _ = nn.kneighbors(boundary_np)
    k_distances = distances[:, -1]  # 第 k 近邻距离（包含 self 的 0 距离，因此用 k+1）
    eps = float(np.quantile(k_distances, 0.3))  # 可在 0.2~0.4 间调
    eps = max(eps, 0.05)  

    def _run_dbscan(eps_val: float, min_samples_val: int):
        db = DBSCAN(eps=float(eps_val), min_samples=int(min_samples_val), metric=metric)
        return db.fit_predict(boundary_np)

    labels = _run_dbscan(eps, min_samples)
    unique_labels = np.unique(labels)
    n_clusters = int(np.sum(unique_labels != -1))

    if n_clusters == 0:
        # 回退：逐步放宽 eps、略降低 min_samples
        for mult in (1.5, 2.0, 3.0):
            labels_try = _run_dbscan(eps * mult, max(2, min_samples - 1))
            unique_try = np.unique(labels_try)
            n_clusters_try = int(np.sum(unique_try != -1))
            if n_clusters_try > 0:
                labels = labels_try
                unique_labels = unique_try
                n_clusters = n_clusters_try
                eps = eps * mult
                min_samples = max(2, min_samples - 1)
                break

    print(f"【DBSCAN结果】metric={metric} eps={eps:.4f} min_samples={min_samples} | 共发现 {n_clusters} 个簇，噪声点: {int(np.sum(labels == -1))} 个")

    cluster_centers = []
    for label in unique_labels:
        if label == -1:
            continue  # 跳过噪声点
        cluster_points = boundary_np[labels == label]  # [m, D]
        if len(cluster_points) == 1:  
            center = cluster_points[0]
        else:
            # 用 medoid：簇内总距离最小的样本
            dmat = pairwise_distances(cluster_points, metric=metric)
            core_idx = int(np.argmin(dmat.sum(axis=1)))
            center = cluster_points[core_idx]

        if metric == "cosine":
            center = center / np.clip(np.linalg.norm(center), 1e-12, None)
        cluster_centers.append(center)

    if len(cluster_centers) == 0:
        # 无有效簇
        print("【警告】DBSCAN未发现有效簇，使用全体边界点均值作为唯一簇中心")
        center = boundary_np.mean(axis=0)
        if metric == "cosine":
            center = center / np.clip(np.linalg.norm(center), 1e-12, None)
        cluster_centers = [center]

    cluster_centers = np.array(cluster_centers, dtype=np.float32)

    unknown_centroids = torch.tensor(
        cluster_centers,
        dtype=boundary_feats.dtype,
        device=boundary_feats.device
    )

    print(f"【最终结果】成功生成 {len(unknown_centroids)} 个自适应簇间质心，形状：{unknown_centroids.shape}")
    return unknown_centroids