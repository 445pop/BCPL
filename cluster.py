import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

# ===================== 核心函数：客观判定最优聚类数（无可视化） =====================
def find_optimal_clusters(X, max_k=15, random_state=42, target_k=9):
    """
    无可视化版：通过量化指标确定最优聚类数
    参数：
        X: 边界点特征（归一化后）
        max_k: 最大尝试聚类数
        random_state: 随机种子（保证结果可复现）
        target_k: 目标聚类数（原9类，用于兜底）
    返回：
        optimal_k: 最优聚类数
        metrics: 各指标结果（可选日志输出）
    """
    # 初始化指标存储
    metrics = {
        'inertia': [],       # 肘部法则：平方和误差
        'silhouette': [],    # 轮廓系数
        'calinski_harabasz': []  # CH指数
    }
    k_range = range(2, max_k+1)  # 至少聚2类
    
    # 遍历计算各k值的指标
    for k in k_range:
        if len(X) < k:  # 样本数不足时终止
            break
        # 服务器环境建议n_init='auto'，避免警告
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        labels = kmeans.fit_predict(X)
        
        # 计算核心指标
        metrics['inertia'].append(kmeans.inertia_)
        metrics['silhouette'].append(silhouette_score(X, labels))
        metrics['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
    
    # 1. 肘部法则找拐点（下降率骤减处）
    inertia_diff = np.diff(metrics['inertia']) / metrics['inertia'][:-1]
    elbow_k = k_range[np.argmin(inertia_diff) + 1]
    
    # 2. 轮廓系数（最大值对应k）
    silhouette_k = k_range[np.argmax(metrics['silhouette'])]
    
    # 3. CH指数（最大值对应k）
    ch_k = k_range[np.argmax(metrics['calinski_harabasz'])]
    
    # 综合判定最优k
    if silhouette_k == ch_k:
        optimal_k = silhouette_k
    else:
        optimal_k = elbow_k
    
    # 兜底：限制最优k在target_k±2范围内（避免偏离原9类过远）
    optimal_k = np.clip(optimal_k, max(2, target_k-2), min(max_k, target_k+2))
    
    # 服务器日志输出（可选，便于调试）
    print(f"【聚类数判定】肘部法则推荐k={elbow_k}，轮廓系数推荐k={silhouette_k}，CH指数推荐k={ch_k}，最终最优k={optimal_k}")
    
    return optimal_k, metrics






        # ===================== 主流程：簇间点筛选+最优聚类 =====================
        # 假设前置变量已定义：all_feats_tensor [N, 768]，centroids [9, 768]
        # -------------------- 1. 基础边界点筛选 --------------------
        # 计算余弦距离（1 - 余弦相似度）
def cubian_cluster(all_feats_tensor,centroids,real_target=9):   
        all_distances = 1 - torch.matmul(all_feats_tensor, centroids.T)  # [N, 9]

        # 每个样本前2近质心（距离+索引）
        top2_distances, top2_indices = torch.topk(all_distances, k=2, largest=False)
        distance_diff = top2_distances[:, 1] - top2_distances[:, 0]

        # 动态阈值（基于百分位，适配数据分布）
        threshold = torch.quantile(distance_diff, 0.1)
        boundary_mask = (distance_diff < threshold)

        # 筛选边界点核心信息
        boundary_feats = all_feats_tensor[boundary_mask]  # [M, 768]
        boundary_top2_indices = top2_indices[boundary_mask]
        boundary_diff = distance_diff[boundary_mask]

        # 边界点数量校验（服务器环境防错）
        if len(boundary_feats) < 2:
            raise ValueError(f"边界点数量不足（仅{len(boundary_feats)}个），无法聚类，请降低阈值或检查数据！")

        # -------------------- 2. 数据预处理（服务器环境必做） --------------------
        boundary_np = boundary_feats.cpu().numpy()
        # 归一化消除量纲影响
        scaler = StandardScaler()
        boundary_scaled = scaler.fit_transform(boundary_np)

        # -------------------- 3. 判定最优聚类数 --------------------
        target_k = real_target  # 原需求9类
        optimal_k, _ = find_optimal_clusters(boundary_scaled, max_k=15, target_k=target_k)

        # -------------------- 4. 按最优k聚类 --------------------
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        kmeans_final.fit(boundary_scaled)

        # 反归一化，还原到原始特征空间
        cluster_centers_scaled = kmeans_final.cluster_centers_
        cluster_centers = scaler.inverse_transform(cluster_centers_scaled)

        # 转换为torch张量（匹配原特征的设备和数据类型）
        unknown_centroids = torch.tensor(
            cluster_centers,
            dtype=all_feats_tensor.dtype,
            device=all_feats_tensor.device
        )

        # -------------------- 5. 适配9个簇间质心（核心兜底逻辑） --------------------
        if optimal_k != target_k:
            labels = kmeans_final.labels_  # 各边界点的聚类标签
            if optimal_k < target_k:
                # 情况1：最优k < 9 → 拆分样本最多的簇，补足到9个
                cluster_sizes = np.bincount(labels)
                while len(unknown_centroids) < target_k:
                    # 找样本数最多的簇
                    largest_cluster_idx = np.argmax(cluster_sizes)
                    # 提取该簇的所有样本
                    largest_cluster_samples = boundary_feats[labels == largest_cluster_idx]
                    # 拆分为2个子簇
                    sub_kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
                    sub_kmeans.fit(largest_cluster_samples.cpu().numpy())
                    # 转换子簇中心为tensor
                    sub_centers = torch.tensor(
                        sub_kmeans.cluster_centers_,
                        dtype=unknown_centroids.dtype,
                        device=unknown_centroids.device
                    )
                    # 替换原簇中心，增加1个中心
                    unknown_centroids = torch.cat([
                        unknown_centroids[:largest_cluster_idx],
                        sub_centers,
                        unknown_centroids[largest_cluster_idx+1:]
                    ])
                    # 更新簇大小（删除原簇，添加2个子簇）
                    cluster_sizes = np.delete(cluster_sizes, largest_cluster_idx)
                    cluster_sizes = np.append(cluster_sizes, [len(largest_cluster_samples)//2]*2)
            
            else:
                # 情况2：最优k > 9 → 筛选最分散的9个中心（距离和最大）
                # 计算各中心间的两两距离和（越大越分散）
                centroid_dist = torch.cdist(unknown_centroids, unknown_centroids, p=2)
                centroid_density = centroid_dist.sum(dim=1)
                # 选距离和最大的9个
                top9_idx = torch.topk(centroid_density, k=target_k, largest=True).indices
                unknown_centroids = unknown_centroids[top9_idx]
                print(f"【最终结果】成功生成{len(unknown_centroids)}个簇间质心（目标9个），形状：{unknown_centroids.shape}")
        return unknown_centroids

# 最终校验：确保数量严格为9
# assert len(unknown_centroids) == 9, f"最终簇间质心数量错误，应为9个，实际{len(unknown_centroids)}个！"
# print(f"【最终结果】成功生成{len(unknown_centroids)}个簇间质心（目标9个），形状：{unknown_centroids.shape}")