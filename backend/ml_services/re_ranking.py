import numpy as np
import logging

logger = logging.getLogger(__name__)

def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    """
    K-Reciprocal Re-ranking.
    """
    # 1. 合并距离矩阵
    original_dist = np.concatenate([q_g_dist.reshape(1, -1), g_g_dist], axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    
    # 3. 计算初始排序
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    # 确保 k1 和 k2 在给定实际邻居数量的情况下是合理的
    # 使用整除以获得整数结果
    adjusted_k1 = min(k1, max(1, all_num // 2))
    adjusted_k2 = min(k2, max(1, all_num // 4))

    # 确保 k1 始终大于 k2，以实现有意义的重排序
    if adjusted_k2 >= adjusted_k1:
        adjusted_k2 = max(1, adjusted_k1 - 1)

    if adjusted_k1 != k1 or adjusted_k2 != k2:
        logger.warning(f"警告: 默认的 k1 ({k1}) 或 k2 ({k2}) 被调整。")
        logger.warning(f"当前可比较的邻居数量 ({all_num})。k1调整为 {adjusted_k1}, k2调整为 {adjusted_k2}。")
    
    k1 = adjusted_k1
    k2 = adjusted_k2

    # 5. 主循环 (现在应该可以安全运行)
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]

        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            half_k1 = int(np.around(k1 / 2))
            candidate_forward_k_neigh_index = initial_rank[candidate, :half_k1 + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :half_k1 + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    

    jaccard_dists_to_query = []
    query_neighborhood = V[0, :] 


    for j in range(all_num):
        if j == 0: # Query vs Query, distance is 0
            jaccard_dists_to_query.append(0.0) # Jaccard distance of an item to itself is 0
            continue

        gallery_neighborhood = V[j, :]
        
        intersection_sum = np.sum(np.minimum(query_neighborhood, gallery_neighborhood))
        union_sum = np.sum(np.maximum(query_neighborhood, gallery_neighborhood))
        
        if union_sum == 0: # Avoid division by zero
            jaccard_dists_to_query.append(1.0) # Max distance if no common elements
        else:
            jaccard_dists_to_query.append(1.0 - (intersection_sum / union_sum))


    jaccard_dist = np.array(jaccard_dists_to_query)[1:] # Skip the query-to-query distance
    
    final_dist = jaccard_dist * (1 - lambda_value) + q_g_dist.flatten() * lambda_value
    
    return final_dist