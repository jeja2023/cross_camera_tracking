import numpy as np
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging
from ..database_conn import Person, FollowedPerson, Individual
from .. import crud
from backend.config import settings

logger = logging.getLogger(__name__)

def calculate_similarity(feature_vector1: List[float], feature_vector2: List[float]) -> float:
    """
    计算两个特征向量之间的余弦相似度。
    假设特征向量已经是归一化的。
    """
    vec1 = np.array(feature_vector1)
    vec2 = np.array(feature_vector2)
    # 避免除以零
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def perform_realtime_comparison(
    db: Session,
    detected_person_feature_vector: List[float],
    current_user_id: int,
    followed_persons_list: List[FollowedPerson] # 新增：接收已过滤的关注人员列表
) -> Optional[Dict[str, Any]]:
    """
    对检测到的人物进行实时比对，与当前用户关注的人物进行比对。
    返回匹配到的关注人物信息，如果未匹配到则返回 None。
    """
    logger.info(f"执行实时比对：检测到的人脸特征向量长度 {len(detected_person_feature_vector)}")
    
    if not followed_persons_list:
        logger.info("传入的关注人物列表为空，跳过实时比对。")
        return None

    best_match = None
    max_similarity = settings.REALTIME_COMPARISON_THRESHOLD # 设置比对阈值

    for followed_person in followed_persons_list:
        # 确保 individual 对象存在且特征向量有效
        if followed_person.individual and followed_person.individual.persons:
            # 找到该 Individual 下的最新或最可靠的 Person 记录来获取特征向量
            # 简单起见，这里假设第一个 Person 记录包含我们需要的特征向量
            # 实际应用中可能需要更复杂的逻辑来选择最佳特征向量
            # 优先使用已训练的或置信度最高的特征向量
            individual_persons = sorted(
                [p for p in followed_person.individual.persons if p.feature_vector is not None],
                key=lambda p: p.created_at, # 可以根据 created_at, is_trained, confidence_score 等排序
                reverse=True
            )

            if not individual_persons:
                continue

            # 使用最新人物的特征向量进行比对
            followed_person_feature_vector = individual_persons[0].feature_vector
            
            try:
                # 特征向量可能存储为 JSON 字符串，需要解析
                if isinstance(followed_person_feature_vector, str):
                    followed_person_feature_vector = np.array(list(map(float, followed_person_feature_vector.strip('[]').split(',')))) # 将字符串转换为浮点数列表
                elif isinstance(followed_person_feature_vector, list):
                    followed_person_feature_vector = np.array(followed_person_feature_vector)
                else:
                    logger.warning(f"关注人物 {followed_person.individual.name} ({followed_person.individual.uuid}) 的特征向量格式未知: {type(followed_person_feature_vector)}")
                    continue

                current_similarity = calculate_similarity(detected_person_feature_vector, followed_person_feature_vector.tolist())
                logger.debug(f"与关注人物 {followed_person.individual.name} 相似度: {current_similarity}")

                if current_similarity > max_similarity:
                    max_similarity = current_similarity
                    best_match = {
                        "individual_id": followed_person.individual.id,
                        "individual_uuid": followed_person.individual.uuid,
                        "individual_name": followed_person.individual.name,
                        "individual_id_card": followed_person.individual.id_card,
                        "similarity": current_similarity,
                        "is_followed": True
                    }
            except Exception as e:
                logger.error(f"计算相似度时出错：{e}", exc_info=True)
                continue

    return best_match