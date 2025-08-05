# ml_logic.py

import cv2
import numpy as np
import onnxruntime as ort
import os
import json
from sqlalchemy.orm import Session
from typing import Optional, List, Any, Dict, Tuple
import logging
import uuid
import time
from tqdm import tqdm
import shutil
from fastapi import HTTPException
import threading
from datetime import datetime
import faiss
from typing import Any
from pymysql import ProgrammingError

from ultralytics import YOLO
from .sort import Sort
from ..database_conn import SessionLocal, Person, Stream, SystemConfig
from .. import crud
from .re_ranking import re_ranking
from .. import schemas
from backend.config import settings
from . import reid_trainer
from backend.schemas import Person, PersonCreate
from backend.utils import media_processing
import aiofiles # 导入 aiofiles 用于异步文件操作

logger = logging.getLogger(__name__)

# --- 配置 ---
# 移除旧的全局路径变量
# CROP_DIR = settings.DATABASE_CROPS_DIR
# FULL_FRAME_DIR = settings.DATABASE_FULL_FRAMES_DIR
REID_INPUT_WIDTH = settings.REID_INPUT_WIDTH
REID_INPUT_HEIGHT = settings.REID_INPUT_HEIGHT
FEATURE_DIM = settings.FEATURE_DIM
FACE_FEATURE_DIM = settings.FACE_FEATURE_DIM
K1 = settings.K1
K2 = settings.K2
LAMBDA_VALUE = settings.LAMBDA_VALUE

# 多模态融合权重 (可以根据实际效果调整)
REID_WEIGHT = settings.REID_WEIGHT
FACE_WEIGHT = settings.FACE_WEIGHT
GAIT_WEIGHT = settings.GAIT_WEIGHT

# 新增：步态特征维度，假设与 Re-ID 特征维度相同
GAIT_FEATURE_DIM = settings.GAIT_FEATURE_DIM

# 全局 Re-ID 模型实例和锁，用于热加载
_reid_session_instance: Optional[ort.InferenceSession] = None
_reid_model_lock = threading.RLock() # Changed to RLock

# 新增：全局人脸识别模型实例和锁
_face_recognition_session_instance: Optional[ort.InferenceSession] = None
_face_recognition_model_lock = threading.RLock() # Changed to RLock

# 新增：全局人脸检测模型实例和锁
_face_detection_model_instance: Optional[YOLO] = None
_face_detection_model_lock = threading.RLock() # Changed to RLock

# 新增：全局步态识别模型实例和锁
_gait_recognition_session_instance: Optional[ort.InferenceSession] = None
_gait_recognition_model_lock = threading.RLock() # Changed to RLock

# 新增：全局姿态检测模型实例和锁
_detection_pose_model_instance: Optional[YOLO] = None
_detection_pose_model_lock = threading.RLock() # Changed to RLock

# 新增：Faiss 索引实例和锁
_faiss_index_instance: Optional[faiss.IndexFlatL2] = None
_faiss_index_lock = threading.RLock() # Changed to RLock
_faiss_person_uuids: List[str] = []

def initialize_faiss_index(db: Session):
    """
    初始化 Faiss 索引，并从数据库加载所有人物特征向量。
    """
    import backend.crud as crud # 移动导入到函数内部
    global _faiss_index_instance, _faiss_person_uuids
    with _faiss_index_lock:
        if _faiss_index_instance is None:
            logger.info("开始初始化 Faiss 索引...")
            try:
                # 创建一个空的 Faiss 索引
                logger.info(f"initialize_faiss_index: 尝试创建 Faiss IndexFlatL2 实例，维度: {settings.FEATURE_DIM}...")
                _faiss_index_instance = faiss.IndexFlatL2(settings.FEATURE_DIM)
                logger.info("initialize_faiss_index: Faiss IndexFlatL2 实例创建成功。")
                _faiss_person_uuids = []

                # 从数据库加载所有人物特征向量
                logger.info("initialize_faiss_index: 尝试从数据库加载所有人物特征...")
                try:
                    all_persons = crud.get_all_persons(db) # Now returns List[Person] ORM objects
                    logger.info(f"initialize_faiss_index: 成功从数据库加载 {len(all_persons)} 个人物特征。")
                except Exception as e:
                    # 如果表不存在 (pymysql.err.ProgrammingError)，则记录警告并继续，索引将保持为空
                    if isinstance(e, ProgrammingError) and "Table" in str(e) and "doesn't exist" in str(e):
                        logger.warning(f"数据库表 'persons' 不存在，Faiss 索引将初始化为空。错误: {e}")
                        all_persons = []
                    else:
                        logger.error(f"从数据库加载人物特征失败: {e}", exc_info=True)
                        raise
                features_to_add = []
                person_uuids_to_add = []

                for person in all_persons:
                    # Directly access attributes of the Person ORM object
                    if person.feature_vector:
                        try:
                            feature = person.feature_vector
                            if isinstance(feature, str):
                                feature = json.loads(feature)
                            
                            if isinstance(feature, list) and len(feature) == settings.FEATURE_DIM:
                                features_to_add.append(feature)
                                person_uuids_to_add.append(person.uuid)
                            else:
                                logger.warning(f"人物 {person.uuid} 的特征向量格式不正确或维度不符，跳过加载到 Faiss。")
                        except json.JSONDecodeError:
                            logger.warning(f"人物 {person.uuid} 的特征向量不是有效的 JSON，跳过加载到 Faiss。")

                if features_to_add:
                    features_array = np.array(features_to_add, dtype=np.float32)
                    _faiss_index_instance.add(features_array)
                    _faiss_person_uuids.extend(person_uuids_to_add)
                    logger.info(f"Faiss 索引初始化成功，已添加 {len(features_to_add)} 个人物特征。")
                else:
                    logger.info("Faiss 索引初始化完成，但数据库中没有可加载的人物特征。")

            except Exception as e:
                logger.error(f"初始化 Faiss 索引失败: {e}", exc_info=True)
                _faiss_index_instance = None
                _faiss_person_uuids = []
                raise RuntimeError(f"初始化 Faiss 索引失败: {e}")

def add_person_feature_to_faiss(person_uuid: str, feature_vector: np.ndarray):
    """
    将新人物的特征向量添加到 Faiss 索引中。
    """
    global _faiss_index_instance, _faiss_person_uuids
    with _faiss_index_lock:
        if _faiss_index_instance is None:
            logger.warning("Faiss 索引未初始化，无法添加新人物特征。")
            return
        try:
            # 确保特征向量是二维 numpy 数组，即使只有一个向量
            if feature_vector.ndim == 1:
                feature_vector = np.expand_dims(feature_vector, axis=0)
            
            _faiss_index_instance.add(feature_vector)
            _faiss_person_uuids.append(person_uuid)
            logger.info(f"人物 {person_uuid} 的特征已成功添加到 Faiss 索引。当前索引大小: {_faiss_index_instance.ntotal}")
        except Exception as e:
            logger.error(f"添加人物 {person_uuid} 特征到 Faiss 索引失败: {e}", exc_info=True)

def get_face_detection_model(db: Session) -> YOLO:
    global _face_detection_model_instance
    with _face_detection_model_lock:
        if _face_detection_model_instance is None:
            try:
                model_path = settings.FACE_DETECTION_MODEL_PATH
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"人脸检测模型未找到: {model_path}")
                logger.info(f"加载人脸检测模型: {model_path}")
                _face_detection_model_instance = YOLO(model_path, task='detect')
                logger.info("人脸检测模型加载成功。")
            except Exception as e:
                logger.error(f"加载人脸检测模型失败: {e}", exc_info=True)
                raise RuntimeError(f"加载人脸检测模型失败: {e}")
        return _face_detection_model_instance

def get_face_recognition_session(db: Session) -> ort.InferenceSession:
    global _face_recognition_session_instance
    with _face_recognition_model_lock:
        if _face_recognition_session_instance is None:
            try:
                model_path = settings.FACE_RECOGNITION_MODEL_PATH
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"人脸识别模型未找到: {model_path}")
                logger.info(f"加载人脸识别模型: {model_path}")
                _face_recognition_session_instance = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                logger.info("人脸识别模型加载成功。")
            except Exception as e:
                logger.error(f"加载人脸识别模型失败: {e}", exc_info=True)
                raise RuntimeError(f"加载人脸识别模型失败: {e}")
        return _face_recognition_session_instance

def get_gait_recognition_session(db: Session) -> Optional[ort.InferenceSession]:
    global _gait_recognition_session_instance
    with _gait_recognition_model_lock:
        if _gait_recognition_session_instance is None:
            model_path = settings.GAIT_RECOGNITION_MODEL_PATH
            if not model_path:
                logger.info("步态识别模型路径未设置，跳过加载步态识别模型。")
                return None
            try:
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"步态识别模型未找到: {model_path}")
                logger.info(f"加载步态识别模型: {model_path}")
                _gait_recognition_session_instance = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                logger.info("步态识别模型加载成功。")
            except Exception as e:
                logger.error(f"加载步态识别模型失败: {e}", exc_info=True)
                raise RuntimeError(f"加载步态识别模型失败: {e}")
        return _gait_recognition_session_instance


def get_detection_model(db: Session) -> YOLO:
    """
    获取姿态检测模型的单例实例。
    """
    global _detection_pose_model_instance
    with _detection_pose_model_lock:
        if _detection_pose_model_instance is None:
            try:
                model_path = settings.POSE_MODEL_PATH
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"姿态检测模型未找到: {model_path}")
                logger.info(f"加载姿态检测模型: {model_path}")
                _detection_pose_model_instance = YOLO(model_path, task='pose')
                logger.info("姿态检测模型加载成功。")
            except Exception as e:
                logger.error(f"加载姿态检测模型失败: {e}", exc_info=True)
                raise RuntimeError(f"加载姿态检测模型失败: {e}")
        return _detection_pose_model_instance

def preprocess_face_for_recognition(face_crop: np.ndarray) -> np.ndarray:
    """
    预处理人脸裁剪图，使其符合人脸识别模型的输入要求。
    通常是缩放和归一化。
    """
    input_size = (112, 112)
    resized_face = cv2.resize(face_crop, input_size)
    rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
    normalized_face = rgb_face.astype(np.float32) / 255.0
    batch_face = np.expand_dims(normalized_face, axis=0)
    return batch_face

def get_face_feature(face_recognition_session: ort.InferenceSession, face_crop: np.ndarray) -> np.ndarray:
    """
    从人脸裁剪图中提取特征向量。
    """
    preprocessed_face = preprocess_face_for_recognition(face_crop)
    input_name = face_recognition_session.get_inputs()[0].name
    face_feature = face_recognition_session.run(None, {input_name: preprocessed_face})[0]
    face_feature = face_feature.flatten()
    return face_feature

def get_clothing_attributes(frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> Optional[Dict[str, Any]]:
    """
    从人物裁剪图中提取衣着属性。
    这是一个占位符函数，需要根据实际的衣着属性模型进行实现。
    返回一个字典，包含衣着属性，例如：{"top_color": "red", "bottom_type": "jeans"}
    """
    logger.warning("衣着属性提取功能尚未实现，返回空数据。")
    # x1, y1, x2, y2 = person_bbox
    # person_crop = frame[y1:y2, x1:x2]
    # 在这里添加调用衣着属性模型的逻辑
    # 例如：model.predict(person_crop) -> attributes_dict
    return {"top_color": "N/A", "bottom_type": "N/A"} # 示例返回数据

def preprocess_for_gait(gait_sequence: List[np.ndarray]) -> np.ndarray:
    """
    预处理步态序列，使其符合步态识别模型的输入要求。
    步态模型通常需要多帧剪影图作为输入。
    假设 GaitSet 模型需要 (Batch, Frames, Height, Width) 的输入，其中帧是 64x64 的灰度剪影。
    """
    if not gait_sequence: 
        return np.array([])

    processed_frames = []
    for frame_idx, frame in enumerate(gait_sequence):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
        
        resized_frame = cv2.resize(gray_frame, (64, 64))
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        processed_frames.append(np.expand_dims(normalized_frame, axis=0))
    
    gait_input = np.stack(processed_frames, axis=0)
    gait_input = gait_input.squeeze(axis=1)
    gait_input = np.expand_dims(gait_input, axis=0)
    
    logger.debug(f"步态序列预处理完成，输入形状: {gait_input.shape}")
    return gait_input

def get_gait_feature(gait_recognition_session: ort.InferenceSession, gait_sequence: List[np.ndarray]) -> np.ndarray:
    """
    从步态序列中提取特征向量。
    """
    if not gait_sequence:
        logger.warning("步态序列为空，无法提取特征。")
        return np.array([])
        
    preprocessed_gait = preprocess_for_gait(gait_sequence)
    if preprocessed_gait.size == 0:
        logger.warning("预处理后的步态数据为空，无法提取特征。")
        return np.array([])

    input_name = gait_recognition_session.get_inputs()[0].name
    gait_feature = gait_recognition_session.run(None, {input_name: preprocessed_gait})[0]
    gait_feature = gait_feature.flatten()
    gait_feature_norm = np.linalg.norm(gait_feature)
    gait_feature = gait_feature / (gait_feature_norm + 1e-12)
    return gait_feature


def calculate_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def get_reid_session(db: Session) -> ort.InferenceSession:
    global _reid_session_instance
    with _reid_model_lock:
        if _reid_session_instance is None:
            try:
                model_path = settings.ACTIVE_REID_MODEL_PATH
                if not os.path.exists(model_path):
                    model_path = settings.REID_MODEL_PATH
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Re-ID 模型未找到: {settings.ACTIVE_REID_MODEL_PATH} 或 {settings.REID_MODEL_PATH}")
                    logger.warning(f"Re-ID 激活模型未找到，回退到默认模型路径: {model_path}")

                logger.info(f"加载 Re-ID 模型: {model_path}")
                _reid_session_instance = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                logger.info("Re-ID 模型加载成功。")
            except Exception as e:
                logger.error(f"加载 Re-ID 模型失败: {e}", exc_info=True)
                raise RuntimeError(f"加载 Re-ID 模型失败: {e}")
        return _reid_session_instance

def preprocess_for_reid(image_crop):
    """
    预处理人物裁剪图，使其符合Re-ID模型的输入要求。
    """
    resized_image = cv2.resize(image_crop, (settings.REID_INPUT_WIDTH, settings.REID_INPUT_HEIGHT))
    normalized_image = resized_image.astype(np.float32) / 255.0
    chw_image = np.transpose(normalized_image, (2, 0, 1))
    batch_image = np.expand_dims(chw_image, axis=0)
    return batch_image

def get_person_feature(reid_session: ort.InferenceSession, person_crop: np.ndarray) -> np.ndarray:
    """
    从人物裁剪图中提取特征向量并进行归一化。确保返回一维NumPy数组。
    """
    preprocessed_crop = preprocess_for_reid(person_crop)
    input_name = reid_session.get_inputs()[0].name
    person_feature = reid_session.run(None, {input_name: preprocessed_crop})[0]
    person_feature = person_feature.flatten()
    person_feature_norm = np.linalg.norm(person_feature)
    person_feature = person_feature / (person_feature_norm + 1e-12)
    return person_feature


def retrain_reid_model(
    persons_for_retrain: List[schemas.Person],
    model_version: str,
    db: Session,
    celery_task = None,
    epochs: int = 10, # 从调用方获取 epochs
    batch_size: int = 32, # 从调用方获取 batch_size
    learning_rate: float = 0.001, # 从调用方获取 learning_rate
    feature_dim: int = 512 # 从调用方获取 feature_dim
) -> Optional[str]:
    logger.info(f"开始 Re-ID 模型再训练，模型版本: {model_version}")
    
    if celery_task:
        celery_task.update_state(state='PREPARING_DATA', meta={'progress': 10, 'status': 'Preparing training data'})

    try:
        all_features = []
        all_labels = []
        unique_person_ids = {}
        label_counter = 0

        # 过滤掉纠正类型为 'misdetection' 的人物
        filtered_persons_for_retrain = []
        for p in persons_for_retrain:
            correction_details = p.get('correction_details') if isinstance(p, dict) else p.correction_details
            if correction_details is None or (isinstance(correction_details, str) and "misdetection" not in correction_details.lower()):
                filtered_persons_for_retrain.append(p)

        if not filtered_persons_for_retrain:
            logger.info("过滤后没有有效的人物数据用于再训练，跳过模型再训练。")
            if celery_task:
                celery_task.update_state(state='SKIPPED', meta={'progress': 100, 'status': 'No valid data for retraining after filtering'})
            return None

        # 初始化数据收集列表 (将这两行移动到此处)
        all_features = []
        all_labels = []

        total_persons_to_process = len(filtered_persons_for_retrain)
        
        # 报告数据准备初始进度 (新的)
        if celery_task:
            celery_task.update_state(state='PREPARING_DATA', meta={'progress': 5, 'status': 'Starting data preparation'})

        for idx, person in enumerate(filtered_persons_for_retrain): # 使用 enumerate 获取索引
            # 获取需要再训练的人物数据
            if isinstance(person, dict):
                feature = person.get('feature_vector')
            else:
                feature = person.feature_vector
            
            if feature:
                try:
                    if isinstance(feature, str):
                        feature = json.loads(feature)
                    
                    if isinstance(feature, list) and isinstance(feature[0], list):
                        feature = feature[0]
                    
                    if not isinstance(feature, list) or not all(isinstance(x, (int, float)) for x in feature):
                        logger.warning(f"人物 {person.uuid} 的特征向量格式不正确，跳过再训练。")
                        continue

                    all_features.append(np.array(feature, dtype=np.float32))
                    
                    person_id = person.get('id') if isinstance(person, dict) else person.id
                    if person_id is not None and person_id not in unique_person_ids:
                        unique_person_ids[person_id] = label_counter
                        label_counter += 1
                    all_labels.append(unique_person_ids[person_id])
                except json.JSONDecodeError:
                    logger.warning(f"人物 {person.uuid} 的特征向量不是有效的 JSON，跳过再训练。")
                    continue
            else:
                logger.warning(f"人物 {person.uuid} 没有有效的特征向量，跳过再训练。")
            
            # 更新数据准备进度 (从 10% 到 30% 线性增长)
            if celery_task and total_persons_to_process > 0:
                # 确保进度在 10% 到 30% 之间
                current_data_prep_progress = 10 + int((idx / total_persons_to_process) * 20) 
                celery_task.update_state(state='PREPARING_DATA', meta={'progress': current_data_prep_progress, 'status': f'Processing data for person {idx+1}/{total_persons_to_process}'})
            
            # 添加日志，反映数据准备进度
            if idx % (max(1, total_persons_to_process // 10)) == 0 or idx == total_persons_to_process - 1: # 每处理10%或最后一个人时记录
                logger.info(f"数据准备进度: {current_data_prep_progress}% (已处理人物 {idx+1}/{total_persons_to_process})")

        if not all_features:
            logger.warning("没有有效的特征向量用于再训练，跳过再训练。")
            if celery_task:
                celery_task.update_state(state='SKIPPED', meta={'progress': 100, 'status': 'No data for retraining'})
            return None

        # 数据准备完成，更新到 30%
        if celery_task:
            celery_task.update_state(state='PREPARING_DATA', meta={'progress': 30, 'status': 'Data preparation complete'})

        features_array = np.array(all_features)
        labels_array = np.array(all_labels)

        logger.info(f"准备用于再训练的数据：特征数量={features_array.shape[0]}, 维度={features_array.shape[1]}, 身份数量={label_counter}")
        
        # 在这里添加模型训练的模拟进度报告
        training_progress_start = 30 # 训练开始的进度
        training_progress_end = 70 # 训练结束的进度
        
        # 报告模型训练开始
        if celery_task:
            celery_task.update_state(state='TRAINING_MODEL', meta={'progress': training_progress_start, 'status': 'Starting model training'})

        for i in range(epochs): # 模拟训练的 epoch 循环
            current_progress = training_progress_start + int((i / epochs) * (training_progress_end - training_progress_start)) # 计算当前进度百分比
            if celery_task:
                celery_task.update_state(state='TRAINING_MODEL', meta={'progress': current_progress, 'status': f'Training epoch {i+1}/{epochs}'})
            logger.info(f"模型训练进度: {current_progress}% (Epoch {i+1}/{epochs})") # 将日志消息修改为"模型训练进度"
            time.sleep(0.5) # 模拟训练耗时

        output_model_path = reid_trainer.train_reid_model_pytorch(
            persons_for_retrain=persons_for_retrain,
            new_model_path=settings.ACTIVE_REID_MODEL_PATH,
            db=db,
            celery_task=celery_task, # 传递 Celery 任务对象
            epochs=epochs, # 示例 epochs
            batch_size=batch_size, # 从 settings 获取 batch_size
            learning_rate=learning_rate, # 从 settings 获取学习率
            feature_dim=feature_dim # 传递特征维度
        )

        if output_model_path:
            if celery_task:
                celery_task.update_state(state='UPDATING_CONFIG', meta={'progress': 80, 'status': 'Updating system configuration'})
            
            crud.update_system_config(db, key="ACTIVE_REID_MODEL_PATH", value=output_model_path)
            db.commit()
            settings.ACTIVE_REID_MODEL_PATH = output_model_path
            logger.info(f"Re-ID 模型再训练完成，新激活模型路径: {output_model_path}")
            
            if celery_task:
                celery_task.update_state(state='RELOADING_FAISS', meta={'progress': 90, 'status': 'Reloading Faiss index'})

            global _faiss_index_instance, _faiss_person_uuids, _reid_session_instance
            with _faiss_index_lock:
                _faiss_index_instance = None
                _faiss_person_uuids = []
                _reid_session_instance = None
            
            initialize_faiss_index(db)
            logger.info("Faiss 索引已清空并重新加载。")

            if celery_task:
                celery_task.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Re-ID model retraining completed successfully'})
            return output_model_path
        else:
            logger.error("Re-ID 模型再训练失败，未生成新的模型路径。")
            if celery_task:
                celery_task.update_state(state='FAILURE', meta={'progress': 100, 'status': 'Re-ID model retraining failed'})
            return None

    except Exception as e:
        logger.error(f"Re-ID 模型再训练过程中发生错误: {e}", exc_info=True)
        if celery_task:
            celery_task.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed', 'error': str(e)})
        return None


# --- 查询逻辑函数 ---
def find_similar_people(db: Session, query_image_path: Optional[str] = None, threshold: float = 0.5, video_uuid: Optional[str] = None, stream_uuid: Optional[str] = None, current_user: schemas.User = None, skip: int = 0, limit: int = 20, query_person_uuids: Optional[List[str]] = None, target_person_uuids: Optional[List[str]] = None):
    """
    在人物特征图库中查找相似的人物。

    :param db: 数据库会话
    :param query_image_path: 查询图片的路径 (如果以图搜图)
    :param threshold: 相似度阈值 (0.0 - 1.0)
    :param video_uuid: 可选：限定在某个视频内搜索
    :param stream_uuid: 可选：限定在某个视频流内搜索
    :param current_user: 当前用户
    :param skip: 分页跳过的数量
    :param limit: 分页返回的数量
    :param query_person_uuids: 可选：查询人物的UUID列表 (如果以UUID搜人)
    :param target_person_uuids: 可选：限定搜索的目标人物UUID列表 (如果限制在特定人物集内搜索)
    :return: 包含相似人物列表和总数的字典
    """
    logger.info(f"find_similar_people: 开始执行搜人操作。查询图片: {query_image_path}, 阈值: {threshold}, 查询人物UUIDs: {query_person_uuids}, 目标人物UUIDs: {target_person_uuids}")

    if not query_image_path and not query_person_uuids:
        logger.error("find_similar_people: 必须提供 query_image_path 或 query_person_uuids。")
        raise HTTPException(status_code=400, detail="必须提供 query_image_path 或 query_person_uuids。")

    reid_session = get_reid_session(db)
    face_detection_model = get_face_detection_model(db)
    face_recognition_session = get_face_recognition_session(db)
    gait_recognition_session = get_gait_recognition_session(db)

    grouped_results = []
    overall_unique_results_uuids = set() # 跟踪所有查询人物的不重复结果

    query_sources = []

    # Step 1: 优先处理 query_person_uuids (如果前端已解析出人物UUID)
    if query_person_uuids:
        logger.info(f"find_similar_people: 使用提供的 {len(query_person_uuids)} 个人物UUID进行搜索。")
        for person_uuid in query_person_uuids:
            person_in_db = crud.get_person_by_uuid(db, person_uuid)
            if person_in_db and person_in_db.feature_vector:
                try:
                    feature = person_in_db.feature_vector
                    if isinstance(feature, str):
                        feature = json.loads(feature)
                    if isinstance(feature, list):
                        query_sources.append({
                            "uuid": person_in_db.uuid,
                            "feature": np.array(feature, dtype=np.float32),
                            "crop_image_path": person_in_db.crop_image_path,
                            "full_frame_image_path": person_in_db.full_frame_image_path,
                        })
                    else:
                        logger.warning(f"人物 {person_uuid} 的特征向量格式不正确，跳过处理。")
                except json.JSONDecodeError:
                    logger.warning(f"人物 {person_uuid} 的特征向量不是有效的 JSON，跳过处理。")
            else:
                logger.warning(f"find_similar_people: 数据库中未找到 UUID 为 {person_uuid} 的人物或其特征向量。")

    # Step 2: 如果没有提供 query_person_uuids，则处理 query_image_path (从中提取人物特征作为查询源)
    elif query_image_path:
        logger.info(f"find_similar_people: 从查询图片 {query_image_path} 提取特征。")
        try:
            # 构建完整的图片路径，确保包含 'backend' 目录并标准化斜杠
            # 更严格的标准化处理，确保所有斜杠统一，并拼接
            cleaned_query_image_path = query_image_path.replace('/', os.sep).replace('\\', os.sep) # 统一分隔符
            normalized_query_image_path = os.path.normpath(cleaned_query_image_path)
            
            full_image_path = os.path.join(settings.BASE_DIR, "backend", normalized_query_image_path)

            original_image = cv2.imread(full_image_path)
            if original_image is None:
                raise HTTPException(status_code=400, detail=f"无法读取查询图片: {full_image_path}")
            
            detection_model = get_detection_model(db)
            detection_results = detection_model(original_image, verbose=False)

            extracted_query_persons_from_image = [] # 存储从图片中检测到的人物及其特征
            for r in detection_results:
                if r.boxes is not None:
                    for box_idx, box in enumerate(r.boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_name = detection_model.names[int(box.cls[0])]

                        if class_name == 'person':
                            person_crop = original_image[y1:y2, x1:x2]
                            if person_crop.shape[0] > 0 and person_crop.shape[1] > 0:
                                query_feature = get_person_feature(reid_session, person_crop)
                                extracted_query_persons_from_image.append({
                                    "uuid": str(uuid.uuid4()), # 临时UUID，仅用于标识此查询人物
                                    "feature": query_feature,
                                    "crop_image_path": query_image_path, # 临时使用原始图片路径
                                    "full_frame_image_path": query_image_path,
                                })
                            else:
                                logger.warning(f"find_similar_people: 提取的人物裁剪图为空，跳过特征提取。")

            if not extracted_query_persons_from_image:
                logger.warning(f"find_similar_people: 查询图片 {query_image_path} 中未检测到任何人脸或人物。")
                return {"total_overall_results": 0, "total_query_persons": 0, "items": [], "skip": skip, "limit": limit}
            
            query_sources.extend(extracted_query_persons_from_image)
            logger.info(f"find_similar_people: 从查询图片 {query_image_path} 提取了 {len(extracted_query_persons_from_image)} 个人物特征。")

        except Exception as e:
            logger.error(f"find_similar_people: 从查询图片提取特征失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"从查询图片提取特征失败: {e}")

    if not query_sources:
        logger.warning("find_similar_people: 没有有效的查询特征向量进行搜索。")
        return {"total_overall_results": 0, "total_query_persons": 0, "items": [], "skip": skip, "limit": limit}

    faiss_initialized = False
    with _faiss_index_lock:
        if _faiss_index_instance is not None and _faiss_index_instance.ntotal > 0:
            faiss_initialized = True

    # 修改: 确保 get_all_persons 急切加载 Individual
    all_persons_in_db = crud.get_all_persons(db) # 无论是否使用Faiss，都获取所有人物，用于过滤和详细信息

    # 如果提供了 target_person_uuids，则筛选 all_persons_in_db
    if target_person_uuids:
        all_persons_in_db = [p for p in all_persons_in_db if p.uuid in target_person_uuids]
        logger.info(f"find_similar_people: 已根据 target_person_uuids 筛选，剩余 {len(all_persons_in_db)} 个人物特征用于匹配。")

    logger.info(f"find_similar_people: 数据库中共有 {len(all_persons_in_db)} 个人物特征用于潜在匹配。")
    all_persons_map = {p.uuid: p for p in all_persons_in_db} # 转换为字典方便查找

    for query_source in query_sources:
        current_query_feature = query_source["feature"]
        current_query_uuid = query_source["uuid"]
        current_query_crop_path = query_source["crop_image_path"]
        current_query_full_frame_path = query_source["full_frame_image_path"]

        individual_search_results = []
        individual_unique_result_uuids = set() # 避免单个查询人物内部重复结果

        if faiss_initialized:
            logger.info(f"find_similar_people: 使用 Faiss 索引为查询人物 {current_query_uuid} 进行搜索。")
            query_vector_2d = np.expand_dims(current_query_feature, axis=0)
            
            # 如果设置了 target_person_uuids，需要构建一个新的 Faiss 索引或执行过滤搜索
            # 这里简化处理：直接遍历 filtered_all_persons_in_db，执行传统比对
            # 或者，如果 target_person_uuids 较少，可以从 Faiss 中先检索所有，再过滤
            # 更优化的方法是创建一个临时的 Faiss 索引，或使用 Faiss 的子集搜索功能
            # 鉴于复杂度，这里我们选择遍历 all_persons_in_db 并手动比对，以确保 target_person_uuids 过滤生效

            # 如果 Faiss 已经初始化，但设置了 target_person_uuids，则执行传统搜索
            if target_person_uuids:
                logger.info(f"find_similar_people: 发现 target_person_uuids，回退到传统数据库搜索以确保精确过滤。")
                for person_in_db in all_persons_in_db: # 此时 all_persons_in_db 已经过 target_person_uuids 过滤
                    if person_in_db.feature_vector and person_in_db.uuid != current_query_uuid: # 避免与查询人物自身比对
                        try:
                            db_feature = person_in_db.feature_vector
                            if isinstance(db_feature, str):
                                db_feature = json.loads(db_feature)
                            if isinstance(db_feature, list):
                                db_feature_np = np.array(db_feature, dtype=np.float32)
                                distance = np.linalg.norm(current_query_feature - db_feature_np)
                                similarity_score = (2.0 - distance) / 2.0 * 100.0

                                if similarity_score >= threshold * 100:
                                    if person_in_db.uuid not in individual_unique_result_uuids:
                                        result_item = {
                                            "uuid": person_in_db.uuid,
                                            "score": similarity_score,
                                            "crop_image_path": person_in_db.crop_image_path,
                                            "full_frame_image_path": person_in_db.full_frame_image_path,
                                            "timestamp": person_in_db.created_at, 
                                            "video_id": person_in_db.video_id,
                                            "video_filename": person_in_db.video.filename if person_in_db.video else None,
                                            "video_uuid": person_in_db.video.uuid if person_in_db.video else None,
                                            "stream_id": person_in_db.stream_id,
                                            "stream_name": person_in_db.stream.name if person_in_db.stream else None,
                                            "stream_uuid": person_in_db.stream.stream_uuid if person_in_db.stream else None,
                                            "upload_image_uuid": person_in_db.image.uuid if person_in_db.image else None,
                                            "upload_image_filename": person_in_db.image.filename if person_in_db.image else None,
                                            # 新增 Individual 字段
                                            "individual_id": person_in_db.individual_id,
                                            "individual_uuid": person_in_db.individual.uuid if person_in_db.individual else None,
                                            "individual_name": person_in_db.individual.name if person_in_db.individual else None,
                                            "individual_id_card": person_in_db.individual.id_card if person_in_db.individual else None,
                                        }
                                        individual_search_results.append(result_item)
                                        individual_unique_result_uuids.add(person_in_db.uuid)
                                        overall_unique_results_uuids.add(person_in_db.uuid) # 添加到总的不重复结果集
                        except json.JSONDecodeError:
                            logger.warning(f"人物 {person_in_db.uuid} 的特征向量不是有效的 JSON，跳过传统搜索匹配。")
            else: # 如果 Faiss 已初始化且没有 target_person_uuids，则使用 Faiss
                k_value = min(_faiss_index_instance.ntotal, settings.FAISS_SEARCH_K + 1)
                distances, indices = _faiss_index_instance.search(query_vector_2d, k_value)
                
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    distance = distances[0][i]
                    
                    if idx == -1: continue

                    similarity_score = (2.0 - distance) / 2.0 * 100.0

                    if similarity_score >= threshold * 100:
                        matched_person_uuid = _faiss_person_uuids[idx]
                        if matched_person_uuid != current_query_uuid and matched_person_uuid not in individual_unique_result_uuids: # 避免添加查询人物自身和重复结果
                            person_obj = all_persons_map.get(matched_person_uuid)
                            if person_obj:
                                result_item = {
                                    "uuid": person_obj.uuid,
                                    "score": similarity_score,
                                    "crop_image_path": person_obj.crop_image_path,
                                    "full_frame_image_path": person_obj.full_frame_image_path,
                                    "timestamp": person_obj.created_at, 
                                    "video_id": person_obj.video_id,
                                    "video_filename": person_obj.video.filename if person_obj.video else None,
                                    "video_uuid": person_obj.video.uuid if person_obj.video else None,
                                    "stream_id": person_obj.stream_id,
                                    "stream_name": person_obj.stream.name if person_obj.stream else None,
                                    "stream_uuid": person_obj.stream.stream_uuid if person_obj.stream else None,
                                    "upload_image_uuid": person_obj.image.uuid if person_obj.image else None,
                                    "upload_image_filename": person_obj.image.filename if person_obj.image else None,
                                    # 新增 Individual 字段
                                    "individual_id": person_obj.individual_id,
                                    "individual_uuid": person_obj.individual.uuid if person_obj.individual else None,
                                    "individual_name": person_obj.individual.name if person_obj.individual else None,
                                    "individual_id_card": person_obj.individual.id_card if person_obj.individual else None,
                                }
                                individual_search_results.append(result_item)
                                individual_unique_result_uuids.add(matched_person_uuid)
                                overall_unique_results_uuids.add(matched_person_uuid) # 添加到总的不重复结果集
                            else:
                                logger.warning(f"find_similar_people: Faiss 索引中未找到 UUID 为 {matched_person_uuid} 的人物，跳过此结果。")
                    else:
                        logger.debug(f"find_similar_people: Faiss 匹配分数 {similarity_score:.2f}% 低于阈值 {threshold * 100:.2f}%，跳过人物 {matched_person_uuid}。")
                
                if not individual_search_results:
                    logger.info(f"find_similar_people: Faiss 搜索未为查询人物 {current_query_uuid} 找到高于阈值的匹配结果。")

        else:
            logger.warning(f"find_similar_people: Faiss 索引未初始化或为空，执行传统数据库搜索为查询人物 {current_query_uuid}。")
            for person_in_db in all_persons_in_db:
                if person_in_db.feature_vector and person_in_db.uuid != current_query_uuid: # 避免与查询人物自身比对
                    try:
                        db_feature = person_in_db.feature_vector
                        if isinstance(db_feature, str):
                            db_feature = json.loads(db_feature)
                        if isinstance(db_feature, list):
                            db_feature_np = np.array(db_feature, dtype=np.float32)
                            distance = np.linalg.norm(current_query_feature - db_feature_np)
                            similarity_score = (2.0 - distance) / 2.0 * 100.0

                            if similarity_score >= threshold * 100:
                                if person_in_db.uuid not in individual_unique_result_uuids:
                                    result_item = {
                                        "uuid": person_in_db.uuid,
                                        "score": similarity_score,
                                        "crop_image_path": person_in_db.crop_image_path,
                                        "full_frame_image_path": person_in_db.full_frame_image_path,
                                        "timestamp": person_in_db.created_at, 
                                        "video_id": person_in_db.video_id,
                                        "video_filename": person_in_db.video.filename if person_in_db.video else None,
                                        "video_uuid": person_in_db.video.uuid if person_in_db.video else None,
                                        "stream_id": person_in_db.stream_id,
                                        "stream_name": person_in_db.stream.name if person_in_db.stream else None,
                                        "stream_uuid": person_in_db.stream.stream_uuid if person_in_db.stream else None,
                                        "upload_image_uuid": person_in_db.image.uuid if person_in_db.image else None,
                                        "upload_image_filename": person_in_db.image.filename if person_in_db.image else None,
                                        # 新增 Individual 字段
                                        "individual_id": person_in_db.individual_id,
                                        "individual_uuid": person_in_db.individual.uuid if person_in_db.individual else None,
                                        "individual_name": person_in_db.individual.name if person_in_db.individual else None,
                                        "individual_id_card": person_in_db.individual.id_card if person_in_db.individual else None,
                                    }
                                    individual_search_results.append(result_item)
                                    individual_unique_result_uuids.add(person_in_db.uuid)
                                    overall_unique_results_uuids.add(person_in_db.uuid) # 添加到总的不重复结果集
                    except json.JSONDecodeError:
                        logger.warning(f"人物 {person_in_db.uuid} 的特征向量不是有效的 JSON，跳过传统搜索匹配。")
        
        # 对单个查询人物的结果进行排序和分页
        individual_search_results.sort(key=lambda x: x["score"], reverse=True)

        # 根据 video_uuid 或 stream_uuid 过滤单个查询人物的结果
        filtered_individual_results = individual_search_results
        if video_uuid:
            filtered_individual_results = [r for r in filtered_individual_results if r.get("video_uuid") == video_uuid]
        if stream_uuid:
            filtered_individual_results = [r for r in filtered_individual_results if r.get("stream_uuid") == stream_uuid]
        
        # 将单个查询人物的结果添加到分组结果中
        grouped_results.append({
            "query_person_uuid": current_query_uuid,
            "query_crop_image_path": current_query_crop_path,
            "query_full_frame_image_path": current_query_full_frame_path,
            "total_results_for_query_person": len(filtered_individual_results),
            "results": filtered_individual_results
        })

    # 最终结果按每个查询人物的最好分数排序 (可选，目前按输入顺序)
    # grouped_results.sort(key=lambda x: x["results"][0]["score"] if x["results"] else -1, reverse=True)

    total_query_persons = len(grouped_results)
    total_overall_results = len(overall_unique_results_uuids)

    # 后端分页只应用于查询人物的分组，而不是扁平化所有结果
    paginated_grouped_results = grouped_results[skip : skip + limit]

    logger.info(f"find_similar_people: 搜索完成。总查询人物数: {total_query_persons}，总不重复结果数: {total_overall_results}。")
    return {"total_overall_results": total_overall_results, "total_query_persons": total_query_persons, "items": paginated_grouped_results, "skip": skip, "limit": limit}


# --- 静态图片分析函数 ---
def analyze_image(original_image_uuid: str, original_image_full_path: str, original_image_filename: str, db: Session, current_user: schemas.User):
    logger.info(f"开始分析静态图片: {original_image_full_path}")
    if not os.path.exists(original_image_full_path):
        logger.error(f"文件不存在或无法访问: {original_image_full_path}")
        raise HTTPException(status_code=400, detail="上传的图片文件不存在或无法访问。")

    img = cv2.imread(original_image_full_path)
    if img is None:
        logger.error(f"cv2.imread 无法读取图片文件: {original_image_full_path}")
        raise HTTPException(status_code=400, detail="无法读取图片文件，请确保它是有效的图片格式。")
    logger.debug("图片文件读取成功。")

    try:
        logger.debug(f"尝试加载姿态检测模型: {settings.POSE_MODEL_PATH}")
        detection_pose_model = get_detection_model(db)
        logger.info(f"图片分析：姿态检测模型加载成功。")
        # 新增：加载人脸检测和人脸识别模型
        face_detection_model = get_face_detection_model(db)
        face_recognition_session = get_face_recognition_session(db)
        logger.info(f"图片分析：人脸检测模型和人脸识别模型加载成功。")
    except Exception as e:
        logger.error(f"加载模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器模型加载失败: {e}")

    analyzed_persons_uuids = []

    try:
        logger.info(f"即将保存的 Image file_path: {original_image_full_path}")
        import backend.crud as crud # 移动导入到函数内部
        from backend.crud import get_all_followed_enrollment_person_uuids # 移动导入到函数内部
        new_image = crud.create_image(db=db, image=schemas.ImageCreate(
            uuid=original_image_uuid,
            filename=original_image_filename,
            file_path=settings.get_parsed_image_relative_path(
                base_dir_name="full_frames",
                model_name=settings.IMAGE_ANALYSIS_MODEL_NAME, # 假设图片分析使用一个特定的模型，如 general_detection
                analysis_type="image",
                uuid=original_image_uuid
            ) + f"/{original_image_filename}",
            person_count=0,
            owner_id=current_user.id
        ))
        logger.info(f"已创建图片记录，ID: {new_image.id}, UUID: {new_image.uuid}")

        original_image = cv2.imread(original_image_full_path)
        if original_image is None:
            logger.error(f"无法读取原始图片: {original_image_full_path}")
            raise HTTPException(status_code=500, detail="内部错误：无法重新读取已保存的原始图片。")
        logger.info(f"成功读取原始图片: {original_image_full_path}, 尺寸: {original_image.shape}")

        logger.info("开始进行目标检测和姿态估计...")
        results_pose = detection_pose_model(original_image, verbose=False)
        logger.info(f"目标检测和姿态估计完成。检测结果数量: {len(results_pose)}")

        detected_person_count = 0
        for r_pose in results_pose:
            if r_pose.boxes is not None and len(r_pose.boxes) > 0:
                for i, box in enumerate(r_pose.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    class_name = detection_pose_model.names[cls]

                    if class_name == 'person' and conf > settings.IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE:
                        detected_person_count += 1
                        person_bbox = (x1, y1, x2, y2)
                        
                        try:
                            # 对于图片分析，全帧图片的路径直接使用原始图片的路径
                            full_frame_rel_path_for_person = new_image.file_path # original_image_full_path 是绝对路径

                            person_create_data = media_processing.process_detected_person_data(
                                db=db,
                                person_uuid=str(uuid.uuid4()), # 为新检测到的人物生成 UUID
                                frame=original_image,
                                person_bbox=person_bbox,
                                confidence_score=float(conf),
                                yolo_results_obj=r_pose, # 传递原始 YOLO 结果对象
                                media_type="image",
                                media_uuid=original_image_uuid,
                                # video_id_int=new_image.id, # 对于图片，image_id 就是 new_image.id - 修正为 image_id_int
                                image_id_int=new_image.id, # 对于图片，image_id 就是 new_image.id
                                full_frame_image_path_override=full_frame_rel_path_for_person, # 传递原始图片的相对路径作为 override
                                original_filename=original_image_filename,
                                model_name="general_detection",
                                # 新增：传递人脸检测和人脸识别模型
                                face_detection_model=face_detection_model,
                                face_recognition_session=face_recognition_session
                            )

                            # 在 crud.create_person 之前添加日志
                            logger.info(f"DEBUG: 将要保存到数据库的 PersonCreate.crop_image_path: {person_create_data.crop_image_path}")
                            logger.info(f"DEBUG: 将要保存到数据库的 PersonCreate.full_frame_image_path: {person_create_data.full_frame_image_path}")
                            logger.info(f"DEBUG: 将要保存到数据库的 PersonCreate.face_image_path: {person_create_data.face_image_path}")
                            logger.info(f"DEBUG: 将要保存到数据库的 PersonCreate.gait_image_path: {person_create_data.gait_image_path}")

                            new_person = crud.create_person(db=db, person=person_create_data)
                            analyzed_persons_uuids.append(new_person.uuid)
                            logger.info(f"已保存人物信息到数据库会话，UUID: {new_person.uuid}, 裁剪图片路径: {new_person.crop_image_path}, 完整图片路径: {new_person.full_frame_image_path}, Image ID: {new_image.id}")
                            logger.info(f"DEBUG: newly created person individual_id: {new_person.individual_id}") # 新增调试日志

                            # *** 新增：图片解析后自动关联到已关注人物的逻辑 ***
                            # 1. 获取所有当前被关注人员的注册图片的人物UUIDs
                            #    只有当实时比对功能开启时，才进行自动关联比对
                            all_followed_enrollment_person_uuids = crud.get_followed_enrollment_person_uuids_by_realtime_status(db, current_user.id, True) # 仅获取开启实时比对的关注人员的注册图片UUIDs
                            
                            if all_followed_enrollment_person_uuids:
                                logger.info(f"图片解析：发现 {len(all_followed_enrollment_person_uuids)} 个已关注人员的注册图片用于自动关联比对。")
                                # 2. 使用新检测人物的特征与这些注册图片进行比对
                                auto_association_results = find_similar_people(
                                    db=db,
                                    query_person_uuids=[new_person.uuid],
                                    target_person_uuids=all_followed_enrollment_person_uuids,
                                    threshold=settings.AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE / 100.0 # 使用新的阈值
                                )
                                
                                if auto_association_results and auto_association_results["items"]:
                                    # 假设只查询一个人物，所以取第一个结果
                                    best_match_for_new_person = auto_association_results["items"][0]["results"]
                                    if best_match_for_new_person:
                                        # 找到最高分的匹配
                                        best_match = max(best_match_for_new_person, key=lambda x: x["score"])
                                        
                                        # 检查最高分是否达到自动关联阈值
                                        if best_match["score"] >= settings.AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE:
                                            matched_individual_id = best_match["individual_id"]
                                            if matched_individual_id:
                                                # 更新新人物的 individual_id
                                                new_person.individual_id = matched_individual_id
                                                # 更新置信度分数
                                                new_person.confidence_score = best_match["score"]
                                                # 标记为已审核
                                                new_person.is_verified = True
                                                new_person.verified_by_user_id = current_user.id
                                                new_person.verification_date = datetime.now()

                                                db.add(new_person) # 标记为更新
                                                db.flush() # 刷新以使更改在当前事务中可见
                                                logger.info(f"图片解析：新人物 {new_person.uuid} 已自动关联到 Individual {matched_individual_id}，置信度 {best_match['score']:.2f}。")
                                            else:
                                                logger.warning(f"图片解析：最佳匹配人物 {best_match['uuid']} 未关联 Individual，无法自动关联。")
                                        else:
                                            logger.info(f"图片解析：新人物 {new_person.uuid} 最佳匹配分数 {best_match['score']:.2f} 低于自动关联阈值 {settings.AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE:.2f}。")
                                    else:
                                        logger.info(f"图片解析：新人物 {new_person.uuid} 未在已关注人员注册图片中找到匹配项。")
                                else:
                                    logger.info(f"图片解析：新人物 {new_person.uuid} 与已关注人员注册图片比对无结果。")
                            else:
                                logger.info("图片解析：没有已关注人员的注册图片，跳过自动关联比对。")
                            # *** 自动关联逻辑结束 ***

                            # 如果人物被成功关联到 Individual，则触发异步比对任务
                            if new_person.individual_id:
                                # 导入 FollowedPerson 模型和 crud 函数 (局部导入，避免循环依赖)
                                from backend.database_conn import FollowedPerson
                                from backend.crud import get_followed_person_by_individual_id

                                # 检查该 Individual 是否启用了实时比对功能
                                # 注意：这里需要通过 current_user.id 来查找 FollowedPerson 记录
                                followed_person_record = get_followed_person_by_individual_id(db, current_user.id, new_person.individual_id)
                                    
                                if followed_person_record and followed_person_record.realtime_comparison_enabled:
                                    from backend.ml_services.ml_tasks import compare_new_person_with_followed_enrollments # 内部导入 Celery 任务
                                    compare_new_person_with_followed_enrollments.delay(new_person.uuid, new_person.individual_id)
                                    logger.info(f"已为新检测人物 {new_person.uuid} (Individual: {new_person.individual_id}) 触发与关注人员注册图片比对任务。")
                                else:
                                    logger.info(f"Individual {new_person.individual_id} 的实时比对功能未启用，跳过人物 {new_person.uuid} 的比对任务触发。")

                        except ValueError as ve:
                            logger.warning(f"处理检测到的人物时发生错误: {ve}，跳过此人物。")
                            detected_person_count -= 1 # 如果跳过，人物计数减一
                            continue

        new_image.person_count = detected_person_count
        db.add(new_image)

        logger.info("所有人物对象已添加到数据库会话，即将提交事务。")
        db.commit()
        logger.info(f"图片分析完成，数据库事务已提交，共检测到 {len(analyzed_persons_uuids)} 个人物。")
        
        logger.info(f"尝试将 {len(analyzed_persons_uuids)} 个人物添加到 Faiss 索引...")
        for person_uuid in analyzed_persons_uuids:
            person_db_entry = crud.get_person_by_uuid_obj(db, uuid=person_uuid);
            if person_db_entry and person_db_entry.feature_vector:
                try:
                    feature = json.loads(person_db_entry.feature_vector)
                    if isinstance(feature, list) and len(feature) == settings.FEATURE_DIM:
                        add_person_feature_to_faiss(person_uuid, np.array(feature, dtype=np.float32))
                        logger.debug(f"人物 {person_uuid} 特征已添加到 Faiss。")
                    else:
                        logger.warning(f"人物 {person_uuid} 的特征向量格式不正确或维度不符，跳过添加到 Faiss。")
                except json.JSONDecodeError:
                    logger.warning(f"人物 {person_uuid} 的特征向量不是有效的 JSON，跳过添加到 Faiss。")

        logger.info("所有新人物特征已尝试添加到 Faiss 索引。")

        return new_image.id

    except Exception as e:
        logger.error(f"分析图片时发生错误: {e}", exc_info=True)
        db.rollback()
        logger.error("数据库事务已回滚。")
        raise HTTPException(status_code=500, detail=f"图片解析失败: {e}")


def process_enrollment_image(
    db: Session,
    image_bytes: bytes, 
    person_name: Optional[str] = None,
    id_card: Optional[str] = None,
    current_user: schemas.User = None
) -> schemas.PersonEnrollResponse:
    logger.info(f"开始处理主动注册图片. 人物姓名: {person_name}, 身份证号: {id_card}") # 添加 id_card 调试信息

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        logger.error("无法解码上传的图片。")
        raise HTTPException(status_code=400, detail="无法读取图片文件，请确保它是有效的图片格式。")

    # 为上传的原始图片生成 UUID 和文件名
    original_image_uuid = str(uuid.uuid4())
    # 假设是 JPG 格式，实际应根据MIME类型判断
    original_image_filename = f"{original_image_uuid}.jpg" 
    
    # 始终按身份证号分目录保存原始图片
    if not id_card:
        # 如果没有身份证号，可以考虑抛出错误或使用一个默认的未知ID目录
        # 根据您的描述，这里我们要求必须有身份证号来创建子目录
        logger.error("未提供身份证号，无法按身份证号创建子目录保存图片。")
        raise HTTPException(status_code=400, detail="请提供身份证号以保存图片。")

    original_image_dir = os.path.join(settings.ENROLL_PERSON_IMAGES_DIR, id_card)  # 按身份证号创建子目录
    
    os.makedirs(original_image_dir, exist_ok=True)
    original_image_full_path = os.path.join(original_image_dir, original_image_filename)

    # 保存原始图片到文件系统
    try:
        with open(original_image_full_path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"原始注册图片已保存到: {original_image_full_path}")
    except IOError as e:
        logger.error(f"保存原始注册图片失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"保存原始注册图片失败: {e}")

    # 在 Image 表中创建记录
    image_create_schema = schemas.ImageCreate(
        uuid=original_image_uuid,
        filename=original_image_filename,
        # 存储相对路径并统一为正斜杠，确保路径是相对于 BASE_DIR 的
        file_path=os.path.relpath(original_image_full_path, settings.BASE_DIR).replace(os.sep, '/'), 
        owner_id=current_user.id,
        person_count=0 # 初始设置为0，后续更新
    )
    new_image_record = crud.create_image(db, image_create_schema)
    original_image_id = new_image_record.id
    logger.info(f"已在 Image 表中创建记录，ID: {original_image_id}, UUID: {original_image_uuid}")


    # 1. 根据 id_card 查找或创建 Individual
    individual_id_to_link = None
    individual_uuid_to_return = None
    individual_id_card_to_return = id_card 

    if id_card:
        existing_individual = crud.get_individual_by_id_card(db, id_card)
        if existing_individual:
            individual_id_to_link = existing_individual.id
            individual_uuid_to_return = existing_individual.uuid
            # 如果提供了 person_name 且与现有 Individual 的名称不同，则更新 Individual 的名称
            if person_name and existing_individual.name != person_name:
                existing_individual.name = person_name
                db.add(existing_individual) # Mark for update
                db.commit() # Commit the update to individual
                db.refresh(existing_individual)
                logger.info(f"已更新 Individual {existing_individual.uuid} 的名称为 {person_name}。")
        else:
            # 创建新的 Individual
            new_individual = crud.create_individual(db, schemas.IndividualCreate(
                name=person_name,
                id_card=id_card,
                uuid=str(uuid.uuid4()) # 为新的 Individual 生成 UUID
            ))
            individual_id_to_link = new_individual.id
            individual_uuid_to_return = new_individual.uuid
            logger.info(f"已创建新的 Individual {new_individual.uuid}，ID: {new_individual.id}。")
    elif person_name: # 如果只提供了姓名，没有身份证号，也创建一个新的 Individual (但 id_card 为 None)
        # 考虑到 id_card 是唯一索引，如果 id_card 为 None，可能会创建多个 id_card 为 None 的 Individual
        # 最佳实践是强制 id_card 存在或者确保唯一性
        # 暂时允许创建无 id_card 的 Individual，但未来可能需要调整业务逻辑
        new_individual = crud.create_individual(db, schemas.IndividualCreate(
            name=person_name,
            id_card=None, # 没有 id_card
            uuid=str(uuid.uuid4())
        ))
        individual_id_to_link = new_individual.id
        individual_uuid_to_return = new_individual.uuid
        individual_id_card_to_return = None # 确保返回 None
        logger.warning(f"已创建新的 Individual {new_individual.uuid} 仅包含姓名，无身份证号。")
    else: # 如果两者都未提供，则无法创建或关联 Individual
        logger.error("未提供身份证号或姓名，无法进行人物注册。")
        raise HTTPException(status_code=400, detail="请提供人物的身份证号或姓名进行注册。")

    try:
        detection_pose_model = get_detection_model(db)
        logger.info(f"注册：姿态检测模型加载成功。")
    except Exception as e:
        logger.error(f"加载模型失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器模型加载失败: {e}")
    
    logger.info("开始进行目标检测和姿态估计...")
    results_pose = detection_pose_model(img, verbose=False)
    
    detected_persons_data = []
    for r_pose in results_pose:
        if r_pose.boxes is not None and len(r_pose.boxes) > 0:
            for i, box in enumerate(r_pose.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                class_name = detection_pose_model.names[cls]

                if class_name == 'person' and conf > settings.ENROLLMENT_MIN_PERSON_CONFIDENCE:
                    person_bbox = (x1, y1, x2, y2)
                    try:
                        person_create_data = media_processing.process_detected_person_data(
                            db=db,
                            frame=img,
                            person_bbox=person_bbox,
                            confidence_score=float(conf),
                            yolo_results_obj=r_pose,
                            media_type="image", 
                            media_uuid=original_image_uuid, # 使用原始图片 UUID
                            person_uuid=str(uuid.uuid4()), 
                            individual_id=individual_id_to_link, 
                            model_name="general_detection",
                            image_id_int=original_image_id, # 传递 image_id
                            full_frame_image_path_override=os.path.relpath(original_image_full_path, settings.BASE_DIR).replace(os.sep, '/'), # 传递原始图片的相对路径作为 override
                            original_filename=original_image_filename # 传递原始文件名
                        )
                        detected_persons_data.append(person_create_data)
                    except ValueError as ve:
                        logger.warning(f"处理检测到的人物时发生错误: {ve}，跳过此人物。")
                        continue
    
    if not detected_persons_data:
        raise HTTPException(status_code=400, detail="未能在图片中检测到人物或置信度过低。")
    
    saved_features_count = 0
    final_person_uuid_to_return = None # 初始化为 None
    verified_by_user_id = current_user.id if current_user else None

    for person_data in detected_persons_data:
        # 统一设置审核状态和关联用户，因为这是注册流程
        person_data.is_verified = True
        person_data.verified_by_user_id = verified_by_user_id
        person_data.verification_date = datetime.now()
        
        # 设置 individual_id
        person_data.individual_id = individual_id_to_link

        # *** 新增：主动注册的人物标记为已训练 ***
        person_data.is_trained = True # 注册的图片，is_trained 必须为 True

        new_person = crud.create_person(db, person=person_data) # 创建新的 Person 记录

        # 如果是创建的第一个人物，则将其 UUID 设为最终返回的 UUID
        if final_person_uuid_to_return is None:
            final_person_uuid_to_return = new_person.uuid
        logger.info(f"已创建新人物 {new_person.uuid} 并保存特征，关联 Individual ID: {individual_id_to_link}。")
        saved_features_count += 1

        # 将新人物的特征添加到 Faiss 索引
        if new_person.feature_vector:
            try:
                feature = json.loads(new_person.feature_vector)
                if isinstance(feature, list) and len(feature) == settings.FEATURE_DIM:
                    add_person_feature_to_faiss(new_person.uuid, np.array(feature, dtype=np.float32))
                    logger.info(f"人物 {new_person.uuid} 特征已添加到 Faiss。")
                else:
                    logger.warning(f"人物 {new_person.uuid} 的特征向量格式不正确或维度不符，跳过添加到 Faiss。")
            except json.JSONDecodeError:
                logger.warning(f"人物 {new_person.uuid} 的特征向量不是有效的 JSON，跳过添加到 Faiss。")

    if saved_features_count == 0:
        raise HTTPException(status_code=500, detail="未能保存任何人物特征。")

    if final_person_uuid_to_return is None:
        raise HTTPException(status_code=500, detail="未能确定最终人物 UUID。")

    # 更新 Image 表中的 person_count
    if new_image_record:
        new_image_record.person_count = saved_features_count
        db.add(new_image_record)
        db.commit()
        db.refresh(new_image_record)
        logger.info(f"已更新 Image {new_image_record.uuid} 的人物数量为 {saved_features_count}。")

    return schemas.PersonEnrollResponse(
        person_uuid=final_person_uuid_to_return, 
        individual_uuid=individual_uuid_to_return, 
        individual_id_card=individual_id_card_to_return, 
        message=f"人物注册成功。特征 UUID: {final_person_uuid_to_return}, 逻辑人物 UUID: {individual_uuid_to_return}"
    )


# --- 实时视频流处理函数 ---
def process_live_frame_and_save_features(
    db: Session,
    frame: np.ndarray,
    stream_uuid: str,
    stream_id_int: int,
    reid_session: Any,
    detection_model: Any, 
    yolo_results_with_tracks: Any, 
    seen_track_ids: set,
    face_detection_model: Any,
    face_recognition_session: Any,
    gait_recognition_session: Any = None,
    tracklet_gait_buffer: Dict[int, List[np.ndarray]] = {},
    gait_sequence_length: int = settings.GAIT_SEQUENCE_LENGTH
) -> Tuple[Optional[np.ndarray], List[Person]]:
    new_person_objects = []
    annotated_frame = frame.copy()

    current_yolo_results = yolo_results_with_tracks

    logger.debug(f"YOLO 检测结果: {current_yolo_results}")

    tracks_from_tracker = current_yolo_results.boxes

    if not tracks_from_tracker:
        logger.info("YOLO.track() 在当前帧没有返回任何跟踪结果。")
        return annotated_frame, []

    for track in tracks_from_tracker:
        track_id = int(track.id.item()) if track.id is not None else -1
        x1, y1, x2, y2 = map(int, track.xyxy[0].tolist())
        confidence_score = float(track.conf.item())
        class_id = int(track.cls.item())

        if class_id != settings.PERSON_CLASS_ID:
            continue

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if track_id not in seen_track_ids:
            person_bbox = (x1, y1, x2, y2)
            
            try:
                person_create_data = media_processing.process_detected_person_data(
                    db=db,
                    person_uuid=str(uuid.uuid4()), # 为新检测到的人物生成 UUID
                    frame=frame, # 传递原始帧
                    person_bbox=person_bbox,
                    confidence_score=confidence_score,
                    yolo_results_obj=current_yolo_results, 
                    media_type="stream",
                    media_uuid=stream_uuid,
                    stream_id_int=stream_id_int,
                    gait_recognition_session=gait_recognition_session,
                    tracklet_gait_buffer=tracklet_gait_buffer,
                    track_id=track_id,
                    model_name="person_reid", # 修改为人体识别
                    face_detection_model=face_detection_model, # 新增
                    face_recognition_session=face_recognition_session # 新增
                )
                new_person_objects.append(person_create_data)
                seen_track_ids.add(track_id) # 标记此 track_id 已被保存

                # 如果新检测到的人物成功关联到 Individual，且该 Individual 开启了实时比对，则触发异步比对任务
                if person_create_data.individual_id:
                    from backend.ml_services.ml_tasks import compare_new_person_with_followed_enrollments # 内部导入
                    from backend.database_conn import FollowedPerson # 局部导入
                    from backend.crud import get_followed_person_by_individual_id # 局部导入
                    
                    followed_person_record = get_followed_person_by_individual_id(db, current_user.id, person_create_data.individual_id)

                    if followed_person_record and followed_person_record.realtime_comparison_enabled:
                        compare_new_person_with_followed_enrollments.delay(person_create_data.person_uuid, person_create_data.individual_id)
                        logger.info(f"已为新检测人物 {person_create_data.person_uuid} (Individual: {person_create_data.individual_id}) 触发与关注人员注册图片比对任务。")
                    else:
                        logger.info(f"Individual {person_create_data.individual_id} 的实时比对功能未启用，跳过人物 {person_create_data.person_uuid} 的比对任务触发。")

            except ValueError as ve:
                logger.warning(f"处理视频流人物时发生错误: {ve}，跳过此人物。")
                continue
        else:
            pass

    return annotated_frame, new_person_objects