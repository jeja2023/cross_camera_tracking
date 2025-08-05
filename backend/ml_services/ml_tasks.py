# backend/ml_services/ml_tasks.py
import cv2
import numpy as np
import onnxruntime as ort
import os
import uuid
import json
from tqdm import tqdm
from typing import Optional, List
import time
import logging
import subprocess
import dataclasses
import threading
from ultralytics import YOLO
import signal
from datetime import datetime
import pytz
from sqlalchemy.orm import Session

# 导入 Celery 应用实例
from backend.celery_app import celery_app
# 导入数据库和 Schemas
from ..database_conn import SessionLocal, Person, Video, Stream, User, Image, Individual, FollowedPerson # 导入所有需要的模型
from .. import schemas
# from .. import crud # 导入 crud 模块 # 移除此行

# 导入 stream_manager
from backend.stream_manager import save_frame_to_redis, get_frame_from_redis

# 从 ml_logic 中导入公共的辅助函数和配置，确保它们在 ml_logic.py 中仍然存在
from . import ml_logic
from backend.config import settings

# 新增：导入 BOTSORT
from ultralytics.trackers.bot_sort import BOTSORT

# 导入 media_processing 模块
from backend.utils import media_processing

logger = logging.getLogger(__name__)

# 定义一个类来模拟 BoT-SORT 所需的 args 对象
@dataclasses.dataclass
class TrackerArgs:
    # 跟踪参数 (BoT-SORT 期望的参数)
    # 如果 ACTIVE_REID_MODEL_PATH 为空，则 with_reid 应该为 False
    with_reid: bool = False # 禁用 BoT-SORT 内部 Re-ID，我们使用 ml_logic 中的 Re-ID 模型
    model: str = settings.ACTIVE_REID_MODEL_PATH if settings.ACTIVE_REID_MODEL_PATH else "" # Re-ID 模型路径，直接使用 ONNX 模型路径
    fuse_score: bool = True # 融合置信度分数
    gmc_method: str = 'orb' # 全局运动补偿方法 (options: 'orb', 'ecc', 'files')
    proximity_thresh: float = settings.TRACKER_PROXIMITY_THRESH # 接近度阈值 (IoU)
    appearance_thresh: float = settings.TRACKER_APPEARANCE_THRESH # 外观相似度阈值 (Re-ID)
    track_buffer: int = settings.TRACKER_TRACK_BUFFER # 轨迹保留帧数，用于计算 max_time_lost
    high_thresh: float = settings.TRACKER_HIGH_THRESH  # 新增：高置信度阈值
    low_thresh: float = settings.TRACKER_LOW_THRESH    # 新增：低置信度阈值
    new_track_thresh: float = settings.TRACKER_NEW_TRACK_THRESH # 新增：新轨迹初始化阈值
    min_hits: int = settings.TRACKER_MIN_HITS # 新增：轨迹初始化所需的最低检测次数

# 全局模型实例，在 Celery Worker 启动时加载一次
detection_model = None
reid_session = None # 保持此变量，如果 BoT-SORT 内部 ReID 无法加载 ONNX，我们仍然需要外部 ReID
tracker_video = None # For video processing task
tracker_stream = None # For live stream processing task
face_detection_model = None # 新增：全局人脸检测模型
face_recognition_session = None # 新增：全局人脸识别模型
gait_recognition_session = None # 新增：全局步态识别模型

# 移除 load_models_globally 函数及其所有调用
# def load_models_globally():
#     """
#     在Celery worker启动时加载模型和tracker，避免重复加载。
#     如果模型已经加载，则不会重新加载。如果加载失败，将抛出异常。
#     """
#     global detection_model, reid_session, tracker_video, tracker_stream, face_detection_model, face_recognition_session, gait_recognition_session
    
#     # 如果模型已经加载，则直接返回，避免重复加载
#     if detection_model is not None and reid_session is not None and tracker_video is not None and tracker_stream is not None and face_detection_model is not None and face_recognition_session is not None:
#         logger.info("ML 模型和 BoT-SORT Tracker 已在当前进程中加载，跳过重复加载。")
#         return
#     try:
#         logger.info("尝试加载 ML 模型和 BoT-SORT Tracker...")
#         detection_model = ml_logic.YOLO(ml_logic.settings.POSE_MODEL_PATH, task='detect') # 使用姿态模型
#         db = SessionLocal() # 获取数据库会话
#         try:
#             reid_session = ml_logic.get_reid_session(db) # 保持加载原始 Re-ID session，以防 BoT-SORT 的 Re-ID 兼容性问题
#             face_recognition_session = ml_logic.get_face_recognition_session(db) # 加载人脸识别模型
#             gait_recognition_session = ml_logic.get_gait_recognition_session(db) # 新增：加载步态识别模型
#         finally:
#             db.close()
#         # 初始化 BoT-SORT 跟踪器
#         tracker_args = TrackerArgs() # 使用默认参数实例化 args
#         # 为视频处理和实时流处理创建 BoT-SORT 实例
#         # 注意：BoT-SORT 的 frame_rate 参数通常在 __init__ 中设置
#         tracker_video = BOTSORT(tracker_args, frame_rate=settings.VIDEO_PROCESSING_FRAME_RATE) # 使用配置的视频帧率
#         tracker_stream = BOTSORT(tracker_args, frame_rate=settings.STREAM_PROCESSING_FRAME_RATE) # 使用配置的实时流帧率
#         face_detection_model = ml_logic.YOLO(ml_logic.settings.FACE_DETECTION_MODEL_PATH, task='detect') # 加载人脸检测模型，明确指定task为'detect'
#         logger.info("ML 模型和 BoT-SORT Tracker 已成功加载到 Celery Worker。")
#     except Exception as e:
#         logger.error(f"Celery Worker 初始化时加载模型失败: {e}", exc_info=True)
#         raise # 重新抛出异常，确保外部调用者知道加载失败
#         # 根据需要，可以选择在这里抛出异常，阻止worker启动，或者记录错误并继续
#         # sys.exit(1) # 如果模型加载是强制性的，可以退出

# 在模块加载时（即 Celery Worker 启动时）调用加载函数
# load_models_globally() # 移除此行，避免在导入时就尝试加载模型

@celery_app.task(bind=True) # 使用 Celery 装饰器，并绑定 self 参数
def process_video_for_extraction_task(self, video_path: str, video_id: int):
    """
    Celery 任务：处理视频并提取特征。
    """
    try:
        # 在任务开始时初始化进度为0
        progress = 0 
        terminated_by_signal = False # 初始化为 False

        db = SessionLocal()
        
        # 确保 Faiss 索引在任务开始时被初始化 (移到这里)
        try:
            ml_logic.initialize_faiss_index(db) # 确保 Faiss 索引在任务开始时初始化
        except Exception as e:
            logger.error(f"视频 (ID: {video_id}): 初始化 Faiss 索引失败。任务无法继续。错误: {e}\n", exc_info=True)
            db.query(Video).filter(Video.id == video_id).update({"status": "failed"})
            db.commit()
            self.update_state(state="FAILED", meta={'progress': progress, 'status': 'failed'})
            return

        # 确保模型已加载 (如果未加载或加载失败，则尝试加载)
        try:
            # 直接从 ml_logic 获取模型实例，不再依赖全局变量
            current_detection_model = ml_logic.get_detection_model(db) # 使用新的获取函数
            current_reid_session = ml_logic.get_reid_session(db)
            face_detection_model_instance = ml_logic.get_face_detection_model(db)
            face_recognition_session_instance = ml_logic.get_face_recognition_session(db)
            gait_recognition_session_instance = ml_logic.get_gait_recognition_session(db) # 获取步态识别模型

            # 动态设置 TrackerArgs 中的 Re-ID 模型路径和 with_reid 标志
            use_reid_with_tracker = False
            reid_model_for_tracker_path = ""
            if settings.ACTIVE_REID_MODEL_PATH and os.path.exists(settings.ACTIVE_REID_MODEL_PATH):
                use_reid_with_tracker = True
                reid_model_for_tracker_path = settings.ACTIVE_REID_MODEL_PATH
            else:
                logger.warning(f"BoT-SORT 跟踪器 Re-ID 模型路径 '{settings.ACTIVE_REID_MODEL_PATH}' 无效或文件不存在，将禁用 BoT-SORT 内部 Re-ID。\n")

            tracker_args = TrackerArgs(
                with_reid=False, # 强制禁用 BoT-SORT 内部 Re-ID
                model="" # 确保不传递任何模型路径给 BoT-SORT 的内部 Re-ID
            )
            current_tracker = BOTSORT(tracker_args, frame_rate=settings.VIDEO_PROCESSING_FRAME_RATE) # 为视频处理创建 BoT-SORT 实例

            # 检查所有模型是否加载成功
            if current_detection_model is None or current_reid_session is None or current_tracker is None or face_detection_model_instance is None or face_recognition_session_instance is None:
                raise Exception("核心 ML 模型或 Tracker 未能成功加载。")
            logger.info(f"视频 (ID: {video_id}): AI模型加载成功。\n")
        except Exception as e:
            logger.error(f"视频 (ID: {video_id}): ML 模型或 Tracker 加载失败，任务无法继续。错误: {e}\n", exc_info=True)
            db.query(Video).filter(Video.id == video_id).update({"status": "failed"})
            db.commit()
            self.update_state(state="FAILED", meta={'progress': progress, 'status': 'failed'})
            return

        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            logger.error(f"视频 (ID: {video_id}) 不存在。\n")
            return

        logger.info(f"Celery 任务开始：处理视频 {video_path} (Video ID: {video_id})\n")

        def update_status_and_commit(status: str, progress: int = 0):
            """辅助函数，用于更新状态并提交，同时更新 Celery 任务状态"""
            db.query(Video).filter(Video.id == video_id).update({"status": status})
            db.commit()
            self.update_state(state=status.upper(), meta={'progress': progress, 'status': status})
            logger.info(f"视频 (ID: {video_id}) 状态已更新为 {status} 并已提交。 Celery 任务状态: {status.upper()}.\n")

        session_id_map = {} # 用于存储 track_id 到 person_uuid 的映射
        seen_track_ids = set() # 跟踪已经保存过特征的 track_id
        terminated_by_signal = False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}\n")
            update_status_and_commit("failed")
            return

        # 获取视频的 UUID 用于构建保存路径
        db_video_info = db.query(Video).filter(Video.id == video_id).first()
        if not db_video_info or not db_video_info.uuid:
            # 如果视频没有UUID，或者获取失败，则生成一个新的UUID并更新到数据库
            if not db_video_info: # 如果连视频记录都找不到，直接失败
                logger.error(f"视频 ID {video_id} 不存在，无法启动任务。\n")
                update_status_and_commit("failed")
                return
            else: # 视频记录存在但没有UUID，则生成并更新
                logger.warning(f"视频 {video_id} 没有关联的UUID，将生成并更新。\n")
                video_uuid = str(uuid.uuid4())
                db_video_info.uuid = video_uuid
                db.commit() # 提交UUID更新
                db.refresh(db_video_info)
                logger.info(f"视频 {video_id} 的UUID已更新为 {video_uuid}。\n")
        else:
            video_uuid = db_video_info.uuid

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            logger.warning(f"视频 (ID: {video_id}) 总帧数为0，可能已损坏。\n")
            update_status_and_commit("failed")
            return

        pbar = tqdm(total=total_frames, desc=f"后台提取 {os.path.basename(video_path)}", unit="frame")
        last_progress_reported = -1
        frame_count = 0 # 初始化帧计数器
        
        COMMIT_BATCH_SIZE = settings.VIDEO_COMMIT_BATCH_SIZE # 从配置中获取批处理大小
        frames_processed_since_last_commit = 0
        persons_to_add = [] # 收集待添加的人物对象

        while cap.isOpened():
            # 检查终止信号
            video_status_check = db.query(Video).filter(Video.id == video_id).first()
            if not video_status_check:
                logger.warning(f"视频 (ID: {video_id}) 在处理中被删除，终止任务。\n")
                terminated_by_signal = True
                break
            if video_status_check.status == "terminated":
                logger.info(f"检测到终止信号，停止处理视频 (ID: {video_id})。\n")
                terminated_by_signal = True
                break
            
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1 # 增加帧计数
            # 每处理 500 帧或达到总帧数的 1% 时记录一次
            if frame_count % 500 == 0 or (total_frames > 0 and frame_count % (total_frames // 100) == 0 and total_frames // 100 > 0):
                progress = int((frame_count / total_frames) * 100)
                logger.info(f"视频 (ID: {video_id}): 已处理 {frame_count} / {total_frames} 帧. (Progress: {progress}%)\n")

            # 使用 YOLOv8 模型的 .track() 方法进行检测和跟踪
            results = current_detection_model.track(
                frame,
                persist=True, # 启用跨帧跟踪
                classes=settings.PERSON_CLASS_ID,
                conf=settings.DETECTION_CONFIDENCE_THRESHOLD,
                verbose=False
            )
            # track 方法返回一个 Results 对象列表，我们通常只取第一个
            results_yolo = results[0] # 获取第一个结果对象，包含检测和跟踪信息
            
            # 直接从 results_yolo 中获取 tracks 和 boxes 数据
            tracks = results_yolo.boxes # 获取包含跟踪ID的 Boxes 对象

            for track in tracks: # 遍历每个跟踪结果 (这里 track 是一个 Box 对象)
                # track 是一个 Box 对象，包含 xyxy, conf, cls, id 等属性
                track_id = int(track.id.item()) if track.id is not None else -1 # 获取跟踪ID
                x1, y1, x2, y2 = map(int, track.xyxy[0].tolist()) # 获取边界框坐标
                confidence_score = float(track.conf.item()) # 获取置信度分数
                class_id = int(track.cls.item()) # 获取类别ID
                
                # 只处理人形检测
                if class_id != settings.PERSON_CLASS_ID:
                    continue

                # 检查 track_id 是否已经被保存过 (即是否是新人物)
                if track_id not in session_id_map and track_id not in seen_track_ids: # 确保在当前会话中和所有已保存人物中都是新的
                    person_bbox = (x1, y1, x2, y2)
                    try:
                        person_obj = media_processing.process_detected_person_data(
                            db=db,
                            person_uuid=str(uuid.uuid4()), # 为新检测到的人物生成 UUID
                            frame=frame, # 传递原始帧
                            person_bbox=person_bbox,
                            confidence_score=confidence_score,
                            yolo_results_obj=results_yolo, # 传递YOLO结果
                            media_type="video",
                            media_uuid=video_uuid,
                            video_id_int=video_id,
                            # 新增：传递人脸识别和步态识别模型
                            face_detection_model=face_detection_model_instance,
                            face_recognition_session=face_recognition_session_instance,
                            gait_recognition_session=gait_recognition_session_instance, # 如果为None，则传递None
                            track_id=track_id
                        )
                        # 直接创建 Person 对象并添加到会话
                        db_person_obj = Person(
                            uuid=person_obj.uuid,
                            feature_vector=json.dumps(person_obj.feature_vector),
                            crop_image_path=person_obj.crop_image_path,
                            full_frame_image_path=person_obj.full_frame_image_path,
                            video_id=person_obj.video_id,
                            stream_id=person_obj.stream_id,
                            image_id=person_obj.image_id,
                            is_verified=person_obj.is_verified,
                            verified_by_user_id=person_obj.verified_by_user_id,
                            verification_date=person_obj.verification_date,
                            correction_details=person_obj.correction_details,
                            marked_for_retrain=person_obj.marked_for_retrain,
                            confidence_score=person_obj.confidence_score,
                            pose_keypoints=json.dumps(person_obj.pose_keypoints) if person_obj.pose_keypoints else None,
                            face_image_path=person_obj.face_image_path,
                            face_feature_vector=json.dumps(person_obj.face_feature_vector) if person_obj.face_feature_vector else None,
                            face_id=person_obj.face_id,
                            clothing_attributes=json.dumps(person_obj.clothing_attributes) if person_obj.clothing_attributes else None,
                            gait_feature_vector=json.dumps(person_obj.gait_feature_vector) if person_obj.gait_feature_vector else None,
                            gait_image_path=person_obj.gait_image_path,
                            individual_id=person_obj.individual_id,
                            is_trained=person_obj.is_trained
                        )
                        db.add(db_person_obj)
                        db.flush() # Flush to get ID for newly created person if needed
                        persons_to_add.append(db_person_obj)
                        session_id_map[track_id] = person_obj.uuid
                        seen_track_ids.add(track_id)
                        logger.info(f"视频 (ID: {video_id}) 新人物 {person_obj.uuid} (Track ID: {track_id}) 已添加到待处理列表。\n")
                    except ValueError as ve:
                        logger.warning(f"处理视频人物时发生错误: {ve}，跳过此人物。\n")
                        continue
            
            pbar.update(1)
            frames_processed_since_last_commit += 1
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = int((current_frame / total_frames) * 100)

            if progress > last_progress_reported:
                db.query(Video).filter(Video.id == video_id).update({"progress": progress})
                db.commit()
                self.update_state(state='PROCESSING', meta={'progress': progress, 'status': 'processing'})
                last_progress_reported = progress
            
            if frames_processed_since_last_commit >= COMMIT_BATCH_SIZE:
                try:
                    if persons_to_add: # 如果有待提交的人物
                        db.commit()
                        for p in persons_to_add:
                            ml_logic.add_person_feature_to_faiss(p.uuid, np.array(json.loads(p.feature_vector), dtype=np.float32))
                            db.refresh(p)
                        persons_to_add = [] # 清空列表
                        logger.info(f"视频 (ID: {video_id}) 批量提交 {COMMIT_BATCH_SIZE} 帧的数据。 当前已处理总帧数: {frame_count}。\n")
                    else:
                        db.rollback()
                except Exception as commit_e:
                    db.rollback()
                    logger.error(f"视频 (ID: {video_id}) 批量提交时发生错误: {commit_e}\n", exc_info=True)
                    persons_to_add = [] 
                frames_processed_since_last_commit = 0
        
        pbar.close()
        cap.release()

        if persons_to_add:
            try:
                db.commit()
                for p in persons_to_add:
                    ml_logic.add_person_feature_to_faiss(p.uuid, np.array(json.loads(p.feature_vector), dtype=np.float32))
                    db.refresh(p)
                logger.info(f"视频 (ID: {video_id}) 循环结束后提交剩余 {len(persons_to_add)} 个人物数据。\n")
            except Exception as commit_e:
                db.rollback()
                logger.error(f"视频 (ID: {video_id}) 循环结束后提交剩余数据时发生错误: {commit_e}\n", exc_info=True)

        final_status = "terminated" if terminated_by_signal else "completed"
        db.query(Video).filter(Video.id == video_id).update({"status": final_status})
        db.commit()
        self.update_state(state=final_status.upper(), meta={'progress': 100, 'status': final_status})


    except Exception as e:
        logger.error(f"处理视频 {video_path} (ID: {video_id}) 时发生未捕获的严重错误: {e}\n", exc_info=True)
        if db.is_active:
            try:
                db.rollback() # 在异常情况下回滚所有未提交的更改
                logger.error(f"Attempting to set video {video_id} status to 'failed' due to unhandled exception.\n")
                db.query(Video).filter(Video.id == video_id).update({"status": "failed"})
                db.commit()
                self.update_state(state="FAILED", meta={'progress': progress, 'status': 'failed'})
            except Exception as commit_e:
                logger.error(f"视频 (ID: {video_id}) 异常处理中更新状态失败: {commit_e}\n", exc_info=True)
    finally:
        db.close()


# 新增 Celery 任务：再训练 Re-ID 模型
@celery_app.task(bind=True)
def retrain_reid_model_task(self, model_version: str = "v2", person_uuids: Optional[List[str]] = None):
    """
    Celery 任务：使用指定数据或标记为再训练的数据微调 Re-ID 模型。
    model_version 参数用于指定新模型的版本，例如 'v2', 'v3'。
    person_uuids 如果提供，则只使用这些人物的数据进行再训练。
    """
    logger.info(f"Celery 任务开始：再训练 Re-ID 模型 (版本: {model_version}), 指定人物UUIDs: {person_uuids}\n")
    self.update_state(state='STARTED', meta={'progress': 0, 'status': 'retraining'})
    db = SessionLocal()
    try:
        # 1. 根据 person_uuids 参数获取人物数据
        if person_uuids:
            persons_for_retrain = db.query(Person).filter(Person.uuid.in_(person_uuids)).all()
            logger.info(f"使用指定的 {len(person_uuids)} 个人物 UUID 进行再训练。实际找到 {len(persons_for_retrain)} 个人物。\n")
        else:
            # 如果没有指定 UUIDs，则获取所有标记为 'marked_for_retrain' 的人物数据
            persons_for_retrain = db.query(Person).filter(Person.marked_for_retrain == True).all()
            logger.info(f"没有指定人物 UUID，将使用所有标记为再训练的 {len(persons_for_retrain)} 个人物数据。\n")
        
        if not persons_for_retrain:
            logger.info("没有有效的人物数据用于再训练，跳过模型再训练。\n")
            self.update_state(state='SKIPPED', meta={'progress': 100, 'status': 'skipped'})
            return False

        # 转换为可序列化的字典列表，避免 ORM 对象在多进程中传递的问题
        serializable_persons_data = []
        for person in persons_for_retrain:
            serializable_persons_data.append({
                "id": person.id, # 添加 id
                "uuid": person.uuid,
                "feature_vector": person.feature_vector,
                "individual_id": person.individual_id,
                "crop_image_path": person.crop_image_path # 添加 crop_image_path
            })

        # 2. 调用 ml_logic 中的再训练函数
        # 传递序列化后的数据，而不是 ORM 对象
        new_model_path = ml_logic.retrain_reid_model(
            serializable_persons_data,
            model_version,
            db, # 传递db会话，以便在ml_logic中处理数据库操作（例如更新is_verified）
            celery_task=self # 传递 Celery 任务实例，以便更新进度
        )

        if new_model_path:
            # 再训练成功后，批量更新所有参与再训练的人物状态
            person_uuids_to_update = [p.uuid for p in persons_for_retrain]
            db.query(Person).filter(Person.uuid.in_(person_uuids_to_update)).update(
                {
                    Person.marked_for_retrain: False,
                    Person.is_trained: True,
                    Person.is_verified: True,
                    Person.verified_by_user_id: 1,
                    Person.verification_date: datetime.now(pytz.timezone('Asia/Shanghai'))
                },
                synchronize_session="fetch"
            )
            db.commit()

            logger.info(f"Re-ID 模型再训练成功，新模型路径: {new_model_path}。已更新所有参与再训练的人物标记和训练状态。\n")
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'completed', 'new_model': new_model_path})
            return True
        else:
            logger.warning("Re-ID 模型再训练未完成或失败。\n")
            self.update_state(state='FAILED', meta={'progress': 100, 'status': 'failed'})
            return False

    except Exception as e:
        logger.error(f"Re-ID 模型再训练任务失败: {e}\n", exc_info=True)
        self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed', 'error': str(e)})
        return False
    finally:
        db.close()


@celery_app.task(bind=True)
def process_live_stream_task(self, stream_id: int, stream_url: str):
    """
    Celery 任务：处理实时视频流并提取特征，保存标注视频。
    """
    db = SessionLocal()
    
    # 确保 Faiss 索引在任务开始时被初始化 (移到这里)
    try:
        ml_logic.initialize_faiss_index(db) # 确保 Faiss 索引在任务开始时初始化
    except Exception as e:
        logger.error(f"视频流 (ID: {stream_id}): 初始化 Faiss 索引失败。任务无法继续。错误: {e}\n", exc_info=True)
        db.query(Stream).filter(Stream.id == stream_id).update({"status": "failed"})
        db.commit()
        self.update_state(state="FAILED", meta={'progress': 0, 'status': 'failed'})
        return

    cap: Optional[cv2.VideoCapture] = None
    stream_uuid: str = "" # 用于构建保存路径
    terminated_by_signal: bool = False # 确保在任何退出路径上都已定义

    # 新增：用于存储每个 track_id 的步态序列 (裁剪图列表)
    tracklet_gait_buffer: dict[int, list[np.ndarray]] = {}
    # 步态序列长度阈值 (与 ml_logic 中定义的一致)
    gait_sequence_length = settings.GAIT_SEQUENCE_LENGTH # 步态序列长度 (与 ml_logic 中定义的一致)

    try:
        logger.info(f"Celery 任务开始：处理视频流 {stream_url} (Stream ID: {stream_id})\n")

        # 获取流的 UUID 用于构建保存路径
        db_stream_info = db.query(Stream).filter(Stream.id == stream_id).first()
        if db_stream_info:
            stream_uuid = db_stream_info.stream_uuid
        else:
            logger.error(f"Stream ID {stream_id} 不存在，无法启动任务。\n")
            db.query(Stream).filter(Stream.id == stream_id).update({"status": "failed"})
            db.commit()
            return
        
        if not stream_uuid:
            logger.error(f"视频流 (ID: {stream_id}) 的 UUID 未获取到。\n")
            db.query(Stream).filter(Stream.id == stream_id).update({"status": "failed"})
            db.commit()
            return

        # 加载模型
        try:
            # 直接从 ml_logic 获取模型实例
            current_detection_model = ml_logic.get_detection_model(db)
            current_reid_session = ml_logic.get_reid_session(db)
            face_detection_model_instance = ml_logic.get_face_detection_model(db)
            face_recognition_session_instance = ml_logic.get_face_recognition_session(db)
            gait_recognition_session_instance = ml_logic.get_gait_recognition_session(db)

            # 动态设置 TrackerArgs 中的 Re-ID 模型路径和 with_reid 标志
            use_reid_with_tracker = False
            reid_model_for_tracker_path = ""
            if settings.ACTIVE_REID_MODEL_PATH and os.path.exists(settings.ACTIVE_REID_MODEL_PATH):
                use_reid_with_tracker = True
                reid_model_for_tracker_path = settings.ACTIVE_REID_MODEL_PATH
            else:
                logger.warning(f"视频流 BoT-SORT 跟踪器 Re-ID 模型路径 '{settings.ACTIVE_REID_MODEL_PATH}' 无效或文件不存在，将禁用 BoT-SORT 内部 Re-ID。\n")

            tracker_args = TrackerArgs(
                with_reid=False, # 强制禁用 BoT-SORT 内部 Re-ID
                model="" # 确保不传递任何模型路径给 BoT-SORT 的内部 Re-ID
            )
            if current_detection_model is None or current_reid_session is None or face_detection_model_instance is None or face_recognition_session_instance is None:
                raise Exception("核心 ML 模型或 Tracker 未能成功加载。")
            logger.info(f"视频流 {stream_id}: AI模型加载成功。\n")
        except Exception as e:
            logger.error(f"视频流 {stream_id}: 加载AI模型失败: {e}\n", exc_info=True)
            db.query(Stream).filter(Stream.id == stream_id).update({"status": "failed"})
            db.commit()
            return

        # 确保 Faiss 索引在任务开始时被初始化 (移到这里)
        try:
            ml_logic.initialize_faiss_index(db) # 确保 Faiss 索引在任务开始时初始化
        except Exception as e:
            logger.error(f"视频流 (ID: {stream_id}): 初始化 Faiss 索引失败。任务无法继续。错误: {e}\n", exc_info=True)
            db.query(Stream).filter(Stream.id == stream_id).update({"status": "failed"})
            db.commit()
            self.update_state(state="FAILED", meta={'progress': 0, 'status': 'failed'})
            return

        # 新增：模型加载成功后，更新流状态为 'processing'
        logger.info(f"Attempting to set stream {stream_id} status to 'processing' after model load.\n")
        db.query(Stream).filter(Stream.id == stream_id).update({"status": "processing"})
        db.commit()
        logger.info(f"Stream {stream_id} status updated to 'processing' successfully.\n")

        seen_track_ids = set() # 跟踪已经保存过特征的人物ID

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            logger.error(f"无法打开视频流: {stream_url}\n")
            db.query(Stream).filter(Stream.id == stream_id).update({"status": "failed"})
            db.commit()
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25 # 如果获取不到FPS，默认为25

        # 确定保存视频的路径和文件名
        stream_obj = db.query(Stream).filter(Stream.id == stream_id).first()
        if not stream_obj or not stream_obj.stream_uuid:
            logger.error(f"视频流 {stream_id}: 无法获取流的UUID，终止任务。\n")
            db.query(Stream).filter(Stream.id == stream_id).update({"status": "failed"})
            db.commit()
            return
        stream_uuid = stream_obj.stream_uuid

        save_dir = os.path.join(settings.SAVED_STREAMS_DIR, stream_uuid)
        os.makedirs(save_dir, exist_ok=True)
        output_video_filename = f"{stream_uuid}.mp4"
        absolute_output_video_path = os.path.join(save_dir, output_video_filename)
        relative_output_video_path = os.path.join(stream_uuid, output_video_filename).replace(os.sep, '/')

        fourcc = cv2.VideoWriter_fourcc(*'avc1') # 改为H.264编码，提升浏览器兼容性
        out = cv2.VideoWriter(absolute_output_video_path, fourcc, fps, (frame_width, frame_height))
        
        if not out.isOpened():
            logger.error(f"视频流 {stream_id}: 无法初始化视频写入器到 {absolute_output_video_path}。请检查路径、权限或编解码器。\n")
            db.query(Stream).filter(Stream.id == stream_id).update({"status": "failed"})
            db.commit()
            return # 无法写入视频，提前退出任务

        logger.info(f"视频流 {stream_id}: 初始化视频保存到 {absolute_output_video_path}\n")

        COMMIT_BATCH_SIZE = 100 # 每100帧提交一次数据库
        frames_processed_since_last_commit = 0
        persons_to_add_live = [] # 收集待添加的人物对象 (直播流专用)
        frame_count = 0 # 初始化帧计数器

        while cap.isOpened():
            # 检查终止信号
            stream_status_check = db.query(Stream).filter(Stream.id == stream_id).first()
            if not stream_status_check:
                logger.warning(f"视频流 (ID: {stream_id}) 在处理中被删除，终止任务。\n")
                terminated_by_signal = True
                break
            if stream_status_check.status == "terminated" or stream_status_check.status == "stopped": # 检查 terminated 或 stopped
                logger.info(f"检测到终止信号，停止处理视频流 (ID: {stream_id})。状态: {stream_status_check.status}\n")
                terminated_by_signal = True
                break
            
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"视频流 {stream_id} 读取完毕或断开连接。\n")
                break

            # 修复 NameError：确保 annotated_frame 总是被定义
            annotated_frame = frame.copy()

            frame_count += 1 # 增加帧计数
            # 每处理 500 帧时记录一次
            if frame_count % 500 == 0:
                logger.debug(f"视频流 (ID: {stream_id}): 已处理 {frame_count} 帧。\n")

            results_live = current_detection_model.track(source=frame, persist=True, tracker="botsort.yaml", verbose=False)
            if results_live and results_live[0].boxes.id is not None:
                annotated_frame = results_live[0].plot()
                
                for box in results_live[0].boxes:
                    track_id = int(box.id.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence_score = float(box.conf.item())
                    class_id = int(box.cls.item())

                    if class_id != settings.PERSON_CLASS_ID:
                        continue

                    if track_id not in seen_track_ids:
                        person_bbox = (x1, y1, x2, y2)
                        try:
                            person_create_data = media_processing.process_detected_person_data(
                                db=db, frame=frame, person_bbox=person_bbox,
                                confidence_score=confidence_score, yolo_results_obj=results_live[0],
                                media_type="stream", media_uuid=stream_uuid, person_uuid=str(uuid.uuid4()),
                                stream_id_int=stream_id,
                                gait_recognition_session=gait_recognition_session_instance,
                                tracklet_gait_buffer=tracklet_gait_buffer, track_id=track_id,
                                face_detection_model=face_detection_model_instance,
                                face_recognition_session=face_recognition_session_instance
                            )
                            if person_create_data:
                                # 直接创建 Person 对象并添加到会话
                                db_person_obj = Person(
                                    uuid=person_create_data.uuid,
                                    feature_vector=json.dumps(person_create_data.feature_vector),
                                    crop_image_path=person_create_data.crop_image_path,
                                    full_frame_image_path=person_create_data.full_frame_image_path,
                                    video_id=person_create_data.video_id,
                                    stream_id=person_create_data.stream_id,
                                    image_id=person_create_data.image_id,
                                    is_verified=person_create_data.is_verified,
                                    verified_by_user_id=person_create_data.verified_by_user_id,
                                    verification_date=person_create_data.verification_date,
                                    correction_details=person_create_data.correction_details,
                                    marked_for_retrain=person_create_data.marked_for_retrain,
                                    confidence_score=person_create_data.confidence_score,
                                    pose_keypoints=json.dumps(person_create_data.pose_keypoints) if person_create_data.pose_keypoints else None,
                                    face_image_path=person_create_data.face_image_path,
                                    face_feature_vector=json.dumps(person_create_data.face_feature_vector) if person_create_data.face_feature_vector else None,
                                    face_id=person_create_data.face_id,
                                    clothing_attributes=json.dumps(person_create_data.clothing_attributes) if person_create_data.clothing_attributes else None,
                                    gait_feature_vector=json.dumps(person_create_data.gait_feature_vector) if person_create_data.gait_feature_vector else None,
                                    gait_image_path=person_create_data.gait_image_path,
                                    individual_id=person_create_data.individual_id,
                                    is_trained=person_create_data.is_trained
                                )
                                db.add(db_person_obj)
                                db.flush() # Flush to get ID for newly created person if needed
                                persons_to_add_live.append(db_person_obj)
                                seen_track_ids.add(track_id)
                        except ValueError as ve:
                            logger.warning(f"处理视频流人物时发生错误: {ve}，跳过此人物。\n")
                            continue

            out.write(annotated_frame)
            if annotated_frame is not None:
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    save_frame_to_redis(stream_uuid, buffer.tobytes())

            frames_processed_since_last_commit += 1
            if frames_processed_since_last_commit >= COMMIT_BATCH_SIZE:
                try:
                    if persons_to_add_live:
                        db.commit()
                        for p in persons_to_add_live:
                            ml_logic.add_person_feature_to_faiss(p.uuid, np.array(json.loads(p.feature_vector), dtype=np.float32))
                        persons_to_add_live = []
                except Exception as commit_e:
                    db.rollback()
                    logger.error(f"视频流 (ID: {stream_id}) 批量提交时发生错误: {commit_e}\n", exc_info=True)
                frames_processed_since_last_commit = 0

        cap.release()
        out.release()
        logger.info(f"视频流 {stream_id}: 视频捕获和写入完成。\n")

        try:
            ffprobe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,profile,width,height,avg_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                absolute_output_video_path
            ]
            logger.debug(f"视频流 {stream_id} 正在运行 ffprobe 命令: {ffprobe_cmd}\n")
            ffprobe_output = subprocess.check_output(ffprobe_cmd, stderr=subprocess.STDOUT, text=True)
            logger.info(f"视频流 {stream_id} 输出视频信息:\n{ffprobe_output}\n")
        except subprocess.CalledProcessError as e:
            logger.error(f"视频流 {stream_id} 运行 ffprobe 失败 (返回码: {e.returncode}): {e.output}\n", exc_info=True)
        except FileNotFoundError:
            logger.error(f"ffprobe 未找到。请确保已安装 FFmpeg 并将其添加到 PATH 中。\n")
        except Exception as e:
            logger.error(f"视频流 {stream_id} 获取视频信息时发生未知错误: {e}\n", exc_info=True)

        final_status = "terminated" if terminated_by_signal else "completed"
        logger.info(f"Attempting to set stream {stream_id} final status to '{final_status}'.\n")
        db.query(Stream).filter(Stream.id == stream_id).update({"status": final_status})
        db.query(Stream).filter(Stream.id == stream_id).update({"output_video_path": relative_output_video_path})
        db.commit()
        logger.info(f"Stream {stream_id} final status updated to '{final_status}' and output path updated successfully.\n")
        self.update_state(state=final_status.upper(), meta={'progress': 100, 'status': final_status})

        if persons_to_add_live:
            try:
                for person_data in persons_to_add_live:
                    # 直接操作 ORM 对象，不再通过 crud.create_person
                    # 注意：person_data 已经是 PersonCreate schema 对象，需要转换为 Person ORM 对象
                    db_person_obj = Person(
                        uuid=person_data.uuid,
                        feature_vector=json.dumps(person_data.feature_vector),
                        crop_image_path=person_data.crop_image_path,
                        full_frame_image_path=person_data.full_frame_image_path,
                        video_id=person_data.video_id,
                        stream_id=person_data.stream_id,
                        image_id=person_data.image_id,
                        is_verified=person_data.is_verified,
                        verified_by_user_id=person_data.verified_by_user_id,
                        verification_date=person_data.verification_date,
                        correction_details=person_data.correction_details,
                        marked_for_retrain=person_data.marked_for_retrain,
                        confidence_score=person_data.confidence_score,
                        pose_keypoints=json.dumps(person_data.pose_keypoints) if person_data.pose_keypoints else None,
                        face_image_path=person_data.face_image_path,
                        face_feature_vector=json.dumps(person_data.face_feature_vector) if person_data.face_feature_vector else None,
                        face_id=person_data.face_id,
                        clothing_attributes=json.dumps(person_data.clothing_attributes) if person_data.clothing_attributes else None,
                        gait_feature_vector=json.dumps(person_data.gait_feature_vector) if person_data.gait_feature_vector else None,
                        gait_image_path=person_data.gait_image_path,
                        individual_id=person_data.individual_id,
                        is_trained=person_data.is_trained
                    )
                    db.add(db_person_obj)
                    ml_logic.add_person_feature_to_faiss(db_person_obj.uuid, np.array(json.loads(db_person_obj.feature_vector), dtype=np.float32))
                    db.refresh(db_person_obj)
                db.commit()
                logger.info(f"视频流 (ID: {stream_id}) 循环结束后提交剩余 {len(persons_to_add_live)} 个人物数据。\n")
            except Exception as commit_e:
                db.rollback()
                logger.error(f"视频流 (ID: {stream_id}) 循环结束后提交剩余数据时发生错误: {commit_e}\n", exc_info=True)

        logger.info(f"视频流 {stream_id} 的处理任务已完成。最终状态: {final_status}。\n")

    except Exception as e:
        logger.error(f"处理视频流 {stream_id} 时发生未捕获的严重错误: {e}\n", exc_info=True)
        if db.is_active:
            try:
                db.rollback() # Rollback any pending changes
                logger.error(f"Attempting to set stream {stream_id} status to 'failed' due to unhandled exception.\n")
                db.query(Stream).filter(Stream.id == stream_id).update({"status": "failed"})
                db.commit()
                logger.error(f"Stream {stream_id} status updated to 'failed' due to unhandled exception.\n")
            except Exception as commit_e:
                logger.error(f"Failed to update stream {stream_id} status to 'failed' in exception handler: {commit_e}\n", exc_info=True)
    finally:
        if cap and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()
        db.close()


@celery_app.task(bind=True)
def analyze_image_task(self, original_image_uuid: str, original_image_full_path: str, original_image_filename: str, current_user_id: int):
    """
    Celery 任务：处理图片并提取特征。
    """
    logger.info(f"Celery 任务开始：分析图片 {original_image_filename} (UUID: {original_image_uuid})，用户 ID: {current_user_id}\n")
    self.update_state(state='STARTED', meta={'progress': 0, 'status': 'started'})
    db = SessionLocal()
    try:
        # 获取用户对象
        current_user = db.query(User).filter(User.id == current_user_id).first()
        if not current_user:
            logger.error(f"用户 ID {current_user_id} 不存在，无法执行图片分析任务。\n")
            self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed', 'error': 'User not found'})
            return None

        # 确保 Faiss 索引在任务开始时被初始化
        try:
            ml_logic.initialize_faiss_index(db)
        except Exception as e:
            logger.error(f"图片 (UUID: {original_image_uuid}): 初始化 Faiss 索引失败。任务无法继续。错误: {e}\n", exc_info=True)
            self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed', 'error': f"Faiss index initialization failed: {e}"})
            return None

        # 调用 ml_logic 中的图片分析函数
        created_image_id = ml_logic.analyze_image(
            original_image_uuid=original_image_uuid,
            original_image_full_path=original_image_full_path,
            original_image_filename=original_image_filename,
            db=db,
            current_user=current_user
        )

        if created_image_id:
            # 获取刚刚创建的 Image 对象，以便获取 person_count
            image_entry = db.query(Image).filter(Image.id == created_image_id).first()
            person_count = image_entry.person_count if image_entry else 0
            
            logger.info(f"图片 (UUID: {original_image_uuid}) 分析成功，检测到 {person_count} 个人物。\n")
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'completed', 'image_id': created_image_id, 'image_uuid': original_image_uuid, 'person_count': person_count})
            return created_image_id
        else:
            logger.error(f"图片 (UUID: {original_image_uuid}) 分析失败，未返回图片 ID。\n")
            self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed', 'error': 'Image analysis failed'})
            return None

    except Exception as e:
        logger.error(f"图片分析任务 (UUID: {original_image_uuid}) 发生未捕获的错误: {e}\n", exc_info=True)
        self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed', 'error': str(e)})
        return None
    finally:
        db.close()


# 定义一个新的 Celery 任务，用于在后台执行全库比对
@celery_app.task(bind=True)
def run_full_database_comparison(self, person_uuid: str):
    """
    在后台对指定人物进行全库比对，并更新相关人物的置信度得分。
    """
    self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting full database comparison'})
    
    db: Optional[Session] = None
    try:
        db = SessionLocal()
        
        query_person = db.query(Person).filter(Person.uuid == person_uuid).first()
        if not query_person:
            logger.warning(f"全库比对任务失败：未找到 UUID 为 {person_uuid} 的人物。\n")
            self.update_state(state='FAILURE', meta={'progress': 100, 'status': 'Person not found'})
            return
        
        if not query_person.feature_vector:
            logger.warning(f"人物 {person_uuid} 没有特征向量，跳过全库比对。\n")
            self.update_state(state='FAILURE', meta={'progress': 100, 'status': 'No feature vector for person'})
            return

        query_feature_vector = ml_logic.json.loads(query_person.feature_vector)
        
        logger.info(f"开始对人物 {person_uuid} 进行全库比对...\n")
        
        all_persons = db.query(Person).all()
        
        if not all_persons:
            logger.info("数据库中没有人物特征，跳过全库比对。\n")
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'No persons in database for comparison'})
            return

        search_results = ml_logic.find_similar_people(
            db=db,
            query_person_uuids=[person_uuid],
            threshold=settings.ALERT_IMAGE_MIN_CONFIDENCE_SCORE / 100.0
        )
        
        if search_results and search_results["items"]:
            comparison_results = search_results["items"][0]["results"]
            logger.info(f"人物 {person_uuid} 的全库比对完成，找到 {len(comparison_results)} 个匹配结果。\n")

            total_matches = len(comparison_results)
            for i, result_item in enumerate(comparison_results):
                matched_uuid = result_item["uuid"]
                confidence_score = result_item["score"]

                if matched_uuid != person_uuid:
                    matched_person = db.query(Person).filter(Person.uuid == matched_uuid).first()
                    if matched_person:
                        if matched_person.individual_id is None:
                            matched_person.individual_id = query_person.individual_id
                            logger.info(f"匹配人物 {matched_uuid} 已关联到 Individual {query_person.individual_id}。\n")

                        if matched_person.confidence_score is None or confidence_score > matched_person.confidence_score:
                            matched_person.confidence_score = confidence_score
                            db.add(matched_person)
                            db.commit()
                            db.refresh(matched_person)
                            logger.info(f"已更新人物 {matched_uuid} 的置信度得分到 {confidence_score:.2f}。\n")

                progress = int((i + 1) / total_matches * 100)
                self.update_state(state='PROGRESS', meta={'progress': progress, 'status': f'Processed {i+1}/{total_matches} matches'})
            
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Full database comparison completed'})

        else:
            logger.info(f"人物 {person_uuid} 的全库比对未找到任何匹配结果。\n")
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'No matches found for full database comparison'})

    except Exception as e:
        logger.error(f"全库比对任务执行失败: {e}\n", exc_info=True)
        self.update_state(state='FAILURE', meta={'progress': 0, 'status': 'Failed', 'error': str(e)})
    finally:
        if db:
            db.close()


# 新增：Celery 任务用于处理新检测到的人物特征与关注人员的注册图片进行比对
@celery_app.task(bind=True)
def compare_new_person_with_followed_enrollments(self, new_person_uuid: str, individual_id: int):
    """
    将新检测到的人物特征与指定 Individual 的所有注册图片进行比对，并更新置信度得分。
    仅当该 Individual 的 realtime_comparison_enabled 为 True 时才执行比对。
    """
    self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting comparison with followed enrollments'})

    db: Optional[Session] = None
    try:
        db = SessionLocal()
        
        # 获取关注人员记录以检查 realtime_comparison_enabled 状态
        followed_person_record = db.query(FollowedPerson).filter(
            FollowedPerson.individual_id == individual_id,
            FollowedPerson.unfollowed_at.is_(None) # 确保是当前关注的记录
        ).first()

        if not followed_person_record or not followed_person_record.realtime_comparison_enabled:
            logger.info(f"Individual {individual_id} 的实时比对功能未启用，跳过人物 {new_person_uuid} 的比对任务。\n")
            self.update_state(state='SKIPPED', meta={'progress': 100, 'status': 'Realtime comparison disabled for individual'})
            return

        new_person = db.query(Person).filter(Person.uuid == new_person_uuid).first()
        if not new_person or not new_person.feature_vector:
            logger.warning(f"比对任务失败：未找到 UUID 为 {new_person_uuid} 的人物或其特征向量。\n")
            self.update_state(state='FAILURE', meta={'progress': 100, 'status': 'New person not found or no feature'})
            return
        
        # 获取指定 Individual 的所有注册图片
        enrollment_persons = db.query(Person).filter(
            Person.individual_id == individual_id,
            Person.image_id.isnot(None)
        ).all()
        if not enrollment_persons:
            logger.info(f"Individual {individual_id} 没有注册图片，跳过比对。\n")
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'No enrollment images for individual'})
            return

        # 将注册图片的 UUIDs 列表传递给 find_similar_people 作为查询源
        enrollment_uuids = [p.uuid for p in enrollment_persons]
        
        search_results = ml_logic.find_similar_people(
            db=db,
            query_person_uuids=[new_person_uuid],
            target_person_uuids=enrollment_uuids,
            threshold=settings.ALERT_IMAGE_MIN_CONFIDENCE_SCORE / 100.0
        )

        if search_results and search_results["items"]:
            comparison_results = search_results["items"][0]["results"]
            logger.info(f"新人物 {new_person_uuid} 与关注人员注册图片比对完成，找到 {len(comparison_results)} 个匹配结果。\n")

            max_confidence_score = 0.0
            for result_item in comparison_results:
                matched_uuid = result_item["uuid"]
                confidence_score = result_item["score"]

                if confidence_score > max_confidence_score:
                    max_confidence_score = confidence_score
            
            if max_confidence_score >= settings.ALERT_IMAGE_MIN_CONFIDENCE_SCORE:
                new_person.confidence_score = max_confidence_score
                if new_person.individual_id is None:
                    new_person.individual_id = individual_id
                db.add(new_person)
                db.commit()
                db.refresh(new_person)
                logger.info(f"已更新新人物 {new_person_uuid} 的置信度得分到 {max_confidence_score:.2f}。\n")
            else:
                logger.info(f"新人物 {new_person_uuid} 与注册图片比对最高分低于阈值 {settings.ALERT_IMAGE_MIN_CONFIDENCE_SCORE:.2f}。\n")
            
        else:
            logger.info(f"新人物 {new_person_uuid} 与注册图片比对未找到任何匹配结果。\n")
        
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Comparison with followed enrollments completed'})

    except Exception as e:
        logger.error(f"新人物与关注人员注册图片比对任务执行失败: {e}\n", exc_info=True)
        self.update_state(state='FAILURE', meta={'progress': 0, 'status': 'Failed', 'error': str(e)})
    finally:
        if db:
            db.close()


@celery_app.task(bind=True)
def run_scheduled_full_database_comparison(self):
    self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Starting scheduled full database comparison'})
    db: Optional[Session] = None
    try:
        db = SessionLocal()
        # 获取所有已关注且未取消关注的人员
        followed_persons = db.query(FollowedPerson).filter(FollowedPerson.unfollowed_at.is_(None)).all()
        if not followed_persons:
            logger.info("没有关注人员，跳过定时全库比对。\n")
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'No followed persons'})
            return
        
        total_followed = len(followed_persons)
        for i, followed_person in enumerate(followed_persons):
            individual_id = followed_person.individual_id
            # 仅对启用了定时全库比对的关注人员执行比对
            if followed_person.realtime_comparison_enabled: # 检查 individual-specific 的实时比对开关
                enrollment_persons = db.query(Person).filter(
                    Person.individual_id == individual_id,
                    Person.image_id.isnot(None)
                ).all()
                if enrollment_persons:
                    for person in enrollment_persons:
                        if person.uuid:
                            run_full_database_comparison.delay(person.uuid)
                            logger.info(f"已为人物 {person.uuid} 触发定时全库比对 Celery 任务。\n")
                else:
                    logger.info(f"Individual {individual_id} 没有注册图片，跳过定时全库比对任务。\n")
            else:
                logger.info(f"Individual {individual_id} 的定时全库比对功能未启用，跳过此人物。\n")

            progress = int((i + 1) / total_followed * 100)
            self.update_state(state='PROGRESS', meta={'progress': progress, 'status': f'Processed {i+1}/{total_followed} followed persons'})
        
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'Scheduled full database comparison completed'})
    except Exception as e:
        logger.error(f"定时全库比对任务执行失败: {e}\n", exc_info=True)
        self.update_state(state='FAILURE', meta={'progress': 0, 'status': 'Failed', 'error': str(e)})
    finally:
        if db:
            db.close()
