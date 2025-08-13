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
import dataclasses # 新增：导入 dataclasses
import threading # 新增：导入 threading
from ultralytics import YOLO # 新增：导入 YOLO
import signal # 新增：导入 signal
from datetime import datetime # 新增：导入 datetime
import pytz # 新增：导入 pytz

# 导入 Celery 应用实例
from backend.celery_app import celery_app
# 导入数据库和 CRUD 操作
from ..database_conn import SessionLocal
import backend.crud # 确保导入了整个 crud 模块
import backend.crud_match as crud_match # 明确导入 crud_match 模块
# 导入 stream_manager
from backend.stream_manager import save_frame_to_redis, get_frame_from_redis # 新增导入

# 导入 schemas
from .. import schemas # 新增：导入 schemas 模块

# 从 ml_logic 中导入公共的辅助函数和配置，确保它们在 ml_logic.py 中仍然存在
# 请根据您 ml_logic.py 实际保留的内容来调整这些导入
from . import ml_logic # 修改为导入 ml_logic 模块
from backend.config import settings # 导入 settings

# 新增：导入 BOTSORT
from ultralytics.trackers.bot_sort import BOTSORT

# 导入 media_processing 模块
from backend.utils import media_processing

from backend.ml_services.ml_logic import initialize_faiss_index, get_reid_session, get_person_feature, _faiss_index_instance, _faiss_person_uuids # 新增导入 Faiss 相关的全局变量

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
        frames_processed_since_last_commit = 0 # 重新添加此行，确保变量被初始化

        db = SessionLocal()
        
        # 确保 Faiss 索引在任务开始时被初始化 (移到这里)
        try:
            ml_logic.initialize_faiss_index(db) # 确保 Faiss 索引在任务开始时初始化
        except Exception as e:
            logger.error(f"视频 (ID: {video_id}): 初始化 Faiss 索引失败。任务无法继续。错误: {e}", exc_info=True)
            backend.crud.update_video_status(db, video_id=video_id, status="failed")
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
                logger.warning(f"BoT-SORT 跟踪器 Re-ID 模型路径 '{settings.ACTIVE_REID_MODEL_PATH}' 无效或文件不存在，将禁用 BoT-SORT 内部 Re-ID。")

            tracker_args = TrackerArgs(
                with_reid=False, # 强制禁用 BoT-SORT 内部 Re-ID
                model="" # 确保不传递任何模型路径给 BoT-SORT 的内部 Re-ID
            )
            current_tracker = BOTSORT(tracker_args, frame_rate=settings.VIDEO_PROCESSING_FRAME_RATE) # 为视频处理创建 BoT-SORT 实例

            # 检查所有模型是否加载成功
            if current_detection_model is None or current_reid_session is None or current_tracker is None or face_detection_model_instance is None or face_recognition_session_instance is None:
                raise Exception("核心 ML 模型或 Tracker 未能成功加载。")
            logger.info(f"视频 (ID: {video_id}): AI模型加载成功。")
        except Exception as e:
            logger.error(f"视频 (ID: {video_id}): ML 模型或 Tracker 加载失败，任务无法继续。错误: {e}", exc_info=True)
            backend.crud.update_video_status(db, video_id=video_id, status="failed")
            db.commit()
            self.update_state(state="FAILED", meta={'progress': progress, 'status': 'failed'})
            return

        video = backend.crud.get_video(db, video_id=video_id)
        if not video:
            logger.error(f"视频 (ID: {video_id}) 不存在。")
            return

        logger.info(f"Celery 任务开始：处理视频 {video_path} (Video ID: {video_id})")

        def update_status_and_commit(status: str, progress: int = 0):
            """辅助函数，用于更新状态并提交，同时更新 Celery 任务状态"""
            backend.crud.update_video_status(db, video_id=video_id, status=status)
            db.commit()
            self.update_state(state=status.upper(), meta={'progress': progress, 'status': status})
            logger.info(f"视频 (ID: {video_id}) 状态已更新为 {status} 并已提交。 Celery 任务状态: {status.upper()}.")

        session_id_map = {} # 用于存储 track_id 到 person_uuid 的映射
        seen_track_ids = set() # 跟踪已经保存过特征的 track_id
        terminated_by_signal = False

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频文件: {video_path}")
            update_status_and_commit("failed")
            return

        # 获取视频的 UUID 用于构建保存路径
        db_video_info = backend.crud.get_video(db, video_id=video_id)
        if not db_video_info or not db_video_info.uuid:
            # 如果视频没有UUID，或者获取失败，则生成一个新的UUID并更新到数据库
            if not db_video_info: # 如果连视频记录都找不到，直接失败
                logger.error(f"视频 ID {video_id} 不存在，无法启动任务。")
                update_status_and_commit("failed")
                return
            else: # 视频记录存在但没有UUID，则生成并更新
                logger.warning(f"视频 {video_id} 没有关联的UUID，将生成并更新。")
                video_uuid = str(uuid.uuid4())
                db_video_info.uuid = video_uuid
                db.commit() # 提交UUID更新
                db.refresh(db_video_info)
                logger.info(f"视频 {video_id} 的UUID已更新为 {video_uuid}。")
        else:
            video_uuid = db_video_info.uuid

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            logger.warning(f"视频 (ID: {video_id}) 总帧数为0，可能已损坏。")
            update_status_and_commit("failed")
            return

        pbar = tqdm(total=total_frames, desc=f"后台提取 {os.path.basename(video_path)}", unit="frame")
        last_progress_reported = -1
        frame_count = 0 # 初始化帧计数器
        
        # frames_processed_since_last_commit = 0 # 移除此行
        # persons_to_add = [] # 移除此行

        while cap.isOpened():
            # 检查终止信号
            video_status_check = backend.crud.get_video(db, video_id=video_id)
            if not video_status_check:
                logger.warning(f"视频 (ID: {video_id}) 在处理中被删除，终止任务。")
                terminated_by_signal = True
                break
            if video_status_check.status == "terminated":
                logger.info(f"检测到终止信号，停止处理视频 (ID: {video_id})。")
                terminated_by_signal = True
                break
            
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1 # 增加帧计数
            # 每处理 500 帧或达到总帧数的 1% 时记录一次
            if frame_count % 500 == 0 or (total_frames > 0 and frame_count % (total_frames // 100) == 0 and total_frames // 100 > 0):
                progress = int((frame_count / total_frames) * 100)
                logger.info(f"视频 (ID: {video_id}): 已处理 {frame_count} / {total_frames} 帧. (Progress: {progress}%)")

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
                            # 新增：传递人脸识别和步态识别模型实例
                            face_detection_model=face_detection_model_instance,
                            face_recognition_session=face_recognition_session_instance,
                            gait_recognition_session=gait_recognition_session_instance, # 如果为None，则传递None
                            track_id=track_id # 确保 track_id 也被传递
                        )

                        # 在创建 real_time_match_alert 之前，立即创建并提交 person 对象
                        if person_obj:
                            created_person = backend.crud.create_person(db, person=person_obj)
                            db.commit() # 立即提交人物记录
                            db.refresh(created_person) # 刷新以获取完整的对象，包括ID
                            ml_logic.add_person_feature_to_faiss(created_person.uuid, np.array(json.loads(created_person.feature_vector), dtype=np.float32)) # 将新人物添加到 Faiss 索引
                            logger.info(f"视频 (ID: {video_id}) 新人物 {created_person.uuid} (Track ID: {track_id}) 已立即提交到数据库并添加到 Faiss 索引。")
                        else:
                            logger.warning(f"视频 (ID: {video_id}) 未能成功处理检测到的人物 (Track ID: {track_id})，跳过。")
                            continue

                        # 新增：实时比对逻辑
                        video_owner = backend.crud.get_user(db, user_id=video.owner_id) # 获取视频所有者
                        followed_persons_list = []
                        if video_owner:
                            followed_persons_list = backend.crud.get_followed_persons_by_user(db, user_id=video_owner.id, skip=0, limit=settings.REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS)
                        # 过滤掉未开启实时比对的关注人物
                        followed_persons_list = [fp for fp in followed_persons_list if fp.individual and fp.individual.is_realtime_comparison_enabled]

                        followed_persons_count = len(followed_persons_list)

                        should_perform_realtime_comparison = (
                            settings.REALTIME_COMPARISON_THRESHOLD > 0 and
                            followed_persons_count > 0
                        )

                        if should_perform_realtime_comparison:
                            detected_feature_vector = json.loads(person_obj.feature_vector) if isinstance(person_obj.feature_vector, str) else person_obj.feature_vector
                            matched_followed_person = ml_logic.perform_realtime_comparison(db, detected_feature_vector, video_owner.id, followed_persons_list)

                            if matched_followed_person:
                                logger.info(f"【实时比对 (视频解析)】比对成功：检测到的人物与关注人员 {matched_followed_person['individual_name']} ({matched_followed_person['individual_uuid']}) 匹配，相似度：{matched_followed_person['similarity']:.2f}")
                                person_obj.individual_id = matched_followed_person['individual_id']
                                person_obj.is_verified = True
                                person_obj.verified_by_user_id = video_owner.id
                                person_obj.verification_date = datetime.now(pytz.timezone('Asia/Shanghai'))
                                person_obj.correction_details = f"Realtime comparison matched with followed person {matched_followed_person['individual_uuid']} (similarity: {matched_followed_person['similarity']:.2f})"
                                person_obj.correction_type_display = "已纠正（实时比对）"
                                
                                # 更新已创建的人物对象（在同一个事务中）
                                backend.crud.update_person(db, person_id=created_person.id, person_update_data=schemas.PersonUpdate(**person_obj.dict()))
                                db.commit() # 提交人物更新
                                db.refresh(created_person) # 刷新以获取更新后的对象

                                alert_data = schemas.RealtimeMatchAlert(
                                    person_uuid=created_person.uuid, # 使用已提交人物的 UUID
                                    matched_individual_id=matched_followed_person['individual_id'],
                                    matched_individual_uuid=matched_followed_person['individual_uuid'],
                                    matched_individual_name=matched_followed_person['individual_name'],
                                    similarity_score=matched_followed_person['similarity'],
                                    timestamp=datetime.now(pytz.timezone('Asia/Shanghai')),
                                    source_media_type="video", # 来源类型为视频
                                    source_media_uuid=video_uuid,
                                    user_id=video_owner.id,
                                    cropped_image_path=created_person.crop_image_path, # 新增
                                    full_frame_image_path=created_person.full_frame_image_path # 新增
                                )
                                crud_match.create_realtime_match_alert(db, alert_data)
                                db.commit() # 提交预警记录
                                logger.info(f"【实时比对 (视频解析)】已记录实时比对预警：人物 {created_person.uuid} 与关注人员 {matched_followed_person['individual_uuid']} (视频 {video_uuid}) 匹配。")
                            else:
                                logger.info("【实时比对 (视频解析)】比对未匹配到关注人员。")
                        else:
                            reason = []
                            if settings.REALTIME_COMPARISON_THRESHOLD <= 0:
                                reason.append(f"实时比对阈值 ({settings.REALTIME_COMPARISON_THRESHOLD}) 小于等于 0")
                            if followed_persons_count == 0:
                                reason.append("未关注任何人物")
                            logger.info(f"【实时比对 (视频解析)】条件不满足，未执行实时比对。原因: {', '.join(reason)}。")

                        # persons_to_add.append(created_person) # 移除此行
                        session_id_map[track_id] = created_person.uuid # 使用 created_person.uuid
                        seen_track_ids.add(track_id) # 将 track_id 添加到已保存集合
                        # logger.info(f"视频 (ID: {video_id}) 新人物 {person_obj.uuid} (Track ID: {track_id}) 已添加到待处理列表。") # 移除此行，因为已立即提交
                    except ValueError as ve:
                        logger.warning(f"处理视频人物时发生错误: {ve}，跳过此人物。")
                        db.rollback() # 在捕获 ValueError 时回滚当前事务
                        continue
            
            pbar.update(1)
            frames_processed_since_last_commit += 1
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            progress = int((current_frame / total_frames) * 100)

            if progress > last_progress_reported:
                backend.crud.update_video_progress(db, video_id=video_id, progress=progress)
                self.update_state(state='PROCESSING', meta={'progress': progress, 'status': 'processing'})
                last_progress_reported = progress
            
            # 移除批量提交逻辑，因为人物已经单独提交
            # if frames_processed_since_last_commit >= COMMIT_BATCH_SIZE:
            #     try:
            #         if persons_to_add: # 如果有待提交的人物
            #             db.commit() # 批量提交
            #             for p in persons_to_add:
            #                 ml_logic.add_person_feature_to_faiss(p.uuid, np.array(json.loads(p.feature_vector), dtype=np.float32)) # 将新人物添加到 Faiss 索引
            #                 db.refresh(p) # 刷新以获取完整的对象，包括ID
            #             persons_to_add = [] # 清空列表
            #             logger.info(f"视频 (ID: {video_id}) 批量提交 {COMMIT_BATCH_SIZE} 帧的数据。 当前已处理总帧数: {frame_count}。") # 更新日志，包含总帧数
            #         else:
            #             db.rollback() # 如果没有待提交的人物，回滚
            #     except Exception as commit_e:
            #         db.rollback()
            #         logger.error(f"视频 (ID: {video_id}) 批量提交时发生错误: {commit_e}", exc_info=True)
            #     frames_processed_since_last_commit = 0
        
        pbar.close()
        cap.release()

        # 移除循环结束后提交剩余人物的逻辑
        # if persons_to_add:
        #     try:
        #         db.commit()
        #         for p in persons_to_add:
        #             ml_logic.add_person_feature_to_faiss(p.uuid, np.array(json.loads(p.feature_vector), dtype=np.float32)) # 将新人物添加到 Faiss 索引
        #             db.refresh(p)
        #         logger.info(f"视频 (ID: {video_id}) 循环结束后提交剩余 {len(persons_to_add)} 个人物数据。")
        #     except Exception as commit_e:
        #         db.rollback()
        #         logger.error(f"视频 (ID: {video_id}) 循环结束后提交剩余数据时发生错误: {commit_e}", exc_info=True)

        final_status = "terminated" if terminated_by_signal else "completed"
        backend.crud.update_video_status(db, video_id=video_id, status=final_status)
        db.commit() # 最终状态和最后进度提交
        self.update_state(state=final_status.upper(), meta={'progress': 100, 'status': final_status})


    except Exception as e:
        logger.error(f"处理视频 {video_path} (ID: {video_id}) 时发生未捕获的严重错误: {e}", exc_info=True)
        if db.is_active:
            try:
                db.rollback() # 在异常情况下回滚所有未提交的更改
                logger.error(f"Attempting to set video {video_id} status to 'failed' due to unhandled exception.") # 更新日志信息
                backend.crud.update_video_status(db, video_id=video_id, status="failed")
                db.commit() # 提交失败状态
                self.update_state(state="FAILED", meta={'progress': progress, 'status': 'failed'})
            except Exception as commit_e:
                logger.error(f"视频 (ID: {video_id}) 异常处理中更新状态失败: {commit_e}", exc_info=True)
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
    logger.info(f"Celery 任务开始：再训练 Re-ID 模型 (版本: {model_version}), 指定人物UUIDs: {person_uuids}")
    # self.update_state(state='STARTED', meta={'progress': 0, 'status': 'retraining'})
    db = SessionLocal()
    settings.reload_from_db(db) # 强制从数据库重新加载 settings
    try:
        # 1. 根据 person_uuids 参数获取人物数据
        if person_uuids:
            persons_for_retrain = backend.crud.get_persons_by_uuids(db, person_uuids) # 使用传入的 UUIDs
            logger.info(f"使用指定的 {len(person_uuids)} 个人物 UUID 进行再训练。实际找到 {len(persons_for_retrain)} 个人物。")
        else:
            # 如果没有指定 UUIDs，则获取所有标记为 'marked_for_retrain' 的人物数据
            persons_for_retrain = backend.crud.get_persons_marked_for_retrain(db) # 调用新函数
            logger.info(f"没有指定人物 UUID，将使用所有标记为再训练的 {len(persons_for_retrain)} 个人物数据。")
        
        if not persons_for_retrain:
            logger.info("没有有效的人物数据用于再训练，跳过模型再训练。")
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
            backend.crud.update_persons_trained_status(db, person_uuids_to_update) # 调用新的批量更新函数

            logger.info(f"Re-ID 模型再训练成功，新模型路径: {new_model_path}。已更新所有参与再训练的人物标记和训练状态。")
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'completed', 'new_model': new_model_path})
            return True
        else:
            logger.warning("Re-ID 模型再训练未完成或失败。")
            self.update_state(state='FAILED', meta={'progress': 100, 'status': 'failed'})
            return False

    except Exception as e:
        logger.error(f"Re-ID 模型再训练任务失败: {e}", exc_info=True)
        self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed'})
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
        logger.error(f"视频流 (ID: {stream_id}): 初始化 Faiss 索引失败。任务无法继续。错误: {e}", exc_info=True)
        backend.crud.update_stream_status(db, stream_id=stream_id, status="failed")
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

    # 确定保存视频和裁剪图的路径
    # 这些路径的生成现在由 media_processing.generate_media_paths 负责，这里只需确保基目录存在
    # save_dir = os.path.join(settings.SAVED_STREAMS_DIR, stream_uuid) # 移除
    # os.makedirs(save_dir, exist_ok=True) # 移除
    # output_video_filename = f"{stream_uuid}.mp4" # 移除
    # absolute_output_video_path = os.path.join(save_dir, output_video_filename) # 移除
    # relative_output_video_path = os.path.join(stream_uuid, output_video_filename).replace(os.sep, '/') # 移除

    # 为人物裁剪图和完整帧创建目录
    # stream_crops_dir = os.path.join(settings.DATABASE_CROPS_IMAGE_ANALYSIS_DIR, "stream", stream_uuid) # 移除
    # os.makedirs(stream_crops_dir, exist_ok=True) # 移除

    # stream_full_frames_dir = os.path.join(settings.DATABASE_FULL_FRAMES_IMAGE_ANALYSIS_DIR, "stream", stream_uuid) # 移除
    # os.makedirs(stream_full_frames_dir, exist_ok=True) # 移除

    try:
        logger.info(f"Celery 任务开始：处理视频流 {stream_url} (Stream ID: {stream_id})")

        # 获取流的 UUID 用于构建保存路径
        db_stream_info = backend.crud.get_stream(db, stream_id=stream_id)
        if db_stream_info:
            stream_uuid = db_stream_info.stream_uuid
        else:
            logger.error(f"Stream ID {stream_id} 不存在，无法启动任务。")
            backend.crud.update_stream_status(db, stream_id=stream_id, status="failed")
            db.commit()
            return
        
        # 确保目录在任务开始时由 media_processing.generate_media_paths 创建
        # 不需要在这里显式创建，因为 generate_media_paths 内部会创建
        # 只需要确保 stream_uuid 已经获取到
        if not stream_uuid:
            logger.error(f"视频流 (ID: {stream_id}) 的 UUID 未获取到。")
            backend.crud.update_stream_status(db, stream_id=stream_id, status="failed")
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
                logger.warning(f"视频流 BoT-SORT 跟踪器 Re-ID 模型路径 '{settings.ACTIVE_REID_MODEL_PATH}' 无效或文件不存在，将禁用 BoT-SORT 内部 Re-ID。")

            tracker_args = TrackerArgs(
                with_reid=False, # 强制禁用 BoT-SORT 内部 Re-ID
                model="" # 确保不传递任何模型路径给 BoT-SORT 的内部 Re-ID
            )
            if current_detection_model is None or current_reid_session is None or face_detection_model_instance is None or face_recognition_session_instance is None:
                raise Exception("核心 ML 模型或 Tracker 未能成功加载。")
            logger.info(f"视频流 {stream_id}: AI模型加载成功。")
        except Exception as e:
            logger.error(f"视频流 {stream_id}: 加载AI模型失败: {e}", exc_info=True)
            backend.crud.update_stream_status(db, stream_id=stream_id, status="failed")
            db.commit()
            return

        # 确保 Faiss 索引在任务开始时被初始化 (移到这里)
        try:
            ml_logic.initialize_faiss_index(db) # 确保 Faiss 索引在任务开始时初始化
        except Exception as e:
            logger.error(f"视频流 (ID: {stream_id}): 初始化 Faiss 索引失败。任务无法继续。错误: {e}", exc_info=True)
            backend.crud.update_stream_status(db, stream_id=stream_id, status="failed")
            db.commit()
            self.update_state(state="FAILED", meta={'progress': 0, 'status': 'failed'})
            return

        # 再次检查模型加载状态，以防 Faiss 初始化失败后模型状态被改变 (此段不再需要，因为已直接加载)
        # try:
        #     load_models_globally()
        # except Exception as e:
        #     logger.error(f"视频流 (ID: {stream_id}): ML 模型或 Tracker 未加载成功，任务无法继续。错误: {e}", exc_info=True)
        #     update_stream_status(db, stream_id=stream_id, status="failed")
        #     db.commit()
        #     return

        # 新增：模型加载成功后，更新流状态为 'processing'
        logger.info(f"Attempting to set stream {stream_id} status to 'processing' after model load.")
        backend.crud.update_stream_status(db, stream_id=stream_id, status="processing")
        db.commit()
        logger.info(f"Stream {stream_id} status updated to 'processing' successfully.")

        seen_track_ids = set() # 跟踪已经保存过特征的人物ID

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            logger.error(f"无法打开视频流: {stream_url}")
            backend.crud.update_stream_status(db, stream_id=stream_id, status="failed")
            db.commit()
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25 # 如果获取不到FPS，默认为25

        # 确定保存视频的路径和文件名
        stream_obj = backend.crud.get_stream(db, stream_id=stream_id)
        if not stream_obj or not stream_obj.stream_uuid:
            logger.error(f"视频流 {stream_id}: 无法获取流的UUID，终止任务。")
            backend.crud.update_stream_status(db, stream_id=stream_id, status="failed")
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
            logger.error(f"视频流 {stream_id}: 无法初始化视频写入器到 {absolute_output_video_path}。请检查路径、权限或编解码器。")
            backend.crud.update_stream_status(db, stream_id=stream_id, status="failed")
            db.commit()
            return # 无法写入视频，提前退出任务

        logger.info(f"视频流 {stream_id}: 初始化视频保存到 {absolute_output_video_path}")

        COMMIT_BATCH_SIZE = 100 # 每100帧提交一次数据库
        frames_processed_since_last_commit = 0
        persons_to_add_live = [] # 收集待添加的人物对象 (直播流专用)
        frame_count = 0 # 初始化帧计数器

        while cap.isOpened():
            # 检查终止信号
            stream_status_check = backend.crud.get_stream(db, stream_id=stream_id)
            if not stream_status_check:
                logger.warning(f"视频流 (ID: {stream_id}) 在处理中被删除，终止任务。")
                terminated_by_signal = True
                break
            if stream_status_check.status == "terminated" or stream_status_check.status == "stopped": # 检查 terminated 或 stopped
                logger.info(f"检测到终止信号，停止处理视频流 (ID: {stream_id})。状态: {stream_status_check.status}")
                terminated_by_signal = True
                break
            
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"视频流 {stream_id} 读取完毕或断开连接。")
                break

            # 修复 NameError：确保 annotated_frame 总是被定义
            annotated_frame = frame.copy()

            frame_count += 1 # 增加帧计数
            # 每处理 500 帧时记录一次
            if frame_count % 500 == 0:
                logger.debug(f"视频流 (ID: {stream_id}): 已处理 {frame_count} 帧。") # 将 INFO 改为 DEBUG

            # ... (此处省略检测和跟踪逻辑，因为它在原始代码中存在但未在视图中完全显示)
            # 假设检测和跟踪逻辑会更新 annotated_frame
            results_live = current_detection_model.track(source=frame, persist=True, tracker="botsort.yaml", verbose=False)
            if results_live and results_live[0].boxes.id is not None:
                annotated_frame = results_live[0].plot()
                
                # 直接从 track() 的结果中获取跟踪信息
                for box in results_live[0].boxes:
                    track_id = int(box.id.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence_score = float(box.conf.item())

                    if track_id not in seen_track_ids:
                        person_bbox = (x1, y1, x2, y2)
                        try:
                            person_create_data = media_processing.process_detected_person_data(
                                db=db, frame=frame, person_bbox=person_bbox,
                                confidence_score=confidence_score, yolo_results_obj=results_live[0],
                                media_type="stream", media_uuid=stream_uuid, person_uuid=str(uuid.uuid4()),
                                stream_id_int=stream_id,
                                gait_recognition_session=gait_recognition_session_instance, # 如果为None，则传递None
                                tracklet_gait_buffer=tracklet_gait_buffer, track_id=track_id,
                                face_detection_model=face_detection_model_instance, # 新增
                                face_recognition_session=face_recognition_session_instance # 新增
                            )
                            if person_create_data:
                                created_person = backend.crud.create_person(db, person=person_create_data)
                                db.commit() # 立即提交人物记录
                                db.refresh(created_person) # 刷新以获取完整的对象，包括ID
                                ml_logic.add_person_feature_to_faiss(created_person.uuid, np.array(json.loads(created_person.feature_vector), dtype=np.float32)) # 将新人物添加到 Faiss 索引
                                logger.info(f"视频流 (ID: {stream_id}) 新人物 {created_person.uuid} (Track ID: {track_id}) 已立即提交到数据库并添加到 Faiss 索引。")

                                # 新增：实时比对逻辑
                                stream_owner = backend.crud.get_user(db, user_id=stream_obj.owner_id) # 获取视频流所有者
                                followed_persons_list = []
                                if stream_owner:
                                    followed_persons_list = backend.crud.get_followed_persons_by_user(db, user_id=stream_owner.id, skip=0, limit=settings.REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS)
                                # 过滤掉未开启实时比对的关注人物
                                followed_persons_list = [fp for fp in followed_persons_list if fp.individual and fp.individual.is_realtime_comparison_enabled]

                                followed_persons_count = len(followed_persons_list)

                                should_perform_realtime_comparison = (
                                    settings.REALTIME_COMPARISON_THRESHOLD > 0 and
                                    followed_persons_count > 0
                                )

                                if should_perform_realtime_comparison:
                                    detected_feature_vector = json.loads(created_person.feature_vector) if isinstance(created_person.feature_vector, str) else created_person.feature_vector
                                    matched_followed_person = ml_logic.perform_realtime_comparison(db, detected_feature_vector, stream_owner.id, followed_persons_list)

                                    if matched_followed_person:
                                        logger.info(f"【实时比对 (视频流解析)】比对成功：检测到的人物与关注人员 {matched_followed_person['individual_name']} ({matched_followed_person['individual_uuid']}) 匹配，相似度：{matched_followed_person['similarity']:.2f}")
                                        created_person.individual_id = matched_followed_person['individual_id']
                                        created_person.is_verified = True
                                        created_person.verified_by_user_id = stream_owner.id
                                        created_person.verification_date = datetime.now(pytz.timezone('Asia/Shanghai'))
                                        created_person.correction_details = f"Realtime comparison matched with followed person {matched_followed_person['individual_uuid']} (similarity: {matched_followed_person['similarity']:.2f})"
                                        created_person.correction_type_display = "已纠正（实时比对）"
                                        
                                        # 更新已创建的人物对象（在同一个事务中）
                                        person_update_data = schemas.PersonUpdate(
                                            individual_id=created_person.individual_id,
                                            is_verified=created_person.is_verified,
                                            verified_by_user_id=created_person.verified_by_user_id,
                                            verification_date=created_person.verification_date,
                                            correction_details=created_person.correction_details,
                                            correction_type_display=created_person.correction_type_display
                                        )
                                        backend.crud.update_person(db, person_id=created_person.id, person_update_data=person_update_data)
                                        db.commit() # 提交人物更新
                                        db.refresh(created_person) # 刷新以获取更新后的对象

                                        alert_data = schemas.RealtimeMatchAlert(
                                            person_uuid=created_person.uuid, # 使用已提交人物的 UUID
                                            matched_individual_id=matched_followed_person['individual_id'],
                                            matched_individual_uuid=matched_followed_person['individual_uuid'],
                                            matched_individual_name=matched_followed_person['individual_name'],
                                            similarity_score=matched_followed_person['similarity'],
                                            timestamp=datetime.now(pytz.timezone('Asia/Shanghai')),
                                            source_media_type="stream", # 来源类型为视频流
                                            source_media_uuid=stream_uuid,
                                            user_id=stream_owner.id,
                                            cropped_image_path=created_person.crop_image_path, # 新增
                                            full_frame_image_path=created_person.full_frame_image_path # 新增
                                        )
                                        crud_match.create_realtime_match_alert(db, alert_data)
                                        db.commit() # 提交预警记录
                                        logger.info(f"【实时比对 (视频流解析)】已记录实时比对预警：人物 {created_person.uuid} 与关注人员 {matched_followed_person['individual_uuid']} (视频流 {stream_uuid}) 匹配。")
                                    else:
                                        logger.info("【实时比对 (视频流解析)】比对未匹配到关注人员。")
                                else:
                                    reason = []
                                    if settings.REALTIME_COMPARISON_THRESHOLD <= 0:
                                        reason.append(f"实时比对阈值 ({settings.REALTIME_COMPARISON_THRESHOLD}) 小于等于 0")
                                    if followed_persons_count == 0:
                                        reason.append("未关注任何人物")
                                    logger.info(f"【实时比对 (视频流解析)】条件不满足，未执行实时比对。原因: {', '.join(reason)}。")
                                
                                seen_track_ids.add(track_id) # 将 track_id 添加到已保存集合
                            else:
                                logger.warning(f"视频流 (ID: {stream_id}) 未能成功处理检测到的人物 (Track ID: {track_id})，跳过。")
                                continue
                        except ValueError as ve:
                            logger.warning(f"处理视频流人物时发生错误: {ve}，跳过此人物。")
                            db.rollback() # 在捕获 ValueError 时回滚当前事务
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
                        for person_data in persons_to_add_live:
                            created_person = backend.crud.create_person(db, person=person_data)
                            if created_person:
                                ml_logic.add_person_feature_to_faiss(created_person.uuid, np.array(json.loads(created_person.feature_vector), dtype=np.float32))
                        db.commit()
                        persons_to_add_live = []
                except Exception as commit_e:
                    db.rollback()
                    logger.error(f"视频流 (ID: {stream_id}) 批量提交时发生错误: {commit_e}", exc_info=True)
                frames_processed_since_last_commit = 0

        cap.release()
        out.release()
        logger.info(f"视频流 {stream_id}: 视频捕获和写入完成。")

        try:
            ffprobe_cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,profile,width,height,avg_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                absolute_output_video_path
            ]
            logger.debug(f"视频流 {stream_id} 正在运行 ffprobe 命令: {ffprobe_cmd}") # 记录完整的ffprobe命令
            ffprobe_output = subprocess.check_output(ffprobe_cmd, stderr=subprocess.STDOUT, text=True)
            logger.info(f"视频流 {stream_id} 输出视频信息:\n{ffprobe_output}")
        except subprocess.CalledProcessError as e:
            logger.error(f"视频流 {stream_id} 运行 ffprobe 失败 (返回码: {e.returncode}): {e.output}", exc_info=True)
        except FileNotFoundError:
            logger.error(f"ffprobe 未找到。请确保已安装 FFmpeg 并将其添加到 PATH 中。")
        except Exception as e:
            logger.error(f"视频流 {stream_id} 获取视频信息时发生未知错误: {e}", exc_info=True)

        final_status = "terminated" if terminated_by_signal else "completed"
        logger.info(f"Attempting to set stream {stream_id} final status to '{final_status}'.")
        backend.crud.update_stream_status(db, stream_id=stream_id, status=final_status)
        backend.crud.update_stream_output_video_path(db, stream_id=stream_id, output_video_path=relative_output_video_path)
        db.commit() # 确保所有更新都已提交
        logger.info(f"Stream {stream_id} final status updated to '{final_status}' and output path updated successfully.")
        self.update_state(state=final_status.upper(), meta={'progress': 100, 'status': final_status})

        if persons_to_add_live:
            try:
                for person_data in persons_to_add_live:
                    created_person = backend.crud.create_person(db, person=person_data) # 使用 crud.create_person
                    ml_logic.add_person_feature_to_faiss(created_person.uuid, np.array(json.loads(created_person.feature_vector), dtype=np.float32))
                    db.refresh(created_person)
                db.commit()
                logger.info(f"视频流 (ID: {stream_id}) 循环结束后提交剩余 {len(persons_to_add_live)} 个人物数据。")
            except Exception as commit_e:
                db.rollback()
                logger.error(f"视频流 (ID: {stream_id}) 循环结束后提交剩余数据时发生错误: {commit_e}", exc_info=True)

        logger.info(f"视频流 {stream_id} 的处理任务已完成。最终状态: {final_status}。")

    except Exception as e:
        logger.error(f"处理视频流 {stream_id} 时发生未捕获的严重错误: {e}", exc_info=True)
        if db.is_active:
            try:
                db.rollback() # Rollback any pending changes
                logger.error(f"Attempting to set stream {stream_id} status to 'failed' due to unhandled exception.")
                backend.crud.update_stream_status(db, stream_id=stream_id, status="failed")
                db.commit()
                logger.error(f"Stream {stream_id} status updated to 'failed' due to unhandled exception.")
            except Exception as commit_e:
                logger.error(f"Failed to update stream {stream_id} status to 'failed' in exception handler: {commit_e}", exc_info=True)
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
    logger.info(f"Celery 任务开始：分析图片 {original_image_filename} (UUID: {original_image_uuid})，用户 ID: {current_user_id}")
    logger.info("analyze_image_task: Running with updated logic for real-time comparison.") # 新增日志
    self.update_state(state='STARTED', meta={'progress': 0, 'status': 'started'})
    db = SessionLocal()
    try:
        # 获取用户对象
        current_user = backend.crud.get_user(db, user_id=current_user_id)
        if not current_user:
            logger.error(f"用户 ID {current_user_id} 不存在，无法执行图片分析任务。")
            self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed'})
            return None

        # 确保 Faiss 索引在任务开始时被初始化
        try:
            ml_logic.initialize_faiss_index(db)
        except Exception as e:
            logger.error(f"图片 (UUID: {original_image_uuid}): 初始化 Faiss 索引失败。任务无法继续。错误: {e}", exc_info=True)
            self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed'})
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
            image_entry = backend.crud.get_image(db, image_id=created_image_id)
            person_count = image_entry.person_count if image_entry else 0
            
            logger.info(f"图片 (UUID: {original_image_uuid}) 分析成功，检测到 {person_count} 个人物。")
            self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'completed', 'image_id': created_image_id, 'image_uuid': original_image_uuid, 'person_count': person_count})
            return created_image_id
        else:
            logger.error(f"图片 (UUID: {original_image_uuid}) 分析失败，未返回图片 ID。")
            self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed'})
            return None

    except Exception as e:
        logger.error(f"图片分析任务 (UUID: {original_image_uuid}) 发生未捕获的错误: {e}", exc_info=True)
        self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed'})
        return None
    finally:
        db.close()


@celery_app.task(bind=True)
def run_global_search_for_followed_person(self, individual_id: int, user_id: int, is_initial_search: bool = False):
    """
    Celery 任务：为新关注的人物执行一次全局搜索比对。
    将比对结果中置信度90%以上的图片显示在关注人员历史轨迹页面。
    """
    logger.info(f"Celery 任务开始：为人物 {individual_id} (用户: {user_id}) 执行全局搜索比对。是否初始搜索: {is_initial_search}")
    self.update_state(state='STARTED', meta={'progress': 0, 'status': 'started'})
    db = SessionLocal()
    try:
        # 确保 Faiss 索引在任务开始时被初始化
        ml_logic.initialize_faiss_index(db)
        reid_session = ml_logic.get_reid_session(db) # 获取 Re-ID session
        if reid_session is None:
            logger.error(f"Re-ID 模型加载失败，全局搜索比对任务无法继续。")
            self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed'})
            return

        # 1. 获取关注人物的注册图片
        followed_person_individual = backend.crud.get_individual(db, individual_id)
        if not followed_person_individual:
            logger.error(f"无法找到 Individual ID {individual_id} 对应的关注人物，任务中止。")
            self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed'})
            return
            
        followed_person_individual_id = followed_person_individual.id # 获取被关注人物的 Individual ID
        
        enrollment_images = backend.crud.get_enrollment_images_by_individual_id(db, individual_id)
        if not enrollment_images:
            logger.warning(f"人物 {individual_id} 没有注册图片，跳过全局搜索比对。")
            self.update_state(state='SKIPPED', meta={'progress': 100, 'status': 'skipped', 'message': 'No enrollment images found.'})
            return
        
        # 2. 遍历每张注册图片，进行全局搜索
        all_global_search_results = []
        total_images = len(enrollment_images)
        for i, image_path in enumerate(enrollment_images):
            progress = int(((i + 1) / total_images) * 100 * 0.5) # 假设搜索占50%进度
            self.update_state(state='PROCESSING', meta={'progress': progress, 'status': f'processing image {i+1}/{total_images}'})
            
            try:
                # 提取特征向量
                img_to_process = cv2.imread(os.path.join(settings.BASE_DIR, image_path.lstrip('/'))) # 移除开头的斜杠
                if img_to_process is None:
                    logger.warning(f"无法读取图片 {image_path}，跳过。")
                    continue

                feature_vector = ml_logic.get_person_feature(reid_session, img_to_process)
                if feature_vector is None:
                    logger.warning(f"无法从图片 {image_path} 提取特征，跳过。")
                    continue

                # 使用 Faiss 索引进行搜索
                # 确保 _faiss_index_instance 已被初始化
                if ml_logic._faiss_index_instance is None:
                    logger.error("Faiss 索引未初始化，无法进行全局搜索。")
                    self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed'})
                    return
                
                query_vector_2d = np.expand_dims(feature_vector, axis=0)
                k_value = min(ml_logic._faiss_index_instance.ntotal, settings.FAISS_SEARCH_K)
                if k_value == 0: # 如果索引中没有数据，则k_value可能为0
                    logger.warning("Faiss 索引中没有人物特征，跳过搜索。")
                    continue

                distances, indices = ml_logic._faiss_index_instance.search(query_vector_2d, k_value)

                if indices is None or len(indices) == 0:
                    logger.info(f"图片 {image_path} 未找到匹配结果。")
                    continue
                
                # 获取 Faiss 索引中存储的 person_uuid 列表
                # 需要从 ml_logic 中获取或直接访问，这里假设 ml_logic.faiss_index_to_uuid 存在
                # 注意：Faiss 索引中的 ID 是整数，需要映射回 person_uuid
                indexed_person_uuids = ml_logic._faiss_person_uuids # 获取 Faiss 索引中所有 UUID

                # 遍历搜索结果
                for i in range(len(indices[0])):
                    idx = indices[0][i]
                    distance = distances[0][i]

                    if idx == -1: # 无效索引，跳过
                        continue

                    # 从全局 UUID 列表中获取匹配的人物 UUID
                    matched_person_uuid = ml_logic._faiss_person_uuids[idx]
                    confidence = 1 - distance # 将距离转换为置信度 (0-1之间)

                    # 过滤置信度90%以上的图片
                    if confidence >= settings.HUMAN_REVIEW_CONFIDENCE_THRESHOLD: # 使用配置的阈值
                        # 获取匹配人物的详细信息 (包括 crop_image_path)
                        matched_person = backend.crud.get_person_by_uuid(db, matched_person_uuid)
                        if matched_person:
                            # 排除与查询人物属于同一个 individual_id 的结果 (即排除用户自己上传的图片)
                            if matched_person.individual_id == followed_person_individual_id:
                                logger.debug(f"排除人物 {matched_person.uuid}，因为它与查询人物属于同一 Individual ({followed_person_individual_id})。")
                                continue
                            all_global_search_results.append({
                                "followed_person_id": individual_id, # 这里可能需要改为 FollowedPerson 实例的 ID
                                "matched_person_uuid": matched_person.uuid,
                                "matched_person_id": matched_person.id,
                                "matched_image_path": matched_person.crop_image_path, # 通常匹配的是人物的裁剪图
                                "confidence": confidence,
                                "search_time": datetime.now(pytz.utc), # 记录搜索时间
                                "user_id": user_id, # 添加 user_id
                                "is_initial_search": is_initial_search # 添加 is_initial_search
                            })
            except Exception as e:
                logger.error(f"处理图片 {image_path} 时发生错误: {e}", exc_info=True)
                continue

        # 3. 将结果存储到数据库
        logger.info(f"为人物 {individual_id} (用户: {user_id}) 发现 {len(all_global_search_results)} 个全局搜索比对结果。")
        if all_global_search_results:
            crud_match.create_multiple_global_search_results(db, all_global_search_results)

        # 4. 更新 Celery 任务状态
        self.update_state(state='SUCCESS', meta={'progress': 100, 'status': 'completed', 'results_count': len(all_global_search_results)})

    except Exception as e:
        logger.error(f"为人物 {individual_id} (用户: {user_id}) 执行全局搜索比对任务失败: {e}", exc_info=True)
        self.update_state(state='FAILED', meta={'progress': 0, 'status': 'failed'})
    finally:
        db.close()
