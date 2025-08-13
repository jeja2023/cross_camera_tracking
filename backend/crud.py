import json
import numpy as np
import os
from sqlalchemy.orm import Session, joinedload
from .database_conn import Person, Video, User, Log, Stream, Image, CorrectionLog, SystemConfig, Individual, FollowedPerson, RealtimeMatchAlert # 导入 RealtimeMatchAlert
from . import auth
from . import schemas
import logging
from datetime import datetime
from typing import Optional, List, Dict, Union
import pytz
from backend.config import settings # 导入 settings
import shutil
import uuid
from sqlalchemy import or_
import backend.crud_match as crud_match # 新增导入 crud_match
from sqlalchemy import desc # 新增导入 desc
from sqlalchemy import func # 新增导入 func

# --- 裁剪图片配置 ---
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # 移除冗余行
CROP_DIR = settings.DATABASE_CROPS_DIR # 使用统一配置的裁剪图片目录

logger = logging.getLogger(__name__)

def _prepare_person_for_schema(person_obj: Person, db: Session, current_user_id: Optional[int] = None) -> schemas.Person:
    """
    Helper function to prepare a SQLAlchemy Person object for Pydantic serialization,
    populating related fields like video_uuid, stream_uuid, and upload_image_uuid/filename.
    Adds is_followed status based on current_user_id.
    """
    person_dict = {
        "id": person_obj.id,
        "uuid": person_obj.uuid,
        "feature_vector": json.loads(person_obj.feature_vector) if isinstance(person_obj.feature_vector, str) else person_obj.feature_vector,
        "crop_image_path": person_obj.crop_image_path.replace(os.sep, '/') if person_obj.crop_image_path is not None else '', # 直接使用数据库路径，确保斜杠方向正确
        "full_frame_image_path": person_obj.full_frame_image_path.replace(os.sep, '/') if person_obj.full_frame_image_path else None, # 直接使用数据库路径，确保斜杠方向正确
        "created_at": person_obj.created_at,
        "video_id": person_obj.video_id,
        "stream_id": person_obj.stream_id,
        "upload_image_id": person_obj.image_id, # For image analysis
        "video_uuid": None,
        "video_name": None,
        "stream_uuid": None,
        "stream_name": None,
        "upload_image_uuid": None,
        "upload_image_filename": None,
        "is_verified": person_obj.is_verified, # 新增
        "verified_by_user_id": person_obj.verified_by_user_id, # 新增
        "verification_date": person_obj.verification_date, # 新增
        "correction_details": person_obj.correction_details, # 新增
        "marked_for_retrain": person_obj.marked_for_retrain, # 新增
        "confidence_score": person_obj.confidence_score, # 新增
        "pose_keypoints": json.loads(person_obj.pose_keypoints) if isinstance(person_obj.pose_keypoints, str) and person_obj.pose_keypoints else None,
        "face_image_path": person_obj.face_image_path.replace(os.sep, '/') if person_obj.face_image_path else None, # 直接使用数据库路径
        "face_feature_vector": json.loads(person_obj.face_feature_vector) if isinstance(person_obj.face_feature_vector, str) and person_obj.face_feature_vector else None,
        "face_id": person_obj.face_id,
        "clothing_attributes": json.loads(person_obj.clothing_attributes) if isinstance(person_obj.clothing_attributes, str) and person_obj.clothing_attributes else None,
        "gait_feature_vector": json.loads(person_obj.gait_feature_vector) if isinstance(person_obj.gait_feature_vector, str) and person_obj.gait_feature_vector else None,
        "gait_image_path": person_obj.gait_image_path.replace(os.sep, '/') if person_obj.gait_image_path else None,
        # New Individual fields
        "individual_id": person_obj.individual_id,
        "individual_uuid": None,
        "individual_name": None,
        "individual_id_card": None,
        "name": None, # Will be populated from individual
        "id_card": None, # Will be populated from individual
        "is_trained": person_obj.is_trained, # 新增：添加 is_trained 字段
        "correction_type_display": person_obj.correction_type_display, # 新增：直接从 ORM 对象获取
        "uploaded_by_username": None, # 新增：初始化为 None
        "is_followed": False, # 默认值，将在下方根据实际情况更新
    }

    logger.debug(f"DEBUG: _prepare_person_for_schema - Processing person UUID: {person_obj.uuid}")
    logger.debug(f"DEBUG: _prepare_person_for_schema - person_obj.image_id: {person_obj.image_id}")
    logger.debug(f"DEBUG: _prepare_person_for_schema - person_obj.image: {person_obj.image}")
    if person_obj.image:
        logger.debug(f"DEBUG: _prepare_person_for_schema - person_obj.image.owner: {person_obj.image.owner}")
        if person_obj.image.owner:
            logger.debug(f"DEBUG: _prepare_person_for_schema - person_obj.image.owner.username: {person_obj.image.owner.username}")

    # Populate name and id_card from Individual
    if person_obj.individual:
        person_dict['name'] = person_obj.individual.name
        person_dict['id_card'] = person_obj.individual.id_card
        person_dict['individual_uuid'] = person_obj.individual.uuid
        person_dict['individual_name'] = person_obj.individual.name
        person_dict['individual_id_card'] = person_obj.individual.id_card

        # 检查人物是否被当前用户关注
        if current_user_id is not None:
            is_followed = crud_match.check_is_followed(db, person_obj.individual_id, current_user_id)
            person_dict['is_followed'] = is_followed

    # Populate uploaded_by_username directly from ORM relationships
    if person_obj.video and person_obj.video.owner:
        person_dict['uploaded_by_username'] = person_obj.video.owner.username
    elif person_obj.stream and person_obj.stream.owner:
        person_dict['uploaded_by_username'] = person_obj.stream.owner.username
    elif person_obj.image and person_obj.image.owner:
        person_dict['uploaded_by_username'] = person_obj.image.owner.username

    # Populate correction_type_display from the latest CorrectionLog
    # 移除此段逻辑，因为 correction_type_display 已经直接从 person_obj 中获取
    # if person_obj.correction_logs:
    #     latest_correction = max(person_obj.correction_logs, key=lambda log: log.correction_date)
    #     if latest_correction:
    #         correction_type = latest_correction.correction_type
    #         if correction_type == "merge":
    #             person_dict['correction_type_display'] = "已纠正 (合并)"
    #         elif correction_type == "misdetection":
    #             person_dict['correction_type_display'] = "已纠正 (误检)"
    #         elif correction_type == "relabel":
    #             person_dict['correction_type_display'] = "已纠正 (重新标注)"
    #         elif correction_type == "other":
    #             person_dict['correction_type_display'] = "已纠正 (其他)"
    #         else:
    #             person_dict['correction_type_display'] = f"已纠正 ({correction_type})"

    # 关联视频、流或上传图片信息
    if person_obj.video:
        person_dict['video_uuid'] = person_obj.video.uuid
        person_dict['video_name'] = person_obj.video.filename
    elif person_obj.stream:
        person_dict['stream_uuid'] = person_obj.stream.stream_uuid
        person_dict['stream_name'] = person_obj.stream.name
    elif person_obj.image:
        person_dict['upload_image_uuid'] = person_obj.image.uuid
        person_dict['upload_image_filename'] = person_obj.image.filename

    logger.info(f"DEBUG: _prepare_person_for_schema - person_dict before validation: {person_dict.get('crop_image_path')}, Individual Name: {person_dict.get('name')}") # 新增日志
    logger.info(f"DEBUG: _prepare_person_for_schema - is_trained value: {person_dict.get('is_trained')}") # 新增调试日志
    return schemas.Person.model_validate(person_dict)

# --- 用户 CRUD 操作 ---
def get_user(db: Session, user_id: int):
    """通过ID查询单个用户信息。"""
    return db.query(User).filter(User.id == user_id).first()

def get_all_users(db: Session, is_active: Optional[bool] = None):
    """获取数据库中所有用户的列表，可按活跃状态筛选。"""
    query = db.query(User)
    if is_active is not None:
        query = query.filter(User.is_active == is_active)
    return query.all()

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = auth.get_password_hash(user.password)
    db_user = User(username=user.username, hashed_password=hashed_password, role=user.role, unit=user.unit, phone_number=user.phone_number, is_active=user.is_active)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def update_user_role(db: Session, user_id: int, new_role: schemas.UserRole):
    """更新指定用户的角色。"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db_user.role = new_role
        db.commit()
        db.refresh(db_user)
    return db_user

def delete_user(db: Session, user_id: int):
    """删除指定用户。"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db.delete(db_user)
        db.commit()
        return True
    return False

def update_user_status(db: Session, user_id: int, is_active: bool):
    """更新指定用户的活跃状态。"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db_user.is_active = is_active
        db.commit()
        db.refresh(db_user)
    return db_user

def update_user_profile(db: Session, user_id: int, unit: Optional[str] = None, phone_number: Optional[str] = None):
    """更新指定用户的个人资料（单位和手机号码）。"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        if unit is not None: # 允许 unit 为空字符串来清空
            db_user.unit = unit
        if phone_number is not None: # 允许 phone_number 为空字符串来清空
            db_user.phone_number = phone_number
        db.commit()
        db.refresh(db_user)
    return db_user

def update_user_password(db: Session, user_id: int, hashed_password: str):
    """更新指定用户的密码。"""
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db_user.hashed_password = hashed_password
        db.commit()
        db.refresh(db_user)
    return db_user

# --- 新增 Individual CRUD operations ---
def create_individual(db: Session, individual: schemas.IndividualCreate) -> Individual:
    db_individual = Individual(
        uuid=individual.uuid if individual.uuid else str(uuid.uuid4()),
        name=individual.name,
        id_card=individual.id_card,
        created_at=datetime.now(pytz.timezone('Asia/Shanghai'))
    )
    db.add(db_individual)
    db.commit()
    db.refresh(db_individual)
    return db_individual

def get_individual(db: Session, individual_id: int) -> Optional[Individual]:
    return db.query(Individual).filter(Individual.id == individual_id).first()

def get_individual_by_uuid(db: Session, individual_uuid: str) -> Optional[Individual]:
    return db.query(Individual).filter(Individual.uuid == individual_uuid).first()

def get_individual_by_id_card(db: Session, id_card: str) -> Optional[Individual]:
    return db.query(Individual).filter(Individual.id_card == id_card).first()

def get_individuals(db: Session, skip: int = 0, limit: int = 100) -> List[Individual]:
    return db.query(Individual).order_by(Individual.created_at.desc()).offset(skip).limit(limit).all()


# --- 视频 CRUD 操作 (仅处理文件上传视频) ---
def create_video(db: Session, filename: str, owner_id: int, file_path: str) -> Video:
    db_video = Video(
        uuid=str(uuid.uuid4()),
        filename=filename, 
        status="processing", 
        owner_id=owner_id, 
        file_path=file_path,
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)
    return db_video

def update_video_status(db: Session, video_id: int, status: str):
    """更新视频的处理状态。"""
    db_video = db.query(Video).filter(Video.id == video_id).first()
    if db_video:
        db_video.status = status
        # db.commit() # 移除此行，由调用方批量提交
    return db_video

def get_all_videos(db: Session, status: Optional[str] = None, skip: int = 0, limit: int = 100) -> List[Video]:
    """获取数据库中所有视频的列表，可按状态筛选，支持分页。"""
    query = db.query(Video).options(joinedload(Video.owner)) # 急切加载 owner 关系
    if status:
        query = query.filter(Video.status == status)
    return query.order_by(Video.id.desc()).offset(skip).limit(limit).all()

def get_total_videos_count(db: Session, status: Optional[str] = None) -> int:
    """获取数据库中所有视频的总数，可按状态筛选。"""
    query = db.query(Video)
    if status:
        query = query.filter(Video.status == status)
    return query.count()

def update_video_progress(db: Session, video_id: int, progress: int):
    """更新指定视频的处理进度。"""
    db_video = db.query(Video).filter(Video.id == video_id).first()
    if db_video:
        db_video.progress = progress
        # db.commit() # 移除此行，由调用方批量提交
    return db_video

def get_video(db: Session, video_id: int) -> Optional[Video]:
    """获取单个视频的信息。"""
    return db.query(Video).filter(Video.id == video_id).first()

def get_video_by_uuid(db: Session, uuid: str) -> Optional[Video]:
    """通过 UUID 获取单个视频的信息。"""
    return db.query(Video).filter(Video.uuid == uuid).first()

def get_videos_by_owner_id(db: Session, owner_id: int, status: Optional[str] = None, skip: int = 0, limit: int = 100) -> List[Video]:
    """获取指定用户上传的所有视频，可按状态筛选，支持分页。"""
    query = db.query(Video).options(joinedload(Video.owner)).filter(Video.owner_id == owner_id) # 急切加载 owner 关系
    if status:
        query = query.filter(Video.status == status)
    return query.order_by(Video.id.desc()).offset(skip).limit(limit).all()

def get_total_videos_count_by_owner_id(db: Session, owner_id: int, status: Optional[str] = None) -> int:
    """获取指定用户上传的所有视频的总数，可按状态筛选。"""
    query = db.query(Video).filter(Video.owner_id == owner_id)
    if status:
        query = query.filter(Video.status == status)
    return query.count()

# --- 视频流 CRUD 操作 (新的) ---
def create_stream(db: Session, name: str, stream_url: str, owner_id: int, stream_uuid: Optional[str] = None) -> Stream:
    db_stream = Stream(
        name=name,
        stream_url=stream_url,
        owner_id=owner_id,
        status="processing", # 默认状态改为正在处理
        stream_uuid=stream_uuid
    )
    db.add(db_stream)
    db.commit()
    db.refresh(db_stream)
    return db_stream

def get_stream(db: Session, stream_id: int) -> Optional[Stream]:
    """获取单个视频流的信息。"""
    return db.query(Stream).filter(Stream.id == stream_id).first()

def get_stream_by_uuid(db: Session, stream_uuid: str) -> Optional[Stream]:
    """通过 UUID 获取单个视频流的信息。"""
    return db.query(Stream).filter(Stream.stream_uuid == stream_uuid).first()

def get_stream_by_url(db: Session, stream_url: str) -> Optional[Stream]:
    """通过 URL 获取单个视频流的信息。"""
    return db.query(Stream).filter(Stream.stream_url == stream_url).first()

def get_all_streams(db: Session, skip: int = 0, limit: int = 100) -> List[Stream]:
    """获取所有视频流的列表，支持分页。"""
    return db.query(Stream).order_by(Stream.id.desc()).offset(skip).limit(limit).all()

def get_streams_by_owner_id(db: Session, owner_id: int, skip: int = 0, limit: int = 100) -> List[Stream]:
    """获取指定用户拥有的所有视频流，支持分页。"""
    return db.query(Stream).filter(Stream.owner_id == owner_id).order_by(Stream.id.desc()).offset(skip).limit(limit).all()

def update_stream_status(db: Session, stream_id: int, status: str) -> Optional[Stream]:
    """更新视频流的处理状态。"""
    db_stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if db_stream:
        db_stream.status = status
        # db.commit() # 移除此行，由调用方批量提交
        # db.refresh(db_stream) # 移除此行，避免在commit前重置状态
    return db_stream

def update_stream_output_video_path(db: Session, stream_id: int, output_video_path: str) -> Optional[Stream]:
    """更新视频流的输出视频路径。"""
    db_stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if db_stream:
        logger.info(f"尝试更新流 {stream_id} 的 output_video_path 为: {output_video_path}") # 新增日志
        db_stream.output_video_path = output_video_path
        # db.commit() # 移除此行，由调用方批量提交
        # db.refresh(db_stream)
    return db_stream

def update_stream_last_processed_at(db: Session, stream_id: int) -> Optional[Stream]:
    """更新视频流的最后处理时间。"""
    db_stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if db_stream:
        db_stream.last_processed_at = datetime.now(pytz.timezone('Asia/Shanghai'))
        # db.commit() # 移除此行，由调用方批量提交
        db.refresh(db_stream)
    return db_stream

def delete_stream(db: Session, stream_id: int) -> Optional[Stream]:
    """
    删除指定视频流及其关联的人物特征图片文件和数据库记录，以及保存的视频流文件。
    """
    db_stream = db.query(Stream).filter(Stream.id == stream_id).first()
    if db_stream:
        # 1. 获取所有与此视频流关联的人物特征
        associated_persons = db.query(Person).filter(Person.stream_id == stream_id).all()

        # 2. 遍历并删除每个关联人物的特征图片文件和数据库记录
        for person in associated_persons:
            # 特征图片路径现在是 "<stream_uuid>/<person_uuid>.jpg"，需要删除整个子目录
            # 这里不再需要单独删除图片文件，因为将删除整个目录
            db.delete(person) # 删除人物特征的数据库记录

        # 删除 general_detection 模型相关的裁剪图片目录
        if db_stream.stream_uuid:
            general_detection_crop_dir = settings.get_parsed_image_path(
                base_dir=settings.DATABASE_CROPS_DIR,
                model_name="general_detection", # general_detection 模型
                analysis_type="stream",        # 视频流分析
                uuid=db_stream.stream_uuid     # 使用视频流的 UUID 作为目录
            )
            if os.path.isdir(general_detection_crop_dir):
                try:
                    shutil.rmtree(general_detection_crop_dir)
                    logger.info(f"已删除视频流 {db_stream.stream_uuid} 的 general_detection 裁剪图片目录: {general_detection_crop_dir}")
                except OSError as e:
                    logger.error(f"删除视频流 {db_stream.stream_uuid} 的 general_detection 裁剪图片目录 {general_detection_crop_dir} 时出错: {e}")
            else:
                logger.warning(f"视频流 {db_stream.stream_uuid} 的 general_detection 裁剪图片目录 {general_detection_crop_dir} 不存在。")
            
            # 删除 general_detection 模型相关的完整帧图片目录
            general_detection_full_frame_dir = settings.get_parsed_image_path(
                base_dir=settings.DATABASE_FULL_FRAMES_DIR,
                model_name="general_detection", # general_detection 模型
                analysis_type="stream",        # 视频流分析
                uuid=db_stream.stream_uuid     # 使用视频流的 UUID 作为目录
            )
            if os.path.isdir(general_detection_full_frame_dir):
                try:
                    shutil.rmtree(general_detection_full_frame_dir)
                    logger.info(f"已删除视频流 {db_stream.stream_uuid} 的 general_detection 完整帧图片目录: {general_detection_full_frame_dir}")
                except OSError as e:
                    logger.error(f"删除视频流 {db_stream.stream_uuid} 的 general_detection 完整帧图片目录 {general_detection_full_frame_dir} 时出错: {e}")
            else:
                logger.warning(f"视频流 {db_stream.stream_uuid} 的 general_detection 完整帧图片目录 {general_detection_full_frame_dir} 不存在。")

            # 删除 face_recognition 模型相关的裁剪图片目录
            face_recognition_crop_dir = settings.get_parsed_image_path(
                base_dir=settings.DATABASE_CROPS_DIR,
                model_name="face_recognition", # face_recognition 模型
                analysis_type="stream",        # 视频流分析
                uuid=db_stream.stream_uuid     # 使用视频流的 UUID 作为目录
            )
            if os.path.isdir(face_recognition_crop_dir):
                try:
                    shutil.rmtree(face_recognition_crop_dir)
                    logger.info(f"已删除视频流 {db_stream.stream_uuid} 的 face_recognition 裁剪图片目录: {face_recognition_crop_dir}")
                except OSError as e:
                    logger.error(f"删除视频流 {db_stream.stream_uuid} 的 face_recognition 裁剪图片目录 {face_recognition_crop_dir} 时出错: {e}")
            else:
                logger.warning(f"视频流 {db_stream.stream_uuid} 的 face_recognition 裁剪图片目录 {face_recognition_crop_dir} 不存在。")
            
            # 删除 gait_recognition 模型相关的裁剪图片目录
            gait_recognition_crop_dir = settings.get_parsed_image_path(
                base_dir=settings.DATABASE_CROPS_DIR,
                model_name="gait_recognition", # gait_recognition 模型
                analysis_type="stream",        # 视频流分析
                uuid=db_stream.stream_uuid     # 使用视频流的 UUID 作为目录
            )
            if os.path.isdir(gait_recognition_crop_dir):
                try:
                    shutil.rmtree(gait_recognition_crop_dir)
                    logger.info(f"已删除视频流 {db_stream.stream_uuid} 的 gait_recognition 裁剪图片目录: {gait_recognition_crop_dir}")
                except OSError as e:
                    logger.error(f"删除视频流 {db_stream.stream_uuid} 的 gait_recognition 裁剪图片目录 {gait_recognition_crop_dir} 时出错: {e}")
            else:
                logger.warning(f"视频流 {db_stream.stream_uuid} 的 gait_recognition 裁剪图片目录 {gait_recognition_crop_dir} 不存在。")

        # 针对视频流文件，需要从 SAVED_STREAMS_DIR 构建完整路径
        if db_stream.output_video_path:
            full_output_video_path = os.path.join(settings.SAVED_STREAMS_DIR, db_stream.output_video_path)
            if os.path.exists(full_output_video_path):
                try:
                    # 如果 output_video_path 是指向 stream_id 目录下的 MP4 文件，那么需要删除整个目录
                    # 这里的判断逻辑保持不变，但现在操作的是完整的绝对路径
                    stream_save_dir = os.path.dirname(full_output_video_path)
                    # 再次确认路径是否是预期的 stream_uuid 目录
                    if os.path.isdir(stream_save_dir) and stream_save_dir.endswith(db_stream.stream_uuid):
                        shutil.rmtree(stream_save_dir)
                        logger.info(f"已删除视频流保存目录: {stream_save_dir}")
                    else:
                        # 如果不是预期的目录结构，只删除文件
                        os.remove(full_output_video_path)
                        logger.info(f"已删除视频流文件: {full_output_video_path}")
                except OSError as e:
                    logger.error(f"删除视频流文件/目录 {full_output_video_path} 时出错: {e}")
            else:
                logger.warning(f"视频流 {db_stream.stream_uuid} 的保存视频文件 {full_output_video_path} 不存在。")

        # 4. 删除视频流记录
        db.delete(db_stream)
        db.commit()
        logger.info(f"视频流 {stream_id} 及其所有关联数据已成功删除。")
        return db_stream
    return None

def get_total_streams_count(db: Session) -> int:
    """获取所有视频流的总数。"""
    return db.query(Stream).count()

def get_total_streams_count_by_owner_id(db: Session, owner_id: int) -> int:
    """获取指定用户拥有的所有视频流的总数。"""
    return db.query(Stream).filter(Stream.owner_id == owner_id).count()

# --- 图片 CRUD 操作 (新增) ---
def create_image(db: Session, image: schemas.ImageCreate) -> Image:
    """在数据库中创建新的图片记录。"""
    db_image = Image(
        uuid=image.uuid,
        filename=image.filename,
        file_path=image.file_path,
        person_count=image.person_count,
        owner_id=image.owner_id
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image

def get_image(db: Session, image_id: int) -> Optional[Image]:
    """通过 ID 获取单个图片信息。"""
    return db.query(Image).filter(Image.id == image_id).first()

def get_image_by_uuid(db: Session, uuid: str) -> Optional[Image]:
    """通过 UUID 获取单个图片信息。"""
    return db.query(Image).filter(Image.uuid == uuid).first()

# --- 人员 CRUD 操作 (remains linked to Video) ---
def get_person(db: Session, person_id: int, current_user_id: Optional[int] = None) -> Optional[schemas.Person]:
    """通过ID查询单个人物信息。"""
    person_obj = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(Person.id == person_id).first()
    if person_obj:
        return _prepare_person_for_schema(person_obj, db, current_user_id)
    return None

def get_person_by_uuid(db: Session, uuid: str, current_user_id: Optional[int] = None) -> Optional[schemas.Person]:
    """通过 UUID 查询单个人员信息。"""
    person_obj = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(Person.uuid == uuid).first()
    if person_obj:
        return _prepare_person_for_schema(person_obj, db, current_user_id)
    return None

def get_person_by_uuid_obj(db: Session, uuid: str) -> Optional[Person]:
    """通过 UUID 查询单个人员的 SQLAlchemy ORM 对象。"""
    person_obj = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(Person.uuid == uuid).first()
    return person_obj

def create_person(db: Session, person: schemas.PersonCreate) -> Person:
    # Logic to handle individual_id, id_card, person_name from schemas.PersonCreate
    individual_id_to_use = person.individual_id
    if person.id_card: # If id_card is provided, try to find or create an Individual
        existing_individual = get_individual_by_id_card(db, person.id_card)
        if existing_individual:
            individual_id_to_use = existing_individual.id
            # Optionally update existing individual's name if person_name is provided and different
            if person.person_name and existing_individual.name != person.person_name:
                existing_individual.name = person.person_name
                db.add(existing_individual) # Mark for update
        else:
            # Create new individual if not found
            new_individual = create_individual(db, schemas.IndividualCreate(
                name=person.person_name,
                id_card=person.id_card,
                uuid=str(uuid.uuid4()) # Generate UUID for new Individual
            ))
            individual_id_to_use = new_individual.id
    
    # feature_vector already a JSON string, directly store
    db_person = Person(
        uuid=person.uuid,
        feature_vector=person.feature_vector,
        crop_image_path=person.crop_image_path,
        full_frame_image_path=person.full_frame_image_path,
        video_id=person.video_id,
        stream_id=person.stream_id,
        image_id=person.image_id,
        is_verified=person.is_verified,
        verified_by_user_id=person.verified_by_user_id,
        verification_date=person.verification_date,
        correction_details=person.correction_details,
        marked_for_retrain=person.marked_for_retrain,
        confidence_score=person.confidence_score,
        pose_keypoints=person.pose_keypoints,
        face_image_path=person.face_image_path,
        face_feature_vector=person.face_feature_vector,
        face_id=person.face_id,
        clothing_attributes=person.clothing_attributes,
        gait_feature_vector=person.gait_feature_vector,
        gait_image_path=person.gait_image_path,
        individual_id=individual_id_to_use, # Assign individual_id
        is_trained=False # 新增：初始化为未训练
    )
    db.add(db_person)
    db.commit()
    db.refresh(db_person)
    return db_person

# 更新人物的再训练标记状态
def update_person_retrain_status(
    db: Session,
    person_id: int,
    marked_for_retrain: bool
) -> Optional[Person]:
    db_person = db.query(Person).filter(Person.id == person_id).first()
    if db_person:
        db_person.marked_for_retrain = marked_for_retrain
        # 如果是标记为待再训练，并且当前是已审核状态，则取消已审核状态
        # 这一步的逻辑应该由前端或外部调用者控制，这里只更新 retrain 标记
        # if marked_for_retrain and db_person.is_verified:
        #     db_person.is_verified = False
        #     db_person.verified_by_user_id = None
        #     db_person.verification_date = None
        db.commit()
        db.refresh(db_person)
    return db_person

# 获取所有标记为再训练的人物
def get_persons_marked_for_retrain(db: Session, current_user_id: Optional[int] = None) -> List[Person]:
    """获取所有标记为再训练的人物列表。"""
    query = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), 
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(Person.marked_for_retrain == True)
    persons_data = query.all()
    return [ _prepare_person_for_schema(person, db, current_user_id) for person in persons_data ]

# 新增：批量更新人物的训练状态（已训练）
def update_persons_trained_status(
    db: Session,
    person_uuids: List[str]
) -> int:
    """根据 UUID 列表批量更新人物的训练状态为已训练，并清除再训练标记。"""
    updated_count = db.query(Person).filter(Person.uuid.in_(person_uuids)).update(
        {
            Person.marked_for_retrain: False,
            Person.is_trained: True, # 新增：标记为已训练
            Person.is_verified: True, # 自动标记为已审核
            Person.verified_by_user_id: 1, # 系统用户 ID，表示系统自动审核
            Person.verification_date: datetime.now(pytz.timezone('Asia/Shanghai'))
        },
        synchronize_session="fetch"
    )
    db.commit()
    logger.info(f"已批量更新 {updated_count} 个人物的训练状态和再训练标记。")
    return updated_count


# 新增：更新人物审核状态和纠正详情
def update_person_verification_status(
    db: Session,
    person_id: int,
    is_verified: bool,
    verified_by_user_id: Optional[int] = None,
    correction_details: Optional[str] = None,
    marked_for_retrain: Optional[bool] = None,
    is_trained: Optional[bool] = None # 新增参数
) -> Optional[Person]:
    db_person = db.query(Person).filter(Person.id == person_id).first()
    if db_person:
        db_person.is_verified = is_verified
        db_person.verified_by_user_id = verified_by_user_id
        db_person.verification_date = datetime.now(pytz.timezone('Asia/Shanghai')) if is_verified else None
        if correction_details is not None:
            db_person.correction_details = correction_details
        # marked_for_retrain 和 is_trained 不再在这里直接更新，由更上层的逻辑控制
        # if marked_for_retrain is not None:
        #     db_person.marked_for_retrain = marked_for_retrain
        # if is_trained is not None:
        #     db_person.is_trained = is_trained
        db.commit()
        db.refresh(db_person)
    return db_person


# 创建纠正日志
def create_correction_log(db: Session, correction_log: schemas.CorrectionLogCreate) -> CorrectionLog:
    """创建新的纠正日志条目。"""
    # 获取用户名和IP地址，因为 Pydantic 模型不再包含这些字段
    # 这些信息应从前端传入或在路由层获取
    username_from_db = None
    if correction_log.username: # Pydantic模型中仍可包含，但这里主要为了兼容旧数据或测试
        username_from_db = correction_log.username
    elif correction_log.corrected_by_user_id: # 如果 schema 包含这个，可以从这里获取
        user = db.query(User).filter(User.id == correction_log.corrected_by_user_id).first()
        if user: username_from_db = user.username
    
    db_correction_log = CorrectionLog(
        # correction_date=correction_log.timestamp, # 将 timestamp 映射到 correction_date (移除)
        # logger=correction_log.logger, # 移除，不再是 ORM 模型列
        # level=correction_log.level, # 移除，不再是 ORM 模型列
        details=correction_log.details, # message 映射到 details，因为 ORM 模型没有 message 字段
        person_id=correction_log.person_id,
        correction_type=correction_log.correction_type,
        corrected_by_user_id=correction_log.corrected_by_user_id, # 从 Pydantic 模型获取
        corrected_by_username=username_from_db, # 新增
        ip_address=correction_log.ip_address, # 新增
        target_person_uuid=correction_log.target_person_uuid, # 新增：存储目标人物UUID
        target_individual_id=correction_log.target_individual_id # 新增：存储目标 Individual ID
    )
    db.add(db_correction_log)
    db.commit()
    db.refresh(db_correction_log)
    return db_correction_log

def merge_persons(db: Session, source_person_uuid: str, target_person_uuid: str, current_user_id: int) -> Optional[schemas.Person]:
    """
    将源人物（source_person_uuid）的信息合并到目标人物（target_person_uuid）上。
    此操作会更新源人物的姓名、身份证号，并标记为已审核，同时记录合并日志。
    注意：此版本不进行特征向量的物理合并，只更新元数据和创建新的 Person 记录以关联到目标人物。
    """
    source_person = db.query(Person).options(joinedload(Person.individual)).filter(Person.uuid == source_person_uuid).first()
    target_person = db.query(Person).options(joinedload(Person.individual)).filter(Person.uuid == target_person_uuid).first()

    if not source_person or not target_person:
        logger.warning(f"合并人物失败：源人物 {source_person_uuid} 或目标人物 {target_person_uuid} 不存在。")
        return None

    logger.info(f"开始合并人物：源人物 {source_person_uuid} 到目标人物 {target_person_uuid}。")

    try:
        # 实际合并逻辑：将源人物的 individual_id 指向目标人物的 individual_id
        # 如果目标人物没有 individual，则需要创建一个
        if not target_person.individual_id:
            logger.info(f"目标人物 {target_person_uuid} 没有关联的 Individual，正在为其创建新的 Individual。")
            new_individual = create_individual(db, schemas.IndividualCreate(
                name="未知人物", # 直接使用默认值
                id_card=str(uuid.uuid4()), # 为 id_card 生成一个 UUID 占位符
                uuid=str(uuid.uuid4()) # 为新的 Individual 生成 UUID
            ))
            target_person.individual_id = new_individual.id
            db.add(target_person) # 标记目标人物为更新
            db.commit() # 提交目标人物的 individual_id 更改
            db.refresh(target_person)
            logger.info(f"已为目标人物 {target_person_uuid} 创建并关联新的 Individual {new_individual.uuid}。")

        if source_person.individual_id != target_person.individual_id:
            source_person.individual_id = target_person.individual_id
            source_person.is_verified = True # 标记源人物为已审核
            source_person.verified_by_user_id = current_user_id
            source_person.verification_date = datetime.now(pytz.timezone('Asia/Shanghai'))
            source_person.correction_details = f"Merged from {source_person.uuid} to individual {target_person.individual.uuid} (via person {target_person.uuid})"
            source_person.correction_type_display = "已纠正（合并）" # 明确设置纠正类型显示
            source_person.marked_for_retrain = False
            db.add(source_person)

            # 同时，更新目标人物的状态
            # 如果目标人物未被审核，则标记为已审核；如果已审核，则仅更新审核日期和添加细节
            target_person.verified_by_user_id = current_user_id
            target_person.verification_date = datetime.now(pytz.timezone('Asia/Shanghai'))

            new_correction_detail_for_target = f"Identity confirmed/strengthened by merge from person {source_person.uuid}. "
            if target_person.correction_details:
                target_person.correction_details += new_correction_detail_for_target
            else:
                target_person.correction_details = new_correction_detail_for_target
            target_person.correction_type_display = "身份已合并确认" # 新增：明确设置目标人物的纠正类型显示

            if not target_person.is_verified:
                target_person.is_verified = True
                logger.info(f"目标人物 {target_person_uuid} 的状态已更新为已审核（通过合并操作）。")
            else:
                logger.info(f"目标人物 {target_person_uuid} 已是已审核状态，仅更新了纠正详情（通过合并操作）。")

            db.add(target_person) # 标记目标人物为更新
            
            db.commit() # 提交所有更改
            db.refresh(target_person) # 刷新目标人物以获取最新的 Individual 关联

            # 获取与目标 Individual 关联的所有其他 Person 记录，并更新它们的状态
            if target_person.individual_id:
                other_persons_in_individual = db.query(Person).filter(
                    Person.individual_id == target_person.individual_id,
                    Person.uuid != source_person_uuid,
                    Person.uuid != target_person_uuid # 排除已处理的源人物和目标人物
                ).all()

                for person_in_group in other_persons_in_individual:
                    person_in_group.verified_by_user_id = current_user_id
                    person_in_group.verification_date = datetime.now(pytz.timezone('Asia/Shanghai'))

                    # 为这些额外的人物记录更新纠正详情
                    new_correction_detail_for_group = f"Identity confirmed/strengthened by merge of person {source_person.uuid}. "
                    if person_in_group.correction_details:
                        person_in_group.correction_details += new_correction_detail_for_group
                    else:
                        person_in_group.correction_details = new_correction_detail_for_group
                    person_in_group.correction_type_display = "身份已合并确认" # 新增：明确设置纠正类型显示
                    
                    if not person_in_group.is_verified:
                        person_in_group.is_verified = True
                        logger.info(f"人物 {person_in_group.uuid} (属于同一 Individual {target_person.individual.uuid}) 的状态已更新为已审核（通过合并操作）。")
                    else:
                        logger.info(f"人物 {person_in_group.uuid} (属于同一 Individual {target_person.individual.uuid}) 已是已审核状态，仅更新了纠正详情（通过合并操作）。")
                    db.add(person_in_group)
                
                # 再次提交，以保存对其他 Person 记录的更改
                if other_persons_in_individual:
                    db.commit()
                    logger.info(f"已更新与 Individual {target_person.individual.uuid} 关联的 {len(other_persons_in_individual)} 个人物的状态。")

            # 记录合并纠正日志
            log_message = f"人物 {source_person.uuid} 的特征已合并到逻辑人物 {target_person.individual.uuid} (通过人物 {target_person.uuid})。"
            correction_log_data = schemas.CorrectionLogCreate(
                message=log_message,
                person_id=source_person.id,
                correction_type="merge",
                details=f"Source Person UUID: {source_person.uuid}, Target Individual ID: {target_person.individual_id}, Target Individual UUID: {target_person.individual.uuid}",
                target_person_uuid=target_person.uuid, # 保留原有的 target_person_uuid，用于追溯
                target_individual_id=target_person.individual_id, # 新增：记录目标 Individual ID
                corrected_by_user_id=current_user_id
            )
            create_correction_log(db, correction_log=correction_log_data)
            logger.info(f"人物 {source_person.uuid} 已成功合并到逻辑人物 {target_person.individual.uuid}。")
        else:
            logger.info(f"人物 {source_person_uuid} 和 {target_person_uuid} 已经关联到同一个逻辑人物，无需合并。")
        
        # 返回合并后的目标人物信息（其 individual 关联已经更新）
        # 注意：这里返回的 target_person_uuid 仍然是原始的 Person UUID，但是其代表的逻辑人物已更新
        return _prepare_person_for_schema(target_person, db)

    except Exception as e:
        db.rollback()
        logger.error(f"合并人物 {source_person_uuid} 到 {target_person_uuid} 过程中发生错误: {e}", exc_info=True)
        return None

def get_persons(db: Session, skip: int = 0, limit: int = 100, 
                is_verified: Optional[bool] = None, 
                marked_for_retrain: Optional[bool] = None,
                min_confidence: Optional[float] = None,
                max_confidence: Optional[float] = None,
                query: Optional[str] = None,
                is_trained: Optional[bool] = None,
                has_id_card: Optional[bool] = None,
                current_user_id: Optional[int] = None) -> List[schemas.Person]:
    """获取人物列表，支持分页、审核状态、再训练标记、置信度以及模糊搜索。"""
    q = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    )

    if is_verified is not None:
        q = q.filter(Person.is_verified == is_verified)
    
    if marked_for_retrain is not None:
        q = q.filter(Person.marked_for_retrain == marked_for_retrain)

    if is_trained is not None:
        q = q.filter(Person.is_trained == is_trained)

    if min_confidence is not None:
        q = q.filter(Person.confidence_score >= min_confidence)

    if max_confidence is not None:
        q = q.filter(Person.confidence_score <= max_confidence)

    # 新增对 Individual.id_card 的筛选
    if has_id_card is not None:
        # 确保 Individual 表被 join
        q = q.outerjoin(Individual, Person.individual_id == Individual.id)
        if has_id_card:
            q = q.filter(Individual.id_card != None)
        else:
            q = q.filter(Individual.id_card == None)

    if query:
        search_query = f"%{query.lower()}%".lower()
        # Join with Individual table for name/id_card search
        q = q.outerjoin(Individual, Person.individual_id == Individual.id) # Left join to include persons without individual_id
        q = q.filter(or_(
            Individual.name.ilike(search_query), # Search in Individual name
            Individual.id_card.ilike(search_query), # Search in Individual id_card
            Person.uuid.ilike(search_query),
            Person.video.has(Video.filename.ilike(search_query)), 
            Person.stream.has(Stream.name.ilike(search_query)), 
            Person.image.has(Image.filename.ilike(search_query)) 
        ))

    persons_orm = q.order_by(Person.created_at.desc()).offset(skip).limit(limit).all()
    return [ _prepare_person_for_schema(person, db, current_user_id) for person in persons_orm ]


def get_total_persons_count(db: Session, is_verified: Optional[bool] = None, marked_for_retrain: Optional[bool] = None,
                            min_confidence: Optional[float] = None, max_confidence: Optional[float] = None,
                            query: Optional[str] = None,
                            is_trained: Optional[bool] = None,
                            has_id_card: Optional[bool] = None) -> int:
    
    q = db.query(Person).options(
        joinedload(Person.individual) # Eager load individual
    )

    if is_verified is not None:
        q = q.filter(Person.is_verified == is_verified)
    
    if marked_for_retrain is not None:
        q = q.filter(Person.marked_for_retrain == marked_for_retrain)

    if is_trained is not None:
        q = q.filter(Person.is_trained == is_trained)
    
    if min_confidence is not None:
        q = q.filter(Person.confidence_score >= min_confidence)

    if max_confidence is not None:
        q = q.filter(Person.confidence_score <= max_confidence)

    # 新增对 Individual.id_card 的筛选
    if has_id_card is not None:
        q = q.outerjoin(Individual, Person.individual_id == Individual.id)
        if has_id_card:
            q = q.filter(Individual.id_card != None)
        else:
            q = q.filter(Individual.id_card == None)

    if query:
        search_query = f"%{query.lower()}%".lower()
        q = q.outerjoin(Individual, Person.individual_id == Individual.id)
        q = q.filter(or_(
            Individual.name.ilike(search_query),
            Individual.id_card.ilike(search_query),
            Person.uuid.ilike(search_query),
            Person.video.has(Video.filename.ilike(search_query)), 
            Person.stream.has(Stream.name.ilike(search_query)), 
            Person.image.has(Image.filename.ilike(search_query)) 
        ))

    return q.count()

# 新增获取未审核人物的函数
def get_unverified_persons(db: Session, skip: int = 0, limit: int = 100, current_user_id: Optional[int] = None) -> List[schemas.Person]:
    """获取未审核且未标记为再训练的人物列表。"""
    query = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(
        Person.is_verified == False,
        Person.marked_for_retrain == False,
        Person.is_trained == False, # 新增：确保只显示未训练的人物
        (Person.confidence_score == None) | (Person.confidence_score < settings.HUMAN_REVIEW_CONFIDENCE_THRESHOLD) # 修正过滤条件以包含 confidence_score 为 None 的情况
    )
    persons_data = query.order_by(Person.created_at.desc()).offset(skip).limit(limit).all()
    return [_prepare_person_for_schema(p, db, current_user_id) for p in persons_data]

def get_total_unverified_persons_count(db: Session) -> int:
    """获取未审核且未标记为再训练的人物总数。"""
    return db.query(Person).filter(
        Person.is_verified == False,
        Person.marked_for_retrain == False,
        Person.is_trained == False, # 新增：确保只计算未训练的人物
        (Person.confidence_score == None) | (Person.confidence_score < settings.HUMAN_REVIEW_CONFIDENCE_THRESHOLD) # 修正过滤条件以包含 confidence_score 为 None 的情况
    ).count()

# --- 系统配置 CRUD 操作 ---
def get_system_config(db: Session, key: str) -> Optional[SystemConfig]:
    """获取单个系统配置项。"""
    return db.query(SystemConfig).filter(SystemConfig.key == key).first()

def update_system_config(db: Session, key: str, value: str) -> Optional[SystemConfig]:
    """更新系统配置项。"""
    db_config = db.query(SystemConfig).filter(SystemConfig.key == key).first()
    if db_config:
        db_config.value = value
        db_config.last_updated = datetime.now(pytz.timezone('Asia/Shanghai'))
        db.commit()
        db.refresh(db_config)
    return db_config

def set_system_config(db: Session, key: str, value: str) -> SystemConfig:
    """设置系统配置项，如果不存在则创建。"""
    db_config = get_system_config(db, key)
    if db_config:
        db_config.value = value
        db_config.last_updated = datetime.now(pytz.timezone('Asia/Shanghai'))
        db.commit()
        db.refresh(db_config)
        return db_config
    else:
        new_config = SystemConfig(
            key=key,
            value=value,
            last_updated=datetime.now(pytz.timezone('Asia/Shanghai'))
        )
        db.add(new_config)
        db.commit()
        db.refresh(new_config)
        return new_config

def get_all_system_configs(db: Session) -> Dict[str, str]:
    """获取所有系统配置项。"""
    configs = db.query(SystemConfig).all()
    return {config.key: config.value for config in configs}

def set_system_configs(db: Session, configs: Dict[str, Union[str, int, float, bool]]):
    """批量设置系统配置项，如果不存在则创建。"""
    for key, value in configs.items():
        # 将所有值转换为字符串以存储在数据库中
        str_value = str(value)
        set_system_config(db, key, str_value)
    db.commit() # 在所有更新/创建完成后进行一次性提交


def get_persons_by_video_id(db: Session, video_id: int, skip: int = 0, limit: int = 100, current_user_id: Optional[int] = None) -> List[schemas.Person]:
    """获取指定视频ID下的人物列表，支持分页。"""
    query = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(Person.video_id == video_id)
    persons_data = query.order_by(Person.id.desc()).offset(skip).limit(limit).all()
    return [_prepare_person_for_schema(p, db, current_user_id) for p in persons_data]

def get_total_persons_count_by_video_id(db: Session, video_id: int) -> int:
    """获取指定视频ID下的人物总数。"""
    return db.query(Person).filter(Person.video_id == video_id).count()

def get_all_persons_by_video_id(db: Session, video_id: int, current_user_id: Optional[int] = None) -> List[schemas.Person]:
    """获取指定视频ID下的所有人物。"""
    query = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(Person.video_id == video_id)
    persons_data = query.order_by(Person.id.desc()).all()
    return [_prepare_person_for_schema(p, db, current_user_id) for p in persons_data]

def get_all_persons_by_stream_id(db: Session, stream_id: int, current_user_id: Optional[int] = None) -> List[schemas.Person]:
    """获取指定视频流ID下的所有人物。"""
    query = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(Person.stream_id == stream_id)
    persons_data = query.order_by(Person.id.desc()).all()
    return [_prepare_person_for_schema(p, db, current_user_id) for p in persons_data]

def get_all_persons_by_owner_id(db: Session, owner_id: int, current_user_id: Optional[int] = None) -> List[schemas.Person]:
    """获取指定用户拥有的所有人物。"""
    query = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).outerjoin(Video, Person.video_id == Video.id)\
    .outerjoin(Stream, Person.stream_id == Stream.id)\
    .outerjoin(Image, Person.image_id == Image.id)\
    .filter(
        (Video.owner_id == owner_id) |\
        (Stream.owner_id == owner_id) |\
        (Image.owner_id == owner_id)
    )

    persons_data = query.order_by(Person.id.desc()).all()
    return [_prepare_person_for_schema(p, db, current_user_id) for p in persons_data]

def get_features_for_stream_by_stream_id(db: Session, stream_id: int, skip: int = 0, limit: Optional[int] = 100, current_user_id: Optional[int] = None) -> List[schemas.Person]:
    """获取指定视频流ID下，支持分页的特征。用于实时流的特征查询。"""
    query = db.query(Person).options(
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 虽然这里主要针对流，但为了通用性也加载一下image
        joinedload(Person.video).joinedload(Video.owner), # 新增：急切加载 video 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(Person.stream_id == stream_id)

    persons_data = query.order_by(Person.created_at.desc()).offset(skip)

    if limit is not None:
        persons_data = persons_data.limit(limit)
    
    persons_data = persons_data.all()
    return [_prepare_person_for_schema(p, db, current_user_id) for p in persons_data]

def delete_person_by_uuid(db: Session, uuid: str):
    """通过 UUID 删除指定人物以及其关联的裁剪图片和完整帧图片文件。"""
    db_person = db.query(Person).filter(Person.uuid == uuid).first()
    if db_person:
        # 删除裁剪图片文件
        if db_person.crop_image_path and os.path.exists(db_person.crop_image_path):
            try:
                os.remove(db_person.crop_image_path)
                logger.info(f"已删除裁剪图片: {db_person.crop_image_path}")
            except OSError as e:
                logger.error(f"删除裁剪图片 {db_person.crop_image_path} 时出错: {e}")
        
        # 删除完整帧图片文件
        if db_person.full_frame_image_path and os.path.exists(db_person.full_frame_image_path):
            try:
                os.remove(db_person.full_frame_image_path)
                logger.info(f"已删除完整帧图片: {db_person.full_frame_image_path}")
            except OSError as e:
                logger.error(f"删除完整帧图片 {db_person.full_frame_image_path} 时出错: {e}")
        
        # 删除数据库记录
        db.delete(db_person)
        db.commit()
        return True
    return False

def get_latest_person(db: Session):
    """获取最新的人物特征，用于测试或展示。"""
    return db.query(Person).order_by(Person.created_at.desc()).first()

def delete_video(db: Session, video_id: int) -> Optional[Video]:
    """
    删除指定视频及其关联的人物特征图片文件和数据库记录，以及视频文件本身。
    """
    db_video = db.query(Video).filter(Video.id == video_id).first()
    if db_video:
        # 获取所有与此视频关联的人物特征
        associated_persons = db.query(Person).filter(Person.video_id == video_id).all()

        # 优先删除与这些人物关联的实时比对预警记录
        for person in associated_persons:
            db.query(RealtimeMatchAlert).filter(RealtimeMatchAlert.person_uuid == person.uuid).delete()
            logger.info(f"已删除人物 {person.uuid} 关联的所有实时比对预警。")

        # 遍历并删除每个关联人物的数据库记录
        for person in associated_persons:
            # 在这里，我们不再单独删除每个 person 的 crop_image_path 和 full_frame_image_path
            # 因为后续会删除整个 video_uuid 对应的裁剪图目录和原始帧目录。
            # 只需要删除 Person 数据库记录即可。
            db.delete(person) # 删除人物特征的数据库记录
        logger.info(f"已删除视频 {video_id} 关联的 {len(associated_persons)} 个人物数据库记录。")

        # 删除视频文件本身
        if os.path.exists(db_video.file_path):
            try:
                os.remove(db_video.file_path)
                logger.info(f"已删除视频文件: {db_video.file_path}")
            except OSError as e:
                logger.error(f"删除视频文件 {db_video.file_path} 时出错: {e}")
        else:
            logger.warning(f"视频文件 {db_video.file_path} 不存在或不是文件，跳过删除。")

        # 删除 general_detection 模型相关的裁剪图片目录
        if db_video.uuid:
            general_detection_crop_dir = settings.get_parsed_image_path(
                base_dir=settings.DATABASE_CROPS_DIR,
                model_name="general_detection", # general_detection 模型
                analysis_type="video",         # 视频分析
                uuid=db_video.uuid             # 使用视频的 UUID 作为目录
            )
            if os.path.isdir(general_detection_crop_dir):
                try:
                    shutil.rmtree(general_detection_crop_dir)
                    logger.info(f"已删除视频 {db_video.uuid} 的 general_detection 裁剪图片目录: {general_detection_crop_dir}")
                except OSError as e:
                    logger.error(f"删除视频 {db_video.uuid} 的 general_detection 裁剪图片目录 {general_detection_crop_dir} 时出错: {e}")
            else:
                logger.warning(f"视频 {db_video.uuid} 的 general_detection 裁剪图片目录 {general_detection_crop_dir} 不存在。")
            
            # 删除 general_detection 模型相关的完整帧图片目录
            general_detection_full_frame_dir = settings.get_parsed_image_path(
                base_dir=settings.DATABASE_FULL_FRAMES_DIR,
                model_name="general_detection", # general_detection 模型
                analysis_type="video",         # 视频分析
                uuid=db_video.uuid             # 使用视频的 UUID 作为目录
            )
            if os.path.isdir(general_detection_full_frame_dir):
                try:
                    shutil.rmtree(general_detection_full_frame_dir)
                    logger.info(f"已删除视频 {db_video.uuid} 的 general_detection 完整帧图片目录: {general_detection_full_frame_dir}")
                except OSError as e:
                    logger.error(f"删除视频 {db_video.uuid} 的 general_detection 完整帧图片目录 {general_detection_full_frame_dir} 时出错: {e}")
            else:
                logger.warning(f"视频 {db_video.uuid} 的 general_detection 完整帧图片目录 {general_detection_full_frame_dir} 不存在。")

            # 删除 face_recognition 模型相关的裁剪图片目录
            face_recognition_crop_dir = settings.get_parsed_image_path(
                base_dir=settings.DATABASE_CROPS_DIR,
                model_name="face_recognition", # face_recognition 模型
                analysis_type="video",         # 视频分析
                uuid=db_video.uuid             # 使用视频的 UUID 作为目录
            )
            if os.path.isdir(face_recognition_crop_dir):
                try:
                    shutil.rmtree(face_recognition_crop_dir)
                    logger.info(f"已删除视频 {db_video.uuid} 的 face_recognition 裁剪图片目录: {face_recognition_crop_dir}")
                except OSError as e:
                    logger.error(f"删除视频 {db_video.uuid} 的 face_recognition 裁剪图片目录 {face_recognition_crop_dir} 时出错: {e}")
            else:
                logger.warning(f"视频 {db_video.uuid} 的 face_recognition 裁剪图片目录 {face_recognition_crop_dir} 不存在。")
            
            # 删除 gait_recognition 模型相关的裁剪图片目录
            gait_recognition_crop_dir = settings.get_parsed_image_path(
                base_dir=settings.DATABASE_CROPS_DIR,
                model_name="gait_recognition", # gait_recognition 模型
                analysis_type="video",         # 视频分析
                uuid=db_video.uuid             # 使用视频的 UUID 作为目录
            )
            if os.path.isdir(gait_recognition_crop_dir):
                try:
                    shutil.rmtree(gait_recognition_crop_dir)
                    logger.info(f"已删除视频 {db_video.uuid} 的 gait_recognition 裁剪图片目录: {gait_recognition_crop_dir}")
                except OSError as e:
                    logger.error(f"删除视频 {db_video.uuid} 的 gait_recognition 裁剪图片目录 {gait_recognition_crop_dir} 时出错: {e}")
            else:
                logger.warning(f"视频 {db_video.uuid} 的 gait_recognition 裁剪图片目录 {gait_recognition_crop_dir} 不存在。")

        # 删除视频数据库记录
        db.delete(db_video)
        db.commit()
        logger.info(f"视频 {video_id} 及其所有关联数据已成功删除。")
        return db_video
    return None

# --- 日志 CRUD 操作 ---
def create_log(db: Session, log_entry: schemas.LogEntry):
    """创建新的日志条目。"""
    db_log = Log(
        timestamp=log_entry.timestamp,
        logger=log_entry.logger,
        level=log_entry.level,
        message=log_entry.message,
        username=log_entry.username,
        ip_address=log_entry.ip_address
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_logs(db: Session, skip: int = 0, limit: int = 100, 
             level: Optional[str] = None, 
             start_time: Optional[datetime] = None, 
             end_time: Optional[datetime] = None, 
             keyword: Optional[str] = None):
    """获取日志列表，支持筛选和分页。"""
    query = db.query(Log)

    if level:
        query = query.filter(Log.level == level)
    if start_time:
        query = query.filter(Log.timestamp >= start_time)
    if end_time:
        query = query.filter(Log.timestamp <= end_time)
    if keyword:
        query = query.filter(Log.message.like(f'%{keyword}%') |
                             Log.username.like(f'%{keyword}%') |
                             Log.ip_address.like(f'%{keyword}%'))

    total = query.count() # 总数
    logs = query.order_by(Log.timestamp.desc()).offset(skip).limit(limit).all()
    return {"total": total, "logs": logs}

def get_total_persons_count_by_stream_id(db: Session, stream_id: int) -> int:
    """获取指定视频流ID下的人物总数。"""
    return db.query(Person).filter(Person.stream_id == stream_id).count()

def get_all_persons(db: Session) -> List[Person]: # Modified return type to List[Person]
    """
    获取数据库中所有人物的列表。
    """
    query = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    )
    logger.info("crud.get_all_persons: About to execute query.all()...") # New log
    persons_data = query.order_by(Person.id.desc()).all()
    logger.info(f"crud.get_all_persons: Finished executing query.all(), retrieved {len(persons_data)} persons.") # New log
    return persons_data # Return ORM objects directly

def get_persons_by_image_id(db: Session, image_id: int, current_user_id: Optional[int] = None) -> List[schemas.Person]:
    """获取指定图片ID下的人物列表。"""
    query = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual) # Eager load individual
    ).filter(Person.image_id == image_id)
    persons_data = query.order_by(Person.id.desc()).all()
    return [_prepare_person_for_schema(p, db, current_user_id) for p in persons_data]

def get_image_analysis_history(db: Session, owner_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[schemas.ImageResponse]:
    """获取图片解析历史记录，支持分页。"""
    query = db.query(Image).options(joinedload(Image.owner)) # 急切加载 owner 关系
    if owner_id:
        query = query.filter(Image.owner_id == owner_id)
    
    images = query.order_by(Image.created_at.desc()).offset(skip).limit(limit).all()
    
    # 将 Image ORM 对象转换为 ImageResponse Pydantic 模型，并填充 uploader_username
    image_responses = []
    for img in images:
        image_responses.append(schemas.ImageResponse(
            id=img.id,
            uuid=img.uuid,
            filename=img.filename,
            file_path=img.file_path,
            person_count=img.person_count,
            created_at=img.created_at,
            uploader_username=img.owner.username if img.owner else None # 获取上传人用户名
        ))
    return image_responses

def get_total_image_analysis_history_count(db: Session, owner_id: Optional[int] = None) -> int:
    """获取图片解析历史记录的总数。"""
    query = db.query(Image)
    if owner_id:
        query = query.filter(Image.owner_id == owner_id)
    return query.count()

def delete_image_analysis_data(db: Session, image_uuid: str, current_user_id: int, current_user_role: str) -> bool:
    """
    删除指定图片分析数据，包括图片文件、关联的人物特征图片及其数据库记录。
    只有管理员或图片所有者可以删除。
    """
    db_image = db.query(Image).filter(Image.uuid == image_uuid).first()
    if not db_image:
        logger.warning(f"删除图片 {image_uuid} 失败：未找到图片。")
        return False

    # 权限检查：只有管理员或图片所有者可以删除
    if current_user_role != "admin" and db_image.owner_id != current_user_id:
        logger.warning(f"用户 {current_user_id} (角色: {current_user_role}) 试图删除非其所有的图片 {image_uuid}。")
        return False
    
    logger.info(f"开始删除图片 {image_uuid} 及其相关数据。")

    try:
        # 1. 获取所有与此图片关联的人物特征
        associated_persons = db.query(Person).filter(Person.image_id == db_image.id).all()

        # 2. 遍历并删除每个关联人物的特征图片文件和数据库记录
        for person in associated_persons:
            # 在这里，我们不再单独删除每个 person 的 crop_image_path 和 full_frame_image_path
            # 因为后续会删除整个 media_uuid 对应的裁剪图目录和原始图片目录。
            # 只需要删除 Person 数据库记录即可。

            # 删除 Person 数据库记录
            db.delete(person)

        logger.info(f"已删除图片 {image_uuid} 关联的 {len(associated_persons)} 个人物数据库记录。")

        # 3. 删除裁剪图片目录 (所有人物的裁剪图都存储在这个目录下)
        crop_images_full_dir_to_delete = settings.get_parsed_image_path(
            base_dir=settings.DATABASE_CROPS_DIR,
            model_name="general_detection", # 对于图片分析，裁剪图的模型名称是固定的 "general_detection"
            analysis_type="image",         # 对于图片分析，分析类型是固定的 "image"
            uuid=db_image.uuid             # 使用 Image 的 UUID 来构建目录，所有人物裁剪图都在此目录下
        )

        if os.path.isdir(crop_images_full_dir_to_delete):
            try:
                shutil.rmtree(crop_images_full_dir_to_delete)
                logger.info(f"已删除裁剪图片目录: {crop_images_full_dir_to_delete}")
            except OSError as e:
                logger.error(f"删除裁剪图片目录 {crop_images_full_dir_to_delete} 时出错: {e}")
        else:
            logger.warning(f"裁剪图片目录 {crop_images_full_dir_to_delete} 不存在或不是目录，跳过删除。")


        original_image_full_dir_to_delete = settings.get_parsed_image_path(
            base_dir=settings.DATABASE_FULL_FRAMES_DIR,
            model_name="general_detection", # 对于图片分析，模型名称是固定的 "general_detection"
            analysis_type="image",         # 对于图片分析，分析类型是固定的 "image"
            uuid=db_image.uuid             # 使用 Image 的 UUID 来构建目录
        )
        
        if os.path.isdir(original_image_full_dir_to_delete):
            try:
                shutil.rmtree(original_image_full_dir_to_delete)
                logger.info(f"已删除原始上传图片目录: {original_image_full_dir_to_delete}")
            except OSError as e:
                logger.error(f"删除原始上传图片目录 {original_image_full_dir_to_delete} 时出错: {e}")
        else:
            logger.warning(f"原始上传图片目录 {original_image_full_dir_to_delete} 不存在或不是目录。")

        # 5. 删除人脸识别相关的裁剪图片目录
        face_recognition_crops_dir_to_delete = settings.get_parsed_image_path(
            base_dir=settings.DATABASE_CROPS_DIR,
            model_name="face_recognition",
            analysis_type="image",
            uuid=db_image.uuid
        )
        if os.path.isdir(face_recognition_crops_dir_to_delete):
            try:
                shutil.rmtree(face_recognition_crops_dir_to_delete)
                logger.info(f"已删除人脸识别裁剪图片目录: {face_recognition_crops_dir_to_delete}")
            except OSError as e:
                logger.error(f"删除人脸识别裁剪图片目录 {face_recognition_crops_dir_to_delete} 时出错: {e}")
        else:
            logger.warning(f"人脸识别裁剪图片目录 {face_recognition_crops_dir_to_delete} 不存在或不是目录，跳过删除。")

        # 6. 删除步态识别相关的裁剪图片目录
        gait_recognition_crops_dir_to_delete = settings.get_parsed_image_path(
            base_dir=settings.DATABASE_CROPS_DIR,
            model_name="gait_recognition",
            analysis_type="image",
            uuid=db_image.uuid
        )
        if os.path.isdir(gait_recognition_crops_dir_to_delete):
            try:
                shutil.rmtree(gait_recognition_crops_dir_to_delete)
                logger.info(f"已删除步态识别裁剪图片目录: {gait_recognition_crops_dir_to_delete}")
            except OSError as e:
                logger.error(f"删除步态识别裁剪图片目录 {gait_recognition_crops_dir_to_delete} 时出错: {e}")
        else:
            logger.warning(f"步态识别裁剪图片目录 {gait_recognition_crops_dir_to_delete} 不存在或不是目录，跳过删除。")

        # 7. 删除图片数据库记录
        db.delete(db_image)
        db.commit()
        logger.info(f"图片 {image_uuid} 及其所有相关数据库记录已成功删除。")
        return True
    except Exception as e:
        db.rollback() # 发生错误时回滚事务
        logger.error(f"删除图片 {image_uuid} 的数据时发生意外错误: {e}", exc_info=True)
        return False

def get_persons_by_uuids(db: Session, person_uuids: List[str], current_user_id: Optional[int] = None) -> List[Person]:
    """
    根据人物 UUID 列表获取人物对象。
    """
    query = db.query(Person).options(
        joinedload(Person.video).joinedload(Video.owner), # 急切加载 video 及其 owner
        joinedload(Person.stream).joinedload(Stream.owner), # 急切加载 stream 及其 owner
        joinedload(Person.image).joinedload(Image.owner), # 急切加载 image 及其 owner
        joinedload(Person.individual), # Eager load individual
        joinedload(Person.correction_logs).load_only(CorrectionLog.correction_type, CorrectionLog.correction_date) # 急切加载纠正日志
    ).filter(Person.uuid.in_(person_uuids))
    persons_data = query.order_by(Person.id.desc()).all()
    return [_prepare_person_for_schema(p, db, current_user_id) for p in persons_data]

# 新增：更新人物关注状态
def update_person_followed_status(
    db: Session,
    person_id: int,
    is_followed: bool
) -> Optional[Person]:
    db_person = db.query(Person).filter(Person.id == person_id).first()
    if db_person:
        db_person.is_followed = is_followed
        db.commit()
        db.refresh(db_person)
        logger.info(f"人物 {db_person.uuid} 的关注状态已更新为 {is_followed}。")
    return db_person

# 新增 FollowedPerson CRUD operations
def create_followed_person(db: Session, individual_id: int, user_id: int) -> Optional[FollowedPerson]:
    db_followed_person = FollowedPerson(
        individual_id=individual_id,
        user_id=user_id,
        follow_time=datetime.now(pytz.timezone('Asia/Shanghai'))
    )
    db.add(db_followed_person)
    db.commit()
    db.refresh(db_followed_person)
    return db_followed_person

def get_followed_person(db: Session, individual_id: int, user_id: int) -> Optional[FollowedPerson]:
    return db.query(FollowedPerson).filter(
        FollowedPerson.individual_id == individual_id,
        FollowedPerson.user_id == user_id,
        FollowedPerson.unfollow_time == None # Only return currently followed persons
    ).first()

def get_followed_persons_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[FollowedPerson]:
    return db.query(FollowedPerson).options(
        joinedload(FollowedPerson.individual),
        joinedload(FollowedPerson.user)
    ).filter(
        FollowedPerson.user_id == user_id,
        FollowedPerson.unfollow_time == None
    ).order_by(FollowedPerson.follow_time.desc()).offset(skip).limit(limit).all()

def get_total_followed_persons_count_by_user(db: Session, user_id: int) -> int:
    return db.query(FollowedPerson).filter(
        FollowedPerson.user_id == user_id,
        FollowedPerson.unfollow_time == None
    ).count()

def unfollow_person(db: Session, individual_id: int, user_id: int) -> Optional[FollowedPerson]:
    db_followed_person = get_followed_person(db, individual_id, user_id) # Get the currently followed record
    if db_followed_person:
        db_followed_person.unfollow_time = datetime.now(pytz.timezone('Asia/Shanghai'))
        db.add(db_followed_person)
        db.commit()
        db.refresh(db_followed_person)
    return db_followed_person# 辅助函数：在 FollowedPerson 表中查找或创建关注记录
def toggle_follow_status(
    db: Session,
    individual_id: int,
    user_id: int,
    is_followed: bool
) -> bool:
    """
    切换人物的关注状态。如果 is_followed 为 True，则创建关注记录；
    如果为 False，则标记现有关注记录为取消关注。
    """
    existing_follow = db.query(FollowedPerson).filter(
        FollowedPerson.individual_id == individual_id,
        FollowedPerson.user_id == user_id,
        FollowedPerson.unfollow_time == None
    ).first()

    if is_followed:
        if not existing_follow:
            # 如果不存在关注记录，则创建新记录
            create_followed_person(db, individual_id, user_id)
            logger.info(f"用户 {user_id} 关注了人物 {individual_id}。")
            return True
        else:
            logger.info(f"用户 {user_id} 已关注人物 {individual_id}，无需重复操作。")
            return False
    else: # is_followed is False
        if existing_follow:
            # 如果存在关注记录，则标记为取消关注
            unfollow_person(db, individual_id, user_id)
            logger.info(f"用户 {user_id} 取消关注了人物 {individual_id}。")
            return True
        else:
            logger.info(f"用户 {user_id} 未关注人物 {individual_id}，无需取消关注。")
            return False

# 辅助函数：检查人物是否被关注
def check_is_followed(db: Session, individual_id: int, user_id: int) -> bool:
    """检查指定人物是否被当前用户关注。"""
    return db.query(FollowedPerson).filter(
        FollowedPerson.individual_id == individual_id,
        FollowedPerson.user_id == user_id,
        FollowedPerson.unfollow_time == None
    ).first() is not None

# 新增：获取个体注册图片路径的函数
def get_enrollment_images_by_individual_id(db: Session, individual_id: int):
    """
    根据 Individual ID 获取所有关联的人物注册图片路径和 UUID。
    这些图片现在存储在 `database/full_frames` 目录下。
    """
    logger.info(f"正在获取 Individual ID {individual_id} 的注册图片。")
    # 获取与该 individual_id 关联的所有人物 (Person) 记录
    # 筛选出 is_verified 为 True 的人物，这些是主动注册的图片
    persons = db.query(Person).filter(Person.individual_id == individual_id, Person.is_verified == True).all()

    image_paths_and_uuids = []
    for person in persons:
        # 确保 full_frame_image_path 存在且是有效的相对路径
        if person.full_frame_image_path:
            # person.full_frame_image_path 已经是一个相对路径，格式应为 "database/full_frames/model_name/analysis_type/uuid/filename.jpg"
            # 需要将其转换为服务器的绝对路径
            # 从 full_frame_image_path 提取相对于 backend/ 的路径
            relative_to_backend_path = person.full_frame_image_path

            full_path = os.path.join(settings.BASE_DIR, "backend", relative_to_backend_path)
            
            if os.path.exists(full_path):
                image_paths_and_uuids.append({
                    "path": full_path,
                    "uuid": person.uuid, # 返回人物的UUID
                    "image_uuid": person.image.uuid if person.image else None # 如果需要，也可以返回 Image 的 UUID
                })
            else:
                logger.warning(f"人物 {person.uuid} 的完整帧图片文件 {full_path} 不存在。")
        else:
            logger.warning(f"人物 {person.uuid} 没有 full_frame_image_path。")
            
    logger.info(f"为 Individual ID {individual_id} 找到 {len(image_paths_and_uuids)} 张注册图片。")
    return image_paths_and_uuids

# 新增：删除注册图片 ( enrollment_images ) 的函数
def delete_enrollment_image_by_uuid(db: Session, individual_id: int, image_uuid: str, current_user_id: int, current_user_role: str) -> bool:
    """
    删除指定人物的注册图片，包括文件和数据库记录。
    只有该图片的所有者或管理员可以删除。
    """
    db_image = db.query(Image).filter(Image.uuid == image_uuid).first() # 重新插入这行
    logger.debug(f"DEBUG: Attempting to delete enrollment image {image_uuid} for individual {individual_id}.") # Added debug log
    if not db_image:
        logger.warning(f"删除注册图片失败：未找到 UUID 为 {image_uuid} 的图片。")
        return False

    # 1. 验证图片是否属于指定人物档案
    # 注册图片通常不直接关联 Individual ID，而是通过 Person->Individual 关联
    # 但这里假设 image 记录本身有 owner_id，我们可以通过它和 Individual 的 owner 来验证。
    # 最直接的验证方式是：该图片是否曾用于创建与此 Individual 关联的 Person
    # 或者，简单地允许 Individual 的 owner 或管理员删除其档案下的图片

    # 确保图片存在且属于某个 Individual (通过 Person 间接关联)
    # 或者，更简单：如果图片是注册图片 (上传到 ENROLLMENT_IMAGES_DIR)，那么它的 owner_id 必须与 current_user_id 匹配，或者 current_user 是管理员。
    # 由于 Image 模型有 owner_id，我们可以直接用它来检查权限。

    # 权限检查：只有管理员或图片所有者可以删除
    if current_user_role != "admin" and db_image.owner_id != current_user_id:
        logger.warning(f"用户 {current_user_id} (角色: {current_user_role}) 试图删除非其所有的注册图片 {image_uuid}。")
        return False

    # 2. 删除本地文件
    # 获取图片关联的 Person 记录，以便找到其 full_frame_image_path
    person_to_delete = db.query(Person).filter(Person.image_id == db_image.id).first()
    if person_to_delete and person_to_delete.full_frame_image_path:
        # full_frame_image_path 是相对于 backend 目录的路径，例如 database/full_frames/general_detection/image/<person_uuid>/<filename>.jpg
        full_frame_image_absolute_path = os.path.join(settings.BASE_DIR, "backend", person_to_delete.full_frame_image_path)
        
        # 图片文件现在存储在以 person_uuid 命名的目录下，我们需要删除这个目录
        image_dir_to_delete = os.path.dirname(full_frame_image_absolute_path)

        if os.path.isdir(image_dir_to_delete):
            try:
                shutil.rmtree(image_dir_to_delete)
                logger.info(f"已删除注册图片目录: {image_dir_to_delete}")
            except OSError as e:
                logger.error(f"删除注册图片目录 {image_dir_to_delete} 时出错: {e}", exc_info=True)
                return False
        else:
            logger.warning(f"注册图片目录 {image_dir_to_delete} 不存在或不是目录，跳过删除。")
    else:
        logger.warning(f"未能找到与图片 {image_uuid} 关联的人物记录或其 full_frame_image_path。跳过文件删除。")

    # 3. 删除数据库记录
    try:
        # 在删除 Image 记录之前，删除所有关联到此 Image 的 Person 记录
        associated_persons = db.query(Person).filter(Person.image_id == db_image.id).all()
        for person in associated_persons:
            db.delete(person) # 直接删除关联的 Person 记录
        db.commit() # 提交 Person 的删除
        logger.info(f"已删除 {len(associated_persons)} 个人物与图片 {image_uuid} 的关联记录。")

        db.delete(db_image)
        db.commit()
        logger.info(f"注册图片 {image_uuid} 及其数据库记录已成功删除。") # Moved this line to here

        # 检查该 individual 是否还有任何已验证的注册图片 (is_verified=True 的 Person 记录)
        remaining_verified_persons_count = db.query(Person).filter(
            Person.individual_id == individual_id,
            Person.is_verified == True
        ).count()
        logger.debug(f"DEBUG: Individual {individual_id} has {remaining_verified_persons_count} remaining verified persons.") # Added debug log

        if remaining_verified_persons_count == 0:
            # 如果不再有已验证的人物，则取消所有关注该人物的用户
            followed_persons_to_unfollow = db.query(FollowedPerson).filter(
                FollowedPerson.individual_id == individual_id,
                FollowedPerson.unfollow_time == None
            ).all()
            logger.debug(f"DEBUG: Found {len(followed_persons_to_unfollow)} followed persons to unfollow for Individual {individual_id}.") # Added debug log

            for fp in followed_persons_to_unfollow:
                unfollow_person(db, fp.individual_id, fp.user_id) # 直接调用 unfollow_person
                logger.info(f"由于人物档案 {individual_id} 不再有注册图片，已自动取消用户 {fp.user_id} 对其的关注。")
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"删除注册图片 {image_uuid} 的数据库记录时发生错误: {e}", exc_info=True)
        return False


# 新增：获取个体注册图片的函数，现在返回包含 UUID 的列表
def get_enrollment_images_by_individual_id(db: Session, individual_id: int):
    """
    根据 Individual ID 获取所有关联的人物注册图片路径和 UUID。
    这些图片现在存储在 `database/full_frames` 目录下。
    """
    logger.info(f"正在获取 Individual ID {individual_id} 的注册图片。")
    # 获取与该 individual_id 关联的所有人物 (Person) 记录
    # 筛选出 is_verified 为 True 的人物，这些是主动注册的图片
    persons = db.query(Person).filter(Person.individual_id == individual_id, Person.is_verified == True).all()

    image_paths_and_uuids = []
    for person in persons:
        # 确保 full_frame_image_path 存在且是有效的相对路径
        if person.full_frame_image_path:
            # person.full_frame_image_path 已经是一个相对路径，格式应为 "database/full_frames/model_name/analysis_type/uuid/filename.jpg"
            # 需要将其转换为服务器的绝对路径
            # 从 full_frame_image_path 提取相对于 backend/ 的路径
            relative_to_backend_path = person.full_frame_image_path

            full_path = os.path.join(settings.BASE_DIR, "backend", relative_to_backend_path)
            
            if os.path.exists(full_path):
                image_paths_and_uuids.append({
                    "path": full_path,
                    "uuid": person.uuid, # 返回人物的UUID
                    "image_uuid": person.image.uuid if person.image else None # 如果需要，也可以返回 Image 的 UUID
                })
            else:
                logger.warning(f"人物 {person.uuid} 的完整帧图片文件 {full_path} 不存在。")
        else:
            logger.warning(f"人物 {person.uuid} 没有 full_frame_image_path。")
            
    logger.info(f"为 Individual ID {individual_id} 找到 {len(image_paths_and_uuids)} 张注册图片。")
    return image_paths_and_uuids

# 新增：更新人物信息
def update_person(db: Session, person_id: int, person_update_data: schemas.PersonUpdate) -> Optional[Person]:
    """
    更新指定人物的信息。
    """
    db_person = db.query(Person).filter(Person.id == person_id).first()
    if db_person:
        update_data = person_update_data.model_dump(exclude_unset=True) # 使用 exclude_unset=True 只更新提供的字段
        for key, value in update_data.items():
            if key == "feature_vector" and isinstance(value, list): # 特征向量是列表，需要转为JSON字符串
                setattr(db_person, key, json.dumps(value))
            elif key == "pose_keypoints" and isinstance(value, list): # 姿态关键点是列表，需要转为JSON字符串
                setattr(db_person, key, json.dumps(value))
            elif key == "face_feature_vector" and isinstance(value, list): # 人脸特征向量是列表，需要转为JSON字符串
                setattr(db_person, key, json.dumps(value))
            elif key == "clothing_attributes" and isinstance(value, dict): # 衣着属性是字典，需要转为JSON字符串
                setattr(db_person, key, json.dumps(value))
            elif key == "gait_feature_vector" and isinstance(value, list): # 步态特征向量是列表，需要转为JSON字符串
                setattr(db_person, key, json.dumps(value))
            else:
                setattr(db_person, key, value)
        
        db.add(db_person) # Mark for update
        db.commit()
        db.refresh(db_person)
    return db_person

