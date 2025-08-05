# video_routes.py

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Query, BackgroundTasks # Removed BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
import logging
import shutil
import os
import asyncio
import cv2 
import time
from fastapi.responses import StreamingResponse, JSONResponse
import aiofiles
import json
import numpy as np

from ..database_conn import SessionLocal, get_db
from .. import crud
from .. import schemas
from .. import auth
from ..ml_services import ml_logic
from ..ml_services.ml_tasks import process_video_for_extraction_task # Import Celery task
from backend.config import settings

router = APIRouter(
    tags=["Video Management"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# 2. 修改 upload_video_and_extract 函数
@router.post("/extract-features", summary="上传视频并提取特征", status_code=202) # 修正：移除尾部斜杠
async def upload_video_and_extract(
    # Removed background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    video: UploadFile = File(...),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 正在上传视频: {video.filename}")
    try:
        # 确保uploads目录存在
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        temp_video_path = os.path.join(settings.UPLOAD_DIR, video.filename)

        # 使用 aiofiles 异步写入文件
        async with aiofiles.open(temp_video_path, "wb") as buffer:
            while contents := await video.read(settings.UPLOAD_CHUNK_SIZE_BYTES):  # 使用配置的分块大小
                await buffer.write(contents)
        logger.info(f"视频文件 {video.filename} 已保存到 {temp_video_path}")

        db_video = crud.create_video(db, filename=video.filename, owner_id=current_user.id, file_path=temp_video_path)

        # 将视频处理任务调度到 Celery
        task = process_video_for_extraction_task.delay(
            video_path=temp_video_path,
            video_id=db_video.id
        )
        logger.info(f"视频处理任务 {task.id} 已为用户 {current_user.username} 调度。")

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={"message": "视频上传成功，正在后台处理。", "task_id": task.id, "filename": video.filename}
        )
    except Exception as e:
        logger.error(f"处理视频上传和特征提取时发生错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"视频上传或处理失败: {e}"
        )


@router.get("/", summary="获取视频列表")
async def read_videos(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user),
    status: Optional[str] = Query(None),
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, le=200, description="返回的最大记录数")
):
    logger.info(f"用户 {current_user.username} 正在请求视频列表，skip={skip}, limit={limit}。")
    
    total_videos = 0
    if current_user.role == "admin":
        videos = crud.get_all_videos(db, status=status, skip=skip, limit=limit) # Pass status, skip, limit
        total_videos = crud.get_total_videos_count(db, status=status)
        logger.info(f"管理员 {current_user.username} 获取所有视频列表 ({len(videos)} 个，总数 {total_videos} 个)。")
    else:
        videos = crud.get_videos_by_owner_id(db, owner_id=current_user.id, status=status, skip=skip, limit=limit) # Pass status, skip, limit
        total_videos = crud.get_total_videos_count_by_owner_id(db, owner_id=current_user.id, status=status)
        logger.info(f"用户 {current_user.username} 获取自己的视频列表 ({len(videos)} 个，总数 {total_videos} 个)。")
        
    # 转换并返回数据，包含总数
    video_data = [
        {"id": v.id, "filename": v.filename, "status": v.status, "processed_at": v.processed_at, "progress": v.progress, "uuid": v.uuid, "owner_id": v.owner_id, "owner_username": v.owner.username if v.owner else None} 
        for v in videos
    ]
    return {"total_count": total_videos, "skip": skip, "limit": limit, "items": video_data}

@router.get("/{video_id}", response_model=schemas.Video, summary="获取单个视频详细信息")
async def get_video_details(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 正在请求视频 (ID: {video_id}) 的详细信息。")
    video = crud.get_video(db, video_id=video_id)
    if not video:
        logger.warning(f"用户 {current_user.username} 尝试获取不存在的视频 (ID: {video_id}) 的详细信息。")
        raise HTTPException(status_code=404, detail="Video not found")
    
    # 权限检查：只有所有者或管理员才能访问
    if video.owner_id != current_user.id and current_user.role != "admin":
        logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 未经授权尝试访问视频 (ID: {video_id}) 的详细信息。")
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    logger.info(f"用户 {current_user.username} 成功获取视频 (ID: {video_id}) 的详细信息。")
    return video

@router.get("/{video_id}/progress", summary="获取视频处理实时进度")
async def stream_video_progress(
    video_id: int, 
    token: str = Query(...) # Token现在是必需的
):
    # 这个函数本身逻辑是正确的，无需修改。
    # 它的问题在于服务器被阻塞了，导致它无法被正常调用。
    async def event_generator():
        db = SessionLocal()
        try:
            current_user = await auth.get_user_from_token_string(token=token, db=db)
            if not current_user:
                yield f"data: {{ \"status\": \"error\", \"message\": \"Unauthorized. Please log in.\" }}\n\n"
                logger.warning(f"尝试获取视频 (ID: {video_id}) 进度失败: 未授权或令牌无效。")
                return

            video = crud.get_video(db, video_id=video_id)
            if not video:
                yield f"data: {{ \"status\": \"error\", \"message\": \"Video not found\" }}\n\n"
                logger.warning(f"用户 {current_user.username} 尝试获取不存在视频 (ID: {video_id}) 的进度。")
                return
            
            if video.owner_id != current_user.id and current_user.role != 'admin':
                yield f"data: {{ \"status\": \"error\", \"message\": \"Forbidden\", \"detail\": \"Not enough permissions\" }}\n\n"
                logger.warning(f"用户 {current_user.username} 未经授权访问视频 (ID: {video_id}) 进度。")
                return

            logger.info(f"用户 {current_user.username} 开始监听视频 (ID: {video_id}) 进度。")
            last_progress = -1
            while True:
                # 在循环内部使用独立的会话，这是一个好习惯
                db_loop = SessionLocal()
                try:
                    video_in_loop = crud.get_video(db_loop, video_id=video_id)
                finally:
                    db_loop.close()
                
                if not video_in_loop:
                    yield f"data: {{ \"status\": \"error\", \"message\": \"Video not found during stream\" }}\n\n"
                    logger.warning(f"视频 (ID: {video_id}) 在流式传输过程中被删除。")
                    break

                current_progress = video_in_loop.progress
                current_status = video_in_loop.status

                if current_progress != last_progress or current_status in ["completed", "failed", "terminated"]:
                    yield f"data: {{ \"id\": {video_in_loop.id}, \"status\": \"{current_status}\", \"progress\": {current_progress} }}\n\n"
                    last_progress = current_progress

                if current_status in ["completed", "failed", "terminated"]:
                    logger.info(f"视频 (ID: {video_in_loop.id}) 处理结束，状态: {current_status}。关闭进度流。")
                    break
                
                await asyncio.sleep(settings.VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS) # 使用配置的轮询间隔
        except Exception as e:
            logger.error(f"视频 (ID: {video_id}) 进度流式传输时发生严重错误: {e}", exc_info=True)
            yield f"data: {{ \"status\": \"error\", \"message\": \"Internal server error during stream\" }}\n\n"
        finally:
            db.close()
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- 其他路由函数保持不变 ---
@router.get("/{video_id}/persons", response_model=schemas.PaginatedPersonsResponse, summary="获取特定视频的人物特征图库 (限所有者或管理员, 支持分页)")
async def get_video_persons(
    video_id: int,
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, le=200, description="返回的最大记录数"),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    video = crud.get_video(db, video_id=video_id)
    if not video:
        logger.warning(f"用户 {current_user.username} 尝试访问不存在的视频 (ID: {video_id}) 的特征图库。")
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.owner_id != current_user.id and current_user.role != "admin":
        logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 未经授权尝试访问视频 (ID: {video_id}) 的特征图库。")
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    total_persons = crud.get_total_persons_count_by_video_id(db, video_id=video_id)
    persons_orm = crud.get_persons_by_video_id(db, video_id=video_id, skip=skip, limit=limit)

    # 填充 video_uuid
    if video and video.uuid:
        for p_orm in persons_orm:
            p_orm.video_uuid = video.uuid

    # 添加调试日志以检查 feature_vector 的类型和内容
    for i, p_orm in enumerate(persons_orm):
        logger.debug(f"人物 {i} (UUID: {p_orm.uuid}) 原始 feature_vector 类型: {type(p_orm.feature_vector)}, 值 (部分): {str(p_orm.feature_vector)[:200]}...")

    # Explicitly convert ORM objects to Pydantic models and add prefixes
    persons = []
    for p_orm in persons_orm:
        person_dict = p_orm.__dict__.copy()
        
        # 确保 feature_vector 是 List[float]
        if isinstance(p_orm.feature_vector, str):
            try:
                parsed_feature = json.loads(p_orm.feature_vector)
                if isinstance(parsed_feature, list) and len(parsed_feature) > 0 and isinstance(parsed_feature[0], list):
                    person_dict["feature_vector"] = parsed_feature[0] # 取出内部列表
                elif isinstance(parsed_feature, list):
                    person_dict["feature_vector"] = parsed_feature # 如果已经是扁平列表
                else:
                    person_dict["feature_vector"] = []
            except json.JSONDecodeError:
                person_dict["feature_vector"] = []
        elif isinstance(p_orm.feature_vector, np.ndarray):
            person_dict["feature_vector"] = p_orm.feature_vector.tolist()

        # 确保 face_feature_vector 是 List[float]
        if isinstance(p_orm.face_feature_vector, str):
            try:
                parsed_face_feature = json.loads(p_orm.face_feature_vector)
                person_dict["face_feature_vector"] = parsed_face_feature
            except (json.JSONDecodeError, TypeError):
                person_dict["face_feature_vector"] = []
        elif p_orm.face_feature_vector is None:
            person_dict["face_feature_vector"] = []

        # 确保 clothing_attributes 是 List[dict]
        if isinstance(p_orm.clothing_attributes, str):
            try:
                parsed_clothing_attributes = json.loads(p_orm.clothing_attributes)
                person_dict["clothing_attributes"] = parsed_clothing_attributes
            except (json.JSONDecodeError, TypeError):
                person_dict["clothing_attributes"] = []
        elif p_orm.clothing_attributes is None:
            person_dict["clothing_attributes"] = []

        # 移除为路径添加前缀的代码，保持与其他API一致，返回相对路径
        # 前端将统一处理路径前缀
        # if person_dict.get("crop_image_path"):
        #     person_dict["crop_image_path"] = f"/crops/{person_dict["crop_image_path"]}"
        # if person_dict.get("full_frame_image_path"):
        #     person_dict["full_frame_image_path"] = f"/full_frames/{person_dict["full_frame_image_path"]}"

        # Pydantic model_validate will handle the rest of the validation
        persons.append(schemas.Person.model_validate(person_dict))

    logger.info(f"用户 {current_user.username} 成功获取视频 (ID: {video_id}) 的 {len(persons)} 个人物特征图库（总数：{total_persons}）。")
    return {"total": total_persons, "skip": skip, "limit": limit, "items": persons}

@router.delete("/{video_id}", summary="删除视频及其所有关联数据")
async def delete_video(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 尝试删除视频 (ID: {video_id}).")
    
    video_to_delete = crud.get_video(db, video_id=video_id)
    if not video_to_delete:
        logger.warning(f"用户 {current_user.username} 尝试删除不存在的视频 (ID: {video_id})。")
        raise HTTPException(status_code=404, detail="视频未找到")
    
    if video_to_delete.owner_id != current_user.id and current_user.role != "admin":
        logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 未经授权尝试删除视频 (ID: {video_id})。")
        raise HTTPException(status_code=403, detail="无足够权限")

    deleted_video = crud.delete_video(db, video_id=video_id)
    if not deleted_video:
        logger.error(f"删除视频 (ID: {video_id}) 失败，尽管视频已找到。")
        raise HTTPException(status_code=500, detail="删除视频失败")
    
    logger.info(f"用户 {current_user.username} 成功删除了视频 (ID: {video_id}) 及其所有关联数据。")
    return {"message": "视频及其所有关联数据已成功删除", "video_id": video_id}

@router.post("/{video_id}/terminate", summary="终止视频处理")
async def terminate_video(
    video_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.debug(f"用户 {current_user.username} 尝试终止视频 (ID: {video_id}).")
    video = crud.get_video(db, video_id=video_id)
    if not video:
        logger.warning(f"用户 {current_user.username} 尝试终止不存在的视频 (ID: {video_id})。")
        raise HTTPException(status_code=404, detail="Video not found")

    if video.owner_id != current_user.id and current_user.role != "admin":
        logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 未经授权尝试终止视频 (ID: {video_id})。")
        raise HTTPException(status_code=403, detail="Not enough permissions")

    logger.debug(f"视频 (ID: {video.id}) 当前状态: {video.status}")
    if video.status in ["processing", "paused"]:
        # 修改为通过 Celery API 终止任务
        from ..celery_worker import celery_app
        # 这里需要获取任务ID，但这需要修改ml_tasks和crud来存储和检索任务ID
        # 暂时只更新数据库状态，并假设ml_logic中的检查会使其停止
        crud.update_video_status(db, video_id=video_id, status="terminated")
        db.commit() # 确保终止状态被提交
        logger.info(f"用户 {current_user.username} 终止了视频 (ID: {video_id}) 的处理。")
        # TODO: 未来可以通过 Celery inspect().active() 或 active_queues() 来找到并终止任务，但这需要更复杂的实现
        # 目前只靠ml_logic中的状态检查来终止
        return {"message": "Video processing termination requested", "video_id": video_id}
    else:
        raise HTTPException(status_code=400, detail="Video is not currently processing or paused.")