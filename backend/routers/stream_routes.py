from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, Dict
import logging
import os
import shutil
import cv2
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from starlette.responses import RedirectResponse
from datetime import datetime
import uuid
from ..ml_services.ml_tasks import process_live_stream_task
from backend.config import settings
import time
import io
import csv
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from PIL import Image as PILImage
from urllib.parse import quote

# 导入 stream_manager
from backend.stream_manager import get_frame_from_redis

# 导入您项目中的其他模块
from ..database_conn import SessionLocal, get_db
from .. import crud
from .. import schemas
from .. import auth

router = APIRouter(
    tags=["Video Stream Management"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.post("/start", summary="启动实时视频流解析")
async def start_video_stream(
    request: schemas.StreamStartRequest,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_advanced_role_user)
):
    # 移除原有的 role != "admin" 检查，因为新的依赖已经处理了权限。
    # if current_user.role != "admin":
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admins can start video streams.")

    # 每次都创建新的视频流记录
    stream_uuid = str(uuid.uuid4())
    stream_name = request.stream_name or f"Live Stream - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # 根据提供的URL类型确定要使用的流URL
    stream_url_to_use: str
    if request.rtsp_url:
        stream_url_to_use = request.rtsp_url
    elif request.api_stream_url:
        stream_url_to_use = request.api_stream_url
    else:
        # 尽管Pydantic模型已经处理了，这里作为额外的安全检查
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="必须提供rtsp_url或api_stream_url其中一个。")

    db_stream = crud.create_stream(
        db,
        name=stream_name,
        stream_url=stream_url_to_use, # 使用确定的URL
        owner_id=current_user.id,
        stream_uuid=stream_uuid,
    )
    db.commit()
    db.refresh(db_stream)

    try:
        task_id = process_live_stream_task.delay(stream_id=db_stream.id, stream_url=db_stream.stream_url).id
        logger.info(f"视频流处理任务 {task_id} 已为视频流 {db_stream.stream_uuid} 调度。")

        return {"message": "视频流解析已启动，正在后台处理。", "stream_id": db_stream.stream_uuid, "db_stream_id": db_stream.id, "celery_task_id": task_id}
    except Exception as e:
        logger.error(f"启动视频流 {db_stream.stream_uuid} 失败: {e}", exc_info=True)
        crud.update_stream_status(db, stream_id=db_stream.id, status="failed")
        db.commit()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="启动视频流解析时发生内部错误")

@router.post("/stop/{stream_uuid}", summary="停止实时视频流解析")
async def stop_video_stream(
    stream_uuid: str,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_advanced_role_user) # 允许高级用户和管理员
):
    db_stream = crud.get_stream_by_uuid(db, stream_uuid=stream_uuid)
    if not db_stream:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="指定视频流未找到或已停止")

    # 权限检查：
    # 管理员可以停止任何流
    # 高级用户只能停止自己创建的流
    if current_user.role == "admin":
        pass # 管理员拥有所有权限
    elif current_user.role == "advanced" and db_stream.owner_id == current_user.id:
        pass # 高级用户可以停止自己拥有的流
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无足够权限停止此视频流")

    if db_stream.status in ["processing", "active"]:
        logger.info(f"停止视频流 {stream_uuid}: 当前状态为 '{db_stream.status}'。正在更新状态为 'stopped'。")
        crud.update_stream_status(db, stream_id=db_stream.id, status="stopped")
        db.commit()
        logger.info(f"视频流 {stream_uuid} 的处理已请求停止。确认数据库状态已更新为 'stopped'。") # Updated log message
        return {"message": "视频流解析已请求停止", "stream_uuid": stream_uuid, "status": "stopped"} # Return 'stopped' explicitly
    else:
        logger.error(f"尝试停止视频流 {stream_uuid} 失败：当前状态为 '{db_stream.status}'。")
        raise HTTPException(status_code=400, detail=f"视频流当前状态为 '{db_stream.status}'，无法停止。")

@router.post("/resume/{stream_uuid}", summary="恢复已停止或失败的实时视频流解析")
async def resume_video_stream(
    stream_uuid: str,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_advanced_role_user)
):
    # 移除原有的 role != "admin" 检查，因为新的依赖已经处理了权限。
    # if current_user.role != "admin":
    #     raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="只有管理员可以恢复视频流。")

    db_stream = crud.get_stream_by_uuid(db, stream_uuid=stream_uuid)
    if not db_stream:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="指定视频流未找到。")

    if db_stream.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无足够权限恢复此视频流。")

    if db_stream.status in ["stopped", "failed", "inactive", "completed", "terminated"]:
        try:
            # 重新调度 Celery 任务
            task_id = process_live_stream_task.delay(stream_id=db_stream.id, stream_url=db_stream.stream_url).id
            crud.update_stream_status(db, stream_id=db_stream.id, status="active") # 更新状态为活跃或处理中
            db.commit()
            logger.info(f"视频流处理任务 {task_id} 已为视频流 {db_stream.stream_uuid} 重新调度/恢复。")
            return {"message": "视频流解析已成功恢复。", "stream_uuid": db_stream.stream_uuid, "celery_task_id": task_id}
        except Exception as e:
            logger.error(f"恢复视频流 {db_stream.stream_uuid} 失败: {e}", exc_info=True)
            crud.update_stream_status(db, stream_id=db_stream.id, status="failed")
            db.commit()
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail="恢复视频流解析时发生内部错误。")
    else:
        raise HTTPException(status_code=400, detail=f"视频流当前状态为 '{db_stream.status}'，无法恢复。")

@router.get("/feed/{stream_uuid}", summary="获取实时视频流帧 (MJPEG)")
async def video_stream_feed(
    stream_uuid: str,
    db: Session = Depends(get_db)
):
    db_stream = crud.get_stream_by_uuid(db, stream_uuid=stream_uuid)
    if not db_stream:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="视频流未找到")

    # 实时流处理逻辑：从 Redis 获取最新帧
    if db_stream.status in ["processing", "active"]:
        def generate_mjpeg_frames():
            while True:
                # 检查流状态，如果流已停止或终止，则停止生成帧
                current_db_stream = crud.get_stream_by_uuid(db, stream_uuid=stream_uuid)
                if not current_db_stream or current_db_stream.status in ["stopped", "terminated", "failed", "completed"]:
                    logger.info(f"视频流 {stream_uuid} 状态变为 {current_db_stream.status if current_db_stream else '不存在'}，停止MJPEG传输。")
                    break

                frame_data = get_frame_from_redis(stream_uuid)
                if frame_data:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                # 即使没有新帧，也发送一个空的帧或等待，避免前端断开连接
                else:
                    pass

                # 控制帧率，避免过高刷新率导致CPU或带宽占用过高
                time.sleep(1 / settings.MJPEG_STREAM_FPS) # 使用配置的MJPEG帧率

        return StreamingResponse(generate_mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    # 处理非实时流状态（已完成、已停止等），返回保存的MP4或提示信息
    elif db_stream.status in ["completed", "stopped", "terminated"]:
        logger.info(f"视频流 {stream_uuid} 状态为 {db_stream.status}。其 output_video_path 为: {db_stream.output_video_path}") # 新增日志
        if db_stream.output_video_path:
            # output_video_path 现在应该已经是相对于 /saved_streams/ 的路径
            # 例如: "<stream_uuid>/<stream_uuid>.mp4"
            # 直接构建重定向 URL
            redirect_url = f"/saved_streams/{db_stream.output_video_path}" # 确保使用正斜杠
            logger.info(f"视频流 {stream_uuid} 状态为 {db_stream.status}，重定向到保存的视频: {redirect_url}")
            return RedirectResponse(url=redirect_url, status_code=status.HTTP_303_SEE_OTHER)
        else:
            return JSONResponse(status_code=200, content={"message": "视频流已停止或终止，未找到保存的视频文件。", "status": db_stream.status})
    elif db_stream.status == "failed":
        raise HTTPException(status_code=500, detail="视频流处理失败")
    elif db_stream.status == "inactive":
        return JSONResponse(status_code=200, content={"message": "视频流未激活或正在等待处理。", "status": db_stream.status})
    else:
        raise HTTPException(status_code=400, detail=f"视频流状态异常: {db_stream.status}")

@router.get("/saved", response_model=schemas.PaginatedStreamsResponse, summary="获取所有已保存的实时视频流列表") # 修改响应模型以支持分页
async def get_saved_live_streams(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="跳过的记录数"), # 添加 skip 参数
    limit: int = Query(20, ge=1, le=100, description="返回的最大记录数"), # 添加 limit 参数
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    # 强制使会话中的所有对象失效，确保从数据库获取最新数据
    db.expire_all()

    total_streams_count = 0
    try:
        if current_user.role == "admin":
            total_streams_count = crud.get_total_streams_count(db) # 获取总数
            saved_streams = crud.get_all_streams(db, skip=skip, limit=limit) # 传递分页参数
            logger.info(f"管理员 {current_user.username} 获取所有已保存视频流列表 ({len(saved_streams)}/{total_streams_count} 个)。")
        else:
            total_streams_count = crud.get_total_streams_count_by_owner_id(db, owner_id=current_user.id) # 获取总数
            saved_streams = crud.get_streams_by_owner_id(db, owner_id=current_user.id, skip=skip, limit=limit) # 传递分页参数
            logger.info(f"用户 {current_user.username} 获取自己的已保存视频流列表 ({len(saved_streams)}/{total_streams_count} 个)。")

        response_data = []
        for s in saved_streams:
            try:
                processed_output_video_path = None
                if s.output_video_path:
                    logger.debug(f"原始 output_video_path: {s.output_video_path}")
                    # 确保路径使用正斜杠
                    normalized_path = s.output_video_path.replace(os.sep, '/')
                    logger.debug(f"规范化后的路径: {normalized_path}")

                    # 移除 settings.SAVED_STREAMS_DIR 前缀，并确保路径以 /saved_streams/ 开头
                    # os.path.relpath 可以计算相对路径，但我们需要确保它是相对于挂载点的
                    saved_streams_base_url = settings.SAVED_STREAMS_DIR.replace(os.sep, '/')
                    if normalized_path.startswith(saved_streams_base_url):
                        # 提取相对于 SAVED_STREAMS_DIR 的部分
                        relative_to_saved_streams = normalized_path[len(saved_streams_base_url):].lstrip('/')
                        processed_output_video_path = f"/saved_streams/{relative_to_saved_streams}"
                    else:
                        # 如果路径不是以SAVED_STREAMS_DIR开头，说明之前保存的就是相对路径，或者格式不正确
                        # 鉴于ml_tasks中已确保保存的是相对路径，这里直接拼接
                        processed_output_video_path = f"/saved_streams/{s.output_video_path.replace(os.sep, '/')}"
                        # 不再需要警告，因为这是预期的格式
                        # logger.warning(f"视频流 {s.stream_uuid}: output_video_path ({s.output_video_path}) 格式异常，未从 SAVED_STREAMS_DIR 开始。尝试直接拼接。")

                response_data.append({
                    "id": s.id,
                    "name": s.name,
                    "stream_url": s.stream_url,
                    "status": s.status,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "last_processed_at": s.last_processed_at.isoformat() if s.last_processed_at else None,
                    "stream_uuid": s.stream_uuid,
                    "output_video_path": processed_output_video_path, # 使用处理后的路径
                    "is_active": s.status in ["processing", "active"], # 正确设置 is_active 字段
                    "owner_id": s.owner_id # 添加 owner_id
                })
            except Exception as e:
                logger.error(f"处理视频流记录 ID {s.id} ('{s.name}') 时出错: {e}。该记录将被忽略。")

    except Exception as e:
        logger.error(f"获取已保存视频流时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="获取视频流列表失败")

    return {"total": total_streams_count, "skip": skip, "limit": limit, "items": response_data} # 返回分页信息

@router.get("/streams/{stream_uuid}", response_model=schemas.StreamSchema, summary="通过UUID获取单个视频流信息")
async def get_stream_info_by_uuid(
    stream_uuid: str,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    db_stream = crud.get_stream_by_uuid(db, stream_uuid=stream_uuid)
    if not db_stream:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="视频流未找到")

    # 权限检查：只有管理员或流的拥有者可以获取信息
    if current_user.role == "admin" or db_stream.owner_id == current_user.id:
        processed_output_video_path = None
        if db_stream.output_video_path:
            normalized_path = db_stream.output_video_path.replace(os.sep, '/')
            saved_streams_base_url = settings.SAVED_STREAMS_DIR.replace(os.sep, '/')
            if normalized_path.startswith(saved_streams_base_url):
                relative_to_saved_streams = normalized_path[len(saved_streams_base_url):].lstrip('/')
                processed_output_video_path = f"/saved_streams/{relative_to_saved_streams}"
            else:
                processed_output_video_path = f"/saved_streams/{db_stream.output_video_path.replace(os.sep, '/')}"

        return schemas.StreamSchema(
            id=db_stream.id,
            name=db_stream.name,
            stream_url=db_stream.stream_url,
            status=db_stream.status,
            created_at=db_stream.created_at,
            last_processed_at=db_stream.last_processed_at,
            stream_uuid=db_stream.stream_uuid,
            output_video_path=processed_output_video_path,
            is_active=db_stream.status in ["processing", "active"],
            owner_id=db_stream.owner_id
        )
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无足够权限获取此视频流信息")

@router.post("/ping/{stream_uuid}", summary="Ping 实时流以保持活跃")
async def ping_live_stream(
    stream_uuid: str,
    current_user: schemas.User = Depends(auth.get_current_active_advanced_role_user),
    db: Session = Depends(get_db)
):
    db_stream = crud.get_stream_by_uuid(db, stream_uuid=stream_uuid)
    if not db_stream:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="视频流未找到")

    if db_stream.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无足够权限")

    crud.update_stream_last_processed_at(db, stream_id=db_stream.id)
    db.commit()
    return {"message": "Stream pinged successfully", "stream_uuid": stream_uuid, "status": db_stream.status}

@router.delete("/delete/{stream_id}", summary="删除视频流及其所有相关数据和文件")
async def delete_live_stream(
    stream_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_advanced_role_user) # 允许高级用户和管理员
):
    # 权限检查：确保用户有权删除此视频流
    db_stream = crud.get_stream(db, stream_id=stream_id)
    if not db_stream:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="指定视频流未找到。")

    # 管理员可以删除任何流
    # 高级用户只能删除自己创建的流
    if current_user.role == "admin":
        pass # 管理员拥有所有权限
    elif current_user.role == "advanced" and db_stream.owner_id == current_user.id:
        pass # 高级用户可以删除自己拥有的流
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无足够权限删除此视频流。")

    deleted_stream = crud.delete_stream(db, stream_id=stream_id)
    if not deleted_stream:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="删除视频流失败。")

    logger.info(f"视频流 (ID: {stream_id}) 及其相关数据已成功删除。")
    return {"message": f"视频流 {stream_id} 及其所有相关数据和文件已成功删除。", "stream_id": stream_id}

@router.get("/results", summary="根据 stream_uuid 获取实时流解析结果")
async def get_stream_results(
    stream_uuid: str = Query(..., description="要查询的实时流的 UUID"),
    skip: int = Query(0, ge=0, description="跳过的记录数"), # 将 last_id 改为 skip
    limit: int = Query(20, ge=1, le=100, description="返回的最大记录数"), # 调整默认 limit 为 20 以匹配前端
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_advanced_role_user)
):
    # 权限检查已由 Depends(auth.get_current_active_advanced_role_user) 处理，这里无需额外判断

    # 根据 stream_uuid 获取流信息，以便获取 db_stream_id
    db_stream = crud.get_stream_by_uuid(db, stream_uuid=stream_uuid)
    if not db_stream:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="指定视频流未找到。")

    # 验证用户权限：只有流的拥有者和管理员可以获取结果
    if db_stream.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无足够权限访问此视频流结果。")

    try:
        # 获取与该视频流关联的所有人物，并确保 feature_vector 被正确解析
        persons = crud.get_features_for_stream_by_stream_id(db, stream_id=db_stream.id, skip=skip, limit=limit) # 传递 skip 参数
        total_persons_count = crud.get_total_persons_count_by_stream_id(db, stream_id=db_stream.id)

        results_items = []
        for person in persons:
            # crud.get_features_for_stream_by_stream_id 已经返回了 schemas.Person 对象
            # 所以可以直接添加到结果列表中，无需再次构造字典和验证
            # 在这里为图片路径添加前缀，使其可直接用于前端
            person_dict = person.model_dump() # 将 Pydantic 模型转换为字典，以便修改
            # person_dict["crop_image_path"] = f"/crops/{person_dict["crop_image_path"]}"
            # person_dict["full_frame_image_path"] = f"/full_frames/{person_dict["full_frame_image_path"]}"
            if person_dict.get("face_image_path"):
                person_dict["face_image_path"] = f"/crops/{person_dict["face_image_path"]}"
            if person_dict.get("gait_image_path"):
                person_dict["gait_image_path"] = f"/crops/{person_dict["gait_image_path"]}"
            
            results_items.append(schemas.Person(**person_dict)) # 转换为 Person Pydantic 模型

        logger.info(f"成功获取视频流 {stream_uuid} 的 {len(results_items)} 条结果 (总数: {total_persons_count})。")
        return schemas.PaginatedPersonsResponse(
            total=total_persons_count,
            skip=skip, # 这里应该是实际跳过的记录数，如果使用 last_id 作为偏移量
            limit=limit,
            items=results_items
        )

    except Exception as e:
        logger.error(f"获取视频流 {stream_uuid} 的结果时出错: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="无法获取视频流解析结果")