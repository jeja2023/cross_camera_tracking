from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import os
from datetime import datetime

from ..database_conn import get_db
from .. import crud
from .. import schemas
from .. import auth
from backend.celery_app import celery_app # 导入 celery_app
from backend.config import settings # 导入 settings
from ..schemas import ModelConfig, MessageResponse # 导入 ModelConfig 和 MessageResponse

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(auth.get_current_active_admin_user)], # 确保只有管理员可以访问这些路由
    responses={403: {"description": "Not enough permissions"}},
)

logger = logging.getLogger(__name__)

@router.get("/model-configs", response_model=ModelConfig, summary="获取模型参数配置")
async def get_model_configs(
    db: Session = Depends(get_db)
):
    """
    获取系统当前的模型参数配置。
    """
    # 从数据库中加载所有配置
    db_configs = crud.get_all_system_configs(db)
    configs = {}

    # 遍历所有可配置项，从数据库中获取值，并进行适当的类型转换
    for key in settings.CONFIGURABLE_SETTINGS_KEYS:
        db_value = db_configs.get(key)
        if db_value is None: # 如果数据库中没有该配置，则使用 settings 中的默认值
            if hasattr(settings, key):
                value = getattr(settings, key)
            else:
                logger.warning(f"配置项 {key} 既不在数据库中，也不在 settings 中定义。跳过。")
                continue
        else:
            value = db_value # 从数据库获取的值是字符串

        # 根据键名进行类型转换和路径转换
        if key in ["DETECTION_MODEL_FILENAME", "REID_MODEL_FILENAME", "ACTIVE_REID_MODEL_PATH", 
                   "POSE_MODEL_FILENAME", "FACE_DETECTION_MODEL_FILENAME", 
                   "FACE_RECOGNITION_MODEL_FILENAME", "GAIT_RECOGNITION_MODEL_FILENAME", 
                   "CLOTHING_ATTRIBUTE_MODEL_FILENAME"]:
            # 这些是模型文件名，数据库中存储的就是文件名，直接使用
            configs[key] = str(value) if value else ""
        elif key in ["REID_INPUT_WIDTH", "REID_INPUT_HEIGHT", "FEATURE_DIM", "FACE_FEATURE_DIM", "GAIT_FEATURE_DIM",
                    "K1", "K2", "GAIT_SEQUENCE_LENGTH", "MIN_FACE_WIDTH", "MIN_FACE_HEIGHT", 
                    "TRACKER_MIN_HITS", "TRACKER_TRACK_BUFFER", "VIDEO_PROCESSING_FRAME_RATE",
                    "STREAM_PROCESSING_FRAME_RATE", "VIDEO_COMMIT_BATCH_SIZE", "PERSON_CLASS_ID",
                    "EXCEL_EXPORT_MAX_IMAGES", "EXCEL_EXPORT_IMAGE_SIZE_PX", "EXCEL_EXPORT_ROW_HEIGHT_PT",
                    "MJPEG_STREAM_FPS", "FAISS_SEARCH_K", "REID_TRAIN_BATCH_SIZE"]:
            configs[key] = int(value) if value else 0
        elif key in ["REID_WEIGHT", "FACE_WEIGHT", "GAIT_WEIGHT", "LAMBDA_VALUE", 
                    "HUMAN_REVIEW_CONFIDENCE_THRESHOLD", "IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE",
                    "ENROLLMENT_MIN_PERSON_CONFIDENCE", "FACE_DETECTION_CONFIDENCE_THRESHOLD",
                    "TRACKER_PROXIMITY_THRESH", "TRACKER_APPEARANCE_THRESH", "TRACKER_HIGH_THRESH",
                    "TRACKER_LOW_THRESH", "TRACKER_NEW_TRACK_THRESH", "DETECTION_CONFIDENCE_THRESHOLD",
                    "REID_TRAIN_LEARNING_RATE", "REALTIME_COMPARISON_THRESHOLD",
                    "GLOBAL_SEARCH_MIN_CONFIDENCE"]: # 新增：全局搜索最小置信度
            configs[key] = float(value) if value else 0.0
        elif key == "DEVICE_TYPE": # DEVICE_TYPE 是字符串
            configs[key] = str(value) if value else "cpu"
        elif key == "FAISS_METRIC": # FAISS_METRIC 是字符串
            configs[key] = str(value) if value else "L2"
        else:
            configs[key] = str(value) # 默认作为字符串

    return ModelConfig(**configs)

@router.put("/model-configs", response_model=MessageResponse, summary="更新模型参数配置")
async def update_model_configs(
    model_configs: ModelConfig,
    db: Session = Depends(get_db)
):
    """
    更新系统模型参数配置。这将更新数据库中的配置。
    请注意，部分模型参数的更改可能需要重启后端服务才能完全生效。
    """
    configs_to_update = {}
    for key, value in model_configs.model_dump(exclude_unset=True).items():
        # 特殊处理模型路径，将其从文件名转换为完整路径以便存储
        if key in ["DETECTION_MODEL_FILENAME", "REID_MODEL_FILENAME", "POSE_MODEL_FILENAME", 
                   "FACE_DETECTION_MODEL_FILENAME", "FACE_RECOGNITION_MODEL_FILENAME", 
                   "GAIT_RECOGNITION_MODEL_FILENAME", "CLOTHING_ATTRIBUTE_MODEL_FILENAME"]:
            # 如果是模型文件名，我们需要将其转换为完整的模型路径来更新 SystemConfig
            # 注意：这里假设前端只传递文件名，后端负责拼接 MODELS_DIR
            # 如果前端传递的是完整路径，则不需要拼接
            if key == "ACTIVE_REID_MODEL_PATH": # ACTIVE_REID_MODEL_PATH 是一个完整路径
                configs_to_update[key] = str(value)
            else:
                configs_to_update[key] = os.path.join(settings.MODELS_DIR, str(value)) if value else ""
        elif key in ["REID_INPUT_WIDTH", "REID_INPUT_HEIGHT", "FEATURE_DIM", "FACE_FEATURE_DIM", "GAIT_FEATURE_DIM",
                    "K1", "K2", "GAIT_SEQUENCE_LENGTH", "MIN_FACE_WIDTH", "MIN_FACE_HEIGHT", 
                    "TRACKER_MIN_HITS", "TRACKER_TRACK_BUFFER", "VIDEO_PROCESSING_FRAME_RATE",
                    "STREAM_PROCESSING_FRAME_RATE", "VIDEO_COMMIT_BATCH_SIZE", "PERSON_CLASS_ID",
                    "EXCEL_EXPORT_MAX_IMAGES", "EXCEL_EXPORT_IMAGE_SIZE_PX", "EXCEL_EXPORT_ROW_HEIGHT_PT",
                    "MJPEG_STREAM_FPS", "FAISS_SEARCH_K", "REID_TRAIN_BATCH_SIZE"]:
            configs_to_update[key] = str(int(value)) # 转换为整数后转字符串
        elif key in ["REID_WEIGHT", "FACE_WEIGHT", "GAIT_WEIGHT", "LAMBDA_VALUE", 
                    "HUMAN_REVIEW_CONFIDENCE_THRESHOLD", "IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE",
                    "ENROLLMENT_MIN_PERSON_CONFIDENCE", "FACE_DETECTION_CONFIDENCE_THRESHOLD",
                    "TRACKER_PROXIMITY_THRESH", "TRACKER_APPEARANCE_THRESH", "TRACKER_HIGH_THRESH",
                    "TRACKER_LOW_THRESH", "TRACKER_NEW_TRACK_THRESH", "DETECTION_CONFIDENCE_THRESHOLD",
                    "REID_TRAIN_LEARNING_RATE"]:
            configs_to_update[key] = str(float(value)) # 转换为浮点数后转字符串
        else:
            configs_to_update[key] = str(value)

    # 批量更新或创建配置项
    crud.set_system_configs(db, configs_to_update)
    
    # 更新 settings 对象的内存值，这样在不重启服务的情况下，新的请求可以读取到更新后的配置
    # 但需要注意，这只影响当前运行的 Python 进程，如果部署了多个 worker，需要更复杂的同步机制
    settings.reload_from_db(db) # 调用 settings 的方法从数据库重新加载

    logger.info("模型参数配置已更新。")
    return {"message": "模型参数配置更新成功。部分更改可能需要重启服务才能完全生效。"}

@router.get("/logs", response_model=schemas.LogResponse, summary="获取系统日志") # 更改response_model
async def get_system_logs(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, le=1000, description="每页记录数"),
    level: Optional[str] = Query(None, description="日志级别 (INFO, WARNING, ERROR等)"),
    start_time: Optional[datetime] = Query(None, description="开始时间 (YYYY-MM-DDTHH:MM:SS 或 YYYY-MM-DD)"),
    end_time: Optional[datetime] = Query(None, description="结束时间 (YYYY-MM-DDTHH:MM:SS 或 YYYY-MM-DD)"),
    keyword: Optional[str] = Query(None, description="日志消息关键词")
):
    # 调用CRUD操作从数据库获取日志
    logs_data = crud.get_logs(
        db,
        skip=skip,
        limit=limit,
        level=level,
        start_time=start_time,
        end_time=end_time,
        keyword=keyword
    )
    return logs_data

@router.get("/users", response_model=List[schemas.User], summary="获取所有用户列表")
async def get_all_users(db: Session = Depends(get_db), is_active: Optional[bool] = Query(None, description="按活跃状态筛选用户，True为活跃，False为非活跃")):
    users = crud.get_all_users(db, is_active=is_active)
    return users

@router.post("/users", response_model=schemas.User, status_code=201, summary="创建新用户")
async def create_new_user(
    user: schemas.UserCreate,
    db: Session = Depends(get_db)
):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="用户名已被注册")
    return crud.create_user(db=db, user=user)

@router.put("/users/{user_id}/role", response_model=schemas.User, summary="更新用户角色")
async def update_user_role(
    user_id: int,
    role_update: schemas.UserRoleUpdate,
    db: Session = Depends(get_db)
):
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="用户未找到")
    
    # 确保只能更新为合法角色
    if role_update.role not in [schemas.UserRole.USER, schemas.UserRole.ADMIN, schemas.UserRole.ADVANCED]:
        raise HTTPException(status_code=400, detail="无效的用户角色")

    return crud.update_user_role(db=db, user_id=user_id, new_role=role_update.role)

@router.put("/users/{user_id}/status", response_model=schemas.User, summary="更新用户活跃状态")
async def update_user_status(
    user_id: int,
    status_update: schemas.UserStatusUpdate,
    db: Session = Depends(get_db)
):
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="用户未找到")
    
    updated_user = crud.update_user_status(db=db, user_id=user_id, is_active=status_update.is_active)
    if not updated_user:
        raise HTTPException(status_code=500, detail="更新用户状态失败")
    return updated_user

@router.delete("/users/{user_id}", status_code=204, summary="删除用户")
async def delete_existing_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    db_user = crud.get_user(db, user_id=user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="用户未找到")
    
    crud.delete_user(db=db, user_id=user_id)
    return {"message": "用户删除成功"}

@router.post("/update-reid-model-path", summary="更新当前激活的 Re-ID 模型路径")
async def update_active_reid_model_path(
    model_path: str = Query(..., description="新的 Re-ID 模型文件的完整路径"),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_admin_user)
):
    """
    管理员接口：更新数据库中当前激活的 Re-ID 模型路径。
    此操作会使系统后续加载模型时使用新的路径。旧的模型实例将失效，需要重新加载。
    """
    try:
        # 验证路径是否存在且是文件
        if not os.path.exists(model_path) or not os.path.isfile(model_path):
            raise HTTPException(status_code=400, detail=f"模型路径无效或文件不存在: {model_path}")

        # 更新数据库配置
        updated_config = crud.set_system_config(db, key="active_reid_model_path", value=model_path)
        if not updated_config:
            raise HTTPException(status_code=500, detail="更新数据库配置失败。")
        
        logger.info(f"管理员 {current_user.username} 已将活跃 Re-ID 模型路径更新为: {model_path}")
        return {"message": f"活跃 Re-ID 模型路径已更新为: {model_path}"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"更新活跃 Re-ID 模型路径失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误，无法更新模型路径。")

@router.post("/trigger-retrain-reid", summary="触发 Re-ID 模型再训练任务")
async def trigger_reid_model_retrain(
    model_version: str = Query("v2", description="新模型的版本 (例如: v2, v3)"),
    person_uuids: Optional[List[str]] = Query(None, description="可选：指定用于再训练的人物 UUID 列表"),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_admin_user)
):
    """
    管理员接口：触发 Celery 后台任务，使用标记为再训练的数据对 Re-ID 模型进行微调。
    如果提供了 `person_uuids`，则只使用这些人物的数据进行再训练。
    """
    try:
        # 将 person_uuids 传递给 Celery 任务
        task = celery_app.send_task('backend.ml_services.ml_tasks.retrain_reid_model_task', args=[model_version, person_uuids])
        logger.info(f"管理员 {current_user.username} 已触发 Re-ID 模型再训练任务 (ID: {task.id})。新模型版本: {model_version}, 指定人物UUIDs: {person_uuids}")
        return {"message": "Re-ID 模型再训练任务已触发", "task_id": task.id}
    except Exception as e:
        logger.error(f"触发 Re-ID 模型再训练任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="内部服务器错误，无法触发再训练任务。")

@router.get("/task-status/{task_id}", response_model=schemas.TaskStatusResponse, summary="查询 Celery 任务状态和进度")
async def get_celery_task_status(
    task_id: str,
    current_user: schemas.User = Depends(auth.get_current_active_admin_user) # 只有管理员可以查询任务状态
):
    """
    管理员接口：根据任务ID查询 Celery 后台任务的状态和进度。
    """
    task = celery_app.AsyncResult(task_id)
    
    response_data = {
        "task_id": task.id,
        "status": task.status, # PENDING, STARTED, PROGRESS, SUCCESS, FAILURE
        "progress": 0, # 默认进度
        "message": "", # 默认消息
        "result": None
    }

    # 总是尝试从 task.info 中获取 progress 和 status，无论任务状态如何
    # Celery 任务的 meta 数据通常存储在 task.info 中
    if task.info and isinstance(task.info, dict):
        if 'progress' in task.info:
            response_data["progress"] = task.info['progress']
        if 'status' in task.info: # 注意这里是 'status' 而不是 'message'
            response_data["message"] = task.info['status']
        # 如果有错误信息，也从 info 中提取
        if 'error' in task.info:
            response_data["message"] = f"任务失败: {task.info['error']}"


    if task.state == 'PENDING':
        if not response_data["message"]: # 如果 meta 中没有提供更具体的消息
            response_data["message"] = "任务等待中..."
    elif task.state == 'STARTED':
        if not response_data["message"]: # 如果 meta 中没有提供更具体的消息
            response_data["message"] = "任务已启动。"
    elif task.state == 'PROGRESS': # 这个分支可以保持，但现在优先级较低
        if not response_data["message"]:
            response_data["message"] = "任务正在进行中..."
    elif task.state == 'SUCCESS':
        response_data["progress"] = 100
        if not response_data["message"]: # 如果 meta 中没有提供更具体的消息
            response_data["message"] = "任务已成功完成。"
        response_data["result"] = task.result # 成功结果
    elif task.state == 'FAILURE':
        if not response_data["message"]: # 如果 meta 中没有提供更具体的消息
            response_data["message"] = f"任务失败: {task.info.get('error', '未知错误')}"
        # 失败时也尝试获取进度，但优先级低于 task.info 中的 progress
        if 'progress' not in response_data:
            response_data["progress"] = task.info.get('progress', 0)
    elif task.state == 'RETRY':
        if not response_data["message"]:
            response_data["message"] = "任务正在重试中..."
        if 'progress' not in response_data:
            response_data["progress"] = task.info.get('progress', 0)
    elif task.state == 'REVOKED':
        if not response_data["message"]:
            response_data["message"] = "任务已被撤销。"
        response_data["progress"] = 0
    elif task.state == 'TERMINATED':
        if not response_data["message"]:
            response_data["message"] = "任务已被终止。"
        response_data["progress"] = 0
    elif task.state == 'SKIPPED':
        response_data["progress"] = 100
        if not response_data["message"]:
            response_data["message"] = "任务已跳过 (没有有效数据)。"

    # 使用 schemas.TaskStatusResponse 进行验证和返回
    return schemas.TaskStatusResponse(**response_data)