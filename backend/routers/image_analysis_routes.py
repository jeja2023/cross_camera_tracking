from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Annotated, List
import os
import shutil
import logging
import uuid
import json
import aiofiles # Added for async file handling

from backend.auth import get_current_active_user
from backend.ml_services import ml_logic
from backend.config import settings
from backend.database_conn import get_db, SessionLocal
from .. import crud, schemas
from ..ml_services.ml_tasks import analyze_image_task # 导入新的 Celery 任务

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/image_analysis",
    tags=["Image Analysis"],
    responses={404: {"description": "Not found"}},
)

@router.post("/upload_image", response_model=schemas.TaskStatusResponse) # 更改返回模型为 TaskStatusResponse
async def upload_image_for_analysis(
    file: Annotated[UploadFile, File(...)],
    current_user: Annotated[dict, Depends(get_current_active_user)],
    # db: SessionLocal = Depends(get_db) # 不再需要直接的 db 会话，因为任务在 Celery 中执行
):
    logger.info(f"用户 {current_user.username} 正在上传图片进行解析：{file.filename}")
    
    # 生成一个 UUID 作为原始图片的文件名
    original_image_uuid = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1] # 获取原始文件扩展名
    final_original_filename = f"{original_image_uuid}{file_extension}"

    # 构建原始图片保存的绝对路径 (使用新的分层结构)
    # 对于原始图片，它属于“通用检测”模型，且是“image”分析类型，UUID 是其本身的 UUID
    original_image_save_dir = settings.get_parsed_image_path(
        base_dir=settings.DATABASE_FULL_FRAMES_DIR,
        model_name="general_detection", 
        analysis_type="image",
        uuid=original_image_uuid # 使用原始图片的 UUID 作为最终目录
    )
    original_image_full_path = os.path.join(original_image_save_dir, final_original_filename)

    # 确保目标目录存在 (get_parsed_image_path 内部会创建，这里可以移除)
    # os.makedirs(settings.DATABASE_FULL_FRAMES_IMAGE_ANALYSIS_DIR, exist_ok=True)

    try:
        # 将上传的文件内容写入到目标路径
        with open(original_image_full_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"上传的原始图片已保存到: {original_image_full_path}")

        # 将图片分析任务分发到 Celery
        task = analyze_image_task.delay(
            original_image_uuid=original_image_uuid,
            original_image_full_path=original_image_full_path,
            original_image_filename=final_original_filename,
            current_user_id=current_user.id # 修正：将 current_user["id"] 改为 current_user.id
        )

        logger.info(f"图片分析任务已分发到 Celery，任务 ID: {task.id}")
        return schemas.TaskStatusResponse(task_id=task.id, status=task.status) # 返回任务 ID 和当前状态
    except Exception as e:
        logger.error(f"分发图片解析任务失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"图片解析任务分发失败: {e}") 

# 新增路由以查询任务状态
@router.get("/tasks/{task_id}", response_model=schemas.TaskStatusResponse, summary="获取 Celery 任务状态")
async def get_image_analysis_task_status(
    task_id: str,
    current_user: Annotated[dict, Depends(get_current_active_user)]
):
    logger.info(f"用户 {current_user.username} 正在查询图片分析任务状态：{task_id}")
    task = analyze_image_task.AsyncResult(task_id)

    # 打印任务的关键信息，用于调试
    logger.info(f"DEBUG: get_image_analysis_task_status - Task ID: {task_id}, Task Name: {task.name}, State: {task.state}, Result: {task.result}")

    # 如果是 retrain_reid_model_task，需要特殊处理 SUCCESS 状态下的 False 结果
    if task.name == 'backend.ml_services.ml_tasks.retrain_reid_model_task' and task.state == 'SUCCESS' and task.result is False:
        logger.info(f"Re-ID Retrain Task {task_id} state is SUCCESS but result is False, interpreting as FAILED.")
        response = {'task_id': task.id, 'status': 'FAILED', 'progress': 100, 'message': '模型再训练失败：类别数量不足或训练未完成。', 'error': 'Insufficient categories or training not completed.'}
    elif task.state == 'PENDING':
        response = {'task_id': task.id, 'status': task.state, 'progress': 0, 'message': '任务等待中...'}
    elif task.state == 'PROGRESS':
        progress_info = task.info.get('progress', 0) if isinstance(task.info, dict) else 0 # Safely get progress
        status_message = task.info.get('message', '处理中...') if isinstance(task.info, dict) else '处理中...' # 从 message 键获取状态信息
        response = {'task_id': task.id, 'status': task.state, 'progress': progress_info, 'message': status_message}
    elif task.name == 'backend.ml_services.ml_tasks.retrain_reid_model_task' and task.state == 'SUCCESS':
        # 专门处理 Re-ID 模型再训练任务的成功状态
        if task.result is True:
            response = {'task_id': task.id, 'status': 'SUCCESS', 'progress': 100, 'message': '模型再训练成功。'}
        else: # task.result 是 False，表示训练逻辑失败
            response = {'task_id': task.id, 'status': 'FAILED', 'progress': 100, 'message': '模型再训练失败：请检查后端日志。'}
        response['result'] = task.result # 确保 result 字段是布尔值 True/False
    elif task.state == 'SUCCESS':
        # 修正：根据 task.result 的类型来获取 image_id
        if isinstance(task.result, dict):
            # 如果 task.result 是字典，从其内部获取 image_id
            image_id = task.result.get('image_id')
        else:
            # 如果 task.result 不是字典，则认为它本身就是 image_id
            image_id = task.result

        if image_id is None:
            logger.error(f"查询任务 {task.id} 结果时，未从 task.result 中获取到有效的 image_id。Task result: {task.result}") 
            result_data = {"error": "Image ID not found in task result"}
            message_info = "图片ID未找到。"
        else:
            db = SessionLocal() # 获取数据库会话
            try:
                # 调用现有的获取图片解析结果的API逻辑，但直接从crud获取数据
                image_db_entry = crud.get_image(db, image_id=image_id)
                if not image_db_entry:
                    logger.error(f"查询任务 {task.id} 结果时，未找到图片 ID {image_id}。")
                    result_data = {"error": "Image not found in database"}
                    message_info = "图片数据未找到。"
                else:
                    # 复用 get_image_analysis_results 路由中的逻辑来构建返回的数据结构
                    persons_from_db = crud.get_persons_by_image_id(db, image_id=image_db_entry.id)

                    original_image_info = {
                        "id": image_db_entry.id,
                        "uuid": image_db_entry.uuid,
                        "filename": image_db_entry.filename,
                        "created_at": image_db_entry.created_at.isoformat(),
                        "full_frame_image_path": image_db_entry.file_path.replace(os.sep, '/') if image_db_entry.file_path else None,
                        "person_count": image_db_entry.person_count
                    }

                    analyzed_persons_for_response = []
                    for person in persons_from_db:
                        person_data = {
                            "uuid": person.uuid,
                            "crop_image_path": person.crop_image_path.replace(os.sep, '/') if person.crop_image_path else None,
                            "face_image_path": person.face_image_path.replace(os.sep, '/') if person.face_image_path else None,
                            "gait_image_path": person.gait_image_path.replace(os.sep, '/') if person.gait_image_path else None,
                            "feature_vector_preview": person.feature_vector[:5] if person.feature_vector else [],
                            "timestamp": person.created_at.isoformat() if person.created_at else None,
                            "clothing_attributes": person.clothing_attributes,
                            "pose_keypoints": person.pose_keypoints,
                            "name": person.name,
                            "id_card": person.id_card,
                            "is_verified": person.is_verified,
                            "verified_by_user_id": person.verified_by_user_id,
                            "verification_date": person.verification_date.isoformat() if person.verification_date else None
                        }
                        analyzed_persons_for_response.append(person_data)

                    result_data = {
                        "status": "success",
                        "original_image_info": original_image_info,
                        "analyzed_persons": analyzed_persons_for_response,
                        "message": "成功获取图片解析结果。"
                    }
                    message_info = "任务成功完成。"
            except Exception as e:
                logger.error(f"获取图片解析结果失败 (任务ID: {task.id}, 图片ID: {image_id}): {e}", exc_info=True)
                result_data = {"error": f"Failed to retrieve analysis results: {e}"}
                message_info = f"获取解析结果失败: {e}"
            finally:
                db.close()
        
        response = {'task_id': task.id, 'status': task.state, 'progress': 100, 'message': message_info, 'result': result_data}
    elif task.state == 'FAILURE':
        error_message = task.info.get('error', '任务失败。') if isinstance(task.info, dict) else str(task.info) # Safely get error or convert info to string
        response = {'task_id': task.id, 'status': task.state, 'progress': 0, 'message': error_message}
    else: # 处理其他中间状态 (STARTED, RETRY, etc.)
        info_dict = task.info if isinstance(task.info, dict) else {}
        # 尝试从 info 字典中获取 message，如果不存在，则默认为当前状态的描述
        message_info = info_dict.get('message', f'任务状态: {task.state.lower()}...') # 从 message 键获取信息
        progress_info = info_dict.get('progress', 0)
        response = {'task_id': task.id, 'status': task.state, 'progress': progress_info, 'message': message_info}
    
    logger.info(f"任务 {task_id} 状态: {response['status']}, 进度: {response.get('progress', 'N/A')}")
    logger.info(f"请求任务 {task_id} 的状态。 响应数据: {response}") # 额外添加日志，打印整个响应
    return JSONResponse(content=response)

@router.get("/history", response_model=List[schemas.ImageResponse], summary="获取图片解析历史列表") # 返回 ImageResponse 列表
async def get_image_analysis_history_api(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    db: SessionLocal = Depends(get_db),
    skip: int = 0, # 新增分页参数
    limit: int = 100 # 新增分页参数
):
    logger.info(f"用户 {current_user.username} 正在请求图片解析历史列表，skip={skip}, limit={limit}。")
    
    # 根据用户角色调用 crud 函数，并传递 owner_id
    if current_user.role == "admin":
        history = crud.get_image_analysis_history(db, skip=skip, limit=limit) # 管理员查看所有历史
    else:
        history = crud.get_image_analysis_history(db, owner_id=current_user.id, skip=skip, limit=limit) # 普通用户查看自己的历史

    logger.info(f"成功获取 {len(history)} 条图片解析历史记录。")
    # crud.get_image_analysis_history 已经返回了字典列表，可以直接返回
    return history

@router.get("/history/count", summary="获取图片解析历史总数")
async def get_image_analysis_history_count_api(
    current_user: Annotated[dict, Depends(get_current_active_user)],
    db: SessionLocal = Depends(get_db)
):
    logger.info(f"用户 {current_user.username} 正在请求图片解析历史总数。")
    if current_user.role == "admin":
        total_count = crud.get_total_image_analysis_history_count(db)
    else:
        total_count = crud.get_total_image_analysis_history_count(db, owner_id=current_user.id)
    return {"total_count": total_count}

@router.delete("/{image_uuid}", summary="删除特定图片的所有解析数据和文件")
async def delete_image_analysis_data_api(
    image_uuid: str,
    current_user: Annotated[dict, Depends(get_current_active_user)],
    db: SessionLocal = Depends(get_db)
):
    logger.info(f"用户 {current_user.username} 正在请求删除图片 {image_uuid} 的所有解析数据和文件。")

    # 只有管理员或图片所有者可以删除 (现在由 crud.delete_image_analysis_data 来检查所有权和管理员权限)
    # if current_user.role != "admin":
    #     raise HTTPException(status_code=403, detail="只有管理员可以删除图片解析数据。")
    
    success = crud.delete_image_analysis_data(db, image_uuid=image_uuid, current_user_id=current_user.id, current_user_role=current_user.role)

    if success:
        logger.info(f"图片 {image_uuid} 的所有解析数据和文件已成功删除。")
        return JSONResponse(content={"message": "图片及相关解析数据已成功删除。"})
    else:
        logger.error(f"删除图片 {image_uuid} 的数据时发生错误或未找到。")
        raise HTTPException(status_code=500, detail="删除图片解析数据失败或未找到相关数据。")

@router.get("/results/{image_uuid}", summary="获取特定图片的解析结果")
async def get_image_analysis_results(
    image_uuid: str,
    current_user: Annotated[dict, Depends(get_current_active_user)],
    db: SessionLocal = Depends(get_db)
):
    logger.info(f"用户 {current_user.username} 正在请求图片 {image_uuid} 的解析结果。")

    # 首先通过 image_uuid 获取 Image 对象
    image_db_entry = crud.get_image_by_uuid(db, uuid=image_uuid)
    if not image_db_entry:
        logger.error(f"请求图片 {image_uuid} 的解析结果时，未找到该图片。")
        raise HTTPException(status_code=404, detail="未找到该图片的解析结果，或图片不存在。")

    logger.info(f"从数据库获取的原始图片路径: {image_db_entry.file_path}")

    # 然后使用 Image 对象的 ID 来获取相关的 Person 记录
    persons_from_db = crud.get_persons_by_image_id(db, image_id=image_db_entry.id)

    # 获取原始图片信息 (从 Image 记录中提取)
    original_image_info = {
        "id": image_db_entry.id,
        "uuid": image_db_entry.uuid,
        "filename": image_db_entry.filename,
        "created_at": image_db_entry.created_at.isoformat(),
        "full_frame_image_path": image_db_entry.file_path.replace(os.sep, '/') if image_db_entry.file_path else None, # 替换路径分隔符
        "person_count": image_db_entry.person_count
    }

    logger.info(f"发送给前端的原始图片 URL: {original_image_info['full_frame_image_path']}")

    analyzed_persons_for_response = []
    for person in persons_from_db:
        # 新增日志：打印从数据库读取的原始路径
        logger.info(f"DEBUG: 从数据库读取的 person.crop_image_path: {person.crop_image_path}")
        logger.info(f"DEBUG: 从数据库读取的 person.face_image_path: {person.face_image_path}")
        logger.info(f"DEBUG: 从数据库读取的 person.gait_image_path: {person.gait_image_path}")
        person_data = {
            "uuid": person.uuid,
            "crop_image_path": person.crop_image_path.replace(os.sep, '/') if person.crop_image_path else None,
            # full_frame_image_path 已在顶层 original_image_info 中提供，此处不再重复
            "face_image_path": person.face_image_path.replace(os.sep, '/') if person.face_image_path else None,
            "gait_image_path": person.gait_image_path.replace(os.sep, '/') if person.gait_image_path else None,
            "feature_vector_preview": person.feature_vector[:5] if person.feature_vector else [],
            "timestamp": person.created_at.isoformat() if person.created_at else None,
            "clothing_attributes": person.clothing_attributes,
            "pose_keypoints": person.pose_keypoints,
            "name": person.name,
            "id_card": person.id_card,
            "is_verified": person.is_verified,
            "verified_by_user_id": person.verified_by_user_id,
            "verification_date": person.verification_date.isoformat() if person.verification_date else None
        }
        # 新增日志：打印添加到响应前的最终路径
        logger.info(f"DEBUG: 发送给前端的 person_data.crop_image_path: {person_data['crop_image_path']}")
        logger.info(f"DEBUG: 发送给前端的 person_data.face_image_path: {person_data['face_image_path']}")
        logger.info(f"DEBUG: 发送给前端的 person_data.gait_image_path: {person_data['gait_image_path']}")
        analyzed_persons_for_response.append(person_data)
    
    logger.info(f"成功获取图片 {image_uuid} 的解析结果，检测到 {len(analyzed_persons_for_response)} 个人物。")
    logger.info(f"即将发送给前端的 original_image_info: {original_image_info}")
    logger.info(f"即将发送给前端的 analyzed_persons_for_response: {analyzed_persons_for_response}")
    # 新增：详细打印发送给前端的 analyzed_persons 信息
    for i, person in enumerate(analyzed_persons_for_response):
        logger.info(f"DEBUG: analyzed_persons_for_response[{i}] - uuid: {person.get('uuid', 'N/A')}, crop_image_path: {person.get('crop_image_path', 'N/A')}, face_image_path: {person.get('face_image_path', 'N/A')}")

    return JSONResponse(content={
        "status": "success",
        "original_image_info": original_image_info,
        "analyzed_persons": analyzed_persons_for_response,
        "message": "成功获取图片解析结果。"
    }) 