from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json

from .. import crud, schemas, auth
from ..database_conn import get_db, Person
import logging
from backend.ml_services.ml_tasks import retrain_reid_model_task # 新增导入再训练任务

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/hitl", # Human-in-the-Loop
    tags=["Human-in-the-Loop"]
)

@router.get("/persons/unverified", response_model=schemas.PaginatedPersonsResponse)
def get_unverified_persons(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role not in [schemas.UserRole.ADMIN, schemas.UserRole.ADVANCED]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admin or advanced users can access this resource.")
    
    persons = crud.get_unverified_persons(db, skip=skip, limit=limit)
    logger.info(f"get_unverified_persons: Retrieved {len(persons)} persons from CRUD layer.") # 新增日志
    # 打印每个人物的 UUID
    for p in persons:
        logger.info(f"get_unverified_persons: Person UUID from CRUD: {p.uuid}") # 新增日志

    total = crud.get_total_unverified_persons_count(db) # 调用新的总数获取函数
    return {"total": total, "skip": skip, "limit": limit, "items": persons}

@router.post("/persons/{person_uuid}/verify", response_model=schemas.Person)
def verify_person(
    person_uuid: str, # 改为接收 UUID
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role not in [schemas.UserRole.ADMIN, schemas.UserRole.ADVANCED]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admin or advanced users can perform verification.")

    # 通过 UUID 查找 Person 对象
    db_person_by_uuid = crud.get_person_by_uuid_obj(db, uuid=person_uuid) # crud.get_person_by_uuid 返回 schema.Person，这里需要 ORM 对象
    if not db_person_by_uuid:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")

    db_person = crud.update_person_verification_status(
        db=db,
        person_id=db_person_by_uuid.id, # 传入 ORM 对象的 ID
        is_verified=True,
        verified_by_user_id=current_user.id,
        correction_details="", 
        marked_for_retrain=False 
    )
    if not db_person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found after verification attempt.")
    return crud._prepare_person_for_schema(db_person, db)

@router.post("/persons/{person_uuid}/correct", response_model=schemas.Person)
def correct_person(
    person_uuid: str, # 改为接收 UUID
    correction_data: schemas.CorrectionLogCreate,
    request: Request, # 将 Request 放在这里，在所有无默认值参数之后，有默认值参数之前
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role not in [schemas.UserRole.ADMIN, schemas.UserRole.ADVANCED]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admin or advanced users can submit corrections.")
    
    # 通过 UUID 查找 Person 对象
    db_person_by_uuid = crud.get_person_by_uuid_obj(db, uuid=person_uuid) # 获取 ORM 对象
    if not db_person_by_uuid:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")

    # 确保 correction_data 中的 person_id 与路径中的一致
    # person_id 是数据库内部的整数ID，用于关联日志
    correction_data.person_id = db_person_by_uuid.id 
    correction_data.username = current_user.username # 记录操作用户名
    # ip_address 可以从请求头获取，这里暂时留空或从其他地方获取

    if correction_data.correction_type == "merge":
        if not correction_data.target_person_uuid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="合并操作需要提供 target_person_uuid。")
        
        if correction_data.target_person_uuid == person_uuid:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能将人物合并到其自身。")

        merged_target_person = crud.merge_persons(
            db=db,
            source_person_uuid=person_uuid,
            target_person_uuid=correction_data.target_person_uuid,
            current_user_id=current_user.id
        )
        if not merged_target_person:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="人物合并失败。")
        
        # 返回目标人物的信息，因为源人物已经被“合并”
        return merged_target_person
    else: # 其他纠正类型 (misdetection, relabel, other)
        # 如果 original_feature_vector 为空，则从 db_person 中获取
        # 由于 CorrectionLogCreate 不包含 original_feature_vector，此处代码逻辑将导致 AttributeError
        # 因此移除此行，如果需要记录原始特征向量，则应在 schemas.CorrectionLogCreate 中添加此字段
        # if not correction_data.original_feature_vector and db_person_by_uuid.feature_vector:
        #     correction_data.original_feature_vector = db_person_by_uuid.feature_vector
        
        # 记录纠正日志
        correction_data.corrected_by_user_id = current_user.id # 设置纠正用户ID
        correction_data.ip_address = request.client.host if request.client else None # 设置IP地址

        db_correction_log = crud.create_correction_log(
            db=db, 
            correction_log=correction_data,
        )

        # 根据纠正类型设置 is_verified 和 marked_for_retrain
        should_be_marked_for_retrain = True
        if correction_data.correction_type == "misdetection":
            should_be_marked_for_retrain = False # 误检的图片不应再用于 Re-ID 模型再训练
            logger.info(f"人物 {person_uuid} 被标记为误检。marked_for_retrain 设置为 False。")
        else:
            logger.info(f"人物 {person_uuid} 提交纠正类型为 {correction_data.correction_type}。marked_for_retrain 设置为 True。")

        # 更新人物的审核状态和纠正详情
        db_person = crud.update_person_verification_status(
            db=db,
            person_id=db_person_by_uuid.id,
            is_verified=True, # 任何纠正操作都视为已审核
            verified_by_user_id=current_user.id,
            correction_details=correction_data.details, 
            marked_for_retrain=should_be_marked_for_retrain 
        )

        if not db_person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found after correction attempt.")

        logger.info(f"人物 {person_uuid} 已提交纠正 ({correction_data.correction_type})。")
        # retrain_reid_model_task.delay() # 调度 Celery 任务

        return crud._prepare_person_for_schema(db_person, db)

@router.post("/persons/{person_uuid}/mark_for_retrain", response_model=schemas.Person)
def mark_person_for_retrain(
    person_uuid: str, # 改为接收 UUID
    mark: bool,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role not in [schemas.UserRole.ADMIN, schemas.UserRole.ADVANCED]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admin or advanced users can mark persons for retrain.")

    db_person_by_uuid = crud.get_person_by_uuid_obj(db, uuid=person_uuid)
    if not db_person_by_uuid:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found.")

    db_person = crud.update_person_retrain_status(
        db=db,
        person_id=db_person_by_uuid.id,
        marked_for_retrain=mark
    )
    if not db_person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Failed to update person retrain status.")
    
    if mark:
        logger.info(f"人物 {person_uuid} 已标记为再训练。")
        # retrain_reid_model_task.delay() # 调度 Re-ID 模型再训练任务
    else:
        logger.info(f"人物 {person_uuid} 的再训练标记已取消。")

    return crud._prepare_person_for_schema(db_person, db)

@router.post("/retrain_model_manually", response_model=schemas.MessageResponse)
def retrain_model_manually(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role not in [schemas.UserRole.ADMIN, schemas.UserRole.ADVANCED]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admin or advanced users can trigger model retraining.")
    
    # 获取所有标记为再训练的人物
    persons_to_retrain = crud.get_persons_marked_for_retrain(db)
    if not persons_to_retrain:
        logger.info("没有人物被标记为再训练，跳过调度。")
        return schemas.MessageResponse(message="没有人物被标记为再训练，无需训练。")
    
    person_uuids = [p.uuid for p in persons_to_retrain]
    logger.info(f"手动触发 Re-ID 模型再训练。共 {len(person_uuids)} 个人物待训练。")
    
    # 调度 Celery 任务，并将待训练人物的 UUID 列表传递给任务
    background_tasks.add_task(retrain_reid_model_task.delay, person_uuids=person_uuids)

    return schemas.MessageResponse(message=f"已成功调度 Re-ID 模型再训练任务，共 {len(person_uuids)} 个人物。") 