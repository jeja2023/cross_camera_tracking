from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import shutil
import os
import aiofiles
from uuid import UUID

from .. import crud
from .. import schemas
from .. import auth
from ..database_conn import get_db
from ..ml_services import ml_logic
from ..config import settings
import uuid

router = APIRouter(
    tags=["Person Search & Management"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)


@router.post("/enroll", response_model=schemas.PersonEnrollResponse, summary="主动注册人物")
async def enroll_person(
    db: Session = Depends(get_db),
    person_name: Optional[str] = Form(None, description="可选：人物的姓名或标识"),
    id_card: Optional[str] = Form(None, description="可选：身份证号或其他ID"), # 新增：身份证号或ID
    images: List[UploadFile] = File(..., description="包含人物的图片文件 (支持多张)"),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"进入 enroll_person 路由。请求方法: POST")
    logger.info(f"用户 {current_user.username} 正在主动注册人物。姓名: {person_name}, 身份证号/ID: {id_card}, 图片数量: {len(images)}")

    if not images:
        raise HTTPException(status_code=400, detail="请上传至少一张图片。")
    

    # 按身份证号分目录保存图片
    if not id_card:
        raise HTTPException(status_code=400, detail="请填写身份证号。")
    upload_dir = os.path.join(settings.ENROLL_PERSON_IMAGES_DIR, id_card)
    logger.info(f"图片保存目录 upload_dir: {upload_dir}")
    os.makedirs(upload_dir, exist_ok=True)

    enrollment_response = None

    for image_file in images:
        # 使用原始文件名
        original_filename = os.path.splitext(image_file.filename)[0]  
        # 使用身份证号+原始文件名
        id_card_filename = f"{original_filename}.jpg"
        temp_image_path = os.path.join(upload_dir, id_card_filename)
        logger.info(f"准备保存图片到: {temp_image_path}")
        try:
            content = await image_file.read()
            
            # 先让 ml_logic 处理图片并获取数据库存储路径
            enrollment_response = ml_logic.process_enrollment_image(
                db=db,
                image_bytes=content, 
                person_name=person_name,
                id_card=id_card,
                current_user=current_user
            )
            
            # 移除直接保存原始文件名的图片，因为 ml_logic.process_enrollment_image 已保存 UUID 命名的图片
            # async with aiofiles.open(temp_image_path, "wb") as out_file:
            #     await out_file.write(content)
            # logger.info(f"图片已保存到身份证号目录: {temp_image_path}")
            logger.info(f"图片 {image_file.filename} 已由 ml_logic 处理并保存为 UUID 命名文件。")

        except HTTPException as e:
            logger.error(f"处理图片 {image_file.filename} 时发生HTTP错误: {e.detail}", exc_info=True)
            raise e 
        except Exception as e:
            logger.error(f"处理图片 {image_file.filename} 时发生未知错误: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"处理图片 {image_file.filename} 时发生错误: {e}")
        finally:
            # 不再删除临时文件，让图片保持在身份证号目录下
            pass

    if not enrollment_response:
        raise HTTPException(status_code=500, detail="所有图片处理失败或未返回有效响应。")

    return enrollment_response

@router.post("/new_person_search", response_model=schemas.PaginatedGroupedImageSearchResultsResponse, summary="多模态搜人 (以图搜人或以UUID搜人)") # Changed endpoint path
async def search_persons_multi_modal(
    search_request: schemas.PersonSearchRequest, # 接收 Pydantic 模型
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 正在执行多模态搜人查询。查询图片路径: {search_request.query_image_path if search_request.query_image_path else '无'}, 查询人物UUID: {search_request.query_person_uuid if search_request.query_person_uuid else '无'}, 阈值: {search_request.threshold}, 视频UUID: {search_request.video_uuid if search_request.video_uuid else '全局'}, 视频流UUID: {search_request.stream_uuid if search_request.stream_uuid else '全局'}, 限制: {search_request.limit}")

    # Debugging: Print the received search_request object
    logger.info(f"DEBUG: Received search_request object: {search_request.model_dump_json()}")

    if not search_request.query_image_path and not search_request.query_person_uuids:
        logger.error("search_persons_multi_modal: 必须提供 query_image_path 或 query_person_uuids。")
        raise HTTPException(status_code=400, detail="必须提供 query_image_path 或 query_person_uuids。")

    # 处理上传的查询图片（如果存在）
    # query_image_path = None # 此处不再需要，因为前端会先解析图片，再发送人物UUID

    search_response = ml_logic.find_similar_people(
        query_image_path=search_request.query_image_path, # 仍然保留，以防万一直接发送图片路径
        db=db, 
        threshold=search_request.threshold, 
        video_uuid=search_request.video_uuid, 
        stream_uuid=search_request.stream_uuid, 
        current_user=current_user,
        skip=search_request.skip,
        limit=search_request.limit,
        query_person_uuids=search_request.query_person_uuid # 传递人物UUID列表
    )
    
    # 直接返回 ml_logic.find_similar_people 返回的已格式化的数据
    logger.info(f"用户 {current_user.username} 的多模态搜人查询完成，返回总查询人物数: {search_response["total_query_persons"]}，总不重复结果数: {search_response["total_overall_results"]}。")
    return search_response

@router.get("/all", response_model=schemas.PaginatedPersonsResponse, summary="获取全部人物特征图库 (管理员权限，支持分页、筛选和搜索)")
async def get_all_persons_api(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, le=200, description="返回的最大记录数"),
    is_verified: Optional[bool] = Query(None, description="按审核状态筛选 (True: 已审核, False: 未审核)"),
    marked_for_retrain: Optional[bool] = Query(None, description="按再训练标记筛选 (True: 待再训练, False: 不待再训练)"),
    is_trained: Optional[bool] = Query(None, description="按训练状态筛选 (True: 已训练, False: 未训练)"), # 新增：is_trained 参数
    has_id_card: Optional[bool] = Query(None, description="筛选是否有身份证号/ID数据 (True: 有, False: 无)"), # 新增参数
    query: Optional[str] = Query(None, description="根据人物姓名、UUID、身份证号、视频UUID、视频流UUID、图片UUID或名称进行模糊搜索"), 
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role != "admin" and current_user.role != "advanced": 
        logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 未经授权尝试访问全部人物特征图库。")
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    total_persons = crud.get_total_persons_count(db, is_verified=is_verified, marked_for_retrain=marked_for_retrain, is_trained=is_trained, query=query, has_id_card=has_id_card) # 传递 has_id_card
    persons = crud.get_persons(db, skip=skip, limit=limit, is_verified=is_verified, marked_for_retrain=marked_for_retrain, is_trained=is_trained, query=query, has_id_card=has_id_card) # 传递 has_id_card

    logger.info(f"管理员 {current_user.username} 成功获取 {len(persons)} 个人物特征图库（总数：{total_persons}），筛选条件：is_verified={is_verified}, marked_for_retrain={marked_for_retrain}, is_trained={is_trained}, 查询词：{query}, has_id_card={has_id_card}。")
    return {"total": total_persons, "skip": skip, "limit": limit, "items": persons}

@router.get("/latest", response_model=schemas.Person, summary="获取最新创建的人员信息")
async def get_latest_person_api(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can view latest persons")
    
    latest_person = crud.get_latest_person(db)
    if not latest_person:
        raise HTTPException(status_code=404, detail="No persons found")
    
    return latest_person

@router.delete("/{person_uuid}", summary="删除指定人员 (管理员权限)", include_in_schema=False)
async def delete_person(
    person_uuid: str, 
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role != "admin":
         logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 未经授权尝试删除人员 {person_uuid}。")
         raise HTTPException(status_code=403, detail="Only admins can delete persons")
    deleted_person = crud.delete_person_by_uuid(db, uuid=person_uuid)
    if deleted_person is None:
        logger.warning(f"管理员 {current_user.username} 尝试删除不存在的人员 {person_uuid}。")
        raise HTTPException(status_code=404, detail="Person not found")
    logger.info(f"管理员 {current_user.username} 成功删除人员 {person_uuid}。")
    return {"message": "Person deleted successfully", "uuid": person_uuid}

@router.get("/{person_uuid}", response_model=schemas.Person, summary="通过UUID获取指定人员信息")
async def get_person_by_uuid_api(
    person_uuid: UUID,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 正在尝试通过UUID {person_uuid} 获取人员信息。")
    person = crud.get_person_by_uuid(db, uuid=str(person_uuid))
    if not person:
        logger.warning(f"通过UUID {person_uuid} 未找到人员。")
        raise HTTPException(status_code=404, detail="Person not found")
    logger.info(f"成功获取UUID {person_uuid} 的人员信息。")
    return person

@router.get("/by_id_card/{id_card_query}", response_model=schemas.PaginatedPersonsResponse, summary="通过身份证号模糊查询人物")
async def get_persons_by_id_card_api(
    id_card_query: str,
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role not in [schemas.UserRole.ADMIN, schemas.UserRole.ADVANCED]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only admin or advanced users can perform this search.")

    # 使用 crud.get_persons 的 query 参数进行模糊搜索
    total_persons = crud.get_total_persons_count(db, query=id_card_query)
    persons = crud.get_persons(db, query=id_card_query, skip=skip, limit=limit)

    logger.info(f"用户 {current_user.username} 正在通过身份证号模糊查询人物，查询词: {id_card_query}，找到 {len(persons)} 个人物 (总数: {total_persons})。")
    return {"total": total_persons, "skip": skip, "limit": limit, "items": persons}