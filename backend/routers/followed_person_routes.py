from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging

from .. import crud, schemas, auth
from ..database_conn import get_db

router = APIRouter(
    prefix="/followed_persons",
    tags=["Followed Persons Management"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

@router.get("/{individual_id}/is_followed", summary="检查人物是否已被当前用户关注", response_model=bool)
async def check_is_followed(
    individual_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 正在检查 Individual {individual_id} 的关注状态")
    is_followed = crud.is_person_followed(db, current_user.id, individual_id)
    return is_followed

@router.post("/follow/{individual_id}", response_model=schemas.FollowedPerson, summary="关注指定人物")
async def follow_person(
    individual_id: int, # 从 person_uuid 改为 individual_id
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 尝试关注 Individual {individual_id}")
    # 检查 Individual 是否存在
    existing_individual = crud.get_individual(db, individual_id=individual_id) # 假设 crud 中有一个 get_individual 函数
    if not existing_individual:
        raise HTTPException(status_code=404, detail="人物不存在")

    followed_person = crud.create_followed_person(db, follower_id=current_user.id, individual_id=individual_id)
    # 填充 Individual 的 name 和 id_card
    followed_person.individual_name = existing_individual.name
    followed_person.individual_id_card = existing_individual.id_card

    logger.info(f"用户 {current_user.username} 成功关注 Individual {individual_id}")
    return followed_person

@router.delete("/unfollow/{individual_id}", response_model=schemas.FollowedPerson, summary="取消关注指定人物")
async def unfollow_person_route(
    individual_id: int, # 从 person_uuid 改为 individual_id
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 尝试取消关注 Individual {individual_id}")
    unfollowed_person = crud.unfollow_person(db, follower_id=current_user.id, individual_id=individual_id)
    if not unfollowed_person:
        raise HTTPException(status_code=404, detail="未关注此人物或人物不存在")

    # 填充 Individual 的 name 和 id_card (如果需要返回完整对象)
    existing_individual = crud.get_individual(db, individual_id=individual_id)
    if existing_individual:
        unfollowed_person.individual_name = existing_individual.name
        unfollowed_person.individual_id_card = existing_individual.id_card

    logger.info(f"用户 {current_user.username} 成功取消关注 Individual {individual_id}")
    return unfollowed_person

@router.get("/", response_model=schemas.PaginatedFollowedPersonsResponse, summary="获取当前用户关注的所有人物列表")
async def get_followed_persons_route(
    db: Session = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=200),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 正在获取关注人物列表 (skip={skip}, limit={limit})")
    followed_persons_data = crud.get_followed_persons(db, follower_id=current_user.id, skip=skip, limit=limit)
    
    # 填充 Individual 的 name 和 id_card
    for followed_person_item in followed_persons_data["items"]:
        if followed_person_item.followed_individual:
            followed_person_item.individual_name = followed_person_item.followed_individual.name
            followed_person_item.individual_id_card = followed_person_item.followed_individual.id_card
    
    logger.info(f"用户 {current_user.username} 获取关注人物列表成功，总数: {followed_persons_data["total"]}")
    return {"items": followed_persons_data["items"], "total": followed_persons_data["total"], "skip": skip, "limit": limit}

@router.get("/{individual_id}/alerts", response_model=List[schemas.Person], summary="获取指定关注人物的所有预警图片")
async def get_followed_person_alert_images(
    individual_id: int, # 从 person_uuid 改为 individual_id
    min_score: float = Query(90.0, ge=0.0, le=100.0, description="最低比对分值，范围0-100"),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 尝试获取 Individual {individual_id} 的预警图片 (min_score={min_score})")
    # 检查用户是否关注了此人物 (Individual)
    if not crud.is_person_followed(db, current_user.id, individual_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="您未关注此人物，无权查看预警图片")
    
    # 直接调用 crud 函数获取预警图片
    alert_images = crud.get_all_alert_images_for_person(db, individual_id=individual_id, min_score=min_score)
    
    if not alert_images:
        logger.info(f"Individual {individual_id} 没有找到预警图片")
        # 即使没有图片也返回空列表，而不是404
    logger.info(f"成功获取 Individual {individual_id} 的 {len(alert_images)} 张预警图片")
    return alert_images

@router.get("/{individual_id}/enrollments", response_model=List[schemas.Person], summary="获取指定关注人物的所有主动注册图片")
async def get_followed_person_enrollment_images(
    individual_id: int, # 从 person_uuid 改为 individual_id
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    logger.info(f"用户 {current_user.username} 尝试获取 Individual {individual_id} 的主动注册图片")
    # 检查用户是否关注了此人物 (Individual)
    if not crud.is_person_followed(db, current_user.id, individual_id):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="您未关注此人物，无权查看主动注册图片")
    
    # 直接调用 crud 函数获取主动注册图片
    enrollment_images = crud.get_all_enrollment_images_for_person(db, individual_id=individual_id)

    if not enrollment_images:
        logger.info(f"Individual {individual_id} 没有找到主动注册图片")
        # 即使没有图片也返回空列表，而不是404
    logger.info(f"成功获取 Individual {individual_id} 的 {len(enrollment_images)} 张主动注册图片")
    return enrollment_images

@router.get("/{individual_id}/realtime-comparison-enabled", response_model=bool, summary="获取指定关注人物的实时比对功能状态")
async def get_individual_realtime_comparison_status(
    individual_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_admin_user) # 只有管理员可以查看
):
    logger.info(f"管理员 {current_user.username} 正在获取 Individual {individual_id} 的实时比对功能状态")
    followed_person = crud.get_followed_person_by_individual_id(db, current_user.id, individual_id)
    if not followed_person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未关注此人物或人物不存在")
    return followed_person.realtime_comparison_enabled

@router.put("/{individual_id}/toggle-realtime-comparison", response_model=schemas.MessageResponse, summary="切换指定关注人物的实时比对功能状态")
async def toggle_individual_realtime_comparison_route(
    individual_id: int,
    enable: bool = Query(..., description="设置实时比对功能是否开启"),
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_admin_user) # 只有管理员可以切换
):
    logger.info(f"管理员 {current_user.username} 尝试将 Individual {individual_id} 的实时比对功能设置为: {enable}")
    try:
        updated_followed_person = crud.set_individual_realtime_comparison_status(db, current_user.id, individual_id, enable)
        if not updated_followed_person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="未关注此人物或人物不存在")
        logger.info(f"管理员 {current_user.username} 成功将 Individual {individual_id} 的实时比对功能设置为: {enable}")
        return {"message": f"人物 {individual_id} 的实时比对功能已设置为: {enable}"}
    except Exception as e:
        logger.error(f"切换人物 {individual_id} 实时比对功能状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="内部服务器错误，无法切换实时比对功能状态。")