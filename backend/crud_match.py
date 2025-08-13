import json
import numpy as np
import os
from sqlalchemy.orm import Session, joinedload
from .database_conn import Person, Video, User, Log, Stream, Image, CorrectionLog, SystemConfig, Individual, FollowedPerson, RealtimeMatchAlert, GlobalSearchResult
from . import auth
from . import schemas
import logging
from datetime import datetime
from typing import Optional, List, Dict, Union
import pytz
from backend.config import settings
import shutil
import uuid
from sqlalchemy import or_
import backend.crud as crud

logger = logging.getLogger(__name__)

def toggle_follow_status(
    db: Session,
    individual_id: int,
    user_id: int,
    is_followed: bool
) -> bool:
    """
    切换人物的关注状态。
    如果 is_followed 为 True，则关注该人物。
    如果 is_followed 为 False，则取消关注该人物。
    """
    existing_follow = db.query(FollowedPerson).filter(
        FollowedPerson.individual_id == individual_id,
        FollowedPerson.user_id == user_id,
        FollowedPerson.unfollow_time == None
    ).first()

    if is_followed:
        if not existing_follow:
            new_follow = FollowedPerson(
                individual_id=individual_id,
                user_id=user_id,
                follow_time=datetime.now(pytz.utc)
            )
            db.add(new_follow)
            db.commit()
            db.refresh(new_follow)
            logger.info(f"用户 {user_id} 关注了人物 {individual_id}。")
            return True
        else:
            logger.info(f"用户 {user_id} 已关注人物 {individual_id}，无需重复操作。")
            return False
    else:  # is_followed is False
        if existing_follow:
            # 如果存在关注记录，则标记为取消关注
            existing_follow.unfollow_time = datetime.now(pytz.utc)
            db.add(existing_follow)
            db.commit()
            db.refresh(existing_follow)
            logger.info(f"用户 {user_id} 取消关注了人物 {individual_id}。")
            return True
        else:
            logger.info(f"用户 {user_id} 未关注人物 {individual_id}，无需取消关注。")
            return False

def check_is_followed(db: Session, individual_id: int, user_id: int) -> bool:
    """检查指定人物是否被当前用户关注。"""
    return db.query(FollowedPerson).filter(
        FollowedPerson.individual_id == individual_id,
        FollowedPerson.user_id == user_id,
        FollowedPerson.unfollow_time == None
    ).first() is not None

def toggle_individual_realtime_comparison(db: Session, individual_id: int, is_enabled: bool) -> bool:
    """切换指定 Individual 的实时比对功能。"""
    individual = db.query(Individual).filter(Individual.id == individual_id).first()
    if not individual:
        logger.warning(f"尝试切换实时比对状态时，未找到 Individual ID: {individual_id}")
        return False
    
    if individual.is_realtime_comparison_enabled == is_enabled:
        logger.info(f"Individual {individual_id} 的实时比对状态已经是 {is_enabled}，无需更改。")
        return False
    
    individual.is_realtime_comparison_enabled = is_enabled
    db.add(individual)
    db.commit()
    db.refresh(individual)
    logger.info(f"Individual {individual_id} 的实时比对状态已更新为 {is_enabled}。")
    return True

def create_realtime_match_alert(db: Session, alert_data: schemas.RealtimeMatchAlert) -> RealtimeMatchAlert:
    """创建新的实时比对预警记录。"""
    db_alert = RealtimeMatchAlert(
        person_uuid=alert_data.person_uuid,
        matched_individual_id=alert_data.matched_individual_id,
        matched_individual_uuid=alert_data.matched_individual_uuid,
        matched_individual_name=alert_data.matched_individual_name,
        similarity_score=alert_data.similarity_score,
        timestamp=alert_data.timestamp,
        alert_type=alert_data.alert_type,
        source_media_type=alert_data.source_media_type,
        source_media_uuid=alert_data.source_media_uuid,
        user_id=alert_data.user_id,
        cropped_image_path=alert_data.cropped_image_path, # 新增
        full_frame_image_path=alert_data.full_frame_image_path # 新增
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def get_realtime_match_alerts(db: Session, user_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[RealtimeMatchAlert]:
    """获取实时比对预警记录，可按用户筛选，支持分页。"""
    query = db.query(RealtimeMatchAlert).options(
        joinedload(RealtimeMatchAlert.person),
        joinedload(RealtimeMatchAlert.matched_individual),
        joinedload(RealtimeMatchAlert.alert_user)
    )
    if user_id:
        query = query.filter(RealtimeMatchAlert.user_id == user_id)
    return query.order_by(RealtimeMatchAlert.timestamp.desc()).offset(skip).limit(limit).all()

def get_total_realtime_match_alerts_count(db: Session, user_id: Optional[int] = None) -> int:
    """获取实时比对预警记录的总数，可按用户筛选。"""
    query = db.query(RealtimeMatchAlert)
    if user_id:
        query = query.filter(RealtimeMatchAlert.user_id == user_id)
    return query.count()


# 新增 GlobalSearchResult CRUD operations
def create_global_search_result(db: Session, global_search_result: schemas.GlobalSearchResultCreate) -> Optional[GlobalSearchResult]:
    """创建新的全局搜索结果记录。"""
    db_global_search_result = GlobalSearchResult(
        individual_id=global_search_result.individual_id,
        matched_person_uuid=global_search_result.matched_person_uuid,
        matched_person_id=global_search_result.matched_person_id,
        matched_image_path=global_search_result.matched_image_path,
        confidence=global_search_result.confidence,
        search_time=global_search_result.search_time,
        user_id=global_search_result.user_id,
        is_initial_search=global_search_result.is_initial_search
    )
    db.add(db_global_search_result)
    db.commit()
    db.refresh(db_global_search_result)
    logger.info(f"已创建全局搜索结果：individual_id={global_search_result.individual_id}, matched_person_uuid={global_search_result.matched_person_uuid}, confidence={global_search_result.confidence}")
    return db_global_search_result

def create_multiple_global_search_results(db: Session, results_data: List[Dict]) -> List[GlobalSearchResult]:
    """批量创建全局搜索结果记录。"""
    db_results = []
    for result in results_data:
        db_result = GlobalSearchResult(
            individual_id=result["followed_person_id"], # Adjust to match schema field
            matched_person_uuid=result["matched_person_uuid"],
            matched_person_id=result["matched_person_id"],
            matched_image_path=result["matched_image_path"],
            confidence=result["confidence"],
            search_time=result["search_time"],
            user_id=result["user_id"], # Ensure user_id is passed in result dict
            is_initial_search=result.get("is_initial_search", False) # Default to False if not provided
        )
        db.add(db_result)
        db_results.append(db_result)
    db.commit()
    for r in db_results: # Refresh after commit to get IDs
        db.refresh(r)
    logger.info(f"批量创建了 {len(db_results)} 条全局搜索结果。")
    return db_results

def get_global_search_results_by_individual_id(
    db: Session,
    individual_id: int,
    user_id: int,
    skip: int = 0, # 新增 skip 参数
    limit: int = 100, # 新增 limit 参数
    min_confidence: Optional[float] = None,
    is_initial_search: Optional[bool] = None,
    last_query_time: Optional[datetime] = None # 新增参数
) -> List[schemas.GlobalSearchResultResponse]:
    """
    获取指定人物的全局搜索比对结果，支持分页、置信度筛选和是否初始搜索筛选，并支持基于时间戳的增量查询。
    """
    query = db.query(GlobalSearchResult).options(
        joinedload(GlobalSearchResult.individual),
        joinedload(GlobalSearchResult.person).joinedload(Person.individual), # 急切加载匹配人物及其关联的Individual
        joinedload(GlobalSearchResult.user)
    ).filter(
        GlobalSearchResult.individual_id == individual_id,
        GlobalSearchResult.user_id == user_id
    )

    if min_confidence is not None:
        query = query.filter(GlobalSearchResult.confidence >= min_confidence)
    
    if last_query_time is not None: # 新增：根据 last_query_time 过滤
        if last_query_time.tzinfo is None: # 如果是 naive datetime，假设为 UTC
            last_query_time = pytz.utc.localize(last_query_time)
        query = query.filter(GlobalSearchResult.search_time > last_query_time)
    else: # 只有非增量查询时，才考虑 is_initial_search
        if is_initial_search is not None:
            query = query.filter(GlobalSearchResult.is_initial_search == is_initial_search)

    results_orm = query.order_by(GlobalSearchResult.search_time.desc()).offset(skip).limit(limit).all() # 按照时间降序排列，并限制数量
    
    # 将 ORM 对象转换为 Pydantic Schema
    global_search_responses = []
    for result_orm in results_orm:
        # Prepare nested Person schema
        person_schema = None
        if result_orm.person:
            person_schema = crud._prepare_person_for_schema(result_orm.person, db, user_id) # Reuse existing helper

        global_search_responses.append(schemas.GlobalSearchResultResponse(
            id=result_orm.id,
            individual_id=result_orm.individual_id,
            matched_person_uuid=result_orm.matched_person_uuid,
            matched_person_id=result_orm.matched_person_id,
            matched_image_path=result_orm.matched_image_path,
            confidence=result_orm.confidence,
            search_time=result_orm.search_time,
            user_id=result_orm.user_id,
            is_initial_search=result_orm.is_initial_search,
            individual=schemas.Individual.model_validate(result_orm.individual) if result_orm.individual else None,
            person=person_schema,
            user=schemas.User.model_validate(result_orm.user) if result_orm.user else None
        ))
    return global_search_responses

def get_total_global_search_results_count_by_individual_id(
    db: Session,
    individual_id: int,
    user_id: int,
    min_confidence: Optional[float] = None,
    is_initial_search: Optional[bool] = None,
    last_query_time: Optional[datetime] = None # 新增参数
) -> int:
    """
    获取指定人物的全局搜索比对结果总数，支持基于时间戳的增量查询。
    """
    query = db.query(GlobalSearchResult).filter(
        GlobalSearchResult.individual_id == individual_id,
        GlobalSearchResult.user_id == user_id
    )
    if min_confidence is not None:
        query = query.filter(GlobalSearchResult.confidence >= min_confidence)
    
    if last_query_time is not None: # 新增：根据 last_query_time 过滤
        if last_query_time.tzinfo is None: # 如果是 naive datetime，假设为 UTC
            last_query_time = pytz.utc.localize(last_query_time)
        query = query.filter(GlobalSearchResult.search_time > last_query_time)
    else: # 只有非增量查询时，才考虑 is_initial_search
        if is_initial_search is not None:
            query = query.filter(GlobalSearchResult.is_initial_search == is_initial_search)

    return query.count()

def get_alerts_by_individual_id(
    db: Session,
    individual_id: int,
    skip: int = 0,
    limit: int = 100,
) -> List[schemas.Alert]:
    """
    获取指定人物的预警信息列表，支持分页。
    """
    # 首先确保 individual_id 存在
    individual_exists = db.query(Individual).filter(Individual.id == individual_id).first()
    if not individual_exists:
        return []

    query = db.query(RealtimeMatchAlert).options(
        joinedload(RealtimeMatchAlert.person) # 急切加载关联的 Person 对象
    ).filter(
        RealtimeMatchAlert.matched_individual_id == individual_id
    ).order_by(RealtimeMatchAlert.timestamp.desc())

    alerts_orm = query.offset(skip).limit(limit).all()

    alerts_response = []
    for alert_orm in alerts_orm:
        # 直接从 RealtimeMatchAlert ORM 模型中获取图片路径
        cropped_image_path = alert_orm.cropped_image_path
        full_frame_image_path = alert_orm.full_frame_image_path

        alerts_response.append(schemas.Alert(
            id=alert_orm.id,
            individual_id=alert_orm.matched_individual_id,
            person_id=alert_orm.person.id, # 从关联的 Person 对象获取 ID
            person_uuid=alert_orm.person_uuid, # 从 RealtimeMatchAlert 获取人物 UUID
            person_created_at=alert_orm.person.created_at, # 从关联的 Person 对象获取创建时间
            timestamp=alert_orm.timestamp,
            source_media_uuid=alert_orm.source_media_uuid, # 从 RealtimeMatchAlert 获取来源媒体 UUID
            source_media_type=alert_orm.source_media_type, # 新增：从 RealtimeMatchAlert 获取来源媒体类型
            # 移除 location 和 description
            # location="未知地点", # RealtimeMatchAlert 模型中可能没有 location，需要根据实际情况调整
            # description="实时比对预警", # RealtimeMatchAlert 模型中可能没有 description，需要根据实际情况调整
            cropped_image_path=cropped_image_path,
            full_frame_image_path=full_frame_image_path,
            similarity_score=alert_orm.similarity_score # 新增：添加相似度分数
        ))
    return alerts_response

def get_total_alerts_count_by_individual_id(
    db: Session,
    individual_id: int,
) -> int:
    """
    获取指定人物的预警信息总数。
    """
    # 首先确保 individual_id 存在
    individual_exists = db.query(Individual).filter(Individual.id == individual_id).first()
    if not individual_exists:
        return 0

    query = db.query(RealtimeMatchAlert).filter(
        RealtimeMatchAlert.matched_individual_id == individual_id
    )
    return query.count()