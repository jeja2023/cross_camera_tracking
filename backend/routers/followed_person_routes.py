import logging
from typing import List, Optional
from datetime import datetime # 新增：导入 datetime 模块
import os # 新增：导入 os 模块

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from backend import crud, schemas, auth # 恢复对 crud 的直接导入
from backend import crud_match # 新增：导入 crud_match 模块
from backend.database_conn import get_db
from backend.schemas import User
import backend.celery_app # 新增：导入整个 celery_app 模块
from backend.ml_services import ml_tasks # 新增：导入 ml_tasks

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/followed_persons",
    tags=["Followed Persons"],
    responses={404: {"description": "Not found"}},
)

@router.post("/toggle_follow/", response_model=schemas.MessageResponse, summary="切换人物的关注状态（关注/取消关注）")
async def toggle_follow_status_endpoint(
    toggle_request: schemas.FollowedPersonToggleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user),
):
    """
    此API用于切换指定人物的关注状态。
    - 如果 `is_followed` 为 `true`，将尝试关注该人物。
    - 如果 `is_followed` 为 `false`，将尝试取消关注该人物。
    """
    individual_id = toggle_request.individual_id
    is_followed = toggle_request.is_followed
    user_id = current_user.id
    perform_global_search = toggle_request.perform_global_search # 获取新的标志

    # 检查 Individual 是否存在
    individual_obj = crud.get_individual(db, individual_id)
    if not individual_obj:
        raise HTTPException(status_code=404, detail="人物档案不存在。")

    # 切换关注状态
    success = crud_match.toggle_follow_status(db, individual_id, user_id, is_followed)
    
    if success:
        action = "关注" if is_followed else "取消关注"
        # 如果是关注操作且前端请求进行全局搜索，则触发异步任务
        if is_followed and perform_global_search:
            # 触发全局搜索比对的 Celery 任务
            # 这里需要获取注册图片路径，可以从 individual_obj 中获取，或者在 auto_realtime_crud 中处理
            backend.celery_app.celery_app.send_task("backend.ml_services.ml_tasks.run_global_search_for_followed_person",
                                        args=[individual_id, user_id], kwargs={'is_initial_search': True})
            logger.info(f"已为人物 {individual_id} 触发全局搜索比对任务。")

        return schemas.MessageResponse(message=f"人物 {individual_obj.name} (身份证号/ID: {individual_obj.id_card}) 已成功{action}。")
    else:
        action = "关注" if is_followed else "取消关注"
        return schemas.MessageResponse(message=f"人物 {individual_obj.name} (身份证号/ID: {individual_obj.id_card}) 已经是{'关注' if is_followed else '未关注'}状态，无需重复{action}。")

@router.get("/", response_model=schemas.PaginatedFollowedPersonsResponse, summary="获取当前用户关注的人物列表")
async def get_followed_persons_endpoint(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(10, ge=1, le=100, description="返回的记录数限制"),
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user),
):
    """
    获取当前用户关注的人物列表，支持分页。
    """
    user_id = current_user.id
    followed_persons_orm = crud.get_followed_persons_by_user(db, user_id, skip=skip, limit=limit)
    total_count = crud.get_total_followed_persons_count_by_user(db, user_id)

    # 将 ORM 对象转换为 Pydantic Schema，并加载 Individual 和 User 信息
    followed_persons_response = []
    for fp_orm in followed_persons_orm:
        individual_obj = crud.get_individual(db, fp_orm.individual_id) # 确保加载 Individual
        user_obj = crud.get_user(db, fp_orm.user_id) # 确保加载 User

        # 创建临时的 Pydantic Individual 和 User 对象
        individual_schema = schemas.Individual.model_validate(individual_obj) if individual_obj else None
        user_schema = schemas.User.model_validate(user_obj) if user_obj else None

        followed_persons_response.append(schemas.FollowedPersonResponse(
            id=fp_orm.id,
            individual_id=fp_orm.individual_id,
            user_id=fp_orm.user_id,
            follow_time=fp_orm.follow_time,
            unfollow_time=fp_orm.unfollow_time,
            individual=individual_schema,
            user=user_schema
        ))

    return schemas.PaginatedFollowedPersonsResponse(
        total=total_count,
        skip=skip,
        limit=limit,
        items=followed_persons_response
    )

@router.post("/toggle_realtime_comparison/", response_model=schemas.MessageResponse, summary="切换个体实时比对状态（仅限管理员）")
async def toggle_realtime_comparison_endpoint(
    toggle_request: schemas.IndividualRealtimeComparisonToggleRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_admin_user), # 仅限管理员
):
    """
    此API用于切换指定 `individual_id` 的实时比对功能开关。
    - 只有管理员用户可以调用此接口。
    - 如果 `is_enabled` 为 `true`，将尝试开启该人物的实时比对。
    - 如果 `is_enabled` 为 `false`，将尝试关闭该人物的实时比对。
    """
    individual_id = toggle_request.individual_id
    is_enabled = toggle_request.is_enabled

    individual_obj = crud.get_individual(db, individual_id)
    if not individual_obj:
        raise HTTPException(status_code=404, detail="人物档案不存在。")

    if crud_match.toggle_individual_realtime_comparison(db, individual_id, is_enabled):
        action = "开启" if is_enabled else "关闭"
        return schemas.MessageResponse(message=f"人物 {individual_obj.name} (ID: {individual_obj.id_card}) 的实时比对已成功{action}。")
    else:
        action = "开启" if is_enabled else "关闭"
        return schemas.MessageResponse(message=f"人物 {individual_obj.name} (ID: {individual_obj.id_card}) 的实时比对已经是{'开启' if is_enabled else '关闭'}状态，无需重复{action}。")

@router.get("/{individual_id}/is_followed", response_model=bool, summary="检查特定人物是否被当前用户关注")
async def check_person_followed_status(
    individual_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user),
):
    """
    检查指定 `individual_id` 的人物是否被当前用户关注。
    """
    user_id = current_user.id
    return crud_match.check_is_followed(db, individual_id, user_id)

@router.get("/{individual_id}/enrollment_images", response_model=schemas.PaginatedEnrollmentImagesResponse, summary="获取指定人物的注册图片列表")
async def get_individual_enrollment_images(
    individual_id: int,
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, le=1000, description="返回的记录数限制"), # 限制最大返回图片数量
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user), # 所有认证用户可见
):
    """
    此API用于获取指定 `individual_id` 的人物档案下所有注册图片的列表。
    """
    # 确保人物存在
    individual_obj = crud.get_individual(db, individual_id)
    if not individual_obj:
        raise HTTPException(status_code=404, detail="人物档案不存在。")

    # 获取图片路径
    all_image_paths_and_uuids = crud.get_enrollment_images_by_individual_id(db, individual_id)
    total_count = len(all_image_paths_and_uuids)

    formatted_images = []
    for item in all_image_paths_and_uuids:
        full_file_path = item['path']
        person_uuid = item['uuid']  # Person 的 UUID
        image_db_uuid = item['image_uuid']  # Image 的 UUID (可能为 None)

        # 从绝对路径中提取文件名
        filename = os.path.basename(full_file_path)

        # 构建前端可访问的 URL 路径
        # full_file_path 的格式示例：E:\cross_camera_tracking\backend\database\full_frames\general_detection\image\<uuid>\<filename>.jpg
        # 我们需要将其转换为 /database/full_frames/general_detection/image/<uuid>/<filename>.jpg
        
        # 找到 "backend" 目录在路径中的位置
        backend_index = full_file_path.find(os.path.join("backend", ""))
        if backend_index == -1:
            logger.error(f"无法从路径 {full_file_path} 找到 'backend' 目录。")
            continue

        # 获取从 "backend" 目录开始的相对路径
        relative_path_from_backend = full_file_path[backend_index + len("backend") + len(os.sep):]

        # 将路径分隔符转换为 URL 友好的正斜杠
        relative_url_path = "/backend/" + relative_path_from_backend.replace(os.sep, '/')

        # 修正：根据 main.py 中的 StaticFiles 挂载，图片是从 /database/full_frames 开始的
        # 所以我们真正需要的 URL 路径是 /database/full_frames/...
        # relative_url_path 的格式现在是 /backend/database/full_frames/...
        # 我们需要将其修改为 /database/full_frames/...
        if relative_url_path.startswith("/backend/"):
            final_url_path = relative_url_path[len("/backend"):] # 移除 /backend
        else:
            final_url_path = relative_url_path

        formatted_images.append(schemas.EnrollmentImageResponse(
            image_path=final_url_path,
            uuid=person_uuid,
            image_db_uuid=image_db_uuid, # 新增这一行
            filename=filename
        ))
    
    # 应用分页逻辑
    paginated_images = formatted_images[skip : skip + limit]

    # 直接返回包含 UUID 的 Pydantic 模型列表
    return schemas.PaginatedEnrollmentImagesResponse(
        total=total_count,
        skip=skip,
        limit=limit,
        items=paginated_images # 直接使用 paginated_images，它们已经是 EnrollmentImageResponse 对象
    )

@router.delete("/{individual_id}/enrollment_images/{image_uuid}", response_model=schemas.MessageResponse, summary="删除指定人物的注册图片")
async def delete_individual_enrollment_image(
    individual_id: int,
    image_uuid: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user),
):
    """
    此API用于删除指定 `individual_id` 的人物档案下，UUID 为 `image_uuid` 的注册图片。
    只有该图片的所有者或管理员可以删除。
    """
    success = crud.delete_enrollment_image_by_uuid(
        db=db,
        individual_id=individual_id,
        image_uuid=image_uuid,
        current_user_id=current_user.id,
        current_user_role=current_user.role
    )
    if success:
        return schemas.MessageResponse(message=f"人物档案 {individual_id} 下的注册图片 {image_uuid} 已成功删除。")
    else:
        raise HTTPException(status_code=404, detail="图片未找到或您没有权限删除此图片。")

@router.get("/{individual_id}/global_search_results", response_model=schemas.PaginatedGlobalSearchResultsResponse, summary="获取指定关注人员的全局搜索比对结果")
async def get_followed_person_global_search_results(
    individual_id: int,
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(100, ge=1, description="返回的记录数限制"),
    # min_confidence: Optional[float] = Query(0.9, ge=0.0, le=1.0, description="最低置信度阈值"), # 已移除，从系统配置获取
    is_initial_search: Optional[bool] = Query(None, description="是否只返回初始搜索结果"),
    last_query_time: Optional[datetime] = Query(None, description="上次查询时间，用于增量查询"), # 新增参数
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user),
):
    """
    此API用于获取指定 `individual_id` 的关注人员的全局搜索比对结果。
    支持分页、按置信度筛选，并可指定是否只返回初始关注时触发的搜索结果。
    现在也支持基于 `last_query_time` 的增量查询。
    """
    user_id = current_user.id
    results = crud_match.get_global_search_results_by_individual_id(
        db=db,
        individual_id=individual_id,
        user_id=user_id,
        skip=skip, # 传递 skip 参数
        limit=limit, # 传递 limit 参数
        # min_confidence=min_confidence, # 已移除，从系统配置获取
        is_initial_search=is_initial_search,
        last_query_time=last_query_time # 传递新增参数
    )
    total_count = crud_match.get_total_global_search_results_count_by_individual_id(
        db=db,
        individual_id=individual_id,
        user_id=user_id,
        # min_confidence=min_confidence, # 已移除，从系统配置获取
        is_initial_search=is_initial_search,
        last_query_time=last_query_time # 传递新增参数
    )

    return schemas.PaginatedGlobalSearchResultsResponse(
        total=total_count,
        skip=skip, # 恢复 skip 参数的传递
        limit=limit,
        items=results
    )

@router.get("/{individual_id}/alerts", response_model=schemas.PaginatedAlertsResponse, summary="获取指定关注人员的预警信息")
async def get_followed_person_alerts(
    individual_id: int,
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(20, ge=1, le=100, description="返回的记录数限制"),
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user),
):
    """
    此API用于获取指定 `individual_id` 的关注人员的预警信息列表。
    支持分页。
    """
    # 检查 Individual 是否存在
    individual_obj = crud.get_individual(db, individual_id)
    if not individual_obj:
        raise HTTPException(status_code=404, detail="人物档案不存在。")

    alerts = crud_match.get_alerts_by_individual_id(
        db=db,
        individual_id=individual_id,
        skip=skip,
        limit=limit
    )
    total_count = crud_match.get_total_alerts_count_by_individual_id(
        db=db,
        individual_id=individual_id
    )

    return schemas.PaginatedAlertsResponse(
        total=total_count,
        skip=skip,
        limit=limit,
        items=alerts
    )

@router.post("/{individual_id}/trigger_global_search", response_model=schemas.MessageResponse, summary="手动触发指定关注人员的全局搜索比对任务")
async def trigger_global_search_for_followed_person(
    individual_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(auth.get_current_active_user),
):
    """
    此API允许用户手动为指定individual_id的关注人员触发一次新的全局搜索比对任务。
    """
    individual_obj = crud.get_individual(db, individual_id)
    if not individual_obj:
        raise HTTPException(status_code=404, detail="人物档案不存在。")

    # 确保该人物确实是关注人员，尽管不是强制要求，但逻辑上合理
    if not crud_match.check_is_followed(db, individual_id, current_user.id):
        raise HTTPException(status_code=400, detail="该人物未被当前用户关注，无法触发全局搜索任务。")

    backend.celery_app.celery_app.send_task("backend.ml_services.ml_tasks.run_global_search_for_followed_person",
                                args=[individual_id, current_user.id], kwargs={'is_initial_search': False})
    logger.info(f"用户 {current_user.id} 为人物 {individual_id} 手动触发了全局搜索比对任务。")

    return schemas.MessageResponse(message=f"已为人物 {individual_obj.name} (ID: {individual_obj.id_card}) 触发全局搜索比对任务，请稍后刷新页面查看结果。")