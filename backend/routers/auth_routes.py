from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
import logging
from typing import List

from .. import crud
from .. import schemas
from .. import auth
from ..database_conn import get_db

router = APIRouter(
    tags=["认证与用户"],
    responses={404: {"description": "未找到"}},
)

logger = logging.getLogger(__name__)

@router.post("/users/", response_model=schemas.User, summary="创建新用户")
async def create_user_endpoint(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        logger.warning(f"创建用户失败: 用户名 {user.username} 已存在。")
        raise HTTPException(status_code=400, detail="用户名已注册")
    new_user = crud.create_user(db=db, user=user)
    logger.info(f"新用户 {new_user.username} (角色: {new_user.role}, 单位: {new_user.unit}, 手机: {new_user.phone_number}) 已成功创建，等待管理员审核激活。")
    return schemas.User(id=new_user.id, username=new_user.username, role=new_user.role, is_active=new_user.is_active)

@router.post("/users/{user_id}/activate", response_model=schemas.User, summary="管理员激活用户")
async def activate_user_endpoint(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role != "admin":
        logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 未经授权尝试激活用户 {user_id}。")
        raise HTTPException(status_code=403, detail="无足够权限")
    
    activated_user = crud.activate_user(db, user_id=user_id)
    if not activated_user:
        logger.warning(f"管理员 {current_user.username} 尝试激活不存在的用户 (ID: {user_id})。")
        raise HTTPException(status_code=404, detail="用户未找到")
    logger.info(f"管理员 {current_user.username} 成功激活用户 {activated_user.username} (ID: {user_id})。")
    return activated_user

@router.get("/users/inactive", response_model=List[schemas.User], summary="管理员获取所有待激活用户")
async def get_inactive_users_endpoint(
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    if current_user.role != "admin":
        logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 未经授权尝试获取待激活用户列表。")
        raise HTTPException(status_code=403, detail="无足够权限")
    
    inactive_users = crud.get_inactive_users(db)
    logger.info(f"管理员 {current_user.username} 成功获取 {len(inactive_users)} 个待激活用户。")
    return inactive_users

@router.get("/users/me", response_model=schemas.User, summary="获取当前用户的个人信息")
async def read_users_me(current_user: schemas.User = Depends(auth.get_current_active_user)):
    return current_user

@router.put("/users/me", response_model=schemas.User, summary="更新当前用户的个人信息")
async def update_users_me(
    user_update: schemas.UserProfileUpdate,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    # 验证当前密码
    if not auth.verify_password(user_update.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="当前密码不正确，无法更新个人信息。",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    updated_user = crud.update_user_profile(
        db,
        user_id=current_user.id,
        unit=user_update.unit,
        phone_number=user_update.phone_number, # Add this line
    )
    if not updated_user:
        logger.error(f"更新用户 {current_user.username} (ID: {current_user.id}) 个人信息失败。")
        raise HTTPException(status_code=404, detail="用户未找到或更新失败。")
    
    logger.info(f"用户 {current_user.username} (ID: {current_user.id}) 的个人信息已更新。")
    return updated_user

@router.put("/users/change-password", summary="修改当前用户密码")
async def change_password(
    password_change: schemas.PasswordChange,
    db: Session = Depends(get_db),
    current_user: schemas.User = Depends(auth.get_current_active_user)
):
    # 验证当前密码
    if not auth.verify_password(password_change.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="当前密码不正确。",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 检查新密码是否与旧密码相同 (可选但推荐)
    if auth.verify_password(password_change.new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="新密码不能与当前密码相同。",
        )

    # 更新密码
    hashed_new_password = auth.get_password_hash(password_change.new_password)
    updated_user = crud.update_user_password(db, current_user.id, hashed_new_password)

    if not updated_user:
        logger.error(f"用户 {current_user.username} (ID: {current_user.id}) 密码更新失败。")
        raise HTTPException(status_code=500, detail="密码更新失败。")
    
    logger.info(f"用户 {current_user.username} (ID: {current_user.id}) 密码已成功修改。")
    return {"message": "密码修改成功，请使用新密码重新登录。"}

@router.get("/test-auth-routes", summary="测试认证路由是否正常加载")
async def test_auth_routes():
    return {"message": "认证路由正常工作!"}

@router.post("/token", response_model=schemas.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, username=form_data.username)
    
    if not user:
        logger.warning(f"登录失败: 用户名 '{form_data.username}' 不存在。")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码不正确", # 统一错误信息，防止暴露用户名是否存在
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not auth.verify_password(form_data.password, user.hashed_password):
        logger.warning(f"登录失败: 用户 '{form_data.username}' 密码不正确。")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码不正确", # 统一错误信息
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 检查用户是否激活
    if not user.is_active:
        logger.warning(f"用户 {user.username} 尝试登录，但账户未激活。")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账户尚未激活，请联系管理员",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username, "role": user.role, "id": user.id}, expires_delta=access_token_expires
    )
    logger.info(f"用户 {user.username} (ID: {user.id}) 成功登录。") # 新增成功登录日志
    
    return {"access_token": access_token, "token_type": "bearer", "expires_in": auth.ACCESS_TOKEN_EXPIRE_MINUTES * 60}