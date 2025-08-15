from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional
import os
from sqlalchemy.orm import Session
from . import crud, schemas
from .database_conn import get_db
from backend.config import settings # 导入 settings
import logging

SECRET_KEY = settings.SECRET_KEY # 从 config 中获取
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES # 从 config 中获取

logger = logging.getLogger(__name__)

# 使用 CryptContext 并且捕获初始化错误
try:
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    logger.info("CryptContext successfully initialized with bcrypt scheme.")
except Exception as e:
    logger.error(f"Error initializing CryptContext with bcrypt: {e}", exc_info=True)
    # 如果 CryptContext 初始化失败，可以考虑更严重的错误处理，例如退出应用程序
    # 但为了调试，我们先继续
    pwd_context = None # 确保 pwd_context 未初始化时不会被使用

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    if not pwd_context:
        logger.error("CryptContext not initialized. Cannot verify password.")
        return False
    try:
        result = pwd_context.verify(plain_password, hashed_password)
        logger.debug(f"Password verification result: {result} for user (hashed password: {hashed_password[:10]}...)") # Log first 10 chars of hash
        return result
    except Exception as e:
        logger.error(f"Error during password verification: {e}", exc_info=True)
        return False

def get_password_hash(password):
    if not pwd_context:
        logger.error("CryptContext not initialized. Cannot hash password.")
        raise RuntimeError("Password hashing service not available.")
    try:
        hashed = pwd_context.hash(password)
        logger.debug(f"Password hashed successfully. First 10 chars of hash: {hashed[:10]}...")
        return hashed
    except Exception as e:
        logger.error(f"Error during password hashing: {e}", exc_info=True)
        raise

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    if not token:
        logger.warning("Attempted to validate token, but no token was provided.")
        raise credentials_exception

    logger.debug(f"Attempting to validate token: {token[:30]}...")
    logger.debug(f"Full token received: {token}") # Add this line to log the full token

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.debug(f"Token payload decoded: {payload}")
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token payload missing 'sub' (username).")
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error during token decoding: {e}")
        raise credentials_exception

    user = crud.get_user_by_username(db, username=token_data.username)
    if user is None:
        logger.warning(f"User '{token_data.username}' not found in database.")
        raise credentials_exception
    logger.debug(f"Successfully authenticated user: {user.username}")
    return user

async def get_current_active_user(current_user: schemas.User = Depends(get_current_user)):
    if not current_user.is_active:
        logger.warning(f"用户 {current_user.username} 尝试登录但未激活。")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user

# 新增：获取当前活跃的管理员用户
async def get_current_active_admin_user(current_user: schemas.User = Depends(get_current_active_user)):
    if current_user.role != "admin":
        logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 尝试访问管理员功能，但无权限。")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")
    return current_user

# 新增：获取当前活跃的高级用户（管理员或特定高级角色）
async def get_current_active_advanced_role_user(current_user: schemas.User = Depends(get_current_active_user)):
    # 假设“高级用户”可以是“admin”角色，或者一个名为“advanced”的新角色
    # 您可以根据实际需求调整这里的逻辑，例如只允许特定角色
    if current_user.role not in ["admin", "advanced"]:
        logger.warning(f"用户 {current_user.username} (角色: {current_user.role}) 尝试访问高级功能，但无权限。")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions for advanced features")
    return current_user

# 新增：直接从token字符串获取用户（用于EventSource等不能使用Depends的情况）
async def get_user_from_token_string(token: str, db: Session):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        logger.warning("Attempted to validate token string, but no token was provided.")
        raise credentials_exception

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token string payload missing 'sub' (username).")
            raise credentials_exception
        token_data = schemas.TokenData(username=username)
    except JWTError as e:
        logger.warning(f"JWT string validation failed: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error during token string decoding: {e}")
        raise credentials_exception

    user = crud.get_user_by_username(db, username=token_data.username)
    if user is None:
        logger.warning(f"User '{token_data.username}' not found in database from token string.")
        raise credentials_exception
    logger.debug(f"Successfully authenticated user from token string: {user.username}")
    return user