import redis
import logging
from backend.config import settings

logger = logging.getLogger(__name__)

# Redis 客户端实例
# 使用 decode_responses=False 确保我们处理的是字节数据，适合图片帧
try:
    redis_client = redis.StrictRedis.from_url(settings.REDIS_URL, decode_responses=False)
    redis_client.ping() # 测试连接
    logger.info("成功连接到 Redis 服务器。")
except redis.exceptions.ConnectionError as e:
    logger.error(f"无法连接到 Redis 服务器: {e}")
    redis_client = None # 如果连接失败，则将客户端设为None

# 定义 Redis 键的前缀和帧过期时间（秒）
# 帧过期时间不宜过长，确保内存不会被占用过多旧帧
STREAM_FRAME_KEY_PREFIX = "stream_frame:"
FRAME_EXPIRATION_SECONDS = 5 # 帧数据在Redis中保留5秒，足够前端获取

def _get_frame_key(stream_uuid: str) -> str:
    """根据 stream_uuid 生成 Redis 键名。"""
    return f"{STREAM_FRAME_KEY_PREFIX}{stream_uuid}"

def save_frame_to_redis(stream_uuid: str, frame_data: bytes) -> bool:
    """
    将视频帧（字节数据）保存到 Redis，并设置过期时间。
    返回 True 表示成功，False 表示失败。
    """
    if not redis_client:
        logger.error("Redis 客户端未初始化，无法保存帧。")
        return False
    try:
        key = _get_frame_key(stream_uuid)
        # 使用 EX 参数设置过期时间
        redis_client.set(key, frame_data, ex=FRAME_EXPIRATION_SECONDS)
        return True
    except Exception as e:
        logger.error(f"保存视频帧到 Redis 失败 (UUID: {stream_uuid}): {e}", exc_info=True)
        return False

def get_frame_from_redis(stream_uuid: str) -> bytes or None:
    """
    从 Redis 获取视频帧（字节数据）。
    返回帧数据或 None（如果未找到或发生错误）。
    """
    if not redis_client:
        logger.error("Redis 客户端未初始化，无法获取帧。")
        return None
    try:
        key = _get_frame_key(stream_uuid)
        frame_data = redis_client.get(key)
        return frame_data
    except Exception as e:
        logger.error(f"从 Redis 获取视频帧失败 (UUID: {stream_uuid}): {e}", exc_info=True)
        return None 