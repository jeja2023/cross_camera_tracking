import os
import sys
from celery import Celery
from dotenv import load_dotenv
import logging # Import logging

logger = logging.getLogger(__name__)

# 获取当前文件的目录 (backend)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (cross_camera_tracking)
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到 Python 路径，确保在所有进程中都可用
# 这是一个更健壮的方式，尤其是在使用 multiprocessing 或 gevent 时
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added project root to sys.path: {project_root}") # Add log

load_dotenv() # 加载 .env 文件中的环境变量

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 解决 OpenMP 运行时冲突

celery_app = Celery(
    'cross_camera_tracking',  # 应用名称
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"), # 从 .env 文件获取 BROKER_URL
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0"), # 从 .env 文件获取 RESULT_BACKEND
    include=['backend.ml_services.ml_tasks']  # 告诉 Celery worker 在哪里找到任务
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='Asia/Shanghai',
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    # 明确导入 backend 模块，这有时能解决 multiprocessing 相关的导入问题
    imports=('backend', 'backend.ml_services', 'backend.crud', 'backend.schemas', 'backend.database_conn'), # Ensure backend is explicitly imported
    broker_pool_limit=0, # Recommended for gevent or eventlet to prevent blocking issues
)

logger.info(f"Celery app configured with sys.path: {sys.path}") # Log final sys.path 