# backend/celeryconfig.py
import os
from dotenv import load_dotenv
import sys

load_dotenv() # 确保在配置文件中也能加载 .env 变量

# --- 数据库连接 URL (如果 Celery 任务需要直接访问数据库，否则可移除此项) ---
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("错误：Celery 配置中缺少数据库连接字符串 'DATABASE_URL'。请在 .env 文件中设置。", file=sys.stderr)
    sys.exit(1)

# --- Celery Broker 和 Backend URL ---
# 从 .env 文件获取，如果没有设置则强制报错
BROKER_URL = os.getenv("CELERY_BROKER_URL")
if not BROKER_URL:
    print("错误：Celery Broker URL 'CELERY_BROKER_URL' 未在您的 .env 文件中设置。请在 .env 文件中添加。", file=sys.stderr)
    sys.exit(1)

RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")
if not RESULT_BACKEND:
    print("错误：Celery Result Backend 'CELERY_RESULT_BACKEND' 未在您的 .env 文件中设置。请在 .env 文件中添加。", file=sys.stderr)
    sys.exit(1)

# --- Celery 任务序列化和内容接受 ---
TASK_SERIALIZER = os.getenv("CELERY_TASK_SERIALIZER")
if not TASK_SERIALIZER:
    print("错误：Celery TASK_SERIALIZER 未在您的 .env 文件中设置。请在 .env 文件中添加，例如：CELERY_TASK_SERIALIZER='json'", file=sys.stderr)
    sys.exit(1)

RESULT_SERIALIZER = os.getenv("CELERY_RESULT_SERIALIZER")
if not RESULT_SERIALIZER:
    print("错误：Celery RESULT_SERIALIZER 未在您的 .env 文件中设置。请在 .env 文件中添加，例如：CELERY_RESULT_SERIALIZER='json'", file=sys.stderr)
    sys.exit(1)

ACCEPT_CONTENT_STR = os.getenv("CELERY_ACCEPT_CONTENT")
if not ACCEPT_CONTENT_STR:
    print("错误：Celery ACCEPT_CONTENT 未在您的 .env 文件中设置。请在 .env 文件中添加，例如：CELERY_ACCEPT_CONTENT='json'", file=sys.stderr)
    sys.exit(1)
ACCEPT_CONTENT = ACCEPT_CONTENT_STR.split(',')

# --- Celery 时区设置 ---
TIMEZONE = os.getenv("CELERY_TIMEZONE")
if not TIMEZONE:
    print("错误：Celery TIMEZONE 未在您的 .env 文件中设置。请在 .env 文件中添加，例如：CELERY_TIMEZONE='Asia/Shanghai'", file=sys.stderr)
    sys.exit(1)

ENABLE_UTC_STR = os.getenv("CELERY_ENABLE_UTC")
if not ENABLE_UTC_STR:
    print("错误：Celery ENABLE_UTC 未在您的 .env 文件中设置。请在 .env 文件中添加，例如：CELERY_ENABLE_UTC='True'", file=sys.stderr)
    sys.exit(1)
ENABLE_UTC = ENABLE_UTC_STR.lower() == "true"

# --- 其他 Celery 配置 ---
TASK_ACKS_LATE_STR = os.getenv("CELERY_TASK_ACKS_LATE")
if not TASK_ACKS_LATE_STR:
    print("错误：Celery TASK_ACKS_LATE 未在您的 .env 文件中设置。请在 .env 文件中添加，例如：CELERY_TASK_ACKS_LATE='True'", file=sys.stderr)
    sys.exit(1)
TASK_ACKS_LATE = TASK_ACKS_LATE_STR.lower() == "true"

WORKER_PREFETCH_MULTIPLIER_STR = os.getenv("CELERY_WORKER_PREFETCH_MULTIPLIER")
if not WORKER_PREFETCH_MULTIPLIER_STR:
    print("错误：Celery WORKER_PREFETCH_MULTIPLIER 未在您的 .env 文件中设置。请在 .env 文件中添加，例如：CELERY_WORKER_PREFETCH_MULTIPLIER=1", file=sys.stderr)
    sys.exit(1)
try:
    WORKER_PREFETCH_MULTIPLIER = int(WORKER_PREFETCH_MULTIPLIER_STR)
except ValueError:
    print(f"错误：CELERY_WORKER_PREFETCH_MULTIPLIER 的值 '{WORKER_PREFETCH_MULTIPLIER_STR}' 无效。必须是一个整数。", file=sys.stderr)
    sys.exit(1)

# 任务的导入路径
# 确保 Celery worker 知道在哪里找到你的任务
# 例如，如果你的任务在 backend/ml_services/ml_tasks.py 中定义，则添加 'backend.ml_services.ml_tasks'
INCLUDE = ['backend.ml_services.ml_tasks']