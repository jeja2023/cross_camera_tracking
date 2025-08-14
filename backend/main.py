import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import datetime
import logging
import logging.handlers
import sys
from typing import List, Optional
from contextvars import ContextVar # 导入 ContextVar

import uvicorn
from fastapi import FastAPI, Depends, UploadFile, File, BackgroundTasks, Form, HTTPException, status, Query, Body, Request # 导入 Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import sqlalchemy as sa # 新增导入

from fastapi.middleware.cors import CORSMiddleware

from jose import JWTError, jwt # 导入 jwt 和 JWTError

from backend.database_conn import engine, Base, SessionLocal, create_tables, get_db, Log 
from backend.ml_services import ml_logic
from backend import crud
from backend import auth
from backend import schemas
from backend.routers import auth_routes, export_routes, person_routes, video_routes, page_routes, admin_routes, stream_routes, image_analysis_routes, human_in_the_loop_routes, followed_person_routes # 确保导入了所有路由模块，包括stream_routes和image_analysis_routes以及human_in_the_loop_routes 和 followed_person_routes
from backend.config import settings # 导入 settings

# 导入 dotenv 并加载环境变量 (确保在任何使用 os.getenv 的地方之前加载)
from dotenv import load_dotenv
load_dotenv() # 确保在导入 settings 之前加载环境变量

# --- 定义 ContextVar 来存储请求上下文 ---
request_user: ContextVar[Optional[str]] = ContextVar('request_user', default=None)
request_ip: ContextVar[Optional[str]] = ContextVar('request_ip', default=None)

# --- 自定义日志过滤器 ---
class ContextFilter(logging.Filter):
    def filter(self, record):
        record.username = request_user.get()
        record.ip_address = request_ip.get()
        return True

# --- 自定义数据库日志处理程序 ---
class DBLogHandler(logging.Handler):
    def __init__(self, session_local):
        super().__init__()
        self.SessionLocal = session_local
        self.addFilter(ContextFilter()) # 添加自定义过滤器

    def emit(self, record):
        # 只有在应用完全启动后才尝试写入数据库，避免在初始化阶段因DB连接问题而报错
        # 并且只记录我们感兴趣的日志级别
        if record.name.startswith("uvicorn") or record.name.startswith("fastapi"):
            return # 忽略uvicorn和fastapi自身的日志，它们可能过于频繁或重复

        if record.levelname not in ["INFO", "WARNING", "ERROR", "CRITICAL"]:
            return # 只记录特定级别的日志
            
        try:
            db = self.SessionLocal()
            log_entry = schemas.LogEntry(
                timestamp=datetime.datetime.fromtimestamp(record.created),
                logger=record.name,
                level=record.levelname,
                message=self.format(record),
                username=record.username, # 从 record 中获取 username
                ip_address=record.ip_address # 从 record 中获取 ip_address
            )
            crud.create_log(db, log_entry)
            db.close()
        except Exception as e:
            # 记录到标准错误输出，避免递归日志循环
            print(f"Error writing log to database: {e}", file=sys.stderr)

# --- 日志配置 ---
# 清除所有现有的处理程序，避免重复日志
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 创建一个控制台处理器，并设置其级别为 INFO，以显示所有信息
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING) # 修改为 WARNING
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')) # 添加 formatter

# 创建数据库日志处理程序
db_handler = DBLogHandler(SessionLocal)
# 数据库日志处理器将接收所有通过根日志器级别的日志，并根据其内部逻辑进行过滤和记录
# 无需在此处额外设置 db_handler.setLevel，因为其 emit 方法已包含过滤逻辑

# 设置日志基本配置
logging.basicConfig(
    level=settings.LOG_LEVEL, # 从 settings 中读取日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        console_handler, # 在控制台显示 WARNING 及更高级别的日志
        db_handler # 添加数据库日志处理程序
    ]
)
logger = logging.getLogger(__name__)

# 确保必要的目录存在 (使用 settings 中的路径)
# settings.init_directories() # 调用 Settings 类的初始化方法来创建目录，此行是错误的，__init__ 会自动调用

app = FastAPI(title="跨域智能识别系统 API")


# --- 中间件用于设置日志上下文 ---
@app.middleware("http")
async def log_context_middleware(request: Request, call_next):
    username = None
    ip_address = request.client.host if request.client else "N/A"

    # 尝试从认证信息中获取用户名
    try:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]) # 直接解码JWT
            username = payload.get("sub") # 从payload中获取用户名
    except JWTError:
        # 忽略 JWT 错误，例如令牌过期或无效，因为有些路由可能不需要认证
        pass
    except Exception as e:
        # 捕获其他潜在错误，例如头部格式不正确等
        logger.debug(f"Error decoding token in middleware: {e}") # 使用 debug 级别避免过多日志

    token_user = request_user.set(username)
    token_ip = request_ip.set(ip_address)
    try:
        response = await call_next(request)
    finally:
        request_user.reset(token_user)
        request_ip.reset(token_ip)
    return response

# 应用启动时创建数据表并初始化默认管理员账户
async def startup_event():
    logger.info("应用启动中...")
    try:
        logger.info("尝试创建数据库表...")
        create_tables() # Ensure tables are created
        logger.info("数据库表已成功创建或已存在。")
    except Exception as e:
        logger.error(f"创建数据库表失败: {e}", exc_info=True)
        # 如果数据库表创建失败，应用可能无法正常运行，可以考虑在此处退出或采取其他恢复措施
        # sys.exit(1) # 可以取消注释此行以在启动失败时退出

    try:
        logger.info("检查并创建默认管理员账户...")
        db = SessionLocal() # 获取一个新的数据库会话
        # Check if default admin user exists, create if not
        admin_user = crud.get_user_by_username(db, settings.DEFAULT_ADMIN_USERNAME)
        if not admin_user:
            admin_user_create = schemas.UserCreate(
                username=settings.DEFAULT_ADMIN_USERNAME,
                password=settings.DEFAULT_ADMIN_PASSWORD, # 从settings读取明文密码
                role="admin",
                unit=settings.DEFAULT_ADMIN_UNIT,
                phone_number=settings.DEFAULT_ADMIN_PHONE_NUMBER,
                is_active=True
            )
            crud.create_user(db, admin_user_create)
            logger.info("默认管理员账户已创建。")
        else:
            logger.info("管理员账户已存在，跳过创建。")
        db.close()
    except Exception as e:
        logger.error(f"检查或创建管理员账户失败: {e}", exc_info=True)

    # 在应用启动时初始化或加载配置
    logger.info("初始化或从数据库加载系统配置...")
    db_config = SessionLocal() # 获取一个新的数据库会话用于配置加载
    try:
        settings.init_or_load_from_db(db_config)
        logger.info("系统配置已初始化或从数据库加载。")
    except Exception as e:
        logger.error(f"初始化或从数据库加载系统配置失败: {e}", exc_info=True)
        raise RuntimeError(f"应用启动失败：无法加载系统配置: {e}") # 严重错误，阻止应用启动
    finally:
        db_config.close() # 确保关闭数据库会话

    logger.info("初始化 FAISS 索引...")
    try:
        db = SessionLocal() # 获取一个新的数据库会话
        ml_logic.initialize_faiss_index(db)
        db.close()
        logger.info("FAISS 索引初始化完成。")
    except Exception as e:
        logger.error(f"FAISS 索引初始化失败: {e}", exc_info=True)

    logger.info("应用启动完成。")

    # 调试：检查 persons 表结构
    try:
        db_check = SessionLocal()
        inspector = sa.inspect(db_check.bind)
        columns = inspector.get_columns('persons')
        column_names = [col['name'] for col in columns]
        logger.info(f"'persons' 表中的列: {column_names}")
        if 'is_trained' not in column_names:
            logger.error("错误: 'persons' 表中缺少 'is_trained' 列。")
        db_check.close()
    except Exception as e:
        logger.error(f"检查 'persons' 表结构失败: {e}", exc_info=True)

    # Initialize default system configurations if they don't exist
    db = SessionLocal()
    try:
        # Set default for GLOBAL_SEARCH_MIN_CONFIDENCE
        global_search_confidence_config = crud.get_system_config(db, 'GLOBAL_SEARCH_MIN_CONFIDENCE')
        if not global_search_confidence_config:
            crud.set_system_config(db, 'GLOBAL_SEARCH_MIN_CONFIDENCE', '0.9')
            logger.info("Initialized default GLOBAL_SEARCH_MIN_CONFIDENCE to 0.9.")
        else:
            logger.info(f"GLOBAL_SEARCH_MIN_CONFIDENCE already set to {global_search_confidence_config.value}.")

    except Exception as e:
        logger.error(f"Failed to initialize default system configurations: {e}", exc_info=True)
    finally:
        db.close()

app.add_event_handler("startup", startup_event)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(','),  # 从配置中获取允许的前端地址
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# 挂载静态文件目录 (使用 settings 中的路径)
app.mount("/static", StaticFiles(directory=os.path.join(settings.BASE_DIR, "frontend", "static")), name="static")
# 挂载解析后的图片目录，为 crops 和 full_frames 分别创建路由
app.mount("/database/crops", StaticFiles(directory=settings.DATABASE_CROPS_DIR), name="database_crops")
app.mount("/database/full_frames", StaticFiles(directory=settings.DATABASE_FULL_FRAMES_DIR), name="database_full_frames")
app.mount("/saved_streams", StaticFiles(directory=settings.SAVED_STREAMS_DIR), name="saved_streams")

# 包含认证路由
app.include_router(auth_routes.router, prefix="/auth")
# 包含视频路由
app.include_router(video_routes.router, prefix="/videos")
# 包含视频流路由
app.include_router(stream_routes.router, prefix="/streams")

# 包含人员路由
app.include_router(person_routes.router, prefix="/api/persons")
# 包含导出路由
app.include_router(export_routes.router, prefix="/export")

# 临时测试路由，用于调试 /persons/search 的 405 错误
@app.get("/persons/test-search", summary="临时测试路由")
async def test_persons_search():
    return {"message": "Test persons search GET route is working!"}

# 将页面路由和通配符路由放在所有API路由之后
# 包含页面路由
app.include_router(page_routes.router)
# 包含管理员路由
app.include_router(admin_routes.router)

# 包含图片分析路由
app.include_router(image_analysis_routes.router)

# 包含人机回环路由
app.include_router(human_in_the_loop_routes.router)

# 包含关注人员路由
app.include_router(followed_person_routes.router)

# 将根路径重定向到 index.html
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def read_root_html():
    logger.info(f"Serving login page from: {settings.LOGIN_PAGE_PATH}") # 添加日志
    return settings.LOGIN_PAGE_PATH # 直接返回登录页面

# 调试路由：检查登录页面路径
@app.get("/debug-login-path", include_in_schema=False)
async def debug_login_path():
    file_path = os.path.join(settings.BASE_DIR, "frontend", "static", "index.html")
    exists = os.path.exists(file_path)
    is_file = os.path.isfile(file_path)
    print(f"DEBUG: Attempting to serve file: {file_path}")
    print(f"DEBUG: File exists: {exists}")
    print(f"DEBUG: Is a file: {is_file}")
    if exists and is_file:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read(100) # 读取文件前100个字符
        return {"message": "File exists and is readable", "path": file_path, "exists": exists, "is_file": is_file, "content_sample": content}
    return {"message": "File not found or not readable", "path": file_path, "exists": exists, "is_file": is_file}


@app.get("/{page_name}", response_class=FileResponse, include_in_schema=False)
async def serve_html_pages(page_name: str):
    # 动态检查请求的HTML文件是否存在，而不是使用硬编码列表
    # 检查 page_name 是否已包含 .html 后缀
    if page_name.endswith(".html"):
        final_page_name = page_name
    else:
        final_page_name = f"{page_name}.html"

    file_path = os.path.join(settings.BASE_DIR, "frontend", "static", final_page_name)
    if os.path.exists(file_path):
        return file_path
    raise HTTPException(status_code=404, detail="Page not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)