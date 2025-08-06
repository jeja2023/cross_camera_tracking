# 跨域智能识别系统

## 项目简介
本项目旨在构建一个先进的基于视频流的跨摄像机人物追踪系统。它集成了实时视频流处理、人物检测、特征提取、深度学习驱动的跨摄像机ID重识别功能，并提供一个直观的Web管理界面，用于高效地查看、管理视频流及其详尽的分析结果。系统设计注重性能、可扩展性和用户友好性。

## 主要功能
- **视频流管理**：支持添加、启动、停止和删除RTSP/RTMP视频流，并提供灵活的视频录制与保存功能。
- **实时解析与数据存储**：对接入的视频流进行实时高效解析，智能提取人物图像和特征向量。裁剪后的人物图像（`backend/database/crops`）和完整的视频帧（`backend/database/full_frames`）被结构化存储，支持后续的深度分析和可视化呈现。
- **综合智能识别能力**：系统能够自动执行全面的识别任务，包括：**人体检测**、**姿态估计**、**骨骼识别**、**人脸检测**、**人脸识别**以及**粗粒度服装属性识别**，这些功能适用于从图像、视频和视频流中提取数据。所有提取的属性都可以存储在数据库中，以便后续分析和检索。**步态分析**功能被视为未来的增强项。
- **跨摄像机ID重识别**：利用先进的深度学习模型（如OSNet），实现复杂环境下不同摄像机间同一人物的精确ID重识别，显著提升追踪的准确性和鲁棒性。
- **模型再训练**：支持对Re-ID模型进行再训练，以适应新的数据和环境，持续优化识别精度。
- **结果可视化与检索**：提供用户友好的前端界面，用于直观展示已解析视频流的追踪结果和人物裁剪图。系统支持基于人物ID、时间戳、摄像机来源等多维度的高级检索功能。
- **后台任务处理**：利用Celery分布式任务队列进行异步任务处理（如耗时的人物识别和特征提取），确保系统在高并发请求下仍能保持卓越的响应性和可扩展性。
- **用户认证与权限管理**：集成健壮的认证（JWT）和授权机制，保障系统安全性，并可根据不同用户角色精细控制访问权限。
- **图像分析与人机回环**：支持独立的图像文件分析功能，并提供人机回环（Human-in-the-Loop）机制，允许人工介入审核和校正机器学习结果，持续提升系统精度。
- **关注人员管理与智能比对**：支持关注特定人物，对其注册图片进行全库比对。新增了**每位关注人员的实时比对开关**，允许管理员灵活控制新检测到的人物是否自动与关注人员的注册图片进行实时比对，并将高置信度的比对结果（例如90%以上）添加到预警页面。同时，支持**定时全库比对**，确保关注人员数据库的持续更新和比对。
- **自动追踪功能区**：新增自动追踪功能区，包含“历史追踪”和“实时追踪”两个子页面，并集成了统一的系统导航栏进行页面切换。

## 项目结构

- `backend/`: 后端服务，基于FastAPI框架构建，提供核心业务逻辑和API接口。
    - `auth.py`: 负责用户认证（如JWT令牌生成与验证）和授权逻辑，管理用户登录、注册等流程。
    - `crud.py`: 数据库的CRUD（创建、读取、更新、删除）操作封装，提供与数据库交互的通用接口。
    - `ml_services/`: 包含所有机器学习相关的服务和逻辑，如人物检测（YOLOv8）、特征提取、ID重识别模型调用，以及图像预处理和后处理算法。
    - `models/`: 存放预训练的深度学习模型文件（如ONNX格式），供 `ml_services` 调用。
    - `routers/`: 定义了后端API的各个功能模块路由：
        - `admin_routes.py`: 管理员专属API接口，用于系统配置和用户管理。
        - `auth_routes.py`: 用户认证和授权相关API。
        - `stream_routes.py`: 视频流的添加、控制和状态查询API。
        - `person_routes.py`: 人员信息、ID管理和查询API。
        - `video_routes.py`: 视频文件上传、分析和结果查询API。
        - `image_analysis_routes.py`: 独立图像文件上传和分析API。
        - `human_in_the_loop_routes.py`: 人机回环审核流程相关API。
        - `export_routes.py`: 数据导出功能API。
        - `page_routes.py`: 用于渲染静态HTML页面的路由。
        - `__init__.py`: Python包初始化文件。
    - `database/`: 数据库相关文件和存储目录。
        - `crops/`: 存储从视频帧中裁剪出的人物图像。
        - `full_frames/`: 存储原始视频的完整帧图像。
        - `image/`: 图像数据。
        - `stream/`: 流数据。
        - `video/`: 视频数据。
    - `saved_streams/`: 存储已录制的视频流文件。
    - `enroll_person_images/`: 存储主动注册功能上传的图片（位于 `backend/enroll_person_images/`，完全替换原 `uploaded_search_images/` 目录）。
    - `uploads/`: 通用文件上传目录，例如视频上传。
    - `celery_app.py`, `celery_worker.py`, `celeryconfig.py`: Celery异步任务的配置、应用实例和工作者启动脚本，用于处理后台计算密集型任务。
    - `config.py`: 项目的全局配置，如数据库连接字符串、JWT密钥、目录路径等。
    - `database_conn.py`: 数据库连接的初始化和会话管理。
    - `main.py`: FastAPI应用的主入口文件，负责应用初始化、中间件配置、路由注册和事件处理。
    - `schemas.py`: 定义Pydantic模型，用于请求体、响应体和数据库模型的类型校验与序列化。
    - `stream_manager.py`: 负责管理视频流的生命周期和处理逻辑。
- `frontend/`: 前端静态文件，提供用户交互界面和可视化功能。
    - `static/`: 包含所有前端静态资源。
        - `css/`: 包含项目的样式表文件，定义了页面的视觉风格和布局。
        - `js/`: 包含前端JavaScript逻辑文件，处理页面交互、API请求、数据展示等，例如 `auth.js` (认证逻辑), `common.js` (通用工具函数), 各功能模块对应的 `*.js` 文件。
        - `*.html`: 各种功能页面的HTML文件，如 `index.html` (主页), `login.html` (登录页), `admin.html` (管理员界面), `video_results.html` (视频分析结果), `live_stream_results.html` (实时流结果), `image_search.html` (以图搜人页面), `human_review.html` (人机回环审核页) 等。
- `requirements.txt`: Python项目的所有依赖库及其精确版本列表，用于环境搭建。
- `.env`: 环境变量配置文件示例（通常不提交到版本控制），用于敏感信息和环境特定配置。
- `README.md`: 本项目说明文档。
- `venv/`: Python 虚拟环境目录。

```
. # 项目根目录
├── .env
├── backend/
│   ├── __init__.py
│   ├── auth.py
│   ├── celery_app.py
│   ├── celery_worker.py
│   ├── celeryconfig.py
│   ├── config.py
│   ├── crud.py
│   ├── database_conn.py
│   ├── main.py
│   ├── schemas.py
│   ├── stream_manager.py
│   ├── database/
│   │   ├── crops/
│   │   ├── full_frames/
│   │   ├── image/
│   │   ├── stream/
│   │   └── video/
│   ├── ml_services/
│   │   ├── __init__.py
│   │   ├── ml_logic.py
│   │   └── ml_tasks.py
│   ├── models/
│   │   └── *.onnx (模型文件)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── admin_routes.py
│   │   ├── auth_routes.py
│   │   ├── export_routes.py
│   │   ├── human_in_the_loop_routes.py
│   │   ├── image_analysis_routes.py
│   │   ├── page_routes.py
│   │   ├── person_routes.py
│   │   ├── stream_routes.py
│   │   └── video_routes.py
│   ├── saved_streams/
│   │   └── */*.mp4 (录制的视频流)
│   ├── enroll_person_images/  # 主动注册图片目录，位于 backend/enroll_person_images/
│   │   └── *.jpg/*.png (上传的搜索图片)
│   └── uploads/
│       └── *.mp4 (上传的视频文件)
├── frontend/
│   └── static/
│       ├── css/
│       │   └── *.css
│       ├── js/
│       │   └── *.js
│       └── *.html
├── README.md
├── requirements.txt
└── venv/
```

## 技术栈
- **后端**: Python 3.9+, FastAPI, Uvicorn, SQLAlchemy, Celery, Redis (或 RabbitMQ), jose, bcrypt, python-dotenv, OpenCV (opencv-python), scikit-image, numpy, Pillow, ultralytics (YOLOv8), ONNX Runtime, PyTorch (或 torch), filterpy, gevent, ujson, aiofiles。
- **前端**: HTML5, CSS3, JavaScript (原生JS，无大型前端框架依赖), Fetch API。
- **数据库**: MySQL (通过 PyMySQL 驱动), 或 SQLite (开发环境推荐)。
- **异步任务队列**: Redis 或 RabbitMQ (作为 Celery 的消息代理和结果后端)。
- **深度学习模型**: OSNet (ONNX格式), YOLOv8 (用于检测)。

## 安装与运行

### 1. 克隆仓库
```bash
git clone https://github.com/jeja2023/cross_camera_tracking.git
cd cross_camera_tracking
```

### 2. 配置环境变量
在项目根目录下创建 `.env` 文件，并根据您的实际环境配置以下变量。为了方便起见，您可以直接复制 `.env.example` 文件的内容作为起点。

以下是一些关键配置项的示例，详细说明请参考 `.env` 文件本身：
```dotenv
# --- 数据库配置 ---
DATABASE_URL="mysql+pymysql://user:password@host:port/database_name"

# --- Celery 异步任务队列配置 ---
CELERY_BROKER_URL="redis://localhost:6379/0"
CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# --- Redis 服务配置 (直接连接) ---
REDIS_URL="redis://localhost:6379/0"

# --- CORS 跨域资源共享配置 ---
CORS_ORIGINS="*"
CORS_ALLOW_CREDENTIALS="true"
CORS_ALLOW_METHODS="GET,POST,PUT,DELETE,OPTIONS"
CORS_ALLOW_HEADERS="*"

# --- JWT (JSON Web Token) 认证配置 ---
SECRET_KEY="your_super_secret_jwt_key_please_change_this_to_a_strong_random_string"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# --- 默认管理员用户配置 ---
DEFAULT_ADMIN_USERNAME="admin"
DEFAULT_ADMIN_PASSWORD="your_admin_password_here"

# --- 日志配置 ---
LOG_LEVEL="INFO"
LOG_FILE_PATH=""
LOG_FILE_MAX_BYTES=10485760
LOG_FILE_BACKUP_COUNT=5

# --- 模型文件存放配置 ---
MODELS_DIR_NAME="models"
DEVICE_TYPE="CPU"
DETECTION_MODEL_FILENAME="yolov8n.onnx"
REID_MODEL_FILENAME="osnet_ibn_x1_0.onnx"
POSE_MODEL_FILENAME="yolov8n-pose.pt"
FACE_DETECTION_MODEL_FILENAME="yolov8n-face.onnx"
FACE_RECOGNITION_MODEL_FILENAME="arcface.onnx"
GAIT_RECOGNITION_MODEL_FILENAME=
CLOTHING_ATTRIBUTE_MODEL_FILENAME=

# --- 模型输入/输出特征维度配置 ---
REID_INPUT_WIDTH=128
REID_INPUT_HEIGHT=256
FEATURE_DIM=512
FACE_FEATURE_DIM=512
GAIT_FEATURE_DIM=512

# --- 多模态融合权重配置 ---
REID_WEIGHT=0.7
FACE_WEIGHT=0.3
GAIT_WEIGHT=0.0

# --- Re-Ranking 算法参数 ---
K1=20
K2=6
LAMBDA_VALUE=0.3

# --- 步态序列长度配置 ---
GAIT_SEQUENCE_LENGTH=30

# --- Faiss 索引配置 ---
FAISS_INDEX_PATH="faiss_index.bin"
FAISS_ID_MAP_PATH="faiss_id_map.json"
FAISS_DIMS=512
FAISS_METRIC="COSINE"
FAISS_SEARCH_K=20

# --- 人机回环审核配置 ---
HUMAN_REVIEW_CONFIDENCE_THRESHOLD=0.5

# --- 图片分析配置 ---
IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE=0.5

# --- 主动注册配置 ---
ENROLLMENT_MIN_PERSON_CONFIDENCE=0.4

# --- Tracker 和检测相关配置 ---
TRACKER_PROXIMITY_THRESH=0.5
TRACKER_APPEARANCE_THRESH=0.25
TRACKER_HIGH_THRESH=0.1
TRACKER_LOW_THRESH=0.1
TRACKER_NEW_TRACK_THRESH=0.1
TRACKER_MIN_HITS=1
TRACKER_TRACK_BUFFER=30
VIDEO_PROCESSING_FRAME_RATE=25
STREAM_PROCESSING_FRAME_RATE=25
VIDEO_COMMIT_BATCH_SIZE=100
DETECTION_CONFIDENCE_THRESHOLD=0.5
PERSON_CLASS_ID=0

# --- Excel 导出相关配置 ---
EXCEL_EXPORT_MAX_IMAGES=30
EXCEL_EXPORT_IMAGE_SIZE_PX=100
EXCEL_EXPORT_ROW_HEIGHT_PT=75

# --- MJPEG 视频流帧率配置 ---
MJPEG_STREAM_FPS=20

# --- 文件上传和视频进度流配置 ---
UPLOAD_CHUNK_SIZE_BYTES=1048576
VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS=1
```
**注意**: 在生产环境中，请务必设置一个强 `SECRET_KEY` 和 `DEFAULT_ADMIN_PASSWORD`，并根据实际部署情况调整数据库、Redis 等服务连接信息。对于模型文件，请确保将其放置在 `backend/models` 目录下或 `.env` 中 `MODELS_DIR_NAME` 指定的路径。

### 3. 创建并激活Python虚拟环境
```bash
python -m venv venv
# Windows
venv/Scripts/activate
# macOS/Linux
source venv/bin/activate
```

### 4. 安装依赖
```bash
pip install -r requirements.txt
```

### 5. 启动后端服务

#### 5.1 启动 FastAPI 应用
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
（`--reload` 选项适用于开发环境，生产环境应移除以提高性能和稳定性。）

#### 5.2 启动 Celery Worker
如果项目需要异步任务处理（例如视频解析和特征提取），您需要启动 Celery Worker。请确保您已安装并运行了 Redis 或 RabbitMQ 作为消息代理。

**使用 Redis 作为消息代理的示例：**
在 `.env` 文件中配置 `REDIS_BROKER_URL` 和 `REDIS_RESULT_BACKEND`。

然后，在**新的终端**中启动 Celery Worker：
```bash
celery -A backend.celery_app worker -l info -P gevent
```
（`-P gevent` 是为了支持异步操作和协程，如果遇到问题可以尝试替换为 `eventlet` 或不指定以使用默认的 `prefork`。）

### 6. 访问前端
在浏览器中打开以下地址访问系统前端：
```
http://localhost:8000/static/index.html
```
或者，直接访问根路径通常会被重定向到 `index.html`：
```
http://localhost:8000/
```
根据您的实际部署情况，`host` 和 `port` 可能有所不同。

## 新功能与改进 (更新日志)

*   **自动追踪功能区**：
    *   新增“自动追踪”功能区，包含“历史追踪”和“实时追踪”两个可切换页面。
    *   在页面中添加了与页面名称相同的按钮。
    *   将“历史追踪”和“实时追踪”页面切换按钮集成到统一的系统顶部导航栏中，并删除原先独立的导航栏。
    *   优化了“自动追踪”页面的整体布局和样式，使其导航栏与系统其他功能区的设计保持一致。

*   **Re-ID 模型再训练功能**：
    *   新增了通过 Celery 任务触发 Re-ID 模型再训练的功能，支持指定人物 UUIDs 或对所有标记为再训练的人物数据进行训练。
    *   优化了数据准备和模型训练过程的日志输出，提供了更详细的进度信息。

*   **关注人员功能增强与比对策略优化**：
    *   **每人实时比对开关**：在“关注人员”列表中为每位关注人员引入了独立的“实时比对”开关，仅管理员可见，允许精确控制单个人物的实时比对行为。
    *   **定时全库比对**：新增了可配置的定时全库比对功能，支持设定比对间隔，自动对关注人员进行全库比对。
    *   **预警图片过滤**：预警页面默认显示比对分值90%或更高的图片。
    *   **图片解析自动关联**：新增配置项 `AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE`，用于设置图片解析后自动关联到关注人员的最低置信度。
    *   **前端交互优化**：在“关注人员”页面重新添加了“查看注册图片”按钮，并修复了前端页面链接和API调用路径的若干问题，提升了用户体验和功能稳定性。

*   **人脸和步态识别功能**：
    *   新增人脸识别和步态识别功能，增强了多模态识别能力，提高了识别的准确性。
    *   在`.env`文件中增加了`FACE_DETECTION_MODEL_FILENAME`、`FACE_RECOGNITION_MODEL_FILENAME`、`GAIT_RECOGNITION_MODEL_FILENAME`、`FACE_FEATURE_DIM`、`GAIT_FEATURE_DIM`、`FACE_WEIGHT`、`GAIT_WEIGHT`、`GAIT_SEQUENCE_LENGTH`等配置项，用于支持新的模型和特征维度。

*   **人员信息表分页按钮样式调整**：
    *   将“人员信息表”中分页的“上一页”和“下一页”按钮调整为蓝色，提升视觉一致性。

*   **图片解析页面功能增强**：
    *   修复了图片解析页面上传人信息无法正确显示的问题。
    *   在“以图搜人”结果中，上传图片信息现在只显示 UUID，不再显示文件名，提高了显示简洁性。

*   **视频解析页面改进**：
    *   调整了“视频解析”页面中操作按钮的顺序，现在“查看结果”按钮显示在“删除”按钮之前，提升了用户体验。

*   **后端稳定性与数据一致性**：
    *   解决了 `backend/ml_services/ml_logic.py` 中 `CROP_DIR` 和 `FULL_FRAME_DIR` 未正确引用的问题。
    *   修复了 `backend/ml_services/ml_tasks.py` 中 `progress` 变量未初始化可能导致的错误。
    *   解决了 `backend/ml_services/ml_logic.py` 在处理特征向量时 `json.loads` 导致的 `TypeError`。
    *   修复了 `backend/ml_services/ml_logic.py` 中调用 `crud.get_image` 的 `AttributeError`，改为使用正确的 `crud.get_image_by_uuid`。
    *   解决了 `backend/routers/stream_routes.py` 中 `Person` 对象访问 `video` 属性的 `AttributeError`，确保只访问与对象类型匹配的属性。
    *   更新了 `backend/schemas.py` 中 `Person` 和 `ImageSearchResult` 模型，增加了 `upload_image_uuid` 和 `upload_image_filename` 字段，以更好地支持图片关联数据。
    *   改进了 `backend/crud.py` 中的数据加载逻辑，通过急切加载关联信息，提升了数据获取的效率和准确性。
    *   优化了 `backend/routers/person_routes.py` 中的数据处理和转换，减少了冗余步骤。
    *   消除了 `frontend/static/css/video_analysis.css` 中空的 CSS 规则集警告，保持代码整洁。

## API 文档
FastAPI 会自动生成交互式API文档。当后端服务启动后，您可以通过以下地址访问：
- **OpenAPI (Swagger UI)**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 许可证
本项目采用 MIT 许可证。详见 `LICENSE` 文件（如果存在）。

## 贡献
欢迎通过提交问题（Issues）和拉取请求（Pull Requests）为本项目贡献力量。在提交之前，请确保阅读贡献指南（如果存在）。 