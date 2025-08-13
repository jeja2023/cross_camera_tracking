# 跨域智能识别系统

## 项目简介

**跨域智能识别系统** 是一个基于人工智能和多媒体分析的综合性解决方案，旨在实现对人员的智能识别、追踪和管理。系统能够处理来自图片、视频和实时视频流的数据，提取人物特征，并进行高效的相似度比对，广泛应用于安全监控、智能识别、人员管理等领域。

## 最新更新 (2025年8月13日)

我们对系统进行了持续的更新和功能增强，旨在提升用户体验、完善数据管理和修复已知问题。以下是系统主要特性和近期改进的概述：

### 配置管理与模型生命周期

*   **系统配置持久化与动态加载**：
    *   在应用程序首次启动时，系统将自动把 `.env` 文件中的默认配置（包括模型路径、参数阈值等）加载并写入数据库的 `system_configs` 表中。后续应用程序启动时，将优先从数据库加载配置，确保配置的持久化和一致性，避免重启应用时 `.env` 文件覆盖数据库中更新的配置。
    *   前端“系统参数”页面现在直接从数据库实时获取最新配置，确保显示的是当前活跃的准确参数值，无需重启后端服务即可看到配置的变更。
*   **优化模型再训练命名策略**：
    *   新训练的 Re-ID 模型名称现在将始终基于**原始模型名称**（例如 `osnet_ibn_x1_0.onnx`），并在其后附加本次训练的时间戳（例如 `osnet_ibn_x1_0_20250812_102629.onnx`）。这解决了之前模型名称因多次再训练而过长的问题，并有助于模型版本回溯管理。
*   **模型路径存储与加载标准化**：
    *   `ACTIVE_REID_MODEL_PATH`（当前激活的 Re-ID 模型路径）在数据库中存储为相对于模型目录的**相对路径（文件名）**，避免了硬编码绝对路径。在加载模型时，后端会根据 `MODELS_DIR` 正确拼接出完整的模型文件路径。
*   **模型热加载能力增强**：
    *   Celery Worker 在执行模型再训练任务后，会自动刷新其内部的配置实例，确保后续处理任务（如图片分析）时能立即加载并使用新训练的模型，而无需手动重启 Celery Worker 或整个应用。这极大地提升了开发和运维效率。

### 功能增强与优化

*   **全面的图片分析数据删除机制**：
    *   在“已解析图片列表”中删除图片时，系统现在会**彻底删除**数据库中的相关记录，同时清理本地存储的**所有**关联图片文件，包括通用检测（`general_detection`）生成的裁剪图和全帧图片，以及人脸识别（`face_recognition`）和步态识别（`gait_recognition`）模型特有的图片。这确保了数据的一致性和存储空间的有效管理。
*   **优化视频解析数据删除流程**：
    *   在“上传视频解析”功能中删除已处理视频时，除了删除视频文件本身，系统现在也会同步删除该视频生成的所有本地裁剪图片和全帧图片，确保无冗余文件残留。
*   **完善视频流分析数据删除**：
    *   在“视频流解析”中删除已保存视频流时，现在会一并删除 `database` 目录下与该视频流相关的所有裁剪图片、全帧图片，以及保存的视频流文件，实现了更全面的数据清理。
*   **主动注册图片删除交互优化**：
    *   在“主动注册上传图片”页面，为了提供更美观的用户体验，我们将待上传图片旁边的删除按钮替换为**悬浮在图片右上角的“X”形删除图标**。当鼠标悬停在图片上时，该图标会平滑显示，点击即可移除图片，使得界面更加简洁直观。
*   **关注人员注册图片管理**：
    *   在“关注人员查看注册图片”页面，为每张注册图片添加了**删除按钮**。用户现在可以直接在此页面删除不需要的注册图片，系统将同步删除数据库记录和本地存储的文件。

### 错误修复

*   **解决“以图搜人”功能 `AttributeError`**：
    *   修复了在“以图搜人”功能中，当人物对象没有关联视频或流时，可能导致 `AttributeError: 'Person' object has no attribute 'video'` 的问题。现在系统能够更健壮地处理不同来源的人物信息展示。
*   **修复“人物注册”功能 `AttributeError`**：
    *   解决了在“人物注册”功能中，`PersonCreate` 对象尝试使用字典的 `.get()` 方法时导致的 `AttributeError: 'PersonCreate' object has no attribute 'get'` 错误。确保了人物注册流程的顺畅。
*   **依赖库兼容性问题解决**：
    *   处理了 `passlib` 和 `bcrypt` 库之间的兼容性问题，避免了 `AttributeError: module 'bcrypt' has no attribute '__about__'` 错误的发生。
*   **Celery 任务状态更新 `ValueError` 修复**：
    *   解决了 Celery Worker 在更新任务状态时可能出现的 `ValueError: Exception information must include the exception type` 错误，确保任务状态更新机制的稳定性。
*   **Re-ID 模型加载 `TypeError` 修复**：
    *   修复了在 `get_reid_session` 函数中，由于 `crud.get_system_config` 返回 ORM 对象而非字符串值，导致 `os.path.join` 拼接路径时出现的 `TypeError`。现在模型路径能够被正确地提取和拼接。

## 主要功能模块

本系统提供以下核心功能模块：

### 1. 用户认证与管理
- **用户注册**: 新用户可以进行注册，但需经管理员审核激活后方可登录。
- **用户登录**: 提供安全的登录机制，基于JWT (JSON Web Tokens) 进行身份验证。
- **个人信息管理**: 用户可以查看和更新自己的单位、电话号码等个人资料，并修改登录密码。
- **用户角色**: 支持`普通用户`、`高级用户`和`管理员`三种角色，不同角色拥有不同的操作权限。

### 2. 后台管理
- **模型参数配置**: 管理员可以查看和动态调整系统使用的各项AI模型参数，包括模型路径、特征维度、多模态融合权重、各种置信度阈值、追踪器参数等。部分更改可能需要重启服务生效。
- **系统日志查看**: 提供详细的系统运行日志，支持按日志级别、时间范围和关键词进行高级筛选，方便系统运维和故障排查。
- **用户管理**: 管理员可以查看所有注册用户列表，对用户进行激活/停用、修改角色和删除等操作。
- **模型再训练**: 管理员可以手动触发Re-ID模型再训练任务，以优化模型性能。支持通过Celery后台任务异步执行，并实时查询任务进度。

### 3. 人物管理与搜索
- **主动注册人物**: 用户可以上传人物图片，并提供姓名、身份证号等信息，将人物特征注册到系统特征库中。支持实时比对功能。
- **多模态搜人（以图搜人）**:
    - 支持上传图片进行人物特征提取，并在特征库中搜索相似人物。
    - 支持以已有的人物UUID作为查询目标进行搜索。
    - 搜索结果支持按相似度阈值过滤，并提供分页和分组显示。
- **全部特征图库**: 管理员和高级用户可查看系统中所有检测到的人物特征（包括图片、视频和视频流中提取的），支持分页、多条件筛选和模糊搜索。
- **人员档案**: 展示所有已审核且包含身份证号/ID的人物档案，方便管理和导出。
- **人物详情**: 通过UUID获取单个任务的详细信息。
- **人物关注**: 用户可以关注或取消关注特定人物，系统将对关注人物进行实时监控。
- **人物删除**: 管理员可以删除指定人物及其所有关联数据和文件。

### 4. 视频管理与处理
- **视频上传与解析**: 用户可以上传视频文件，系统将视频分发到Celery后台任务进行异步解析，包括人物检测、特征提取和人物跟踪。
- **视频处理进度**: 提供实时进度条和状态更新，用户可以随时查看视频解析的当前状态和进度。
- **视频列表与详情**: 查看所有已上传视频的列表，并支持查看单个视频的详细信息。
- **视频结果查看**: 查看特定视频中检测到的人物列表，包括其裁剪图、UUID、来源信息和入库时间。
- **视频处理终止**: 用户可以终止正在进行中的视频处理任务。
- **视频删除**: 视频所有者或管理员可以删除视频文件及其所有关联的解析数据和人物特征。

### 5. 视频流管理与处理
- **视频流启动与解析**: 高级用户和管理员可以启动RTSP或API视频流进行实时解析，包括人物检测、特征提取和人物跟踪。
- **视频流实时监控**: 前端可以实时显示带有标注框和ID的视频流画面。
- **视频流状态控制**: 支持停止、恢复视频流解析任务。
- **视频流列表**: 查看所有已添加的视频流列表及其状态。
- **视频流结果查看**: 查看特定视频流中检测到的人物列表。
- **视频流删除**: 视频流所有者或管理员可以删除视频流及其所有关联数据和保存的视频文件。
- **心跳机制**: 定期向后端发送请求，以保持视频流处理任务的活跃状态。

### 6. 图片分析与管理
- **图片上传与解析**: 用户可以上传静态图片，系统将对图片进行人物检测、人脸识别、姿态估计等分析，并提取人物特征。
- **图片分析历史**: 查看所有已分析图片的列表，支持分页。
- **图片分析结果**: 查看特定图片的详细分析结果，包括原始图片和其中检测到的人物信息。
- **图片删除**: 管理员可以删除指定图片及其所有关联的分析数据和人物特征。

### 7. 人机回环 (Human-in-the-Loop, HITL) 与模型纠正
- **未审核人物列表**: 提供一个界面，展示所有未经过人工审核或置信度较低的人物特征，等待人工干预。
- **人物审核**: 人工确认模型识别结果是否正确，将人物标记为“确认无误”。
- **人物纠正**:
    - **合并**: 将多个被误识别为不同个体的人物特征合并到同一个逻辑人物（Individual）下。
    - **误检**: 标记为模型误检，该人物将不再用于模型训练。
    - **重新标注**: 对人物进行身份信息的重新标注。
- **标记再训练**: 人工标记特定人物数据用于后续的模型再训练，以提升模型对特定场景或个体的识别能力。
- **纠正日志**: 记录所有人工纠正操作的详细信息。

### 8. 关注人员管理与预警
- **关注/取消关注人物**: 用户可以将特定人物添加到“关注列表”，以便系统进行重点监控。
- **关注人员列表**: 查看所有被当前用户关注的人物列表，并进行操作。
- **实时比对开关**: 管理员可以为特定关注人物启用或禁用实时比对功能。
- **全局搜索比对**: 当关注人物被系统检测到时，自动进行一次全局搜索比对，查找该人物在历史数据中的所有出现记录。
- **预警信息**: 显示关注人物在实时流或视频中被识别时的预警信息，包括时间、地点和相关图片。
- **历史轨迹**: 查看关注人物的历史轨迹（全局搜索比对结果），支持分页和增量查询。

### 9. 数据导出
- **视频结果导出**: 将特定视频的解析结果（人物裁剪图等）导出为Excel文件。
- **图片搜索结果导出**: 将以图搜人或以UUID搜人操作的搜索结果导出为Excel文件。
- **全部人物特征导出**: 将系统特征库中所有人物特征导出为Excel文件。
- **人员档案导出**: 导出已审核且包含身份证号的人物档案信息。
- **视频流分析结果导出**: 将特定视频流的解析结果导出为Excel文件。

## 技术栈

### 后端
- **框架**: FastAPI
- **数据库**: SQLAlchemy (ORM) + MySQL (或其他关系型数据库)
- **任务队列**: Celery + Redis (Broker & Backend)
- **机器学习**:
    - YOLOv8 (目标检测、姿态估计)
    - ONNX Runtime (模型推理)
    - Faiss (向量相似度搜索)
    - PyTorch (模型训练/微调)
    - SORT/BoT-SORT (多目标跟踪)
- **认证**: JWT (JSON Web Tokens)
- **日志**: Python logging (日志文件、控制台、数据库存储)
- **其他**: `python-multipart`, `opencv-python`, `numpy`, `scikit-image`, `scipy`, `imgaug`, `requests`, `pillow`, `PyYAML`, `uvicorn`, `httpx`, `pytz`, `passlib[bcrypt]`

### 前端
- **HTML**: 构建页面结构
- **CSS**: 页面样式 (`common.css`, `index.css` 等)
- **JavaScript**:
    - 原生 JavaScript (ES Modules)
    - `localStorage` (存储认证令牌)
    - `fetch` API (与后端API交互)
    - `WebSocket` / `EventSource` (实时通信，如视频流进度)
    - `FormData` (文件上传)
    - `URLSearchParams` (处理URL参数)
- **其他**: `MutationObserver` (DOM变化监听), Lightbox (图片预览)

## 安装与运行

### 1. 环境准备

- Python 3.8+
- MySQL 数据库 (或兼容的数据库)
- Redis Server
- FFmpeg (用于视频处理和流传输)

### 2. 克隆仓库

```bash
git clone https://github.com/jeja2023/cross_camera_tracking.git
cd cross_camera_tracking
```

### 3. 创建并激活虚拟环境

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 配置环境变量 (`.env` 文件)

在项目根目录下创建 `.env` 文件，并填写以下配置 (请替换为您的实际值)：

```
PROJECT_NAME="Cross Camera Tracking System"
PROJECT_VERSION="1.0.0"

# 数据库配置
DATABASE_URL="mysql+pymysql://user:password@host:port/dbname" # 例如: mysql+pymysql://root:password@localhost:3306/cct_db

# Celery 和 Redis 配置 (用于后台任务)
CELERY_BROKER_URL="redis://localhost:6379/0"
CELERY_RESULT_BACKEND="redis://localhost:6379/0"
REDIS_URL="redis://localhost:6379/0"

# JWT 认证配置
SECRET_KEY="your_super_secret_key_here_please_change_this" # 强烈建议更改为随机字符串
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=1440 # Token过期时间，单位：分钟 (24小时)

# CORS (跨域资源共享) 配置
CORS_ORIGINS="http://127.0.0.1:5500,http://localhost:5500" # 您的前端地址
CORS_ALLOW_CREDENTIALS="true"
CORS_ALLOW_METHODS="GET,POST,PUT,DELETE"
CORS_ALLOW_HEADERS="*"

# 文件存储目录配置 (相对于 backend/ 目录)
UPLOAD_DIR_NAME="uploads" # 用户上传的视频、图片临时存储目录
ENROLLMENT_IMAGES_DIR_NAME="enrollment_images" # 主动注册人物图片存储目录
SAVED_STREAMS_DIR_NAME="saved_streams" # 实时视频流保存目录
DATABASE_CROPS_DIR_NAME="crops" # 裁剪图片存储目录
DATABASE_FULL_FRAMES_DIR_NAME="full_frames" # 原始完整帧图像存储目录
MODELS_DIR_NAME="models" # AI模型文件存储目录
FRONTEND_STATIC_DIR_NAME="frontend/static" # 前端静态文件目录

# AI模型文件名 (放置在 models/ 目录下)
DETECTION_MODEL_FILENAME="yolov8n.pt" # 目标检测模型
REID_MODEL_FILENAME="osnet_x1_0_msmt17.onnx" # 默认Re-ID模型
ACTIVE_REID_MODEL_PATH="" # 当前激活的Re-ID模型路径 (例如: models/finetuned_osnet_v2.onnx) - 如果为空，则使用 REID_MODEL_FILENAME
POSE_MODEL_FILENAME="yolov8n-pose.pt" # 姿态估计模型
FACE_DETECTION_MODEL_FILENAME="yolov8n-face.pt" # 人脸检测模型
FACE_RECOGNITION_MODEL_FILENAME="arcface_resnet18_112x112.onnx" # 人脸识别模型
GAIT_RECOGNITION_MODEL_FILENAME="" # 步态识别模型 (可选，如果未使用则留空)
CLOTHING_ATTRIBUTE_MODEL_FILENAME="" # 衣着属性模型 (可选，如果未使用则留空)

# 模型特征维度配置
REID_INPUT_WIDTH=128
REID_INPUT_HEIGHT=256
FEATURE_DIM=512 # Re-ID特征维度
FACE_FEATURE_DIM=512 # 人脸特征维度
GAIT_FEATURE_DIM=512 # 步态特征维度

# 模型运行设备类型 (cpu 或 cuda)
DEVICE_TYPE="cpu"

# 多模态融合权重
REID_WEIGHT=0.6
FACE_WEIGHT=0.3
GAIT_WEIGHT=0.1

# Re-Ranking 算法参数
K1=20
K2=6
LAMBDA_VALUE=0.3

# 步态序列长度 (用于步态特征提取)
GAIT_SEQUENCE_LENGTH=30

# 人机回环配置
HUMAN_REVIEW_CONFIDENCE_THRESHOLD=0.9 # 低于此置信度的人脸将进入人机回环审核

# 图片分析设置
IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE=0.5 # 图片分析中最低人物置信度

# 注册设置
ENROLLMENT_MIN_PERSON_CONFIDENCE=0.5 # 主动注册中最低人物置信度

# 人脸检测置信度阈值和最小尺寸
FACE_DETECTION_CONFIDENCE_THRESHOLD=0.5
MIN_FACE_WIDTH=50
MIN_FACE_HEIGHT=50

# Tracker 和检测相关配置
TRACKER_PROXIMITY_THRESH=0.3
TRACKER_APPEARANCE_THRESH=0.8 # 外观相似度阈值

# BoT-SORT 跟踪器阈值配置
TRACKER_HIGH_THRESH=0.8
TRACKER_LOW_THRESH=0.3
TRACKER_NEW_TRACK_THRESH=0.7
TRACKER_MIN_HITS=3
TRACKER_TRACK_BUFFER=30

# 视频和流处理帧率
VIDEO_PROCESSING_FRAME_RATE=5 # 每秒处理帧数
STREAM_PROCESSING_FRAME_RATE=5 # 每秒处理帧数
VIDEO_COMMIT_BATCH_SIZE=100 # 每处理多少帧提交一次数据库事务

# 检测置信度阈值和人物类别ID
DETECTION_CONFIDENCE_THRESHOLD=0.5
PERSON_CLASS_ID=0 # YOLOv8 中 'person' 类的ID通常是0

# Excel 导出配置
EXCEL_EXPORT_MAX_IMAGES=500 # Excel导出最多包含多少张图片
EXCEL_EXPORT_IMAGE_SIZE_PX=100 # Excel中导出图片的大小 (像素)
EXCEL_EXPORT_ROW_HEIGHT_PT=75 # Excel中导出图片所在行的高度 (点)

# MJPEG 视频流帧率
MJPEG_STREAM_FPS=10

# Faiss 配置
FAISS_METRIC="L2" # 距离度量: "L2" (欧氏距离) 或 "IP" (内积距离)
FAISS_SEARCH_K=50 # Faiss搜索时返回的最近邻数量

# Re-ID 训练参数
REID_TRAIN_BATCH_SIZE=32
REID_TRAIN_LEARNING_RATE=0.001

# 日志配置
LOG_LEVEL="INFO" # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE_PATH="backend/app.log" # 日志文件路径 (如果为空则不写入文件)
LOG_FILE_MAX_BYTES=10485760 # 10MB
LOG_FILE_BACKUP_COUNT=5

# 默认管理员用户配置 (系统启动时自动创建，仅在数据库无管理员时生效)
DEFAULT_ADMIN_USERNAME="admin"
DEFAULT_ADMIN_PASSWORD="your_admin_password_here" # 请务必更改为强密码
DEFAULT_ADMIN_UNIT="管理部门"
DEFAULT_ADMIN_PHONE_NUMBER="13800000000"
```

### 6. 数据库迁移 (Alembic)

在运行应用前，需要执行数据库迁移以创建或更新数据库表结构。

```bash
alembic upgrade head
```

### 7. 运行后端服务

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
(注：`--reload` 选项仅用于开发环境，生产环境请移除)

### 8. 运行 Celery Worker (后台任务)

打开一个新的终端窗口，激活相同的虚拟环境，并运行 Celery Worker：

```bash
celery -A backend.celery_app.celery_app worker -l info --pool=solo
```
(注：`--pool=solo` 仅用于开发环境和调试，生产环境请使用 `prefork` 或其他池类型以获得并发处理能力)

### 9. 访问前端页面

后端服务和 Celery Worker 启动后，您可以通过浏览器访问：

```
http://localhost:8000
```
系统会自动重定向到登录页面。

## 文件结构概览

```
cross_camera_tracking/
├── alembic/                # Alembic 数据库迁移工具相关文件
├── backend/                # 后端 FastAPI 应用程序
│   ├── __init__.py
│   ├── auth.py             # 认证相关辅助函数
│   ├── celery_app.py       # Celery 应用配置
│   ├── celery_worker.py    # Celery Worker 启动脚本
│   ├── celeryconfig.py     # Celery 配置
│   ├── config.py           # 项目配置 (从 .env 加载，并可从数据库动态加载)
│   ├── crud.py             # 数据库 CRUD (Create, Read, Update, Delete) 操作
│   ├── crud_match.py       # 匹配和比对相关的数据库操作 (如关注人员、预警、全局搜索)
│   ├── database/           # 数据库文件 (如 SQLite) 或用于存放 AI 处理后的图片
│   │   ├── crops/          # 裁剪后的人物图像存储
│   │   └── full_frames/    # 原始完整帧图像存储
│   ├── database_conn.py    # 数据库连接和 SQLAlchemy ORM 模型定义
│   ├── main.py             # FastAPI 主应用，路由注册，启动事件
│   ├── ml_services/        # 机器学习服务模块
│   │   ├── __init__.py
│   │   ├── ml_logic.py     # 核心机器学习逻辑 (模型加载、特征提取、比对、图片分析、视频流处理)
│   │   ├── ml_tasks.py     # Celery 异步任务定义 (视频处理、模型再训练、图片分析、全局搜索)
│   │   ├── re_ranking.py   # 重排序算法实现
│   │   ├── reid_trainer.py # Re-ID 模型训练逻辑
│   │   └── sort.py         # SORT 目标跟踪算法实现
│   ├── models/             # AI 模型文件存储目录
│   ├── routers/            # FastAPI 路由定义
│   │   ├── admin_routes.py # 管理员功能 (用户管理、模型配置、日志)
│   │   ├── auth_routes.py  # 认证和用户注册
│   │   ├── export_routes.py# 数据导出功能 (Excel 报告)
│   │   ├── followed_person_routes.py # 关注人员管理和预警
│   │   ├── human_in_the_loop_routes.py # 人机回环审核与纠正
│   │   ├── image_analysis_routes.py # 图片分析
│   │   ├── page_routes.py  # 前端页面路由
│   │   ├── person_routes.py# 人物注册、搜索和管理
│   │   ├── stream_routes.py# 视频流管理
│   │   └── video_routes.py # 视频管理与处理
│   ├── saved_streams/      # 实时视频流处理后保存的视频文件
│   ├── schemas.py          # Pydantic 数据模型定义 (请求/响应体、数据库模型映射)
│   ├── stream_manager.py   # 视频流管理 (如 Redis 帧缓存)
│   └── uploads/            # 临时上传文件存储 (用户上传的原始视频、图片)
│   └── utils/              # 工具函数
│       ├── media_processing.py # 媒体处理辅助函数
│       └── realtime_comparison.py # 实时比对辅助函数
├── frontend/               # 前端静态文件
│   └── static/             # 静态资源 (HTML, CSS, JS)
│       ├── css/            # 样式表文件
│       ├── js/             # JavaScript 逻辑文件
│       ├── *.html          # HTML 页面文件 (如 index.html, login.html 等)
│       └── images/         # 默认图片等静态资源
├── alembic.ini             # Alembic 配置文件
├── pyproject.toml          # 项目元数据 (PEP 621)
├── README.md               # 项目说明文档
└── requirements.txt        # Python 依赖列表
└── venv/                   # Python 虚拟环境
```

## 贡献

欢迎对本项目进行贡献！如果您有任何建议、Bug 报告或功能请求，请通过 GitHub Issues 提交。如果您想贡献代码，请 Fork 仓库，创建新的分支，并在完成后提交 Pull Request。

## 许可证

本项目基于 [MIT 许可证](LICENSE) 发布。 