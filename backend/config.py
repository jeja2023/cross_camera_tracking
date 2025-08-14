import os
from dotenv import load_dotenv
import sys
from sqlalchemy.orm import Session # 确保导入 Session
from .database_conn import SessionLocal, SystemConfig # 确保 SystemConfig 在顶部导入
import logging
from typing import List # 导入 List 类型

load_dotenv()

logger = logging.getLogger(__name__) # 新增日志记录器

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME")
    if not PROJECT_NAME:
        print("错误：项目名称 'PROJECT_NAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    
    PROJECT_VERSION: str = os.getenv("PROJECT_VERSION")
    if not PROJECT_VERSION:
        print("错误：项目版本 'PROJECT_VERSION' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    # Directories
    if getattr(sys, 'frozen', False):
        BASE_DIR: str = sys._MEIPASS
    else:
        BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 获取目录名称，如果环境变量中没有设置，则使用默认值
    UPLOAD_DIR_NAME: str = os.getenv("UPLOAD_DIR_NAME")
    if not UPLOAD_DIR_NAME:
        print("错误：上传目录名称 'UPLOAD_DIR_NAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    SAVED_STREAMS_DIR_NAME: str = os.getenv("SAVED_STREAMS_DIR_NAME")
    if not SAVED_STREAMS_DIR_NAME:
        print("错误：保存流目录名称 'SAVED_STREAMS_DIR_NAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    DATABASE_CROPS_DIR_NAME: str = os.getenv("DATABASE_CROPS_DIR_NAME")
    if not DATABASE_CROPS_DIR_NAME:
        print("错误：数据库裁剪图片目录名称 'DATABASE_CROPS_DIR_NAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    DATABASE_FULL_FRAMES_DIR_NAME: str = os.getenv("DATABASE_FULL_FRAMES_DIR_NAME")
    if not DATABASE_FULL_FRAMES_DIR_NAME:
        print("错误：数据库完整帧目录名称 'DATABASE_FULL_FRAMES_DIR_NAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    MODELS_DIR_NAME: str = os.getenv("MODELS_DIR_NAME")
    if not MODELS_DIR_NAME:
        print("错误：模型目录名称 'MODELS_DIR_NAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
        
    FRONTEND_STATIC_DIR_NAME: str = os.getenv("FRONTEND_STATIC_DIR_NAME")
    if not FRONTEND_STATIC_DIR_NAME:
        print("错误：前端静态文件目录名称 'FRONTEND_STATIC_DIR_NAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    UPLOAD_DIR: str = os.path.join(BASE_DIR, "backend", UPLOAD_DIR_NAME)
    SAVED_STREAMS_DIR: str = os.path.join(BASE_DIR, "backend", SAVED_STREAMS_DIR_NAME)
    DATABASE_CROPS_DIR: str = os.path.join(BASE_DIR, "backend", "database", DATABASE_CROPS_DIR_NAME)
    DATABASE_FULL_FRAMES_DIR: str = os.path.join(BASE_DIR, "backend", "database", DATABASE_FULL_FRAMES_DIR_NAME)
    DATABASE_FULL_FRAMES_IMAGE_ANALYSIS_DIR: str = os.path.join(BASE_DIR, "backend", "database", DATABASE_FULL_FRAMES_DIR_NAME, "general_detection")

    # 新增：图片分析所使用的模型名称
    IMAGE_ANALYSIS_MODEL_NAME: str = os.getenv("IMAGE_ANALYSIS_MODEL_NAME")
    if not IMAGE_ANALYSIS_MODEL_NAME:
        print("错误：图片分析模型名称 'IMAGE_ANALYSIS_MODEL_NAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    MODELS_DIR: str = os.path.join(BASE_DIR, "backend", MODELS_DIR_NAME)

    # 定义支持的模型名称
    SUPPORTED_MODELS = ["person_reid", "face_recognition", "pose_estimation", "gait_recognition", "general_detection"]
    # 定义支持的解析类型
    SUPPORTED_ANALYSIS_TYPES = ["image", "video", "stream", "enrollment"]

    # 辅助函数：构建解析结果图片的完整保存路径
    def get_parsed_image_path(self, base_dir: str, model_name: str, analysis_type: str, uuid: str) -> str:
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型名称: {model_name}")
        if analysis_type not in self.SUPPORTED_ANALYSIS_TYPES:
            raise ValueError(f"不支持的解析类型: {analysis_type}")

        path = os.path.join(base_dir, model_name, analysis_type, uuid)
        os.makedirs(path, exist_ok=True) # 确保路径存在
        return path

    # 辅助函数：构建解析结果图片的相对保存路径 (用于数据库存储)
    def get_parsed_image_relative_path(self, base_dir_name: str, model_name: str, analysis_type: str, uuid: str) -> str:
        # base_dir_name 应该是 "crops" 或 "full_frames"
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型名称: {model_name}")
        if analysis_type not in self.SUPPORTED_ANALYSIS_TYPES:
            raise ValueError(f"不支持的解析类型: {analysis_type}")

        # 构建相对路径，不包含 BASE_DIR
        relative_path = os.path.join("database", base_dir_name, model_name, analysis_type, uuid)
        return relative_path.replace(os.sep, '/')

    # Model paths
    DETECTION_MODEL_FILENAME: str = os.getenv("DETECTION_MODEL_FILENAME")
    if not DETECTION_MODEL_FILENAME:
        print("错误：目标检测模型文件名 'DETECTION_MODEL_FILENAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    DETECTION_MODEL_PATH: str = os.path.join(MODELS_DIR, DETECTION_MODEL_FILENAME)

    REID_MODEL_FILENAME: str = os.getenv("REID_MODEL_FILENAME")
    if not REID_MODEL_FILENAME:
        print("错误：Re-ID 模型文件名 'REID_MODEL_FILENAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    REID_MODEL_PATH: str = os.path.join(MODELS_DIR, REID_MODEL_FILENAME)

    # 读取 ACTIVE_REID_MODEL_PATH 环境变量，它预计只包含文件名
    _active_reid_model_filename: str = os.getenv("ACTIVE_REID_MODEL_PATH")
    if not _active_reid_model_filename:
        print("错误：主动 Re-ID 模型路径 'ACTIVE_REID_MODEL_PATH' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    # 将模型目录与文件名拼接，得到完整的模型路径
    ACTIVE_REID_MODEL_PATH: str = os.path.join(MODELS_DIR, _active_reid_model_filename)

    POSE_MODEL_FILENAME: str = os.getenv("POSE_MODEL_FILENAME")
    if not POSE_MODEL_FILENAME:
        print("错误：姿态估计模型文件名 'POSE_MODEL_FILENAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    POSE_MODEL_PATH: str = os.path.join(MODELS_DIR, POSE_MODEL_FILENAME)

    FACE_DETECTION_MODEL_FILENAME: str = os.getenv("FACE_DETECTION_MODEL_FILENAME")
    if not FACE_DETECTION_MODEL_FILENAME:
        print("错误：人脸检测模型文件名 'FACE_DETECTION_MODEL_FILENAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    FACE_DETECTION_MODEL_PATH: str = os.path.join(MODELS_DIR, FACE_DETECTION_MODEL_FILENAME)

    FACE_RECOGNITION_MODEL_FILENAME: str = os.getenv("FACE_RECOGNITION_MODEL_FILENAME")
    if not FACE_RECOGNITION_MODEL_FILENAME:
        print("错误：人脸识别模型文件名 'FACE_RECOGNITION_MODEL_FILENAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    FACE_RECOGNITION_MODEL_PATH: str = os.path.join(MODELS_DIR, FACE_RECOGNITION_MODEL_FILENAME)

    _GAIT_RECOGNITION_MODEL_FILENAME = os.getenv("GAIT_RECOGNITION_MODEL_FILENAME", "")
    GAIT_RECOGNITION_MODEL_PATH = os.path.join(MODELS_DIR, _GAIT_RECOGNITION_MODEL_FILENAME) if _GAIT_RECOGNITION_MODEL_FILENAME else ""

    _CLOTHING_ATTRIBUTE_MODEL_FILENAME = os.getenv("CLOTHING_ATTRIBUTE_MODEL_FILENAME", "")
    CLOTHING_ATTRIBUTE_MODEL_PATH = os.path.join(MODELS_DIR, _CLOTHING_ATTRIBUTE_MODEL_FILENAME) if _CLOTHING_ATTRIBUTE_MODEL_FILENAME else ""

    # 模型特征维度配置
    REID_INPUT_WIDTH_STR: str = os.getenv("REID_INPUT_WIDTH")
    if not REID_INPUT_WIDTH_STR:
        print("错误：Re-ID 输入宽度 'REID_INPUT_WIDTH' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    REID_INPUT_WIDTH: int = int(REID_INPUT_WIDTH_STR)

    REID_INPUT_HEIGHT_STR: str = os.getenv("REID_INPUT_HEIGHT")
    if not REID_INPUT_HEIGHT_STR:
        print("错误：Re-ID 输入高度 'REID_INPUT_HEIGHT' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    REID_INPUT_HEIGHT: int = int(REID_INPUT_HEIGHT_STR)

    FEATURE_DIM_STR: str = os.getenv("FEATURE_DIM")
    if not FEATURE_DIM_STR:
        print("错误：特征维度 'FEATURE_DIM' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    FEATURE_DIM: int = int(FEATURE_DIM_STR)

    FACE_FEATURE_DIM_STR: str = os.getenv("FACE_FEATURE_DIM")
    if not FACE_FEATURE_DIM_STR:
        print("错误：人脸特征维度 'FACE_FEATURE_DIM' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    FACE_FEATURE_DIM: int = int(FACE_FEATURE_DIM_STR)

    GAIT_FEATURE_DIM_STR: str = os.getenv("GAIT_FEATURE_DIM")
    if not GAIT_FEATURE_DIM_STR:
        print("错误：步态特征维度 'GAIT_FEATURE_DIM' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    GAIT_FEATURE_DIM: int = int(GAIT_FEATURE_DIM_STR)

    # 新增：模型运行设备类型
    DEVICE_TYPE: str = os.getenv("DEVICE_TYPE")
    if not DEVICE_TYPE:
        print("错误：设备类型 'DEVICE_TYPE' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    # 多模态融合权重 (可以根据实际效果调整)
    REID_WEIGHT_STR: str = os.getenv("REID_WEIGHT")
    if not REID_WEIGHT_STR:
        print("错误：Re-ID 权重 'REID_WEIGHT' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    REID_WEIGHT: float = float(REID_WEIGHT_STR)

    FACE_WEIGHT_STR: str = os.getenv("FACE_WEIGHT")
    if not FACE_WEIGHT_STR:
        print("错误：人脸权重 'FACE_WEIGHT' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    FACE_WEIGHT: float = float(FACE_WEIGHT_STR)

    GAIT_WEIGHT_STR: str = os.getenv("GAIT_WEIGHT")
    if not GAIT_WEIGHT_STR:
        print("错误：步态权重 'GAIT_WEIGHT' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    GAIT_WEIGHT: float = float(GAIT_WEIGHT_STR)

    # Re-Ranking 算法参数
    K1_STR: str = os.getenv("K1")
    if not K1_STR:
        print("错误：Re-Ranking 参数 'K1' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    K1: int = int(K1_STR)

    K2_STR: str = os.getenv("K2")
    if not K2_STR:
        print("错误：Re-Ranking 参数 'K2' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    K2: int = int(K2_STR)

    LAMBDA_VALUE_STR: str = os.getenv("LAMBDA_VALUE")
    if not LAMBDA_VALUE_STR:
        print("错误：Re-Ranking 参数 'LAMBDA_VALUE' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    LAMBDA_VALUE: float = float(LAMBDA_VALUE_STR)

    # 步态序列长度 (用于步态特征提取，例如需要 30-60 帧)
    GAIT_SEQUENCE_LENGTH_STR: str = os.getenv("GAIT_SEQUENCE_LENGTH")
    if not GAIT_SEQUENCE_LENGTH_STR:
        print("错误：步态序列长度 'GAIT_SEQUENCE_LENGTH' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        GAIT_SEQUENCE_LENGTH: int = int(GAIT_SEQUENCE_LENGTH_STR)
    except ValueError:
        print(f"错误：GAIT_SEQUENCE_LENGTH 的值 '{GAIT_SEQUENCE_LENGTH_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    # 定义一个包含所有可持久化配置项的列表
    # 这些键将从 .env 初始化，并在首次启动时写入数据库
    # 它们也将在后续启动时从数据库加载并覆盖 .env 值
    CONFIGURABLE_SETTINGS_KEYS = [
        "DETECTION_MODEL_FILENAME", "REID_MODEL_FILENAME", "ACTIVE_REID_MODEL_PATH", 
        "POSE_MODEL_FILENAME", "FACE_DETECTION_MODEL_FILENAME", "FACE_RECOGNITION_MODEL_FILENAME", 
        "GAIT_RECOGNITION_MODEL_FILENAME", "CLOTHING_ATTRIBUTE_MODEL_FILENAME",
        "REID_INPUT_WIDTH", "REID_INPUT_HEIGHT", "FEATURE_DIM", "FACE_FEATURE_DIM", "GAIT_FEATURE_DIM",
        "DEVICE_TYPE", "REID_WEIGHT", "FACE_WEIGHT", "GAIT_WEIGHT", "K1", "K2", "LAMBDA_VALUE", 
        "GAIT_SEQUENCE_LENGTH", "HUMAN_REVIEW_CONFIDENCE_THRESHOLD", "IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE",
        "ENROLLMENT_MIN_PERSON_CONFIDENCE", "FACE_DETECTION_CONFIDENCE_THRESHOLD", "MIN_FACE_WIDTH", 
        "MIN_FACE_HEIGHT", "TRACKER_PROXIMITY_THRESH", "TRACKER_APPEARANCE_THRESH", "TRACKER_HIGH_THRESH",
        "TRACKER_LOW_THRESH", "TRACKER_NEW_TRACK_THRESH", "TRACKER_MIN_HITS", "TRACKER_TRACK_BUFFER",
        "VIDEO_PROCESSING_FRAME_RATE", "STREAM_PROCESSING_FRAME_RATE", "VIDEO_COMMIT_BATCH_SIZE",
        "DETECTION_CONFIDENCE_THRESHOLD", "PERSON_CLASS_ID", "EXCEL_EXPORT_MAX_IMAGES",
        "EXCEL_EXPORT_IMAGE_SIZE_PX", "EXCEL_EXPORT_ROW_HEIGHT_PT", "MJPEG_STREAM_FPS",
        "FAISS_METRIC", "FAISS_SEARCH_K", "REID_TRAIN_BATCH_SIZE", "REID_TRAIN_LEARNING_RATE",
        "REALTIME_COMPARISON_THRESHOLD", "REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS", "GLOBAL_SEARCH_MIN_CONFIDENCE" # 新增实时比对配置
    ]

    # Ensure directories exist
    def __init__(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.SAVED_STREAMS_DIR, exist_ok=True)
        os.makedirs(self.DATABASE_CROPS_DIR, exist_ok=True)
        os.makedirs(self.DATABASE_FULL_FRAMES_DIR, exist_ok=True) # 确保完整帧目录也存在
        
        # 移除原先确保图片解析特有的子目录也存在的代码
        # os.makedirs(self.DATABASE_FULL_FRAMES_IMAGE_ANALYSIS_DIR, exist_ok=True)
        # os.makedirs(self.DATABASE_CROPS_IMAGE_ANALYSIS_DIR, exist_ok=True)
        # 移除 __init__ 中直接调用 load_from_db，因为这会在应用程序启动前发生，没有数据库会话
        # self.load_from_db() # 增加从数据库加载配置的调用

    def init_or_load_from_db(self, db: Session):
        """
        在应用程序首次启动时，检查数据库中的 SystemConfig 表。
        如果表为空，则将当前 Settings 实例的配置写入数据库。
        如果表不为空，则从数据库加载配置并更新当前 Settings 实例。
        """
        try:
            # 检查 SystemConfig 表是否为空
            if db.query(SystemConfig).count() == 0:
                logger.info("SystemConfig 表为空，将从环境变量加载初始配置并写入数据库。")
                configs_to_write = {}
                for key in self.CONFIGURABLE_SETTINGS_KEYS:
                    if hasattr(self, key):
                        value = getattr(self, key)
                        # 对于模型路径，这里应该存储文件名而不是完整路径
                        if key in ["DETECTION_MODEL_FILENAME", "REID_MODEL_FILENAME", 
                                   "POSE_MODEL_FILENAME", "FACE_DETECTION_MODEL_FILENAME", 
                                   "FACE_RECOGNITION_MODEL_FILENAME", "GAIT_RECOGNITION_MODEL_FILENAME", 
                                   "CLOTHING_ATTRIBUTE_MODEL_FILENAME"]:
                            # 这些在 settings 中已经保存为文件名，ACTIVE_REID_MODEL_PATH 也是相对路径
                            # 所以直接使用 value 即可
                            configs_to_write[key] = str(value) if value is not None else ""
                        elif key == "ACTIVE_REID_MODEL_PATH":
                            # ACTIVE_REID_MODEL_PATH 存储在数据库中应该是相对路径（文件名）
                            # 首次写入时，我们将其初始化为 REID_MODEL_FILENAME
                            configs_to_write[key] = self.REID_MODEL_FILENAME # 确保初始时是原始模型文件名
                        elif isinstance(value, (int, float, bool)):
                            configs_to_write[key] = str(value)
                        else:
                            configs_to_write[key] = value # 默认作为字符串存储
                
                # 特殊处理 ACTIVE_REID_MODEL_PATH 的初始值，应该与 REID_MODEL_FILENAME 相同
                if "ACTIVE_REID_MODEL_PATH" not in configs_to_write or not configs_to_write["ACTIVE_REID_MODEL_PATH"]:
                    configs_to_write["ACTIVE_REID_MODEL_PATH"] = self.REID_MODEL_FILENAME # 确保初始时是原始模型文件名

                from . import crud # 导入 crud 模块
                crud.set_system_configs(db, configs_to_write)
                logger.info("初始配置已成功写入数据库。")
            
            # 从数据库加载配置并更新当前 Settings 实例
            self.load_from_db(db)
            logger.info("配置已从数据库加载并更新 Settings 实例。")

            # 检查并添加数据库中可能缺失的新增配置项
            existing_db_keys = {c.key for c in db.query(SystemConfig.key).all()}

            new_configs_to_add = {}
            for key in self.CONFIGURABLE_SETTINGS_KEYS:
                if key not in existing_db_keys and hasattr(self, key):
                    value = getattr(self, key)
                    # 根据类型进行转换，与初始写入逻辑保持一致
                    if key in ["DETECTION_MODEL_FILENAME", "REID_MODEL_FILENAME", 
                               "POSE_MODEL_FILENAME", "FACE_DETECTION_MODEL_FILENAME", 
                               "FACE_RECOGNITION_MODEL_FILENAME", "GAIT_RECOGNITION_MODEL_FILENAME", 
                               "CLOTHING_ATTRIBUTE_MODEL_FILENAME"]:
                        new_configs_to_add[key] = str(value) if value is not None else ""
                    elif key == "ACTIVE_REID_MODEL_PATH":
                        new_configs_to_add[key] = self.REID_MODEL_FILENAME # 初始时设置为默认Re-ID模型文件名
                    elif isinstance(value, (int, float, bool)):
                        new_configs_to_add[key] = str(value)
                    else:
                        new_configs_to_add[key] = value # 默认作为字符串存储
            
            if new_configs_to_add:
                from . import crud # 导入 crud 模块
                crud.set_system_configs(db, new_configs_to_add)
                logger.info(f"已将 {len(new_configs_to_add)} 个新增配置项写入数据库。")
                # 重新加载以确保新添加的配置也立即生效
                self.load_from_db(db)

        except Exception as e:
            logger.error(f"初始化或从数据库加载配置时发生错误: {e}", exc_info=True)
            raise RuntimeError(f"初始化或从数据库加载配置失败: {e}")

    def load_from_db(self, db: Session): # 新增方法，从数据库加载配置
        """
        从数据库加载配置覆盖环境变量中的值。主要用于模型参数等需要动态更新的配置。
        """
        # 注意：这里不再创建新的 db 会话，而是使用传入的 db 会话
        try:
            db_configs = db.query(SystemConfig).all()
            for config in db_configs:
                if hasattr(self, config.key) and not config.key.startswith("_") and not callable(getattr(self, config.key)):
                    try:
                        current_value = getattr(self, config.key)
                        # 确保正确处理模型路径（从相对路径转换为完整路径，如果需要）
                        if config.key in ["DETECTION_MODEL_PATH", "REID_MODEL_PATH", "POSE_MODEL_PATH", 
                                           "FACE_DETECTION_MODEL_PATH", "FACE_RECOGNITION_MODEL_PATH", 
                                           "GAIT_RECOGNITION_MODEL_PATH", "CLOTHING_ATTRIBUTE_MODEL_PATH"]:
                            # 对于这些，config.value 是文件名，需要拼接 MODELS_DIR
                            setattr(self, config.key, os.path.join(self.MODELS_DIR, config.value)) if config.value else ""
                        elif config.key == "ACTIVE_REID_MODEL_PATH":
                            # ACTIVE_REID_MODEL_PATH 存储的是相对路径 (文件名)，加载时需要拼接 MODELS_DIR
                            setattr(self, config.key, os.path.join(self.MODELS_DIR, config.value)) if config.value else ""
                        elif isinstance(current_value, int):
                            setattr(self, config.key, int(config.value))
                        elif isinstance(current_value, float):
                            setattr(self, config.key, float(config.value))
                        elif isinstance(current_value, bool):
                            setattr(self, config.key, config.value.lower() == "true")
                        else: # 默认为字符串
                            setattr(self, config.key, config.value)
                        logger.debug(f"从数据库加载配置: {config.key} = {getattr(self, config.key)}")
                    except ValueError as e:
                        logger.error(f"从数据库加载配置 {config.key} 失败，类型转换错误: {e}")
        except Exception as e:
            logger.error(f"从数据库加载配置时发生错误: {e}", exc_info=True)
            # 不再关闭 db 会话，由调用方管理
            # if db is not None: 
            #     db.close()

    def reload_from_db(self, db: Session):
        """
        强制从数据库重新加载所有可配置项，并更新当前 settings 实例的属性。
        此方法应在数据库配置更新后调用，以确保当前运行实例使用最新配置。
        """
        logger.info("从数据库重新加载 Settings 配置...")
        self.load_from_db(db) # 调用 load_from_db 方法
        logger.info("Settings 配置重新加载完成。")

    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("错误：数据库连接字符串 'DATABASE_URL' 未在您的 .env 文件中设置。请参照格式添加：DATABASE_URL='mysql+pymysql://user:password@host:port/dbname'", file=sys.stderr)
        sys.exit(1)

    # Celery settings
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL")
    if not CELERY_BROKER_URL:
        print("错误：Celery Broker URL 'CELERY_BROKER_URL' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND")
    if not CELERY_RESULT_BACKEND:
        print("错误：Celery Result Backend 'CELERY_RESULT_BACKEND' 未在您的 .env 文件中设置。请参照格式添加：CELERY_RESULT_BACKEND='redis://localhost:6379/0'", file=sys.stderr)
        sys.exit(1)

    # Redis URL for direct Redis client usage (e.g., stream_manager)
    REDIS_URL: str = os.getenv("REDIS_URL")
    if not REDIS_URL:
        print("错误：Redis URL 'REDIS_URL' 未在您的 .env 文件中设置。请参照格式添加：REDIS_URL='redis://localhost:6379/0'", file=sys.stderr)
        sys.exit(1)

    # JWT settings
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    if not SECRET_KEY:
        print("错误：JWT SECRET_KEY 未在您的 .env 文件中设置。请参照格式添加：SECRET_KEY='<your_secret_key_here>'", file=sys.stderr)
        sys.exit(1)

    ALGORITHM: str = os.getenv("ALGORITHM")
    if not ALGORITHM:
        print("错误：JWT ALGORITHM 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    ACCESS_TOKEN_EXPIRE_MINUTES_STR: str = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
    if not ACCESS_TOKEN_EXPIRE_MINUTES_STR:
        print("错误：JWT ACCESS_TOKEN_EXPIRE_MINUTES 未在您的 .env 文件中设置。请参照格式添加：ACCESS_TOKEN_EXPIRE_MINUTES=1440 (24小时)", file=sys.stderr)
        sys.exit(1)
    try:
        ACCESS_TOKEN_EXPIRE_MINUTES: int = int(ACCESS_TOKEN_EXPIRE_MINUTES_STR)
    except ValueError:
        print(f"错误：ACCESS_TOKEN_EXPIRE_MINUTES 的值 '{ACCESS_TOKEN_EXPIRE_MINUTES_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    # CORS settings
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS")
    if not CORS_ORIGINS:
        print("错误：CORS_ORIGINS 未在您的 .env 文件中设置。请参照格式添加：CORS_ORIGINS='http://127.0.0.1:5500,http://localhost:5500'", file=sys.stderr)
        sys.exit(1)

    CORS_ALLOW_CREDENTIALS_STR: str = os.getenv("CORS_ALLOW_CREDENTIALS")
    if not CORS_ALLOW_CREDENTIALS_STR:
        print("错误：CORS 允许凭据 'CORS_ALLOW_CREDENTIALS' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    CORS_ALLOW_CREDENTIALS: bool = CORS_ALLOW_CREDENTIALS_STR.lower() == "true"

    CORS_ALLOW_METHODS: str = os.getenv("CORS_ALLOW_METHODS")
    if not CORS_ALLOW_METHODS:
        print("错误：CORS 允许方法 'CORS_ALLOW_METHODS' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    CORS_ALLOW_HEADERS: str = os.getenv("CORS_ALLOW_HEADERS")
    if not CORS_ALLOW_HEADERS:
        print("错误：CORS 允许头 'CORS_ALLOW_HEADERS' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    # Human-in-the-Loop settings
    HUMAN_REVIEW_CONFIDENCE_THRESHOLD_STR: str = os.getenv("HUMAN_REVIEW_CONFIDENCE_THRESHOLD")
    if not HUMAN_REVIEW_CONFIDENCE_THRESHOLD_STR:
        print("错误：人机回环置信度阈值 'HUMAN_REVIEW_CONFIDENCE_THRESHOLD' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        HUMAN_REVIEW_CONFIDENCE_THRESHOLD: float = float(HUMAN_REVIEW_CONFIDENCE_THRESHOLD_STR)
    except ValueError:
        print(f"错误：HUMAN_REVIEW_CONFIDENCE_THRESHOLD 的值 '{HUMAN_REVIEW_CONFIDENCE_THRESHOLD_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    # Image Analysis settings
    IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE_STR: str = os.getenv("IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE")
    if not IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE_STR:
        print("错误：图片分析人物最低置信度 'IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE: float = float(IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE_STR)
    except ValueError:
        print(f"错误：IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE 的值 '{IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    # Enrollment settings (主动注册)
    ENROLLMENT_MIN_PERSON_CONFIDENCE_STR: str = os.getenv("ENROLLMENT_MIN_PERSON_CONFIDENCE")
    if not ENROLLMENT_MIN_PERSON_CONFIDENCE_STR:
        print("错误：主动注册人物最低置信度 'ENROLLMENT_MIN_PERSON_CONFIDENCE' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        ENROLLMENT_MIN_PERSON_CONFIDENCE: float = float(ENROLLMENT_MIN_PERSON_CONFIDENCE_STR)
    except ValueError:
        print(f"错误：ENROLLMENT_MIN_PERSON_CONFIDENCE 的值 '{ENROLLMENT_MIN_PERSON_CONFIDENCE_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    # 新增：人脸检测置信度阈值
    FACE_DETECTION_CONFIDENCE_THRESHOLD_STR: str = os.getenv("FACE_DETECTION_CONFIDENCE_THRESHOLD")
    if not FACE_DETECTION_CONFIDENCE_THRESHOLD_STR:
        print("错误：人脸检测置信度阈值 'FACE_DETECTION_CONFIDENCE_THRESHOLD' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        FACE_DETECTION_CONFIDENCE_THRESHOLD: float = float(FACE_DETECTION_CONFIDENCE_THRESHOLD_STR)
    except ValueError:
        print(f"错误：FACE_DETECTION_CONFIDENCE_THRESHOLD 的值 '{FACE_DETECTION_CONFIDENCE_THRESHOLD_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    # 新增：人脸裁剪图的最小尺寸阈值
    MIN_FACE_WIDTH_STR: str = os.getenv("MIN_FACE_WIDTH")
    if not MIN_FACE_WIDTH_STR:
        print("错误：人脸最小宽度 'MIN_FACE_WIDTH' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        MIN_FACE_WIDTH: int = int(MIN_FACE_WIDTH_STR)
    except ValueError:
        print(f"错误：MIN_FACE_WIDTH 的值 '{MIN_FACE_WIDTH_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    MIN_FACE_HEIGHT_STR: str = os.getenv("MIN_FACE_HEIGHT")
    if not MIN_FACE_HEIGHT_STR:
        print("错误：人脸最小高度 'MIN_FACE_HEIGHT' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        MIN_FACE_HEIGHT: int = int(MIN_FACE_HEIGHT_STR)
    except ValueError:
        print(f"错误：MIN_FACE_HEIGHT 的值 '{MIN_FACE_HEIGHT_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    # 新增：Tracker 和检测相关配置
    TRACKER_PROXIMITY_THRESH_STR: str = os.getenv("TRACKER_PROXIMITY_THRESH")
    if not TRACKER_PROXIMITY_THRESH_STR:
        print("错误：Tracker 接近度阈值 'TRACKER_PROXIMITY_THRESH' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        TRACKER_PROXIMITY_THRESH: float = float(TRACKER_PROXIMITY_THRESH_STR)
    except ValueError:
        print(f"错误：TRACKER_PROXIMITY_THRESH 的值 '{TRACKER_PROXIMITY_THRESH_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    TRACKER_APPEARANCE_THRESH_STR: str = os.getenv("TRACKER_APPEARANCE_THRESH")
    if not TRACKER_APPEARANCE_THRESH_STR:
        print("错误：Tracker 外观相似度阈值 'TRACKER_APPEARANCE_THRESH' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        TRACKER_APPEARANCE_THRESH: float = float(TRACKER_APPEARANCE_THRESH_STR)
    except ValueError:
        print(f"错误：TRACKER_APPEARANCE_THRESH 的值 '{TRACKER_APPEARANCE_THRESH_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    # 新增：BoT-SORT 跟踪器阈值配置
    TRACKER_HIGH_THRESH_STR: str = os.getenv("TRACKER_HIGH_THRESH")
    if not TRACKER_HIGH_THRESH_STR:
        print("错误：Tracker 高阈值 'TRACKER_HIGH_THRESH' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        TRACKER_HIGH_THRESH: float = float(TRACKER_HIGH_THRESH_STR)
    except ValueError:
        print(f"错误：TRACKER_HIGH_THRESH 的值 '{TRACKER_HIGH_THRESH_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    TRACKER_LOW_THRESH_STR: str = os.getenv("TRACKER_LOW_THRESH")
    if not TRACKER_LOW_THRESH_STR:
        print("错误：Tracker 低阈值 'TRACKER_LOW_THRESH' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        TRACKER_LOW_THRESH: float = float(TRACKER_LOW_THRESH_STR)
    except ValueError:
        print(f"错误：TRACKER_LOW_THRESH 的值 '{TRACKER_LOW_THRESH_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    TRACKER_NEW_TRACK_THRESH_STR: str = os.getenv("TRACKER_NEW_TRACK_THRESH")
    if not TRACKER_NEW_TRACK_THRESH_STR:
        print("错误：Tracker 新轨迹阈值 'TRACKER_NEW_TRACK_THRESH' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        TRACKER_NEW_TRACK_THRESH: float = float(TRACKER_NEW_TRACK_THRESH_STR)
    except ValueError:
        print(f"错误：TRACKER_NEW_TRACK_THRESH 的值 '{TRACKER_NEW_TRACK_THRESH_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    # 新增：Tracker 初始化最低检测次数 (min_hits)
    TRACKER_MIN_HITS_STR: str = os.getenv("TRACKER_MIN_HITS")
    if not TRACKER_MIN_HITS_STR:
        print("错误：Tracker 最低检测次数 'TRACKER_MIN_HITS' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        TRACKER_MIN_HITS: int = int(TRACKER_MIN_HITS_STR)
    except ValueError:
        print(f"错误：TRACKER_MIN_HITS 的值 '{TRACKER_MIN_HITS_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    # 新增：Tracker 轨迹保留帧数 (track_buffer)
    TRACKER_TRACK_BUFFER_STR: str = os.getenv("TRACKER_TRACK_BUFFER")
    if not TRACKER_TRACK_BUFFER_STR:
        print("错误：Tracker 轨迹保留帧数 'TRACKER_TRACK_BUFFER' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        TRACKER_TRACK_BUFFER: int = int(TRACKER_TRACK_BUFFER_STR)
    except ValueError:
        print(f"错误：TRACKER_TRACK_BUFFER 的值 '{TRACKER_TRACK_BUFFER_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    VIDEO_PROCESSING_FRAME_RATE_STR: str = os.getenv("VIDEO_PROCESSING_FRAME_RATE")
    if not VIDEO_PROCESSING_FRAME_RATE_STR:
        print("错误：视频处理帧率 'VIDEO_PROCESSING_FRAME_RATE' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        VIDEO_PROCESSING_FRAME_RATE: int = int(VIDEO_PROCESSING_FRAME_RATE_STR)
    except ValueError:
        print(f"错误：VIDEO_PROCESSING_FRAME_RATE 的值 '{VIDEO_PROCESSING_FRAME_RATE_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    STREAM_PROCESSING_FRAME_RATE_STR: str = os.getenv("STREAM_PROCESSING_FRAME_RATE")
    if not STREAM_PROCESSING_FRAME_RATE_STR:
        print("错误：实时流处理帧率 'STREAM_PROCESSING_FRAME_RATE' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        STREAM_PROCESSING_FRAME_RATE: int = int(STREAM_PROCESSING_FRAME_RATE_STR)
    except ValueError:
        print(f"错误：STREAM_PROCESSING_FRAME_RATE 的值 '{STREAM_PROCESSING_FRAME_RATE_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    VIDEO_COMMIT_BATCH_SIZE_STR: str = os.getenv("VIDEO_COMMIT_BATCH_SIZE")
    if not VIDEO_COMMIT_BATCH_SIZE_STR:
        print("错误：视频提交批处理大小 'VIDEO_COMMIT_BATCH_SIZE' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        VIDEO_COMMIT_BATCH_SIZE: int = int(VIDEO_COMMIT_BATCH_SIZE_STR)
    except ValueError:
        print(f"错误：VIDEO_COMMIT_BATCH_SIZE 的值 '{VIDEO_COMMIT_BATCH_SIZE_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    DETECTION_CONFIDENCE_THRESHOLD_STR: str = os.getenv("DETECTION_CONFIDENCE_THRESHOLD")
    if not DETECTION_CONFIDENCE_THRESHOLD_STR:
        print("错误：检测置信度阈值 'DETECTION_CONFIDENCE_THRESHOLD' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        DETECTION_CONFIDENCE_THRESHOLD: float = float(DETECTION_CONFIDENCE_THRESHOLD_STR)
    except ValueError:
        print(f"错误：DETECTION_CONFIDENCE_THRESHOLD 的值 '{DETECTION_CONFIDENCE_THRESHOLD_STR}' 无效。必须是一个浮点数。", file=sys.stderr)
        sys.exit(1)

    PERSON_CLASS_ID_STR: str = os.getenv("PERSON_CLASS_ID")
    if not PERSON_CLASS_ID_STR:
        print("错误：人物类别 ID 'PERSON_CLASS_ID' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        PERSON_CLASS_ID: int = int(PERSON_CLASS_ID_STR)
    except ValueError:
        print(f"错误：PERSON_CLASS_ID 的值 '{PERSON_CLASS_ID_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    # 新增：Excel 导出相关配置
    EXCEL_EXPORT_MAX_IMAGES_STR: str = os.getenv("EXCEL_EXPORT_MAX_IMAGES")
    if not EXCEL_EXPORT_MAX_IMAGES_STR:
        print("错误：Excel 导出最大图片数量 'EXCEL_EXPORT_MAX_IMAGES' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        EXCEL_EXPORT_MAX_IMAGES: int = int(EXCEL_EXPORT_MAX_IMAGES_STR)
    except ValueError:
        print(f"错误：EXCEL_EXPORT_MAX_IMAGES 的值 '{EXCEL_EXPORT_MAX_IMAGES_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    EXCEL_EXPORT_IMAGE_SIZE_PX_STR: str = os.getenv("EXCEL_EXPORT_IMAGE_SIZE_PX")
    if not EXCEL_EXPORT_IMAGE_SIZE_PX_STR:
        print("错误：Excel 导出图片大小 'EXCEL_EXPORT_IMAGE_SIZE_PX' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        EXCEL_EXPORT_IMAGE_SIZE_PX: int = int(EXCEL_EXPORT_IMAGE_SIZE_PX_STR)
    except ValueError:
        print(f"错误：EXCEL_EXPORT_IMAGE_SIZE_PX 的值 '{EXCEL_EXPORT_IMAGE_SIZE_PX_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    EXCEL_EXPORT_ROW_HEIGHT_PT_STR: str = os.getenv("EXCEL_EXPORT_ROW_HEIGHT_PT")
    if not EXCEL_EXPORT_ROW_HEIGHT_PT_STR:
        print("错误：Excel 导出行高 'EXCEL_EXPORT_ROW_HEIGHT_PT' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        EXCEL_EXPORT_ROW_HEIGHT_PT: int = int(EXCEL_EXPORT_ROW_HEIGHT_PT_STR)
    except ValueError:
        print(f"错误：EXCEL_EXPORT_ROW_HEIGHT_PT 的值 '{EXCEL_EXPORT_ROW_HEIGHT_PT_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    # 新增：MJPEG 视频流帧率配置
    MJPEG_STREAM_FPS_STR: str = os.getenv("MJPEG_STREAM_FPS")
    if not MJPEG_STREAM_FPS_STR:
        print("错误：MJPEG 流帧率 'MJPEG_STREAM_FPS' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        MJPEG_STREAM_FPS: int = int(MJPEG_STREAM_FPS_STR)
    except ValueError:
        print(f"错误：MJPEG_STREAM_FPS 的值 '{MJPEG_STREAM_FPS_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    # 新增：默认管理员用户配置
    DEFAULT_ADMIN_USERNAME: str = os.getenv("DEFAULT_ADMIN_USERNAME")
    if not DEFAULT_ADMIN_USERNAME:
        print("错误：默认管理员用户名 'DEFAULT_ADMIN_USERNAME' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
        
    DEFAULT_ADMIN_PASSWORD: str = os.getenv("DEFAULT_ADMIN_PASSWORD")
    if not DEFAULT_ADMIN_PASSWORD:
        print("错误：默认管理员密码 'DEFAULT_ADMIN_PASSWORD' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    DEFAULT_ADMIN_UNIT: str = os.getenv("DEFAULT_ADMIN_UNIT")
    if not DEFAULT_ADMIN_UNIT:
        print("错误：默认管理员单位 'DEFAULT_ADMIN_UNIT' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
        
    DEFAULT_ADMIN_PHONE_NUMBER: str = os.getenv("DEFAULT_ADMIN_PHONE_NUMBER")
    if not DEFAULT_ADMIN_PHONE_NUMBER:
        print("错误：默认管理员电话号码 'DEFAULT_ADMIN_PHONE_NUMBER' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    
    # 新增：前端静态页面路径配置
    FRONTEND_STATIC_DIR: str = os.path.join(BASE_DIR, FRONTEND_STATIC_DIR_NAME)
    
    LOGIN_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "login.html")
    VIDEO_ANALYSIS_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "video_analysis.html")
    IMAGE_ANALYSIS_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "image_analysis.html")
    IMAGE_ANALYSIS_RESULTS_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "image_analysis_results.html")
    IMAGE_SEARCH_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "image_search.html")
    VIDEO_RESULTS_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "video_results.html")
    ALL_FEATURES_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "all_features.html")
    PERSON_LIST_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "person_list.html")
    VIDEO_STREAM_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "video_stream.html")
    LIVE_STREAM_RESULTS_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "live_stream_results.html")

    # 新增：文件上传和视频进度流配置
    UPLOAD_CHUNK_SIZE_BYTES_STR: str = os.getenv("UPLOAD_CHUNK_SIZE_BYTES")
    if not UPLOAD_CHUNK_SIZE_BYTES_STR:
        print("错误：文件上传分块大小 'UPLOAD_CHUNK_SIZE_BYTES' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        UPLOAD_CHUNK_SIZE_BYTES: int = int(UPLOAD_CHUNK_SIZE_BYTES_STR)
    except ValueError:
        print(f"错误：UPLOAD_CHUNK_SIZE_BYTES 的值 '{UPLOAD_CHUNK_SIZE_BYTES_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS_STR: str = os.getenv("VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS")
    if not VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS_STR:
        print("错误：视频进度轮询间隔 'VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS: int = int(VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS_STR)
    except ValueError:
        print(f"错误：VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS 的值 '{VIDEO_PROGRESS_POLLING_INTERVAL_SECONDS_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    FAISS_METRIC: str = os.getenv("FAISS_METRIC")
    if not FAISS_METRIC:
        print("错误：Faiss 指标 'FAISS_METRIC' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)

    # 新增：Faiss 搜索时返回的最近邻数量
    FAISS_SEARCH_K_STR: str = os.getenv("FAISS_SEARCH_K")
    if not FAISS_SEARCH_K_STR:
        print("错误：Faiss 搜索 K 值 'FAISS_SEARCH_K' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        FAISS_SEARCH_K: int = int(FAISS_SEARCH_K_STR)
    except ValueError:
        print(f"错误：FAISS_SEARCH_K 的值 '{FAISS_SEARCH_K_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    REID_TRAIN_BATCH_SIZE_STR: str = os.getenv("REID_TRAIN_BATCH_SIZE")
    if not REID_TRAIN_BATCH_SIZE_STR:
        print("警告：Re-ID 训练批量大小 'REID_TRAIN_BATCH_SIZE' 未在您的 .env 文件中设置。将使用默认值 32。", file=sys.stderr)
        REID_TRAIN_BATCH_SIZE: int = 32
    else:
        try:
            REID_TRAIN_BATCH_SIZE: int = int(REID_TRAIN_BATCH_SIZE_STR)
        except ValueError:
            print(f"错误：REID_TRAIN_BATCH_SIZE 的值 '{REID_TRAIN_BATCH_SIZE_STR}' 无效。必须是一个整数。", file=sys.stderr)
            REID_TRAIN_BATCH_SIZE: int = 32

    REID_TRAIN_LEARNING_RATE_STR: str = os.getenv("REID_TRAIN_LEARNING_RATE")
    if not REID_TRAIN_LEARNING_RATE_STR:
        print("警告：Re-ID 训练学习率 'REID_TRAIN_LEARNING_RATE' 未在您的 .env 文件中设置。将使用默认值 0.001。", file=sys.stderr)
        REID_TRAIN_LEARNING_RATE: float = 0.001
    else:
        try:
            REID_TRAIN_LEARNING_RATE: float = float(REID_TRAIN_LEARNING_RATE_STR)
        except ValueError:
            print(f"错误：REID_TRAIN_LEARNING_RATE 的值 '{REID_TRAIN_LEARNING_RATE_STR}' 无效。必须是一个浮点数。将使用默认值 0.001。", file=sys.stderr)
            REID_TRAIN_LEARNING_RATE: float = 0.001

    # 新增：实时比对配置
    REALTIME_COMPARISON_THRESHOLD_STR: str = os.getenv("REALTIME_COMPARISON_THRESHOLD")
    if not REALTIME_COMPARISON_THRESHOLD_STR:
        print("警告：实时比对阈值 \'REALTIME_COMPARISON_THRESHOLD\' 未在您的 .env 文件中设置。将使用默认值 0.6。", file=sys.stderr)
        REALTIME_COMPARISON_THRESHOLD: float = 0.6
    else:
        try:
            REALTIME_COMPARISON_THRESHOLD: float = float(REALTIME_COMPARISON_THRESHOLD_STR)
        except ValueError:
            print(f"错误：REALTIME_COMPARISON_THRESHOLD 的值 \'{REALTIME_COMPARISON_THRESHOLD_STR}\' 无效。必须是一个浮点数。将使用默认值 0.6。", file=sys.stderr)
            REALTIME_COMPARISON_THRESHOLD: float = 0.6

    REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS_STR: str = os.getenv("REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS")
    if not REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS_STR:
        print("警告：实时比对最大关注人数 \'REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS\' 未在您的 .env 文件中设置。将使用默认值 100。", file=sys.stderr)
        REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS: int = 100
    else:
        try:
            REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS: int = int(REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS_STR)
        except ValueError:
            print(f"错误：REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS 的值 \'{REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS_STR}\' 无效。必须是一个整数。将使用默认值 100。", file=sys.stderr)
            REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS: int = 100

    GLOBAL_SEARCH_MIN_CONFIDENCE: float = float(os.getenv("GLOBAL_SEARCH_MIN_CONFIDENCE", "0.9")) # 新增：全局搜索最小置信度

    # --- 人机回环审核配置 ---
    # HUMAN_REVIEW_CONFIDENCE_THRESHOLD: 人机回环审核时，特征置信度阈值。

    # Log settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL")
    if not LOG_LEVEL:
        print("错误：日志级别 'LOG_LEVEL' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
        
    LOG_FILE_PATH: str = os.getenv("LOG_FILE_PATH")
    if LOG_FILE_PATH is None: # 这里允许为空字符串，表示不写入文件
        print("错误：日志文件路径 'LOG_FILE_PATH' 未在您的 .env 文件中设置。如果不需要写入文件，请设置为 LOG_FILE_PATH=''.", file=sys.stderr)
        sys.exit(1)

    LOG_FILE_MAX_BYTES_STR: str = os.getenv("LOG_FILE_MAX_BYTES")
    if not LOG_FILE_MAX_BYTES_STR:
        print("错误：日志文件最大字节数 'LOG_FILE_MAX_BYTES' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        LOG_FILE_MAX_BYTES: int = int(LOG_FILE_MAX_BYTES_STR)
    except ValueError:
        print(f"错误：LOG_FILE_MAX_BYTES 的值 '{LOG_FILE_MAX_BYTES_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1)

    LOG_FILE_BACKUP_COUNT_STR: str = os.getenv("LOG_FILE_BACKUP_COUNT")
    if not LOG_FILE_BACKUP_COUNT_STR:
        print("错误：日志文件备份数量 'LOG_FILE_BACKUP_COUNT' 未在您的 .env 文件中设置。", file=sys.stderr)
        sys.exit(1)
    try:
        LOG_FILE_BACKUP_COUNT: int = int(LOG_FILE_BACKUP_COUNT_STR)
    except ValueError:
        print(f"错误：LOG_FILE_BACKUP_COUNT 的值 '{LOG_FILE_BACKUP_COUNT_STR}' 无效。必须是一个整数。", file=sys.stderr)
        sys.exit(1) 

settings = Settings() 