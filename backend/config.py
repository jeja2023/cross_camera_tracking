import os
from dotenv import load_dotenv
import sys
from sqlalchemy.orm import Session # 新增导入
from .database_conn import SessionLocal, SystemConfig # 新增导入
import logging # 新增导入

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


    ENROLL_PERSON_IMAGES_DIR_NAME: str = os.getenv("ENROLL_PERSON_IMAGES_DIR_NAME")
    if not ENROLL_PERSON_IMAGES_DIR_NAME:
        print("错误：主动注册图片上传目录名称 'ENROLL_PERSON_IMAGES_DIR_NAME' 未在您的 .env 文件中设置。", file=sys.stderr)
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
    ENROLL_PERSON_IMAGES_DIR: str = os.path.join(BASE_DIR, "backend", ENROLL_PERSON_IMAGES_DIR_NAME)
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
    GAIT_SEQUENCE_LENGTH: int = int(GAIT_SEQUENCE_LENGTH_STR)

    # Ensure directories exist
    def __init__(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.ENROLL_PERSON_IMAGES_DIR, exist_ok=True) # 确保主动注册图片目录也存在
        os.makedirs(self.SAVED_STREAMS_DIR, exist_ok=True)
        os.makedirs(self.DATABASE_CROPS_DIR, exist_ok=True)
        os.makedirs(self.DATABASE_FULL_FRAMES_DIR, exist_ok=True) # 确保完整帧目录也存在
        
        # 移除原先确保图片解析特有的子目录也存在的代码
        # os.makedirs(self.DATABASE_FULL_FRAMES_IMAGE_ANALYSIS_DIR, exist_ok=True)
        # os.makedirs(self.DATABASE_CROPS_IMAGE_ANALYSIS_DIR, exist_ok=True)
        self.load_from_db() # 增加从数据库加载配置的调用

    def load_from_db(self, db: Session = None): # 新增方法，从数据库加载配置
        """
        从数据库加载配置覆盖环境变量中的值。主要用于模型参数等需要动态更新的配置。
        """
        if db is None:
            db = SessionLocal() # 获取数据库会话
        try:
            db_configs = db.query(SystemConfig).all()
            for config in db_configs:
                # 检查 config.key 是否存在于 Settings 的属性中，并且不是一个方法或私有变量
                if hasattr(self, config.key) and not config.key.startswith("_") and not callable(getattr(self, config.key)):
                    try:
                        # 尝试将数据库值转换回原始类型
                        # 这里的逻辑需要和 schemas.py 中的 ModelConfig 定义以及 admin_routes.py 中的PUT方法保持一致
                        current_type = type(getattr(self, config.key))
                        if current_type is int:
                            setattr(self, config.key, int(config.value))
                        elif current_type is float:
                            setattr(self, config.key, float(config.value))
                        elif current_type is bool:
                            setattr(self, config.key, config.value.lower() == "true")
                        else: # 默认为字符串
                            setattr(self, config.key, config.value)
                        logger.debug(f"从数据库加载配置: {config.key} = {getattr(self, config.key)}")
                    except ValueError as e:
                        logger.error(f"从数据库加载配置 {config.key} 失败，类型转换错误: {e}")
        except Exception as e:
            logger.error(f"从数据库加载配置时发生错误: {e}", exc_info=True)
        finally:
            if db is not None: # 如果是函数内部创建的 db 会话，则关闭
                db.close()

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
        print("错误：Celery Broker URL 'CELERY_BROKER_URL' 未在您的 .env 文件中设置。请参照格式添加：CELERY_BROKER_URL='redis://localhost:6379/0'", file=sys.stderr)
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
        print("错误：JWT ALGORITHM 未在您的 .env 文件中设置。请参照格式添加：ALGORITHM='HS256'", file=sys.stderr)
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

    # 新增：预警图片最低比对分值
    ALERT_IMAGE_MIN_CONFIDENCE_SCORE_STR: str = os.getenv("ALERT_IMAGE_MIN_CONFIDENCE_SCORE")
    if not ALERT_IMAGE_MIN_CONFIDENCE_SCORE_STR:
        print("错误：预警图片最低比对分值 'ALERT_IMAGE_MIN_CONFIDENCE_SCORE' 未在您的 .env 文件中设置。将使用默认值 90.0。", file=sys.stderr)
        ALERT_IMAGE_MIN_CONFIDENCE_SCORE: float = 90.0 # 提供一个默认值
    else:
        try:
            ALERT_IMAGE_MIN_CONFIDENCE_SCORE: float = float(ALERT_IMAGE_MIN_CONFIDENCE_SCORE_STR)
        except ValueError:
            print(f"错误：ALERT_IMAGE_MIN_CONFIDENCE_SCORE 的值 '{ALERT_IMAGE_MIN_CONFIDENCE_SCORE_STR}' 无效。必须是一个浮点数。将使用默认值 90.0。", file=sys.stderr)
            ALERT_IMAGE_MIN_CONFIDENCE_SCORE: float = 90.0
    
    # 新增：图片解析后自动关联到关注人员的最低置信度
    AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE_STR: str = os.getenv("AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE")
    if not AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE_STR:
        print("错误：图片解析自动关联最低置信度 'AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE' 未在您的 .env 文件中设置。将使用默认值 95.0。", file=sys.stderr)
        AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE: float = 95.0 # 提供一个默认值
    else:
        try:
            AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE: float = float(AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE_STR)
        except ValueError:
            print(f"错误：AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE 的值 '{AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE_STR}' 无效。必须是一个浮点数。将使用默认值 95.0。", file=sys.stderr)
            AUTO_ASSOCIATION_MIN_CONFIDENCE_SCORE: float = 95.0

    # 新增：定时全库比对功能开关
    ENABLE_SCHEDULED_FULL_DATABASE_COMPARISON_STR: str = os.getenv("ENABLE_SCHEDULED_FULL_DATABASE_COMPARISON", "False")
    ENABLE_SCHEDULED_FULL_DATABASE_COMPARISON: bool = ENABLE_SCHEDULED_FULL_DATABASE_COMPARISON_STR.lower() == "true"
    if not ENABLE_SCHEDULED_FULL_DATABASE_COMPARISON:
        print("定时全库比对功能 (ENABLE_SCHEDULED_FULL_DATABASE_COMPARISON) 当前处于禁用状态。如果您想启用，请在 .env 文件中设置为 'True'。", file=sys.stdout)
    
    # 新增：定时全库比对间隔时间（小时）
    SCHEDULED_FULL_DATABASE_COMPARISON_INTERVAL_HOURS_STR: str = os.getenv("SCHEDULED_FULL_DATABASE_COMPARISON_INTERVAL_HOURS", "24")
    try:
        SCHEDULED_FULL_DATABASE_COMPARISON_INTERVAL_HOURS: int = int(SCHEDULED_FULL_DATABASE_COMPARISON_INTERVAL_HOURS_STR)
        if SCHEDULED_FULL_DATABASE_COMPARISON_INTERVAL_HOURS <= 0:
            raise ValueError("间隔时间必须大于0。")
    except ValueError:
        print(f"错误：定时全库比对间隔时间 '{SCHEDULED_FULL_DATABASE_COMPARISON_INTERVAL_HOURS_STR}' 无效。必须是一个正整数。将使用默认值 24 小时。", file=sys.stderr)
        SCHEDULED_FULL_DATABASE_COMPARISON_INTERVAL_HOURS: int = 24

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
    FOLLOWED_PERSONS_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "followed_persons.html") # 新增：关注人员页面路径

    ENROLLMENT_IMAGES_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "enrollment_images.html")
    ALERT_IMAGES_PAGE_PATH: str = os.path.join(FRONTEND_STATIC_DIR, "alert_images.html")

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
            print(f"错误：REID_TRAIN_BATCH_SIZE 的值 '{REID_TRAIN_BATCH_SIZE_STR}' 无效。必须是一个整数。将使用默认值 32。", file=sys.stderr)
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