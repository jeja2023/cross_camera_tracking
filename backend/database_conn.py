import os
import datetime
import sys
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, TIMESTAMP, Boolean, Float
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from dotenv import load_dotenv
import pytz

# 从 .env 文件加载环境变量
load_dotenv()

# --- MySQL特定配置 ---
# 从环境变量中获取 MySQL 数据库连接字符串
DATABASE_URL = os.getenv("DATABASE_URL")

# 检查 DATABASE_URL 是否已设置
if not DATABASE_URL:
    print("错误：数据库连接字符串 'DATABASE_URL' 未在您的 .env 文件中设置。", file=sys.stderr)
    print("请参照格式添加：DATABASE_URL='mysql+pymysql://user:password@host:port/dbname'", file=sys.stderr)
    sys.exit(1) # 如果未设置，则退出程序以避免后续错误

# 创建数据库引擎，专为 MySQL 配置
engine = create_engine(
    DATABASE_URL,
    # 建议在生产环境中为 MySQL 配置连接池
    pool_recycle=3600, # 每小时回收一次连接
    pool_pre_ping=True   # 在每次检出时检查连接的有效性
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 定义东八区时区
beijing_tz = pytz.timezone('Asia/Shanghai')

# 将get_db函数移到这里，作为数据库服务的统一出口
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 新增 Individual ORM 模型
class Individual(Base):
    __tablename__ = "individuals"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=True) # 可选的 UUID
    name = Column(String(255), nullable=True)
    id_card = Column(String(255), unique=True, index=True, nullable=False) # 身份证号，唯一且非空
    created_at = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz))
    is_realtime_comparison_enabled = Column(Boolean, default=False, comment="是否为此个体启用实时比对") # 新增：实时比对开关

    persons = relationship("Person", back_populates="individual", cascade="all, delete-orphan")
    followed_persons = relationship("FollowedPerson", back_populates="individual", cascade="all, delete-orphan") # 新增：关联到 FollowedPerson

    __table_args__ = {'extend_existing': True}

# User ORM 模型
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default='user')
    unit = Column(String(100), nullable=True)
    phone_number = Column(String(20), nullable=True)
    is_active = Column(Boolean, default=False)
    videos = relationship("Video", back_populates="owner")
    streams = relationship("Stream", back_populates="owner")
    images = relationship("Image", back_populates="owner") # 新增：关联到 Image 模型
    verified_persons = relationship("Person", back_populates="verified_by_user", foreign_keys="[Person.verified_by_user_id]") # 修正：明确指定 foreign_keys
    corrections = relationship("CorrectionLog", back_populates="corrected_by_user") # 新增：关联到 CorrectionLog
    followed_persons = relationship("FollowedPerson", back_populates="user", cascade="all, delete-orphan") # 新增：关联到 FollowedPerson

    __table_args__ = {'extend_existing': True}


# Video ORM模型 (for uploaded files)
class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=True) # 新增：视频的唯一标识符
    filename = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    status = Column(String(50), default="completed")
    progress = Column(Integer, default=0)
    processed_at = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz))
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="videos")
    persons = relationship("Person", back_populates="video", cascade="all, delete-orphan")

    __table_args__ = {'extend_existing': True}


# New Stream ORM model (for live streams)
class Stream(Base):
    __tablename__ = "streams"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True) # User-friendly name for the stream
    stream_url = Column(String(255), nullable=False)
    status = Column(String(50), default="inactive") # e.g., "active", "inactive", "processing_frames", "error"
    stream_uuid = Column(String(36), unique=True, index=True, nullable=True) # A unique ID for internal tracking if needed
    created_at = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz))
    last_processed_at = Column(TIMESTAMP, nullable=True) # To track when frames were last extracted/processed
    output_video_path = Column(String(255), nullable=True) # 新增：保存处理后视频的路径

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="streams")
    persons = relationship("Person", back_populates="stream", cascade="all, delete-orphan")

    __table_args__ = {'extend_existing': True}


# Image ORM 模型
class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False) # 原始图片的唯一标识符
    filename = Column(String(255), nullable=False) # 原始图片的文件名
    file_path = Column(String(255), nullable=False) # 原始图片在服务器上的相对路径
    created_at = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz))
    person_count = Column(Integer, default=0) # 记录图片中检测到的人物数量

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="images")
    persons = relationship("Person", back_populates="image", cascade="all, delete-orphan") # 关联到 Person 模型

    __table_args__ = {'extend_existing': True}


# Person ORM模型
class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, index=True, nullable=False)
    # name = Column(String(255), nullable=True) # 移除，由 Individual 管理
    # id_card = Column(String(255), nullable=True) # 移除，由 Individual 管理
    feature_vector = Column(Text, nullable=False)
    crop_image_path = Column(String(255), nullable=False)
    full_frame_image_path = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz))
    is_verified = Column(Boolean, default=False) # 新增：是否经过人工审核
    verified_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # 新增：审核人ID
    verification_date = Column(TIMESTAMP, nullable=True) # 新增：审核时间
    correction_details = Column(Text, nullable=True) # 新增：纠正详情 (JSON)
    marked_for_retrain = Column(Boolean, default=False) # 新增：是否被标记用于再训练
    confidence_score = Column(Float, nullable=True) # 新增：特征置信度分数
    pose_keypoints = Column(Text, nullable=True) # 新增：姿态关键点 (JSON)
    face_image_path = Column(String(255), nullable=True) # 新增：裁剪后的人脸图像路径
    face_feature_vector = Column(Text, nullable=True) # 新增：人脸特征向量 (JSON)
    face_id = Column(String(36), nullable=True) # 新增：人脸ID (用于人脸识别或聚类)
    clothing_attributes = Column(Text, nullable=True) # 新增：衣着属性 (JSON)
    gait_feature_vector = Column(Text, nullable=True) # 新增：步态特征向量 (JSON)
    gait_image_path = Column(String(255), nullable=True) # 新增：步态图像路径

    is_trained = Column(Boolean, default=False) # 新增：是否已用于模型训练
    correction_type_display = Column(String(255), nullable=True) # 新增：最近一次纠正的类型 (用于前端展示)
    is_followed = Column(Boolean, default=False) # 新增：是否被关注

    is_enrollment_image = Column(Boolean, default=False, comment="是否为主动注册的图片") # 新增：标识是否为主动注册图片

    # 新增 individual_id 外键
    individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=True)
    individual = relationship("Individual", back_populates="persons")

    # 将 image_id 字段改为外键，指向 images 表的 id
    image_id = Column(Integer, ForeignKey("images.id"), nullable=True)
    image = relationship("Image", back_populates="persons") # 关联到 Image 模型

    # 将 video_id 设置为可空，并添加 stream_id
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=True)
    video = relationship("Video", back_populates="persons")

    stream_id = Column(Integer, ForeignKey("streams.id"), nullable=True) # 新增：关联到视频流
    stream = relationship("Stream", back_populates="persons") # 新增：关联到视频流
    verified_by_user = relationship("User", back_populates="verified_persons", foreign_keys=[verified_by_user_id]) # 新增：关联到审核用户

    __table_args__ = {'extend_existing': True}


# 新增 CorrectionLog ORM 模型
class CorrectionLog(Base):
    __tablename__ = "correction_logs"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(Integer, ForeignKey("persons.id"), nullable=False) # 被纠正的人物
    original_feature_vector = Column(Text, nullable=True) # 原始特征向量，改为可空
    corrected_feature_vector = Column(Text, nullable=True) # 纠正后的特征向量（如果变化）
    correction_type = Column(String(50), nullable=False) # 纠正类型，例如 'merge', 'split', 'relabel'
    details = Column(Text, nullable=True) # 详细的纠正信息
    corrected_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # 纠正人ID
    corrected_by_username = Column(String(100), nullable=True) # 新增：记录纠正人用户名
    ip_address = Column(String(45), nullable=True) # 新增：记录纠正IP地址
    correction_date = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz), nullable=False) # 纠正时间
    target_person_uuid = Column(String(36), nullable=True) # 新增：存储目标人物UUID，用于合并操作
    target_individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=True) # 新增：存储目标 Individual ID

    person = relationship("Person", back_populates="correction_logs") # 关联到 Person
    corrected_by_user = relationship("User", back_populates="corrections") # 关联到纠正用户

    __table_args__ = {'extend_existing': True}


# Person 模型中添加 correction_logs 关系
Person.correction_logs = relationship("CorrectionLog", back_populates="person", cascade="all, delete-orphan")


# Log ORM 模型
class Log(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz), nullable=False)
    logger = Column(String(255), nullable=False)
    level = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    username = Column(String(100), nullable=True) # 新增用户名列
    ip_address = Column(String(45), nullable=True) # 新增 IP 地址列 (IPv6最大长度45)

    __table_args__ = {'extend_existing': True}


# 新增 SystemConfig ORM 模型
class SystemConfig(Base):
    __tablename__ = "system_configs"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, index=True, nullable=False) # 配置键，例如 'active_reid_model_path'
    value = Column(Text, nullable=False) # 配置值
    last_updated = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz), nullable=False) # 最后更新时间

    __table_args__ = {'extend_existing': True}


# 新增 FollowedPerson ORM 模型
class FollowedPerson(Base):
    __tablename__ = "followed_persons"
    id = Column(Integer, primary_key=True, index=True)
    individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=False) # 关联到 Individual
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False) # 关注用户
    follow_time = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz), nullable=False)
    unfollow_time = Column(TIMESTAMP, nullable=True) # 取消关注时间

    individual = relationship("Individual", back_populates="followed_persons")
    user = relationship("User", back_populates="followed_persons")

    __table_args__ = {'extend_existing': True}


# 新增 RealtimeMatchAlert ORM 模型
class RealtimeMatchAlert(Base):
    __tablename__ = "realtime_match_alerts"
    id = Column(Integer, primary_key=True, index=True)
    person_uuid = Column(String(36), ForeignKey("persons.uuid"), nullable=False, comment="被检测到的人物 UUID")
    matched_individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=False, comment="匹配到的关注人员（逻辑人物）ID")
    matched_individual_uuid = Column(String(36), nullable=False, comment="匹配到的关注人员（逻辑人物）UUID")
    matched_individual_name = Column(String(255), nullable=True, comment="匹配到的关注人员姓名")
    similarity_score = Column(Float, nullable=False, comment="比对的相似度分数")
    timestamp = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz), nullable=False, comment="预警发生的时间")
    alert_type = Column(String(100), default="realtime_followed_person_match", nullable=False, comment="预警类型")
    source_media_type = Column(String(50), nullable=False, comment="来源媒体类型 (image, video, stream)")
    source_media_uuid = Column(String(36), nullable=False, comment="来源媒体的 UUID")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="触发比对的用户 ID 或流所有者 ID")
    cropped_image_path = Column(String(255), nullable=False, comment="预警人物裁剪图片路径") # 新增字段
    full_frame_image_path = Column(String(255), nullable=False, comment="预警原始图片（全帧）路径") # 新增字段

    # 关系定义
    person = relationship("Person", foreign_keys=[person_uuid], primaryjoin="Person.uuid == RealtimeMatchAlert.person_uuid")
    matched_individual = relationship("Individual", foreign_keys=[matched_individual_id])
    alert_user = relationship("User", foreign_keys=[user_id])

    __table_args__ = {'extend_existing': True}


# 新增 GlobalSearchResult ORM 模型
class GlobalSearchResult(Base):
    __tablename__ = "global_search_results"
    id = Column(Integer, primary_key=True, index=True)
    individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=False, comment="关联的 Individual ID")
    matched_person_uuid = Column(String(36), nullable=False, comment="匹配到的人物 UUID")
    matched_person_id = Column(Integer, ForeignKey("persons.id"), nullable=False, comment="匹配到的人物 ID") # 新增 matched_person_id
    matched_image_path = Column(String(255), nullable=False, comment="匹配到的图片路径")
    confidence = Column(Float, nullable=False, comment="比对置信度")
    search_time = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz), nullable=False, comment="搜索时间")
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, comment="发起搜索的用户 ID")
    is_initial_search = Column(Boolean, default=False, nullable=False, comment="是否为首次关注人物时触发的搜索")

    individual = relationship("Individual", backref="global_search_results") # 关联到 Individual
    user = relationship("User", backref="global_search_results") # 关联到 User
    person = relationship("Person", foreign_keys=[matched_person_id], backref="global_search_results_as_matched") # 关联到 Person

    __table_args__ = {'extend_existing': True}


def create_tables():
    Base.metadata.create_all(bind=engine)