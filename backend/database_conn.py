import os
import datetime
import sys
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, TIMESTAMP, Boolean, Float, UniqueConstraint
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

    persons = relationship("Person", back_populates="individual", cascade="all, delete-orphan")
    correction_logs_as_target_individual = relationship("CorrectionLog", back_populates="target_individual") # 新增：作为目标 Individual 的纠正日志
    followed_by = relationship("FollowedPerson", back_populates="followed_individual", primaryjoin="Individual.id == FollowedPerson.individual_id") # 新增：关注此 Individual 的记录，明确 primaryjoin

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
    followed_persons = relationship("FollowedPerson", back_populates="follower") # 新增：用户关注的人员
    persons = relationship("Person", back_populates="user_uploaded", foreign_keys="[Person.user_uploaded_id]") # 修正：为 user_uploaded 关系添加 back_populates

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
    verified_by_user = relationship("User", back_populates="verified_persons", foreign_keys=[verified_by_user_id]) # 新增：关联到审核用户
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

    # Foreign Keys and Relationships
    individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=True)
    individual = relationship("Individual", back_populates="persons")

    video_id = Column(Integer, ForeignKey("videos.id"), nullable=True)
    video = relationship("Video", back_populates="persons")

    stream_id = Column(Integer, ForeignKey("streams.id"), nullable=True) # 新增：关联到视频流
    stream = relationship("Stream", back_populates="persons")

    image_id = Column(Integer, ForeignKey("images.id"), nullable=True)
    image = relationship("Image", back_populates="persons")

    user_uploaded_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user_uploaded = relationship("User", back_populates="persons", foreign_keys=[user_uploaded_id])

    correction_logs = relationship("CorrectionLog", back_populates="person") # 新增：关联的纠正日志
    # followed_by_users = relationship("FollowedPerson", back_populates="followed_person") # 移除：关注此 Person 的记录，因为关注的是 Individual

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
    target_individual = relationship("Individual", back_populates="correction_logs_as_target_individual") # 新增：关联到目标 Individual

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
    follower_id = Column(Integer, ForeignKey("users.id"), nullable=False) # 关注者ID
    individual_id = Column(Integer, ForeignKey(Individual.id), nullable=False) # 被关注的 Individual ID，明确指定外键
    person_uuid = Column(String(36), nullable=True) # 保留，但不再是外键，因为主要关联是 individual_id
    followed_at = Column(TIMESTAMP, default=lambda: datetime.datetime.now(beijing_tz), nullable=False) # 关注时间
    unfollowed_at = Column(TIMESTAMP, nullable=True) # 取消关注时间
    realtime_comparison_enabled = Column(Boolean, default=False, nullable=False) # 新增：实时比对开关

    __table_args__ = (UniqueConstraint('follower_id', 'individual_id', name='_follower_individual_uc'),)

    follower = relationship("User", back_populates="followed_persons")
    followed_individual = relationship("Individual", back_populates="followed_by")
    # followed_person = relationship("Person", back_populates="followed_by_users") # 移除此关系


def create_tables():
    Base.metadata.create_all(bind=engine)