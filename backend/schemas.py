from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union, Any
from datetime import datetime
import os # 导入 os 模块
from pydantic import validator
import json
from enum import Enum # 导入 Enum
import pytz # 导入 pytz 模块

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    task_id: str
    status: TaskStatus
    progress: Optional[int] = Field(None, ge=0, le=100, description="任务进度百分比 (0-100)") # 新增进度字段
    message: Optional[str] = None
    result: Optional[dict] = None

    class Config:
        from_attributes = True

class UserRole(str, Enum):
    USER = "user"
    ADVANCED = "advanced"
    ADMIN = "admin"

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, 
                          errors_messages={'too_short': '密码长度不能少于8个字符！'},
                          description="密码必须至少包含8个字符，且包含大小写字母、数字和特殊字符。")

    @validator('password')
    def password_strength(cls, v):
        if not any(char.isupper() for char in v):
            raise ValueError('密码必须包含至少一个大写字母！')
        if not any(char.islower() for char in v):
            raise ValueError('密码必须包含至少一个小写字母！')
        if not any(char.isdigit() for char in v):
            raise ValueError('密码必须包含至少一个数字！')
        if not any(not char.isalnum() for char in v):
            raise ValueError('密码必须包含至少一个特殊字符！')
        return v

    role: UserRole = UserRole.USER # 使用 Enum 定义角色
    unit: Optional[str] = None
    phone_number: Optional[str] = None
    is_active: bool = False

class User(UserBase):
    id: int
    username: str
    role: UserRole # 使用 Enum 定义角色
    unit: Optional[str] = None
    phone_number: Optional[str] = None
    is_active: bool
    class Config:
        from_attributes = True

class UserRoleUpdate(BaseModel):
    role: UserRole # 使用 Enum 定义角色

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int = Field(..., description="Access Token's expiration time in seconds")

class TokenData(BaseModel):
    username: Optional[str] = None

# 新增 Individual 相关的 Pydantic 模型
class IndividualBase(BaseModel):
    name: Optional[str] = Field(None, description="人物姓名/标识")
    id_card: Optional[str] = Field(None, description="身份证号或其他ID")

class IndividualCreate(IndividualBase):
    uuid: Optional[str] = None # Individual 也可以有一个 UUID

class Individual(IndividualBase):
    id: int
    uuid: Optional[str] = None
    created_at: datetime
    is_realtime_comparison_enabled: Optional[bool] = False # 修正：改为 Optional[bool] 并设置默认值

    class Config:
        from_attributes = True

class Person(BaseModel):
    id: int
    uuid: str
    # name: Optional[str] = Field(None, description="人物姓名/标识") # 移除
    # id_card: Optional[str] = Field(None, description="身份证号或其他ID") # 移除
    feature_vector: Any # Store as JSON string, return as parsed object
    crop_image_path: str
    full_frame_image_path: Optional[str] = None
    created_at: datetime
    video_id: Optional[int] = None
    stream_id: Optional[int] = None
    upload_image_id: Optional[int] = None # For image analysis
    video_uuid: Optional[str] = None # Added for display
    video_name: Optional[str] = None # Added for display
    stream_uuid: Optional[str] = None # Added for display
    stream_name: Optional[str] = None # Added for display
    upload_image_uuid: Optional[str] = None # Added for display
    upload_image_filename: Optional[str] = None # Added for display
    is_verified: bool
    verified_by_user_id: Optional[int] = None
    verification_date: Optional[datetime] = None
    correction_details: Optional[str] = None
    marked_for_retrain: bool
    confidence_score: Optional[float] = None
    pose_keypoints: Optional[Any] = None
    face_image_path: Optional[str] = None
    face_feature_vector: Optional[Any] = None
    face_id: Optional[str] = None
    clothing_attributes: Optional[Any] = None
    gait_feature_vector: Optional[Any] = None
    gait_image_path: Optional[str] = None

    # 新增 individual_id 和 individual 对象
    individual_id: Optional[int] = None
    individual: Optional[Individual] = None # Nested Individual model

    # 为兼容现有 API，重新添加 name 和 id_card 作为计算属性（由后端填充）
    name: Optional[str] = Field(None, description="人物姓名/标识 (从关联的 Individual 获取)") 
    id_card: Optional[str] = Field(None, description="身份证号或其他ID (从关联的 Individual 获取)") 
    is_trained: bool = Field(False, description="是否已用于模型训练") # 修改：显式使用 Field 并设置默认值
    correction_type_display: Optional[str] = Field(None, description="最近一次纠正的类型 (用于前端展示)") # 新增：纠正类型显示字段
    uploaded_by_username: Optional[str] = Field(None, description="上传该人物信息的用户") # 新增：上传用户名字段
    is_followed: Optional[bool] = Field(False, description="人物是否被关注") # 新增：关注状态

    class Config:
        from_attributes = True # Use alias_generator instead of allow_population_by_field_name
        # Add a custom model_validator for name and id_card if they are not directly stored
        # @model_validator(mode='after')
        # def populate_individual_details(self):
        #     if self.individual:
        #         self.name = self.individual.name
        #         self.id_card = self.individual.id_card
        #     return self

    # @model_validator(mode='after')
    # def populate_uploaded_by(self) -> 'Person':
    #     if self.video and self.video.owner:
    #         self.uploaded_by_username = self.video.owner.username
    #     elif self.stream and self.stream.owner:
    #         self.uploaded_by_username = self.stream.owner.username
    #     elif self.image and self.image.owner:
    #         self.uploaded_by_username = self.image.owner.username
    #     return self

class PersonCreate(BaseModel):
    uuid: str
    feature_vector: str # JSON string of numpy array
    crop_image_path: str
    full_frame_image_path: Optional[str] = None # 新增，存储完整帧图片路径
    video_id: Optional[int] = None
    stream_id: Optional[int] = None
    image_id: Optional[int] = None # 用于静态图片分析
    is_verified: bool = False
    verified_by_user_id: Optional[int] = None
    verification_date: Optional[datetime] = None
    correction_details: Optional[str] = None
    marked_for_retrain: bool = False
    confidence_score: Optional[float] = None # 新增置信度分数
    pose_keypoints: Optional[str] = None # 新增：姿态关键点 (JSON 字符串)
    face_image_path: Optional[str] = None # 新增：人脸图像路径
    face_feature_vector: Optional[str] = None # 新增：人脸特征向量 (JSON 字符串)
    face_id: Optional[str] = None # 新增：人脸ID，用于人脸聚类
    clothing_attributes: Optional[str] = None # 新增：衣着属性 (JSON 字符串)
    gait_feature_vector: Optional[str] = None # 新增：步态特征向量 (JSON 字符串)
    gait_image_path: Optional[str] = None # 新增：步态图像路径

    # 新增 Individual 关联字段
    individual_id: Optional[int] = Field(None, description="可选：关联到的逻辑人物ID")
    id_card: Optional[str] = Field(None, description="可选：关联到的逻辑人物身份证号，用于查找或创建 Individual")
    person_name: Optional[str] = Field(None, description="可选：关联到的逻辑人物姓名，用于创建 Individual (当 id_card 存在时)")
    correction_type_display: Optional[str] = Field(None, description="最近一次纠正的类型 (用于前端展示)") # 新增

class PersonUpdate(BaseModel):
    """
    用于更新人物信息的 Pydantic 模型。所有字段都是可选的，以便进行部分更新。
    """
    feature_vector: Optional[str] = None
    crop_image_path: Optional[str] = None
    full_frame_image_path: Optional[str] = None
    video_id: Optional[int] = None
    stream_id: Optional[int] = None
    image_id: Optional[int] = None
    is_verified: Optional[bool] = None
    verified_by_user_id: Optional[int] = None
    verification_date: Optional[datetime] = None
    correction_details: Optional[str] = None
    marked_for_retrain: Optional[bool] = None
    confidence_score: Optional[float] = None
    pose_keypoints: Optional[str] = None
    face_image_path: Optional[str] = None
    face_feature_vector: Optional[str] = None
    face_id: Optional[str] = None
    clothing_attributes: Optional[str] = None
    gait_feature_vector: Optional[str] = None
    gait_image_path: Optional[str] = None
    individual_id: Optional[int] = Field(None, description="可选：关联到的逻辑人物ID")
    id_card: Optional[str] = Field(None, description="可选：关联到的逻辑人物身份证号，用于查找或创建 Individual")
    person_name: Optional[str] = Field(None, description="可选：关联到的逻辑人物姓名，用于创建 Individual (当 id_card 存在时)")
    correction_type_display: Optional[str] = Field(None, description="最近一次纠正的类型 (用于前端展示)")
    is_trained: Optional[bool] = None


class PersonEnrollRequest(BaseModel):
    person_name: Optional[str] = Field(None, description="可选：人物的姓名或标识 (用于创建或更新 Individual)")
    id_card: Optional[str] = Field(None, description="可选：身份证号或其他ID (用于查找或创建 Individual)")
    # target_person_uuid: Optional[str] = Field(None, description="可选：如果关联到已有人物，则为此人物的 UUID") # 移除

class PersonEnrollResponse(BaseModel):
    message: str
    # 这里的 person_uuid 应该是指向 Person 记录的 UUID，如果需要返回 Individual 的 UUID 可以在这里新增
    person_uuid: str 
    individual_uuid: Optional[str] = None # 新增：关联的 Individual 的 UUID
    individual_id_card: Optional[str] = None # 新增：关联的 Individual 的身份证号
    details: Optional[str] = None

class ImageSearchResult(BaseModel):
    uuid: str
    score: float
    crop_image_path: str
    full_frame_image_path: Optional[str] = None # 新增字段
    timestamp: datetime # 新增：特征提取时间
    video_id: Optional[int] = None
    video_filename: Optional[str] = None
    video_uuid: Optional[str] = None
    stream_id: Optional[int] = None
    stream_name: Optional[str] = None
    stream_uuid: Optional[str] = None
    upload_image_uuid: Optional[str] = None # 新增：来自上传图片的UUID
    upload_image_filename: Optional[str] = None # 新增：来自上传图片的文件名

    # 包含 Individual 信息
    individual_id: Optional[int] = None
    individual_uuid: Optional[str] = None
    individual_name: Optional[str] = None
    individual_id_card: Optional[str] = None

    class Config:
        from_attributes = True

class PaginatedPersonsResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[Person]

class PaginatedImageSearchResultsResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[ImageSearchResult]

class ImageSearchGroupedResult(BaseModel):
    query_person_uuid: str
    query_crop_image_path: Optional[str] = None # 查询人物的裁剪图路径
    query_full_frame_image_path: Optional[str] = None # 查询人物的原始图片路径
    total_results_for_query_person: int # 为此查询人物找到的总结果数
    results: List[ImageSearchResult] # 此查询人物的搜索结果

class PaginatedGroupedImageSearchResultsResponse(BaseModel):
    total_overall_results: int # 包含所有查询人物的所有不重复结果的总数
    total_query_persons: int # 查询的人物总数
    items: List[ImageSearchGroupedResult]
    skip: int
    limit: int

# 新增 CorrectionLog 相关的 Pydantic 模型
class CorrectionLogBase(BaseModel):
    person_id: int
    original_feature_vector: Optional[str] = None # 改为可选
    corrected_feature_vector: Optional[str] = None
    correction_type: str
    details: Optional[str] = None
    # corrected_by_user_id: Optional[int] = None # 移除，将在create函数中获取
    # correction_date: Optional[datetime] = None # 移除，将在create函数中获取

class CorrectionLogCreate(BaseModel):
    # timestamp: datetime = Field(default_factory=lambda: datetime.now(pytz.timezone('Asia/Shanghai'))) # 移除，将在后端生成或从Pydantic模型中提取
    # logger: str = "human_in_the_loop" # 移除，后端固定
    # level: str = "INFO" # 移除，后端固定
    message: Optional[str] = None # 改为可选，因为详细信息存储在 details 中
    username: Optional[str] = None # 改为可选，在 crud 层从 user_id 获得
    ip_address: Optional[str] = None # 改为可选，在 crud 层从请求对象获得
    person_id: int # 被纠正的人物ID
    correction_type: str # 纠正类型，例如 'misdetection', 'relabel', 'merge', 'other'
    details: Optional[str] = None # 详细的纠正说明
    target_person_uuid: Optional[str] = Field(None, description="如果纠正类型是合并，则为此目标人物的 UUID") # 新增
    target_individual_id: Optional[int] = Field(None, description="如果纠正类型是合并，则为此目标 Individual 的 ID") # 新增
    corrected_by_user_id: Optional[int] = None # 新增，用于传递给 crud 层

class CorrectionLog(CorrectionLogBase):
    id: int

    class Config:
        from_attributes = True


class LogEntry(BaseModel):
    timestamp: datetime
    logger: str
    level: str
    message: str
    username: Optional[str] = None
    ip_address: Optional[str] = None

class LogResponse(BaseModel):
    total: int
    logs: List[LogEntry]

class UserStatusUpdate(BaseModel):
    is_active: bool

class VideoBase(BaseModel):
    filename: str
    file_path: str
    status: Optional[str] = "completed"
    progress: Optional[int] = 0
    is_live_stream: Optional[bool] = False

class Video(VideoBase):
    id: int
    uuid: str
    processed_at: Optional[datetime]
    owner_id: int

    class Config:
        from_attributes = True

class StreamBase(BaseModel):
    stream_url: str
    name: Optional[str] = None

class StreamCreate(StreamBase):
    pass

class StreamSchema(StreamBase):
    id: int
    status: str
    stream_uuid: Optional[str] = None
    created_at: datetime
    last_processed_at: Optional[datetime]
    owner_id: int
    output_video_path: Optional[str] = None
    is_active: bool # 新增字段

    class Config:
        from_attributes = True

class PaginatedStreamsResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[StreamSchema]

class StreamStartRequest(BaseModel):
    rtsp_url: Optional[str] = None
    api_stream_url: Optional[str] = None # 新增：通过API获取的视频流URL
    camera_id: Optional[str] = None
    stream_name: str

    @model_validator(mode='after')
    def check_at_most_one_url(self):
        if self.rtsp_url is not None and self.api_stream_url is not None:
            raise ValueError('只能提供rtsp_url或api_stream_url其中一个，不能同时提供。')
        if self.rtsp_url is None and self.api_stream_url is None:
            raise ValueError('必须提供rtsp_url或api_stream_url其中一个。')
        return self


class UserProfileUpdate(BaseModel):
    unit: Optional[str] = None
    phone_number: Optional[str] = None
    current_password: str # Required for verification

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, 
                              errors_messages={'too_short': '新密码长度不能少于8个字符！'},
                              description="新密码必须至少包含8个字符，且包含大小写字母、数字和特殊字符。")

    @validator('new_password')
    def new_password_strength(cls, v):
        if not any(char.isupper() for char in v):
            raise ValueError('新密码必须包含至少一个大写字母！')
        if not any(char.islower() for char in v):
            raise ValueError('新密码必须包含至少一个小写字母！')
        if not any(char.isdigit() for char in v):
            raise ValueError('新密码必须包含至少一个数字！')
        if not any(not char.isalnum() for char in v):
            raise ValueError('新密码必须包含至少一个特殊字符！')
        return v

# 新增 Image 相关的 Pydantic 模型
class ImageBase(BaseModel):
    uuid: str
    filename: str
    file_path: str
    person_count: int = 0

class ImageCreate(ImageBase):
    owner_id: int

class ImageResponse(ImageBase):
    id: int
    created_at: datetime
    uploader_username: Optional[str] = None # 新增：上传人用户名

    class Config:
        from_attributes = True

# 新增 SystemConfig 相关的 Pydantic 模型
class SystemConfigBase(BaseModel):
    key: str
    value: str

class SystemConfigCreate(SystemConfigBase):
    pass

class SystemConfig(SystemConfigBase):
    id: int
    last_updated: datetime

    class Config:
        from_attributes = True

# 新增 TaskStatusResponse 模型
class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[int] = None # 任务进度 (0-100)
    message: Optional[str] = None # 任务状态信息或错误信息
    result: Optional[Any] = None # 任务结果 (成功时返回的数据)

# 新增 ModelConfig 模型，用于表示所有可配置的模型参数
class ModelConfig(BaseModel):
    # 模型路径 (只暴露文件名，实际路径在后端拼接)
    DETECTION_MODEL_FILENAME: Optional[str] = None
    REID_MODEL_FILENAME: Optional[str] = None
    ACTIVE_REID_MODEL_PATH: Optional[str] = None # 这个需要特殊处理，因为在 config 中是完整路径
    POSE_MODEL_FILENAME: Optional[str] = None
    FACE_DETECTION_MODEL_FILENAME: Optional[str] = None
    FACE_RECOGNITION_MODEL_FILENAME: Optional[str] = None
    GAIT_RECOGNITION_MODEL_FILENAME: Optional[str] = None
    CLOTHING_ATTRIBUTE_MODEL_FILENAME: Optional[str] = None

    # 模型特征维度配置
    REID_INPUT_WIDTH: Optional[int] = None
    REID_INPUT_HEIGHT: Optional[int] = None
    FEATURE_DIM: Optional[int] = None
    FACE_FEATURE_DIM: Optional[int] = None
    GAIT_FEATURE_DIM: Optional[int] = None

    # 新增：模型运行设备类型
    DEVICE_TYPE: Optional[str] = None

    # 多模态融合权重
    REID_WEIGHT: Optional[float] = None
    FACE_WEIGHT: Optional[float] = None
    GAIT_WEIGHT: Optional[float] = None

    # Re-Ranking 算法参数
    K1: Optional[int] = None
    K2: Optional[int] = None
    LAMBDA_VALUE: Optional[float] = None

    # 步态序列长度
    GAIT_SEQUENCE_LENGTH: Optional[int] = None

    # 人机回环配置
    HUMAN_REVIEW_CONFIDENCE_THRESHOLD: Optional[float] = None

    # 图片分析设置
    IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE: Optional[float] = None

    # 注册设置
    ENROLLMENT_MIN_PERSON_CONFIDENCE: Optional[float] = None

    # 人脸检测置信度阈值和最小尺寸
    FACE_DETECTION_CONFIDENCE_THRESHOLD: Optional[float] = None
    MIN_FACE_WIDTH: Optional[int] = None
    MIN_FACE_HEIGHT: Optional[int] = None

    # Tracker 和检测相关配置
    TRACKER_PROXIMITY_THRESH: Optional[float] = None
    TRACKER_APPEARANCE_THRESH: Optional[float] = None

    # BoT-SORT 跟踪器阈值配置
    TRACKER_HIGH_THRESH: Optional[float] = None
    TRACKER_LOW_THRESH: Optional[float] = None
    TRACKER_NEW_TRACK_THRESH: Optional[float] = None
    TRACKER_MIN_HITS: Optional[int] = None
    TRACKER_TRACK_BUFFER: Optional[int] = None

    # 视频和流处理帧率
    VIDEO_PROCESSING_FRAME_RATE: Optional[int] = None
    STREAM_PROCESSING_FRAME_RATE: Optional[int] = None
    VIDEO_COMMIT_BATCH_SIZE: Optional[int] = None

    # 检测置信度阈值和人物类别ID
    DETECTION_CONFIDENCE_THRESHOLD: Optional[float] = None
    PERSON_CLASS_ID: Optional[int] = None

    # Excel 导出配置
    EXCEL_EXPORT_MAX_IMAGES: Optional[int] = None
    EXCEL_EXPORT_IMAGE_SIZE_PX: Optional[int] = None
    EXCEL_EXPORT_ROW_HEIGHT_PT: Optional[int] = None

    # MJPEG 视频流帧率
    MJPEG_STREAM_FPS: Optional[int] = None

    # Faiss 配置
    FAISS_METRIC: Optional[str] = None
    FAISS_SEARCH_K: Optional[int] = None

    # Re-ID 训练参数
    REID_TRAIN_BATCH_SIZE: Optional[int] = None
    REID_TRAIN_LEARNING_RATE: Optional[float] = None

    # 实时比对配置
    REALTIME_COMPARISON_THRESHOLD: Optional[float] = None
    REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS: Optional[int] = None

    class Config:
        from_attributes = True

# 新增 FollowedPerson 相关的 Pydantic 模型
class FollowedPersonBase(BaseModel):
    individual_id: int

class FollowedPersonCreate(FollowedPersonBase):
    pass

class FollowedPersonResponse(FollowedPersonBase):
    id: int
    user_id: int
    follow_time: datetime
    unfollow_time: Optional[datetime] = None
    
    # 添加关联 Individual 和 User 的信息，方便前端展示
    individual: Optional[Individual] = None
    user: Optional[User] = None

    class Config:
        from_attributes = True

class PaginatedFollowedPersonsResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[FollowedPersonResponse]

class FollowedPersonToggleRequest(BaseModel):
    individual_id: int
    is_followed: bool
    perform_global_search: Optional[bool] = False # 新增字段

class IndividualRealtimeComparisonToggleRequest(BaseModel):
    individual_id: int
    is_enabled: bool

class MessageResponse(BaseModel):
    message: str

class PersonSearchRequest(BaseModel):
    query_image_path: Optional[str] = Field(None, description="查询图片的临时路径")
    query_person_uuid: Optional[List[str]] = Field(None, description="查询人物的 UUID 列表，如果已知人物ID进行搜索") # 修改为列表
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="相似度阈值 (0.0 - 1.0)")
    skip: int = 0
    limit: int = 20

    @validator('query_image_path') # Re-enabled validator
    def check_query_source(cls, v, values):
        if v is None and values.get('query_person_uuid') is None:
            raise ValueError('必须提供 query_image_path 或 query_person_uuid')
        return v

class RealtimeMatchAlert(BaseModel):
    person_uuid: str = Field(..., description="被检测到的人物 UUID")
    matched_individual_id: int = Field(..., description="匹配到的关注人员（逻辑人物）ID")
    matched_individual_uuid: str = Field(..., description="匹配到的关注人员（逻辑人物）UUID")
    matched_individual_name: Optional[str] = Field(None, description="匹配到的关注人员姓名")
    similarity_score: float = Field(..., description="比对的相似度分数")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(pytz.utc), description="预警发生的时间")
    alert_type: str = Field("realtime_followed_person_match", description="预警类型")
    source_media_type: str = Field(..., description="来源媒体类型 (image, video, stream)")
    source_media_uuid: str = Field(..., description="来源媒体的 UUID")
    user_id: int = Field(..., description="触发比对的用户 ID 或流所有者 ID")
    cropped_image_path: str = Field(..., description="预警人物裁剪图片路径") # 新增字段
    full_frame_image_path: str = Field(..., description="预警原始图片（全帧）路径") # 新增字段

    class Config:
        from_attributes = True

class EnrollmentImageResponse(BaseModel):
    image_path: str = Field(..., description="注册图片的相对路径")
    uuid: str = Field(..., description="人物 (Person) 的 UUID") # 明确这个 UUID 指的是 Person 的
    image_db_uuid: Optional[str] = Field(None, description="图片 (Image) 在数据库中的 UUID") # 新增
    filename: str = Field(..., description="注册图片的文件名")

class PaginatedEnrollmentImagesResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[EnrollmentImageResponse]


# 新增 GlobalSearchResult 相关的 Pydantic 模型
class GlobalSearchResultBase(BaseModel):
    individual_id: int
    matched_person_uuid: str
    matched_person_id: int
    matched_image_path: str
    confidence: float
    search_time: datetime
    user_id: int
    is_initial_search: bool

class GlobalSearchResultCreate(GlobalSearchResultBase):
    pass

class GlobalSearchResultResponse(GlobalSearchResultBase):
    id: int
    # 嵌套关联模型
    individual: Optional[Individual] = None
    person: Optional[Person] = None # 匹配到的人物 Person 的详细信息
    user: Optional[User] = None

    class Config:
        from_attributes = True

class PaginatedGlobalSearchResultsResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[GlobalSearchResultResponse]

# 新增 Alert 相关的 Pydantic 模型
class Alert(BaseModel):
    id: int
    individual_id: int
    person_id: int
    person_uuid: str = Field(..., description="被检测到的人物 UUID") # 新增：人物 UUID
    person_created_at: datetime = Field(..., description="被检测到的人物创建时间") # 新增：人物创建时间
    timestamp: datetime # 预警时间
    source_media_uuid: str = Field(..., description="来源媒体的 UUID") # 新增：来源媒体 UUID
    source_media_type: str = Field(..., description="来源媒体类型 (image, video, stream)") # 新增：来源媒体类型
    cropped_image_path: str = Field(..., description="预警人物裁剪图片路径")
    full_frame_image_path: str = Field(..., description="预警原始图片（全帧）路径")
    similarity_score: float = Field(..., description="比对的相似度分数") # 新增：相似度分数

    # 移除 location 和 description
    # location: Optional[str] = None
    # description: Optional[str] = None

    class Config:
        from_attributes = True

class PaginatedAlertsResponse(BaseModel):
    total: int
    skip: int
    limit: int
    items: List[Alert]