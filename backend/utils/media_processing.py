import os
import cv2
import numpy as np
import json
import logging
from typing import Optional, List, Any, Dict, Tuple
import uuid
from datetime import datetime

from backend.config import settings
from backend import schemas
from backend.ml_services import ml_logic

logger = logging.getLogger(__name__)

def generate_media_paths(
    media_type: str,
    media_uuid: str,
    person_uuid: str,
    model_name: str, # 新增参数：模型名称
    original_filename: Optional[str] = None
) -> Dict[str, str]:
    """
    根据媒体类型、UUID 和人物 UUID 生成存储路径。
    返回一个字典，包含裁剪图、完整帧、人脸、步态的绝对路径和相对路径。
    """
    paths = {}

    # 使用 settings 中新的辅助函数构建路径
    # 裁剪图路径
    crop_image_path_abs = settings.get_parsed_image_path(
        base_dir=settings.DATABASE_CROPS_DIR,
        model_name=model_name,
        analysis_type=media_type,
        uuid=media_uuid # 对于裁剪图，使用 media_uuid 作为最终目录
    )
    crop_filename = f"{person_uuid}.jpg" # 裁剪图文件名，可以是任意唯一标识
    final_crop_image_path_abs = os.path.join(crop_image_path_abs, crop_filename)
    crop_image_path_rel = settings.get_parsed_image_relative_path(
        base_dir_name="crops",
        model_name=model_name,
        analysis_type=media_type,
        uuid=media_uuid # 对于裁剪图，使用 media_uuid 作为最终目录
    )
    final_crop_image_path_rel = os.path.join(crop_image_path_rel, crop_filename).replace(os.sep, '/')

    paths.update({
        "crop_image_path_abs": final_crop_image_path_abs,
        "crop_image_path_rel": final_crop_image_path_rel,
    })

    # 完整帧路径 (如果需要)
    full_frame_image_path_abs = None
    full_frame_image_path_rel = None
    full_frame_filename = None

    if media_type in ["image", "video", "stream"]:
        # 对于全帧图片，我们可能希望使用 media_uuid 作为最终目录，而不是 person_uuid
        # 因为一个 media_uuid 对应一个原始图片/视频/流，其中可能包含多个人物
        full_frame_abs_dir = settings.get_parsed_image_path(
            base_dir=settings.DATABASE_FULL_FRAMES_DIR,
            model_name=model_name, # 全帧图也按模型分类
            analysis_type=media_type,
            uuid=media_uuid # 对于全帧图，使用 media_uuid 作为最终目录
        )
        # 如果是图片分析，使用原始文件名；否则，对于视频或流，使用 person_uuid 确保唯一性
        if media_type == "image":
            full_frame_filename = original_filename if original_filename is not None else f"{person_uuid}_full_frame.jpg"
        else:
            full_frame_filename = f"{person_uuid}_full_frame.jpg"
        full_frame_image_path_abs = os.path.join(full_frame_abs_dir, full_frame_filename)
        
        full_frame_image_path_rel = settings.get_parsed_image_relative_path(
            base_dir_name="full_frames",
            model_name=model_name, # 全帧图也按模型分类
            analysis_type=media_type,
            uuid=media_uuid # 对于全帧图，使用 media_uuid 作为最终目录
        )
        full_frame_image_path_rel = os.path.join(full_frame_image_path_rel, full_frame_filename).replace(os.sep, '/')

    paths.update({
        "full_frame_image_path_abs": full_frame_image_path_abs,
        "full_frame_image_path_rel": full_frame_image_path_rel,
        "full_frame_filename": full_frame_filename
    })

    # 人脸裁剪图路径 (同样按模型分类)
    face_abs_dir = settings.get_parsed_image_path(
        base_dir=settings.DATABASE_CROPS_DIR,
        model_name="face_recognition", # 人脸识别模型
        analysis_type=media_type,
        uuid=media_uuid # 人脸裁剪图也用 media_uuid 作为最终目录
    )
    # 人脸裁剪图文件名：使用唯一的人脸 ID
    face_filename = f"{str(uuid.uuid4())}.jpg"
    final_face_image_path_abs = os.path.join(face_abs_dir, face_filename)
    face_image_path_rel = settings.get_parsed_image_relative_path(
        base_dir_name="crops", # 人脸图也存放在 crops 目录下
        model_name="face_recognition",
        analysis_type=media_type,
        uuid=media_uuid
    )
    final_face_image_path_rel = os.path.join(face_image_path_rel, face_filename).replace(os.sep, '/')

    paths.update({
        "face_image_path_abs": final_face_image_path_abs,
        "face_image_path_rel": final_face_image_path_rel,
    })

    # 步态裁剪图路径 (同样按模型分类)
    gait_abs_dir = settings.get_parsed_image_path(
        base_dir=settings.DATABASE_CROPS_DIR,
        model_name="gait_recognition", # 步态识别模型
        analysis_type=media_type,
        uuid=media_uuid # 步态裁剪图也用 media_uuid 作为最终目录
    )
    # 步态裁剪图文件名：使用唯一的步态 ID
    gait_filename = f"{str(uuid.uuid4())}.jpg"
    final_gait_image_path_abs = os.path.join(gait_abs_dir, gait_filename)
    gait_image_path_rel = settings.get_parsed_image_relative_path(
        base_dir_name="crops", # 步态图也存放在 crops 目录下
        model_name="gait_recognition",
        analysis_type=media_type,
        uuid=media_uuid
    )
    final_gait_image_path_rel = os.path.join(gait_image_path_rel, gait_filename).replace(os.sep, '/')

    paths.update({
        "gait_image_path_abs": final_gait_image_path_abs,
        "gait_image_path_rel": final_gait_image_path_rel,
    })

    return paths


from sqlalchemy.orm import Session

def process_detected_person_data(
    db: Session,
    frame: np.ndarray,
    person_bbox: Tuple[int, int, int, int],
    confidence_score: float,
    yolo_results_obj: Any,
    media_type: str,
    media_uuid: str,
    person_uuid: str,
    video_id_int: Optional[int] = None,
    image_id_int: Optional[int] = None, # 新增：图片 ID
    stream_id_int: Optional[int] = None,
    face_detection_model: Optional[Any] = None,
    face_recognition_session: Optional[Any] = None,
    individual_id: Optional[int] = None,
    gait_recognition_session: Optional[Any] = None,
    clothing_attribute_session: Optional[Any] = None, # 新增：衣着属性模型会话
    pose_estimation_model: Optional[Any] = None, # 新增：姿态估计模型
    tracklet_gait_buffer: Optional[Dict[int, List[np.ndarray]]] = None,
    track_id: Optional[int] = None,
    full_frame_image_path_override: Optional[str] = None, # NEW PARAMETER
    original_filename: Optional[str] = None, # NEW PARAMETER
    model_name: Optional[str] = None, # 新增参数
    is_enrollment_image: bool = False # 新增：标识是否为主动注册图片
) -> Optional[schemas.PersonCreate]:
    """
    处理单个检测到的人物数据，提取特征，保存裁剪图，并生成 PersonCreate 模式。
    （已重构：统一使用 generate_media_paths 进行路径管理）
    """
    x1, y1, x2, y2 = person_bbox
    person_crop = frame[y1:y2, x1:x2]
    if person_crop.size == 0:
        logger.warning(f"人物边界框 {person_bbox} 无效或裁剪后图像为空，跳过处理。")
        return None

    # 4. 动态获取模型名称并生成媒体文件路径
    # 从配置中获取图片分析所使用的模型名称
    # model_name = settings.IMAGE_ANALYSIS_MODEL_NAME # 移除这行，现在从参数获取
    effective_model_name = model_name if model_name else settings.IMAGE_ANALYSIS_MODEL_NAME # 使用传入的 model_name，如果为 None 则回退到默认值
    logger.debug(f"使用模型 '{effective_model_name}' 生成路径。")

    paths = generate_media_paths(
        media_type=media_type,
        media_uuid=media_uuid,
        person_uuid=person_uuid,
        model_name=effective_model_name, # 使用 effective_model_name
        original_filename=original_filename # Pass original_filename to generate_media_paths
    )
    crop_image_path_abs = paths["crop_image_path_abs"]
    crop_image_path_rel = paths["crop_image_path_rel"]

    # 确定最终的全帧图片相对路径
    final_full_frame_image_path_rel = full_frame_image_path_override # 优先使用 override
    
    if final_full_frame_image_path_rel is None: # 如果没有 override，则使用 generate_media_paths 生成的路径
        full_frame_image_path_abs = paths.get("full_frame_image_path_abs")
        full_frame_image_path_rel = paths.get("full_frame_image_path_rel")

        # 2. 保存 Re-ID 裁剪图和全帧图
        os.makedirs(os.path.dirname(crop_image_path_abs), exist_ok=True)
        logger.debug(f"尝试保存人物裁剪图到: {crop_image_path_abs}")
        save_crop_success = cv2.imwrite(crop_image_path_abs, person_crop)
        if save_crop_success:
            logger.debug(f"人物裁剪图 (Re-ID) 已保存到: {crop_image_path_abs}")
        else:
            logger.error(f"ERROR: 无法保存人物裁剪图到: {crop_image_path_abs}。cv2.imwrite 返回 False。")

        # 如果没有提供 full_frame_image_path_override，则保存原始完整帧
        if not full_frame_image_path_override:
            os.makedirs(os.path.dirname(full_frame_image_path_abs), exist_ok=True)
            logger.debug(f"尝试保存完整帧图片到: {full_frame_image_path_abs}")
            save_full_frame_success = cv2.imwrite(full_frame_image_path_abs, frame) # <--- 添加这一行来保存原始帧
            if save_full_frame_success:
                logger.debug(f"完整帧图片已保存到: {full_frame_image_path_abs}")
            else:
                logger.error(f"ERROR: 无法保存完整帧图片到: {full_frame_image_path_abs}。cv2.imwrite 返回 False。")
        elif media_type == "image":
            logger.debug(f"图片分析：全帧图片已在上传时保存，无需重复保存。")
        
        final_full_frame_image_path_rel = full_frame_image_path_rel
    else:
        # 如果提供了 override，仅保存裁剪图，不保存全帧图
        os.makedirs(os.path.dirname(crop_image_path_abs), exist_ok=True)
        logger.debug(f"尝试保存人物裁剪图到: {crop_image_path_abs}")
        save_crop_success = cv2.imwrite(crop_image_path_abs, person_crop)
        if save_crop_success:
            logger.debug(f"人物裁剪图 (Re-ID) 已保存到: {crop_image_path_abs}")
        else:
            logger.error(f"ERROR: 无法保存人物裁剪图到: {crop_image_path_abs}。cv2.imwrite 返回 False。")
        logger.debug(f"全帧图片路径通过 override 提供: {final_full_frame_image_path_rel}，跳过保存。")


    # 3. 提取 Re-ID 特征
    reid_session = ml_logic.get_reid_session(db)
    feature_vector_np = ml_logic.get_person_feature(reid_session, person_crop)
    feature_vector_json = json.dumps(feature_vector_np.tolist())
    logger.debug("Re-ID 特征向量已提取。")

    # 4. 提取姿态关键点
    pose_keypoints_data = None
    if pose_estimation_model: # 只有当姿态估计模型加载成功时才执行
        # 在整个帧上运行姿态估计
        pose_results = pose_estimation_model(frame, verbose=False)
        if pose_results and len(pose_results) > 0: # 确保有结果
            for r_pose in pose_results:
                if r_pose.boxes is not None and len(r_pose.boxes) > 0:
                    for i, box in enumerate(r_pose.boxes): # 遍历所有检测到的姿态框
                        x1_p, y1_p, x2_p, y2_p = map(int, box.xyxy[0])
                        # 检查姿态框与人物边界框是否有足够的重叠 (例如，通过 IoU 或中心点)
                        # 这里我们简化，直接检查是否在人物框内或有足够重叠
                        # 更严谨的应该是计算 IoU，但 YOLOv8 的跟踪结果通常匹配很好
                        if (x1_p < x2 and x2_p > x1 and y1_p < y2 and y2_p > y1): # 简单判断是否有重叠
                            if r_pose.keypoints is not None and i < len(r_pose.keypoints):
                                keypoints = r_pose.keypoints[i]
                                pose_keypoints_data = json.dumps(keypoints.xy.cpu().numpy().tolist())
                                logger.debug("姿态关键点已提取。")
                                break # 找到与当前人物框匹配的姿态关键点后退出
                    if pose_keypoints_data: # 如果已经找到了姿态关键点，就跳出外层循环
                        break

    # 5. 提取人脸特征
    face_image_path = None
    face_feature_vector_json = None
    face_id = None
    if face_detection_model and face_recognition_session:
        # 使用人脸检测模型在人物裁剪图范围内检测人脸
        # 首先，提取人物的裁剪图，然后在其上运行人脸检测
        x1, y1, x2, y2 = person_bbox
        person_region = frame[y1:y2, x1:x2]

        if person_region.size > 0:
            face_detection_results = face_detection_model(person_region, verbose=False)
            detected_faces = []
            for r_face in face_detection_results:
                if r_face.boxes is not None and len(r_face.boxes) > 0:
                    for face_box in r_face.boxes:
                        fx1, fy1, fx2, fy2 = map(int, face_box.xyxy[0])
                        face_conf = face_box.conf[0]
                        
                        # 确保人脸边界框有效 (宽度和高度都大于0)
                        if fx2 > fx1 and fy2 > fy1:
                            # 新增：检查人脸尺寸是否大于最小阈值
                            if (fx2 - fx1) >= settings.MIN_FACE_WIDTH and (fy2 - fy1) >= settings.MIN_FACE_HEIGHT:
                                # 确保人脸置信度高于一个合理阈值，例如 0.5
                                if face_conf > settings.FACE_DETECTION_CONFIDENCE_THRESHOLD:
                                    # 裁剪人脸图像 (相对于人物裁剪图的坐标)
                                    face_crop_img = person_region[fy1:fy2, fx1:fx2]
                                    if face_crop_img.size > 0 and face_crop_img.shape[0] > 0 and face_crop_img.shape[1] > 0:
                                        detected_faces.append({"face_crop_img": face_crop_img, "confidence": face_conf})
                                    else:
                                        logger.warning(f"人脸裁剪图为空或尺寸无效，跳过此人脸。bbox: {(fx1, fy1, fx2, fy2)}, 裁剪尺寸: {face_crop_img.shape if face_crop_img.size > 0 else '空'}")
                                else:
                                    logger.debug(f"人脸检测置信度 {face_conf:.2f} 低于阈值 {settings.FACE_DETECTION_CONFIDENCE_THRESHOLD:.2f}，跳过。")
                            else:
                                logger.debug(f"人脸边界框尺寸过小，跳过此人脸。bbox: {(fx1, fy1, fx2, fy2)}, 最小尺寸要求: ({settings.MIN_FACE_WIDTH}, {settings.MIN_FACE_HEIGHT})")
                        else:
                            logger.warning(f"检测到无效人脸边界框 (零宽度或零高度)，跳过此人脸。bbox: {(fx1, fy1, fx2, fy2)}")
                
                if detected_faces:
                    # 选择置信度最高的人脸进行识别（如果有多张脸）
                    best_face = max(detected_faces, key=lambda x: x["confidence"])
                    face_crop_img = best_face["face_crop_img"]

                    face_feature_np = ml_logic.get_face_feature(
                        face_recognition_session=face_recognition_session,
                        face_crop=face_crop_img
                    )
                    if face_feature_np is not None:
                        face_feature_vector_json = json.dumps(face_feature_np.tolist())
                        logger.debug("人脸特征向量已提取。")

                        face_paths = generate_media_paths(media_type, media_uuid, person_uuid, effective_model_name) # 使用动态模型名称
                        face_save_path_abs = face_paths.get("face_image_path_abs") # 使用 generate_media_paths 生成的绝对路径
                        face_image_path = face_paths.get("face_image_path_rel") # 使用 generate_media_paths 生成的相对路径

                        os.makedirs(os.path.dirname(face_save_path_abs), exist_ok=True)
                        # 增加日志：打印即将保存的人脸裁剪图尺寸
                        logger.info(f"DEBUG: 尝试保存人脸图片，路径: {face_save_path_abs}, 尺寸: {face_crop_img.shape}")
                        # 尝试保存人脸裁剪图
                        save_success = cv2.imwrite(face_save_path_abs, face_crop_img)
                        if save_success:
                            logger.debug(f"人脸图片已保存到: {face_save_path_abs}")
                        else:
                            logger.error(f"ERROR: 无法保存人脸图片到: {face_save_path_abs}。cv2.imwrite 返回 False。")
                    else:
                        logger.warning("人脸特征提取失败或为空。")
                else:
                    logger.info("在人物区域内未检测到人脸或置信度过低。")
        else:
            logger.warning("人物区域为空，无法进行人脸检测。")

    # 6. 提取衣着属性
    clothing_attributes_data = [] # Placeholder for clothing attributes
    clothing_attributes_json = None
    if clothing_attribute_session: # 只有当衣着属性模型加载成功时才执行
        try:
            clothing_attributes_data = ml_logic.get_clothing_attributes(clothing_attribute_session, frame, person_bbox) # 传递会话和帧/bbox
            if clothing_attributes_data:
                clothing_attributes_json = json.dumps(clothing_attributes_data)
                logger.debug("衣着属性已提取。")
            else:
                logger.warning("衣着属性提取功能尚未实现或返回空数据。")
        except Exception as e:
            logger.error(f"提取衣着属性失败: {e}", exc_info=True)
            logger.warning("衣着属性提取功能尚未实现，返回空数据。") # 保持原有的警告信息
    else:
        logger.warning("衣着属性模型未加载，跳过衣着属性提取。")

    # 7. 提取步态特征 (仅限流)
    gait_feature_vector_json = None
    gait_image_path = None # 步态图片的相对路径
    if media_type in ["stream"] and gait_recognition_session and tracklet_gait_buffer is not None and track_id is not None:
        if track_id not in tracklet_gait_buffer:
            tracklet_gait_buffer[track_id] = []
        original_person_crop_for_gait = frame[y1:y2, x1:x2]
        if original_person_crop_for_gait.size > 0:
            tracklet_gait_buffer[track_id].append(original_person_crop_for_gait)
            if len(tracklet_gait_buffer[track_id]) > settings.GAIT_SEQUENCE_LENGTH:
                tracklet_gait_buffer[track_id].pop(0)

            if len(tracklet_gait_buffer[track_id]) == settings.GAIT_SEQUENCE_LENGTH:
                gait_feature_np = ml_logic.get_gait_feature(gait_recognition_session, tracklet_gait_buffer[track_id])
                if gait_feature_np is not None:
                    gait_feature_vector_json = json.dumps(gait_feature_np.tolist())
                    gait_paths = generate_media_paths(media_type, media_uuid, person_uuid, effective_model_name) # 使用动态模型名称
                    gait_save_path_abs = paths.get("gait_image_path_abs") # 使用 generate_media_paths 生成的绝对路径
                    gait_image_path = paths.get("gait_image_path_rel") # 使用 generate_media_paths 生成的相对路径

                    os.makedirs(os.path.dirname(gait_save_path_abs), exist_ok=True)
                    cv2.imwrite(gait_save_path_abs, tracklet_gait_buffer[track_id][-1])
                    logger.debug(f"步态图片已保存到: {gait_save_path_abs}")
                else:
                    logger.warning(f"步态特征提取失败或为空。")
    elif media_type not in ["stream"]:
        logger.debug(f"媒体类型 {media_type}，跳过步态特征提取和保存。")

    person_create_data = schemas.PersonCreate(
        uuid=person_uuid, 
        feature_vector=feature_vector_json,
        crop_image_path=crop_image_path_rel,
        full_frame_image_path=final_full_frame_image_path_rel, # 使用确定的全帧图片路径
        confidence_score=float(confidence_score), 
        pose_keypoints=pose_keypoints_data, 
        face_image_path=face_image_path, 
        face_feature_vector=face_feature_vector_json, 
        face_id=face_id, 
        clothing_attributes=clothing_attributes_json,
        gait_feature_vector=gait_feature_vector_json, 
        gait_image_path=gait_image_path,
        image_id=image_id_int if media_type == "image" else None, # 使用 image_id_int
        video_id=video_id_int if media_type == "video" else None,
        stream_id=stream_id_int if media_type == "stream" else None,
        individual_id=individual_id,
        is_verified=True if media_type == "image" else False, 
        verified_by_user_id=None,
        verification_date=datetime.now() if media_type == "image" else None, 
        is_enrollment_image=is_enrollment_image # 新增：传递 is_enrollment_image
    )
    logger.debug(f"已创建 PersonCreate 对象，UUID: {person_create_data.uuid}")
    return person_create_data 