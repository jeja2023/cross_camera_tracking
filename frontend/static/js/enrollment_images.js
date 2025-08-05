import { Auth } from './auth.js';
import { API_BASE_URL, showNotification, initializeAuthenticatedPage, fetchUserProfile, showLightboxWithNav, setupImageGalleryClickEvents, hideLightbox } from './common.js';

// 移除模态框相关的DOM元素引用和全局变量
// let currentImages = [];
// let currentImageIndex = 0;

// const imageModal = document.getElementById('imageModal');
// const closeButton = imageModal.querySelector('.close-button');
// const modalTitle = document.getElementById('modalTitle');
// const fullFrameImage = document.getElementById('fullFrameImage');
// const prevImageBtn = document.getElementById('prevImageBtn');
// const nextImageBtn = document.getElementById('nextImageBtn');
// const imageCounter = document.getElementById('imageCounter');

document.addEventListener('DOMContentLoaded', async () => {
    initializeAuthenticatedPage();
    await fetchUserProfile();

    const urlParams = new URLSearchParams(window.location.search);
    const individualId = urlParams.get('individual_id');
    const personName = urlParams.get('person_name');

    if (!individualId) {
        showNotification('缺少人物ID参数。', 'error');
        document.getElementById('noImagesMessage').style.display = 'block';
        return;
    }

    const personNameDisplay = document.getElementById('personNameDisplay');
    if (personName) {
        personNameDisplay.textContent = personName;
    } else {
        personNameDisplay.textContent = individualId; // 如果没有姓名，显示ID
    }

    // 移除模态框事件监听
    // closeButton.addEventListener('click', () => {
    //     imageModal.style.display = 'none';
    //     fullFrameImage.src = ''; // 清空图片
    //     currentImages = []; // 清空图片列表
    //     currentImageIndex = 0; // 重置索引
    // });

    // prevImageBtn.addEventListener('click', () => navigateModalImage(-1));
    // nextImageBtn.addEventListener('click', () => navigateModalImage(1));

    await fetchEnrollmentImages(individualId);
});

async function fetchEnrollmentImages(individualId) {
    const token = Auth.getToken();
    if (!token) {
        showNotification('未认证或会话已过期，请重新登录。', 'error');
        window.location.href = '/login';
        return;
    }

    try {
        const endpoint = `${API_BASE_URL}/followed_persons/${individualId}/enrollments`;
        const response = await fetch(endpoint, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            if (response.status === 403) {
                showNotification('无权限查看该人员的图片。', 'error');
            } else {
                throw new Error(`HTTP 错误！状态码: ${response.status}`);
            }
            document.getElementById('noImagesMessage').style.display = 'block';
            return;
        }

        const images = await response.json();
        // currentImages = images; // 不再需要全局 currentImages，Lightbox组件会处理
        renderImages(images); // 渲染缩略图

    } catch (error) {
        console.error('获取注册图片失败:', error);
        showNotification(`获取注册图片失败: ${error.message}`, 'error');
        document.getElementById('noImagesMessage').style.display = 'block';
    }
}

function renderImages(images) {
    const imageGallery = document.getElementById('imageGallery');
    imageGallery.innerHTML = ''; // 清空现有内容

    if (images.length === 0) {
        document.getElementById('noImagesMessage').style.display = 'block';
        return;
    }

    document.getElementById('noImagesMessage').style.display = 'none';

    images.forEach((img, index) => {
        const imgContainer = document.createElement('div');
        imgContainer.classList.add('image-item');

        const imgElement = document.createElement('img');
        imgElement.src = img.crop_image_path.startsWith('http') ? img.crop_image_path : `${API_BASE_URL}/${img.crop_image_path}`;
        imgElement.alt = `注册图片 ${img.uuid}`;
        // 为全帧图片路径设置data属性，供Lightbox使用
        imgElement.dataset.fullSize = img.full_frame_image_path.startsWith('http') ? img.full_frame_image_path : `${API_BASE_URL}/${img.full_frame_image_path}`;

        imgElement.title = `UUID: ${img.uuid}\n来源: ${img.video_name || img.stream_name || img.upload_image_filename || '未知来源'}\n时间: ${new Date(img.created_at).toLocaleString()}`;
        // imgElement.dataset.index = index; // 不再需要，由 setupImageGalleryClickEvents 处理
        // imgElement.addEventListener('click', (event) => openImageModal(parseInt(event.target.dataset.index))); // 移除自定义点击事件

        const imgInfo = document.createElement('div');
        imgInfo.classList.add('image-info');
        imgInfo.innerHTML = `
            <p><strong>UUID:</strong> ${img.uuid}</p>
            <p><strong>来源:</strong> ${img.video_name || img.stream_name || img.upload_image_filename || '未知来源'}</p>
            <p><strong>时间:</strong> ${new Date(img.created_at).toLocaleString()}</p>
        `;

        imgContainer.appendChild(imgElement);
        imgContainer.appendChild(imgInfo);
        imageGallery.appendChild(imgContainer);
    });

    // 在所有图片渲染完成后，设置点击事件，使其能通过Lightbox进行导航
    setupImageGalleryClickEvents('#imageGallery', '.image-item img', (img) => img.dataset.fullSize, true); // dynamicContent 设置为 true
}

// 移除旧的模态框相关函数
// function openImageModal(index) {
//     currentImageIndex = index;
//     modalTitle.textContent = `原始图片 (${currentImages[currentImageIndex].individual_name || currentImages[currentImageIndex].individual_id_card || currentImages[currentImageIndex].individual_id})`;
//     renderModalImage();
//     imageModal.style.display = 'flex';
// }

// function renderModalImage() {
//     const image = currentImages[currentImageIndex];
//     fullFrameImage.src = image.full_frame_image_path.startsWith('http') ? image.full_frame_image_path : `${API_BASE_URL}/${image.full_frame_image_path}`;
//     fullFrameImage.alt = `原始图片 ${image.uuid}`;
//     fullFrameImage.title = `UUID: ${image.uuid}\n来源: ${image.video_name || image.stream_name || image.upload_image_filename || '未知来源'}\n时间: ${new Date(image.created_at).toLocaleString()}`;
//     imageCounter.textContent = `${currentImageIndex + 1} / ${currentImages.length}`;

//     prevImageBtn.disabled = currentImageIndex === 0;
//     nextImageBtn.disabled = currentImageIndex === currentImages.length - 1;
// }

// function navigateModalImage(direction) {
//     currentImageIndex += direction;
//     if (currentImageIndex < 0) {
//         currentImageIndex = 0;
//     } else if (currentImageIndex >= currentImages.length) {
//         currentImageIndex = currentImages.length - 1;
//     }
//     renderModalImage();
// }