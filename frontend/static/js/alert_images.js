import { Auth } from './auth.js';
import { API_BASE_URL, showNotification, initializeAuthenticatedPage, fetchUserProfile } from './common.js';

let individualId = null;
let personName = null;

document.addEventListener('DOMContentLoaded', async () => {
    initializeAuthenticatedPage();
    await fetchUserProfile();

    const urlParams = new URLSearchParams(window.location.search);
    individualId = urlParams.get('individual_id');
    personName = urlParams.get('person_name');
    let minScore = parseFloat(urlParams.get('min_score')) || 90.0; // 从URL获取最低分值，默认为90

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

    const minScoreInput = document.getElementById('minScore');
    minScoreInput.value = minScore;

    document.getElementById('applyFilterBtn').addEventListener('click', () => {
        const newMinScore = parseFloat(minScoreInput.value);
        if (!isNaN(newMinScore) && newMinScore >= 0 && newMinScore <= 100) {
            minScore = newMinScore;
            fetchAlertImages(individualId, minScore);
        } else {
            showNotification('请输入有效的比对分值 (0-100)。', 'warning');
        }
    });

    await fetchAlertImages(individualId, minScore);
});

async function fetchAlertImages(individualId, minScore) {
    const token = Auth.getToken();
    if (!token) {
        showNotification('未认证或会话已过期，请重新登录。', 'error');
        window.location.href = '/login';
        return;
    }

    try {
        const endpoint = `${API_BASE_URL}/followed_persons/${individualId}/alerts?min_score=${minScore}`;
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
        renderImages(images);

    } catch (error) {
        console.error('获取预警图片失败:', error);
        showNotification(`获取预警图片失败: ${error.message}`, 'error');
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

    images.forEach(img => {
        const imgContainer = document.createElement('div');
        imgContainer.classList.add('image-item');

        const imgElement = document.createElement('img');
        imgElement.src = img.crop_image_path.startsWith('http') ? img.crop_image_path : `${API_BASE_URL}/${img.crop_image_path}`;
        imgElement.alt = `预警图片 ${img.uuid}`;
        imgElement.title = `UUID: ${img.uuid}\n来源: ${img.video_name || img.stream_name || img.upload_image_filename || '未知来源'}\n时间: ${new Date(img.created_at).toLocaleString()}\n比对分值: ${img.confidence_score !== null ? img.confidence_score.toFixed(2) : 'N/A'}`;

        const imgInfo = document.createElement('div');
        imgInfo.classList.add('image-info');
        imgInfo.innerHTML = `
            <p><strong>UUID:</strong> ${img.uuid}</p>
            <p><strong>来源:</strong> ${img.video_name || img.stream_name || img.upload_image_filename || '未知来源'}</p>
            <p><strong>时间:</strong> ${new Date(img.created_at).toLocaleString()}</p>
            <p><strong>比对分值:</strong> ${img.confidence_score !== null ? img.confidence_score.toFixed(2) : 'N/A'}</p>
        `;

        imgContainer.appendChild(imgElement);
        imgContainer.appendChild(imgInfo);
        imageGallery.appendChild(imgContainer);
    });
}