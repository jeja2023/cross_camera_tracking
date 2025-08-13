// JavaScript for followed_person_alerts.html

import { displayUserInfo, setupLogout, showLightboxWithNav, setupImageGalleryClickEvents } from './common.js';

document.addEventListener('DOMContentLoaded', async () => {
    // 调用通用函数显示用户信息和处理登出
    displayUserInfo();
    setupLogout();

    // 从 URL 获取 followedPersonId
    const params = new URLSearchParams(window.location.search);
    const followedPersonId = params.get('id');
    const personName = params.get('name');

    if (!followedPersonId) {
        console.error('未提供 followedPersonId。');
        document.getElementById('alerts-message').textContent = '未找到关注人员ID。';
        return;
    }

    document.getElementById('individual-name-display').textContent = personName || '未知人员';

    // 移除查询相关的 DOM 元素获取，因为它们将被隐藏
    // const queryAlertsButton = document.getElementById('query-alerts-button');
    // const lastQueryTimeDisplay = document.getElementById('last-query-time-display');

    const alertsResultsContainer = document.getElementById('alerts-results-container');
    const alertsMessage = document.getElementById('alerts-message');

    // 分页控件
    const firstPageButton = document.getElementById('first-page-button');
    const prevPageButton = document.getElementById('prev-page-button');
    const nextPageButton = document.getElementById('next-page-button');
    const lastPageButton = document.getElementById('last-page-button');
    const pageInfoSpan = document.getElementById('page-info');

    let currentPage = 1;
    const itemsPerPage = 20; // 每页显示的图片数量
    let totalPages = 1;

    async function fetchAlerts(page = 1) {
        alertsResultsContainer.innerHTML = '';
        alertsMessage.textContent = '正在加载预警信息...';
        // lastQueryTimeDisplay.textContent = `最近查询时间: ${new Date().toLocaleString()}`;

        try {
            const token = localStorage.getItem('accessToken'); // 修正：使用 'accessToken' 键
            // 更正 API 请求路径
            const response = await fetch(`/followed_persons/${followedPersonId}/alerts?page=${page}&per_page=${itemsPerPage}`, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                if (response.status === 401) {
                    window.location.href = '/static/login.html';
                    return;
                }
                const errorData = await response.json();
                throw new Error(errorData.detail || '无法获取预警信息。');
            }

            const data = await response.json();
            const alerts = data.items;
            totalPages = data.pages;
            currentPage = data.page;

            alertsMessage.textContent = '';

            if (alerts.length === 0) {
                alertsMessage.textContent = '未找到预警信息。';
                return;
            }

            alerts.forEach(alert => {
                const alertCard = document.createElement('div');
                alertCard.className = 'result-item'; // 更改类名为 result-item
                alertCard.innerHTML = `
                    <img src="${alert.cropped_image_path}" alt="预警图片" class="thumbnail" data-full-size="${alert.full_frame_image_path}">
                    <div class="info-section">
                        <p><strong>图片UUID:</strong> ${String(alert.person_uuid)}</p>
                        <p><strong>创建时间:</strong> ${new Date(alert.person_created_at).toLocaleString()}</p>
                        <p><strong>来源UUID:</strong> ${String(alert.source_media_uuid)}</p>
                        <p><strong>来源类型:</strong> ${getSourceTypeDisplayName(alert.source_media_type)}</p>
                        <p><strong>预警时间:</strong> ${new Date(alert.timestamp).toLocaleString()}</p>
                        <p><strong>相似度分值:</strong> <span class="similarity-score">${(alert.similarity_score * 100).toFixed(2)}%</span></p>
                    </div>
                `;
                alertsResultsContainer.appendChild(alertCard);
            });

            // 在数据加载和DOM更新完成后，设置图片库点击事件
            setupImageGalleryClickEvents('#alerts-results-container', '.result-item img', (img) => String(img.dataset.fullSize), true);

            updatePaginationControls();

        } catch (error) {
            console.error('获取预警信息失败:', error);
            alertsMessage.textContent = `加载预警信息失败: ${error.message}`;
        }
    }

    function updatePaginationControls() {
        // 确保 currentPage 和 totalPages 是有效数字，否则显示默认值
        const current = typeof currentPage === 'number' && !isNaN(currentPage) ? currentPage : 1;
        const total = typeof totalPages === 'number' && !isNaN(totalPages) ? totalPages : 1;
        pageInfoSpan.textContent = `页数: ${current} / ${total}`;
        firstPageButton.disabled = current === 1;
        prevPageButton.disabled = current === 1;
        nextPageButton.disabled = current === total;
        lastPageButton.disabled = current === total;
    }

    // 移除按钮事件监听
    // queryAlertsButton.addEventListener('click', () => fetchAlerts());
    firstPageButton.addEventListener('click', () => fetchAlerts(1));
    prevPageButton.addEventListener('click', () => fetchAlerts(currentPage - 1));
    nextPageButton.addEventListener('click', () => fetchAlerts(currentPage + 1));
    lastPageButton.addEventListener('click', () => fetchAlerts(totalPages));

    // 初始加载预警信息
    fetchAlerts();
});

// 新增：根据来源类型返回对应的显示名称
function getSourceTypeDisplayName(sourceType) {
    switch (sourceType) {
        case 'image':
            return '图片解析';
        case 'video':
            return '视频解析';
        case 'stream':
            return '视频流解析';
        case 'image_search':
            return '以图搜人';
        // 可以根据需要添加更多类型
        default:
            return '未知来源';
    }
}