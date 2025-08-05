import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js';
import { showNotification, setupLightbox, initializeAuthenticatedPage, fetchUserProfile } from './common.js';

let currentPage = 1;
const limit = 10; // 每页显示10条记录
let totalPages = 1;
let currentImages = []; // 用于模态框显示的图片列表
let currentImageIndex = 0;

// 全局的实时比对状态文本和按钮（如果HTML中存在的话），现在不再使用，可以移除引用或确保其隐藏
const realtimeComparisonStatusText = document.getElementById('realtimeComparisonStatusText');
const toggleRealtimeComparisonBtn = document.getElementById('toggleRealtimeComparisonBtn');

document.addEventListener('DOMContentLoaded', async () => {
    // 确保用户已认证
    if (!Auth.isAuthenticated()) {
        window.location.href = '/login';
        return;
    }

    initializeAuthenticatedPage(); // 调用 common.js 中的认证页面初始化函数
    // 获取用户 profile，以便在渲染表格时检查管理员权限
    const userProfile = await fetchUserProfile(); 
    setupLightbox(); // 初始化图片预览功能

    // 如果全局的实时比对UI元素存在，确保它们被隐藏，因为功能已改为针对单个关注人
    if (realtimeComparisonStatusText) realtimeComparisonStatusText.style.display = 'none';
    if (toggleRealtimeComparisonBtn) toggleRealtimeComparisonBtn.style.display = 'none';

    // 初始化页面时加载关注人员列表
    await fetchFollowedPersons(currentPage, limit);

    // 分页按钮事件监听
    document.getElementById('firstPageBtn').addEventListener('click', () => {
        if (currentPage !== 1) {
            currentPage = 1;
            fetchFollowedPersons(currentPage, limit);
        }
    });

    document.getElementById('prevPageBtn').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            fetchFollowedPersons(currentPage, limit);
        }
    });

    document.getElementById('nextPageBtn').addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            fetchFollowedPersons(currentPage, limit);
        }
    });

    document.getElementById('lastPageBtn').addEventListener('click', () => {
        if (currentPage !== totalPages) {
            currentPage = totalPages;
            fetchFollowedPersons(currentPage, limit);
        }
    });
});

async function fetchFollowedPersons(page, limit) {
    try {
        const response = await fetchWithAuth(`${API_BASE_URL}/followed_persons/?skip=${(page - 1) * limit}&limit=${limit}`);
        if (response.ok) {
            const data = await response.json();
            const persons = data.items;
            const total = data.total;
            totalPages = Math.ceil(total / limit);

            // 传递 userProfile 到 renderFollowedPersonsTable
            const userProfile = await fetchUserProfile(); 
            renderFollowedPersonsTable(persons, userProfile);
            updatePagination(total, page, limit);
        } else {
            const errorData = await response.json();
            showNotification(`加载关注人员列表失败: ${errorData.detail || response.statusText}`, 'error');
        }
    } catch (error) {
        console.error('获取关注人员列表失败:', error);
        showNotification('网络错误，无法加载关注人员列表。', 'error');
    }
}

function renderFollowedPersonsTable(persons, userProfile) {
    const tableBody = document.getElementById('followedPersonTableBody');
    tableBody.innerHTML = ''; // 清空现有内容

    if (persons.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="5">暂无关注人员。</td></tr>';
        return;
    }

    persons.forEach((person, index) => {
        const row = tableBody.insertRow();
        const displayIndex = (currentPage - 1) * limit + index + 1; // 计算显示序号

        // 格式化关注时间
        const followedAt = new Date(person.followed_at);
        const formattedFollowedAt = followedAt.toLocaleString('zh-CN', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });

        row.insertCell(0).textContent = displayIndex;
        row.insertCell(1).textContent = person.individual_name || '未知姓名';
        row.insertCell(2).textContent = person.individual_id_card || '无身份证/ID';
        row.insertCell(3).textContent = formattedFollowedAt;

        const actionsCell = row.insertCell(4);
        const buttonGroup = document.createElement('div');
        buttonGroup.classList.add('action-buttons-group');

        const viewAlertsButton = document.createElement('button');
        viewAlertsButton.textContent = '查看预警';
        viewAlertsButton.classList.add('button', 'btn-info', 'btn-sm');
        viewAlertsButton.addEventListener('click', () => {
            window.open(`/alert_images?individual_id=${person.individual_id}&individual_name=${encodeURIComponent(person.individual_name || person.individual_id_card || '未知人员')}`, '_blank');
        });
        buttonGroup.appendChild(viewAlertsButton);

        // 添加 "查看注册图片" 按钮
        const viewEnrollmentsButton = document.createElement('button');
        viewEnrollmentsButton.textContent = '查看注册图片';
        viewEnrollmentsButton.classList.add('button', 'btn-primary', 'btn-sm');
        viewEnrollmentsButton.addEventListener('click', () => {
            window.open(`/enrollment_images?individual_id=${person.individual_id}&individual_name=${encodeURIComponent(person.individual_name || person.individual_id_card || '未知人员')}`, '_blank');
        });
        buttonGroup.appendChild(viewEnrollmentsButton);

        // 实时比对功能按钮 (针对单个关注人)
        if (userProfile && userProfile.role === 'admin') {
            const toggleRealtimeComparisonPersonBtn = document.createElement('button');
            toggleRealtimeComparisonPersonBtn.textContent = '加载中...';
            toggleRealtimeComparisonPersonBtn.classList.add('button');
            toggleRealtimeComparisonPersonBtn.dataset.individualId = person.individual_id;
            toggleRealtimeComparisonPersonBtn.dataset.currentStatus = 'unknown'; // 初始状态
            
            buttonGroup.appendChild(toggleRealtimeComparisonPersonBtn);
            
            // 获取此关注人的实时比对状态
            fetchIndividualRealtimeComparisonStatus(person.individual_id, toggleRealtimeComparisonPersonBtn);
            
            // 添加事件监听器
            toggleRealtimeComparisonPersonBtn.addEventListener('click', () => 
                toggleIndividualRealtimeComparison(person.individual_id, toggleRealtimeComparisonPersonBtn));
        }

        const unfollowButton = document.createElement('button');
        unfollowButton.textContent = '取消关注';
        unfollowButton.classList.add('button', 'btn-danger', 'btn-sm');
        unfollowButton.addEventListener('click', () => {
            if (confirm(`确定要取消关注 ${person.individual_name || person.individual_id_card || person.individual_id} 吗？`)) {
                unfollowPerson(person.individual_id);
            }
        });
        buttonGroup.appendChild(unfollowButton);

        actionsCell.appendChild(buttonGroup);
    });
}

// 新增：获取单个关注人员实时比对状态的函数
async function fetchIndividualRealtimeComparisonStatus(individualId, buttonElement) {
    try {
        const response = await fetchWithAuth(`${API_BASE_URL}/followed_persons/${individualId}/realtime-comparison-enabled`);
        if (response.ok) {
            const isEnabled = await response.json();
            updateIndividualRealtimeComparisonUI(isEnabled, buttonElement);
        } else {
            const errorData = await response.json();
            console.error(`获取人物 ${individualId} 实时比对功能状态失败:`, errorData);
            buttonElement.textContent = '加载失败';
            buttonElement.classList.remove('btn-success', 'btn-danger');
            showNotification(`获取人物实时比对功能状态失败: ${errorData.detail || response.statusText}`, 'error');
        }
    } catch (error) {
        console.error(`获取人物 ${individualId} 实时比对功能状态时发生网络错误:`, error);
        buttonElement.textContent = '网络错误';
        buttonElement.classList.remove('btn-success', 'btn-danger');
        showNotification('获取人物实时比对功能状态时发生网络错误。', 'error');
    }
}

// 新增：切换单个关注人员实时比对状态的函数
async function toggleIndividualRealtimeComparison(individualId, buttonElement) {
    const currentStatus = buttonElement.dataset.currentStatus;
    const newStatus = currentStatus === 'true' ? false : true; // 如果当前是true，则新状态是false，反之

    if (confirm(`确定要将人物 ${individualId} 的实时比对功能${newStatus ? '启用' : '停用'}吗？`)) {
        buttonElement.disabled = true; // 禁用按钮，防止重复点击
        buttonElement.textContent = '切换中...';
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/followed_persons/${individualId}/toggle-realtime-comparison?enable=${newStatus}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const result = await response.json();
                showNotification(result.message, 'success');
                updateIndividualRealtimeComparisonUI(newStatus, buttonElement);
            } else {
                const errorData = await response.json();
                showNotification(`切换人物实时比对功能失败: ${errorData.detail || response.statusText}`, 'error');
            }
        } catch (error) {
            console.error(`切换人物 ${individualId} 实时比对功能时发生网络错误:`, error);
            showNotification('切换人物实时比对功能时发生网络错误。', 'error');
        } finally {
            buttonElement.disabled = false; // 重新启用按钮
        }
    }
}

// 新增：更新单个关注人员实时比对UI的函数
function updateIndividualRealtimeComparisonUI(isEnabled, buttonElement) {
    buttonElement.dataset.currentStatus = isEnabled ? 'true' : 'false';
    if (isEnabled) {
        buttonElement.textContent = '实时比对：已启用';
        buttonElement.classList.remove('btn-danger');
        buttonElement.classList.add('btn-success');
    } else {
        buttonElement.textContent = '实时比对：已停用';
        buttonElement.classList.remove('btn-success');
        buttonElement.classList.add('btn-danger');
    }
}

function updatePagination(totalItems, currentPage, limit) {
    const pageInfoSpan = document.getElementById('pageInfo');
    const firstPageBtn = document.getElementById('firstPageBtn');
    const prevPageBtn = document.getElementById('prevPageBtn');
    const nextPageBtn = document.getElementById('nextPageBtn');
    const lastPageBtn = document.getElementById('lastPageBtn');

    totalPages = Math.ceil(totalItems / limit);
    pageInfoSpan.textContent = `第 ${currentPage} 页 / 共 ${totalPages} 页`;

    firstPageBtn.disabled = currentPage === 1;
    prevPageBtn.disabled = currentPage === 1;
    nextPageBtn.disabled = currentPage === totalPages || totalPages === 0;
    lastPageBtn.disabled = currentPage === totalPages || totalPages === 0;
}

async function unfollowPerson(individualId) {
    if (confirm('确定要取消关注此人吗？')) {
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/followed_persons/unfollow/${individualId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                showNotification('取消关注成功！', 'success');
                fetchFollowedPersons(currentPage, limit); // Refresh the list
            } else {
                const errorData = await response.json();
                showNotification(`取消关注失败: ${errorData.detail || response.statusText}`, 'error');
            }
        } catch (error) {
            console.error('取消关注时发生错误:', error);
            showNotification('取消关注时发生网络错误。', 'error');
        }
    }
}

// 模态框逻辑
// const imageModal = document.getElementById('imageModal');
// const closeButton = imageModal.querySelector('.close-button');
// const modalTitle = document.getElementById('modalTitle');
// const imageGallery = document.getElementById('imageGallery');
// const prevImageBtn = document.getElementById('prevImageBtn');
// const nextImageBtn = document.getElementById('nextImageBtn');
// const imageCounter = document.getElementById('imageCounter');

// closeButton.addEventListener('click', () => {
//     imageModal.style.display = 'none';
//     imageGallery.innerHTML = ''; // 清空图片
//     currentImages = []; // 清空图片列表
//     currentImageIndex = 0; // 重置索引
// });

// prevImageBtn.addEventListener('click', () => navigateImage(-1));
// nextImageBtn.addEventListener('click', () => navigateImage(1));

// async function openImageModal(individualId, type, personName) {
//     modalTitle.textContent = `${personName} 的${type === 'alerts' ? '预警图片' : '注册图片'}`;
//     imageGallery.innerHTML = '';
//     currentImages = [];
//     currentImageIndex = 0;
//     imageCounter.textContent = '';

//     const token = Auth.getToken();
//     if (!token) {
//         showNotification('未认证或会话已过期，请重新登录。', 'error');
//         window.location.href = '/login';
//         return;
//     }

//     try {
//         const endpoint = `${API_BASE_URL}/followed_persons/${individualId}/${type}`;
//         const response = await fetch(endpoint, {
//             method: 'GET',
//             headers: {
//                 'Authorization': `Bearer ${token}`
//             }
//         });

//         if (!response.ok) {
//             if (response.status === 403) {
//                 showNotification('无权限查看该人员的图片。', 'error');
//             } else {
//                 throw new Error(`HTTP 错误！状态码: ${response.status}`);
//             }
//             return;
//         }

//         const images = await response.json();
//         currentImages = images.map(img => ({
//             path: img.crop_image_path, // 使用裁剪图片路径
//             timestamp: img.created_at,
//             source: img.video_name || img.stream_name || img.upload_image_filename || '未知来源',
//             uuid: img.uuid
//         }));

//         if (currentImages.length === 0) {
//             imageGallery.innerHTML = '<p>暂无图片。</p>';
//             prevImageBtn.style.display = 'none';
//             nextImageBtn.style.display = 'none';
//             imageCounter.style.display = 'none';
//         } else {
//             renderCurrentImage();
//             prevImageBtn.style.display = 'inline-block';
//             nextImageBtn.style.display = 'inline-block';
//             imageCounter.style.display = 'inline-block';
//         }
//         imageModal.style.display = 'flex'; // 使用 flexbox 居中
//     } catch (error) {
//         console.error(`获取 ${type} 图片失败:`, error);
//         showNotification(`获取图片失败: ${error.message}`, 'error');
//     }
// }

// function renderCurrentImage() {
//     imageGallery.innerHTML = '';
//     if (currentImages.length > 0) {
//         const image = currentImages[currentImageIndex];
//         const imgElement = document.createElement('img');
//         imgElement.src = image.path.startsWith('http') ? image.path : `${API_BASE_URL}/${image.path}`;
//         imgElement.alt = `图片 ${currentImageIndex + 1}`;
//         imgElement.title = `UUID: ${image.uuid}\n来源: ${image.source}\n时间: ${new Date(image.timestamp).toLocaleString()}`;
//         imageGallery.appendChild(imgElement);
//         imageCounter.textContent = `${currentImageIndex + 1} / ${currentImages.length}`;

//         // 更新导航按钮状态
//         prevImageBtn.disabled = currentImageIndex === 0;
//         nextImageBtn.disabled = currentImageIndex === currentImages.length - 1;
//     }
// }

// function navigateImage(direction) {
//     currentImageIndex += direction;
//     if (currentImageIndex < 0) {
//         currentImageIndex = 0;
//     } else if (currentImageIndex >= currentImages.length) {
//         currentImageIndex = currentImages.length - 1;
//     }
//     renderCurrentImage();
// }