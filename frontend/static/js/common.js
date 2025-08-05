import { Auth, fetchWithAuth, handleLogout, API_BASE_URL } from './auth.js';
export { API_BASE_URL }; // 重新导出 API_BASE_URL

console.log("common.js: 文件已加载。"); // 新增：文件加载标志

// --- 全局变量用于Lightbox导航 ---
let currentImages = []; // 存储当前Lightbox显示的图片URL数组
let currentImageIndex = 0; // 当前Lightbox中显示图片的索引

// --- 公共工具函数 ---

/**
 * 显示通知消息的函数
 * @param {string} message - 要显示的通知消息
 * @param {string} type - 通知类型 ('success', 'error', 'info', 'warning')
 */
export function showNotification(message, type = 'info') {
    let notificationContainer = document.getElementById('notification-container');
    if (!notificationContainer) {
        const body = document.querySelector('body');
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notification-container';
        notificationContainer.style.position = 'fixed';
        notificationContainer.style.top = '20px';
        notificationContainer.style.left = '50%'; /* 新增：水平居中 */
        notificationContainer.style.transform = 'translateX(-50%)'; /* 新增：水平居中 */
        notificationContainer.style.zIndex = '1000';
        notificationContainer.style.display = 'flex';
        notificationContainer.style.flexDirection = 'column';
        notificationContainer.style.alignItems = 'center'; /* 新增：使通知内容居中 */
        notificationContainer.style.gap = '10px';
        body.appendChild(notificationContainer);
    }

    const notification = document.createElement('div');
    notification.classList.add('notification', type);
    notification.textContent = message;

    notificationContainer.appendChild(notification);

    // 2秒后自动移除通知
    setTimeout(() => {
        notification.classList.add('hide');
        notification.addEventListener('transitionend', () => {
            notification.remove();
        });
    }, 2000);
}

/**
 * 初始化需要认证的页面（非登录页）逻辑
 */
export function initializeAuthenticatedPage() {
    setupNavBarAndUserInfo();

    // 移除这里动态加载其他页面JS的逻辑，每个页面将自己导入和初始化其JS

    return Promise.resolve(); // 始终返回一个已解析的Promise
}

/**
 * 从后端获取用户个人资料
 */
export async function fetchUserProfile() {
    const token = Auth.getToken();
    if (!token) {
        console.error("未找到认证令牌。");
        return null;
    }

    try {
        const response = await fetchWithAuth(`${API_BASE_URL}/auth/users/me`, {
            method: 'GET'
        });
        if (!response.ok) {
            throw new Error(`获取用户资料失败: ${response.statusText}`);
        }
        const user = await response.json();
        // 可以选择在这里更新 Auth.userInfo 或直接返回
        return user;
    } catch (error) {
        console.error("获取用户资料时发生错误:", error);
        showNotification("无法加载用户资料。", 'error');
        return null;
    }
}

/**
 * 设置导航栏和显示用户信息
 */
function setupNavBarAndUserInfo() {
    const userInfo = Auth.getUserInfo();
    console.log('setupNavBarAndUserInfo: userInfo', userInfo); // 新增日志
    const usernameDisplay = document.getElementById('username-display');
    const roleDisplay = document.getElementById('role-display');
    const logoutButton = document.getElementById('logout-button');

    // 导航链接 (按ID)
    const videoResultsNav = document.getElementById('video-results-nav');
    const allFeaturesNav = document.getElementById('all-features-nav');
    const videoAnalysisNav = document.getElementById('video-analysis-nav');
    const imageSearchNav = document.getElementById('image-search-nav');
    const videoStreamNav = document.getElementById('video-stream-nav');
    const liveStreamResultsNav = document.getElementById('live-stream-results-nav');
    const adminNav = document.getElementById('admin-nav');

    // 功能卡片 (按 data-category 属性)
    const videoAnalysisCard = document.querySelector('[data-category="video-analysis"]');
    const imageSearchCard = document.querySelector('[data-category="image-search"]');
    const userManagementCard = document.querySelector('[data-category="user-management"]');
    const autoTrackingCard = document.querySelector('[data-category="auto-tracking"]');
    const videoStreamCard = document.querySelector('[data-category="video-stream"]');
    const allFeaturesCard = document.querySelector('[data-category="all-features"]');
    const adminPanelCard = document.querySelector('[data-category="admin-panel"]');
    const personListCard = document.getElementById('person-list-card'); 

    // 辅助函数隐藏元素
    const hideElement = (el) => {
        if (el) el.style.display = 'none';
    };

    // 辅助函数显示元素
    const showElement = (el) => {
        if (el) el.style.display = 'inline-block'; // 'block' 或 'flex' 取决于布局需要
    };

    // 初始隐藏所有相关元素
    hideElement(videoResultsNav);
    hideElement(allFeaturesNav);
    hideElement(videoAnalysisNav);
    hideElement(imageSearchNav);
    hideElement(videoStreamNav);
    hideElement(liveStreamResultsNav);
    hideElement(adminNav);
    
    hideElement(videoAnalysisCard);
    hideElement(imageSearchCard);
    hideElement(userManagementCard);
    hideElement(autoTrackingCard);
    hideElement(videoStreamCard);
    hideElement(allFeaturesCard);
    hideElement(adminPanelCard);
    hideElement(personListCard); // 初始隐藏人员信息表卡片

    if (userInfo) {
        if (usernameDisplay) usernameDisplay.textContent = userInfo.username;
        if (roleDisplay) {
            let displayRole = '未知角色';
            switch (userInfo.role) {
                case 'admin':
                    displayRole = '管理员';
                    break;
                case 'advanced': // 已从 'advanced_user' 更改为 'advanced'
                    displayRole = '高级用户';
                    break;
                case 'user':
                    displayRole = '普通用户';
                    break;
            }
            roleDisplay.textContent = displayRole;
        }

        // 所有认证用户可见的功能
        showElement(videoAnalysisCard);
        showElement(imageSearchCard);
        showElement(userManagementCard);
        showElement(videoAnalysisNav);
        showElement(imageSearchNav);

        // 高级用户和管理员可见的功能
        if (userInfo.role === 'advanced' || userInfo.role === 'admin') { 
            showElement(videoStreamCard); 
        }

        // 管理员可见的功能
        if (userInfo.role === 'admin') {
            showElement(allFeaturesCard);
            showElement(adminPanelCard);
            showElement(personListCard); 
            showElement(autoTrackingCard);
            showElement(allFeaturesNav);
            showElement(videoStreamNav);
            showElement(liveStreamResultsNav);
            showElement(adminNav);
        }

        // 视频解析结果页面总是可见 (Nav Link)
        showElement(videoResultsNav);

    } else {
        // 如果没有用户信息（例如token无效），重定向到登录页
        Auth.removeToken(); // 修正：将 clearToken 改为 removeToken
        window.location.href = '/login'; // 直接重定向到登录页
        return; // 防止后续代码执行
    }

    if (logoutButton) logoutButton.addEventListener('click', handleLogout);

    // 设置导航栏激活状态
    const navLinks = document.querySelectorAll('.header nav a');
    navLinks.forEach(link => {
        if (window.location.pathname.includes(link.getAttribute('href').split('/')[1])) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// Lightbox 相关
const lightbox = document.getElementById('lightbox');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxClose = document.querySelector('.lightbox-close');
const lightboxPrev = document.getElementById('lightbox-prev'); // 新增：上一张按钮
const lightboxNext = document.getElementById('lightbox-next'); // 新增：下一张按钮

if (lightboxClose) lightboxClose.addEventListener('click', hideLightbox);
if (lightbox) lightbox.addEventListener('click', (e) => { if (e.target === lightbox) hideLightbox(); });
if (lightboxPrev) lightboxPrev.addEventListener('click', showPreviousImage); // 新增：上一张按钮事件
if (lightboxNext) lightboxNext.addEventListener('click', showNextImage);     // 新增：下一张按钮事件

/**
 * 显示Lightbox并设置图片及导航信息。
 * @param {string} imageUrl - 要显示的第一张图片的URL。
 * @param {string[]} imagesArray - 所有相关图片的URL数组，用于导航。
 * @param {number} initialIndex - 当前图片的索引。
 */
export function showLightboxWithNav(imageUrl, imagesArray, initialIndex) {
    if (lightbox && lightboxImg) {
        currentImages = imagesArray; // 保存图片数组
        currentImageIndex = initialIndex; // 保存当前图片索引

        lightboxImg.src = imageUrl;
        lightbox.classList.remove('hidden');
        updateLightboxNavButtons(); // 更新导航按钮状态
    }
}

/**
 * 显示Lightbox（兼容旧函数名，但不再支持导航）
 * @param {string} imageUrl - 要显示图片的URL。
 */
export function showLightbox(imageUrl) {
    if (lightbox && lightboxImg) {
        console.log(`common.js: 显示Lightbox，原始图片URL: ${imageUrl}`);
        let correctedUrl = imageUrl;

        // 健壮性检查：如果URL是相对路径，则为其添加正确的前缀
        if (correctedUrl && !correctedUrl.startsWith('http') && !correctedUrl.startsWith('/static/')) {
            // 1. 统一处理斜杠
            let sanitizedPath = correctedUrl.replace(/\\/g, '/');
            // 2. 移除可能存在的前导斜杠，以便进行统一判断
            if (sanitizedPath.startsWith('/')) {
                sanitizedPath = sanitizedPath.substring(1);
            }

            // 3. 判断路径是否已经是 'database/...' 格式
            if (sanitizedPath.startsWith('database/')) {
                // 如果是，则直接拼接
                correctedUrl = `${API_BASE_URL}/${sanitizedPath}`;
            } else {
                // 如果不是，则添加 'database/' 前缀
                correctedUrl = `${API_BASE_URL}/database/${sanitizedPath}`;
            }
            console.log(`common.js: URL已修正: ${correctedUrl}`);

        } else if (correctedUrl) {
            // 对于已经是绝对URL或静态资源的路径，只需确保斜杠正确
            correctedUrl = correctedUrl.replace(/\\/g, '/');
        }

        lightboxImg.src = correctedUrl;
        // 调用 showLightboxWithNav 来正确设置导航状态
        showLightboxWithNav(correctedUrl, [correctedUrl], 0);
    }
}

/**
 * 隐藏Lightbox。
 */
export function hideLightbox() {
    if (lightbox) {
        lightbox.classList.add('hidden');
        lightboxImg.src = '';
        // 重置Lightbox导航状态
        currentImages = [];
        currentImageIndex = 0;
    }
}

/**
 * 显示上一张图片。
 */
function showPreviousImage() {
    if (currentImageIndex > 0) {
        currentImageIndex--;
        lightboxImg.src = currentImages[currentImageIndex];
        updateLightboxNavButtons();
    }
}

/**
 * 显示下一张图片。
 */
function showNextImage() {
    if (currentImageIndex < currentImages.length - 1) {
        currentImageIndex++;
        lightboxImg.src = currentImages[currentImageIndex];
        updateLightboxNavButtons();
    }
}

/**
 * 更新Lightbox导航按钮的显示状态（是否隐藏）。
 */
function updateLightboxNavButtons() {
    if (lightboxPrev && lightboxNext) {
        if (currentImages.length <= 1) {
            lightboxPrev.style.display = 'none';
            lightboxNext.style.display = 'none';
        } else {
            // 根据当前图片索引决定显示或隐藏按钮
            lightboxPrev.style.display = (currentImageIndex === 0) ? 'none' : 'block';
            lightboxNext.style.display = (currentImageIndex === currentImages.length - 1) ? 'none' : 'block';
        }
    }
}

/**
 * 为图片库设置点击事件，使其能通过Lightbox进行导航。
 * @param {string} gallerySelector - 包含图片项的容器的选择器 (e.g., '.results-grid', '#personTableBody').
 * @param {string} imageSelector - 图片元素的选择器，相对于 gallerySelector (e.g., '.feature-img', '.person-crop-img').
 * @param {function} getImageUrlFn - 一个函数，接收 imgElement 作为参数，返回其全帧图片URL。例如: (img) => img.dataset.fullImage。
 * @param {boolean} dynamicContent - 指示画廊内容是否会动态变化，如果为true，则每次点击时重新查找图片。
 */
export function setupImageGalleryClickEvents(gallerySelector, imageSelector, getImageUrlFn, dynamicContent = false) {
    const galleryContainer = document.querySelector(gallerySelector);
    if (!galleryContainer) {
        console.warn(`Gallery container not found for selector: ${gallerySelector}`);
        return;
    }

    const setupClick = () => {
        galleryContainer.querySelectorAll(imageSelector).forEach((img) => {
            img.onclick = function() {
                const clickedImageUrl = getImageUrlFn(this);
                let imagesArray = [];
                let initialIndex = 0;

                if (dynamicContent) {
                    // 如果是动态内容，每次点击时重新获取所有图片URL
                    imagesArray = Array.from(galleryContainer.querySelectorAll(imageSelector)).map(getImageUrlFn);
                } else {
                    // 如果是静态内容，可以使用之前缓存的或一次性获取的图片列表
                    // 对于all_features和live_stream_results，可以在loadPersons/loadFeatures时填充全局数组
                    // 这里为了通用性，我们直接重新获取，或者后续优化时传入缓存数组
                    imagesArray = Array.from(galleryContainer.querySelectorAll(imageSelector)).map(getImageUrlFn);
                }
                initialIndex = imagesArray.indexOf(clickedImageUrl);

                showLightboxWithNav(clickedImageUrl, imagesArray, initialIndex);
            };
        });
    };

    // 初始设置
    setupClick();

    // 监听DOM变化，重新绑定事件（例如在内容通过AJAX加载后）
    // 注意：MutationObserver 成本较高，仅在必要时使用或进行优化。
    const observer = new MutationObserver((mutations) => {
        let shouldRescan = false;
        for (let mutation of mutations) {
            if (mutation.type === 'childList' && (mutation.addedNodes.length > 0 || mutation.removedNodes.length > 0)) {
                // 检查是否有新的图片元素被添加或移除
                if (Array.from(mutation.addedNodes).some(node => node.nodeType === 1 && node.matches(imageSelector))) {
                    shouldRescan = true;
                    break;
                }
                if (Array.from(mutation.removedNodes).some(node => node.nodeType === 1 && node.matches(imageSelector))) {
                    shouldRescan = true;
                    break;
                }
            }
        }
        if (shouldRescan) {
            setupClick();
        }
    });

    // 配置观察器观察子节点的变化
    observer.observe(galleryContainer, { childList: true, subtree: true });
} 

// Lightbox 初始化函数，在页面加载时调用一次
export function setupLightbox() {
    // Lightbox 相关的事件监听器已经在文件顶部设置，此处仅作导出标记
    console.log("common.js: Lightbox 初始化完成。");
} 
