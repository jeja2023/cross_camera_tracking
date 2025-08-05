import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js';
import { showLightboxWithNav } from './common.js'; // 导入 common.js 中的 showLightboxWithNav

export function initImageAnalysisResultsPage() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    const imageAnalysisResultsContainer = document.getElementById('imageAnalysisResults');
    const originalImageDisplay = document.getElementById('originalImageDisplay');
    // const originalImageUUIDDisplay = document.getElementById('originalImageUUIDDisplay'); // 已在HTML中移除，JS中无需引用
    const originalImageFilenameDisplay = document.getElementById('originalImageFilenameDisplay');
    const uploadedTimeDisplay = document.getElementById('uploadedTimeDisplay');
    const detectedPersonsCount = document.getElementById('detectedPersonsCount');
    const personGrid = document.getElementById('personGrid');
    const noPersonsMessage = document.getElementById('noPersonsMessage');

    let allPersonImageUrls = []; // 用于Lightbox导航的所有人物裁剪图URL数组

    async function fetchImageAnalysisResults() {
        loadingMessage.style.display = 'block';
        errorMessage.style.display = 'none';
        imageAnalysisResultsContainer.style.display = 'none';
        personGrid.innerHTML = ''; // 清空人物网格
        noPersonsMessage.style.display = 'none';
        allPersonImageUrls = []; // 重置图片URL数组

        const urlParams = new URLSearchParams(window.location.search);
        const imageUUID = urlParams.get('uuid');

        if (!imageUUID) {
            errorMessage.textContent = '错误：未提供图片 UUID。请返回图片解析页面。';
            errorMessage.style.display = 'block';
            loadingMessage.style.display = 'none';
            return;
        }

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/image_analysis/results/${imageUUID}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '无法加载图片解析结果');
            }

            const result = await response.json();
            console.log('图片解析详细结果:', result);

            if (result.status === "success" && result.original_image_info) {
                const originalImage = result.original_image_info;
                const analyzedPersons = result.analyzed_persons;

                // 处理原始图片路径
                // API返回的路径已经是正确的相对路径，只需确保它以'/'开头
                let originalImagePath = originalImage.full_frame_image_path;
                if (originalImagePath) {
                    originalImagePath = API_BASE_URL + '/' + originalImagePath.replace(/\\/g, '/');
                }
                
                console.log("原始图片 URL:", originalImagePath); // 添加调试日志
                originalImageDisplay.src = originalImagePath; // 原始图片URL
                
                detectedPersonsCount.textContent = analyzedPersons.length;

                if (analyzedPersons.length > 0) {
                    noPersonsMessage.style.display = 'none';
                    analyzedPersons.forEach(person => {
                        // 处理人物裁剪图路径
                        // API返回的路径已经是正确的相对路径，只需确保它以'/'开头
                        let cropImagePath = person.crop_image_path;
                        if (cropImagePath) {
                            cropImagePath = API_BASE_URL + '/' + cropImagePath.replace(/\\/g, '/');
                        }
                        
                        const absoluteCropImagePath = cropImagePath;
                        allPersonImageUrls.push(absoluteCropImagePath); // 将人物裁剪图绝对URL添加到数组中
                        console.log("人物裁剪图 URL (absolute):", absoluteCropImagePath); // 添加调试日志

                        const personCard = document.createElement('div');
                        personCard.className = 'person-card';
                        personCard.innerHTML = `
                            <img src="${cropImagePath}" alt="人物裁剪图" data-uuid="${person.uuid}">
                            <p>人物UUID: <span class="uuid-display">${person.uuid}</span></p>
                            <p>入库时间: ${new Date(person.timestamp).toLocaleString()}</p>
                        `;
                        personGrid.appendChild(personCard);
                    });
                } else {
                    noPersonsMessage.style.display = 'block';
                    noPersonsMessage.textContent = '未检测到人物。';
                }

                imageAnalysisResultsContainer.style.display = 'block';

            } else {
                errorMessage.textContent = result.message || '加载图片解析结果失败';
                errorMessage.style.display = 'block';
            }
        } catch (error) {
            console.error('加载图片解析结果错误:', error);
            errorMessage.textContent = `错误：${error.message}`;
            errorMessage.style.display = 'block';
        } finally {
            loadingMessage.style.display = 'none';
        }
    }

    // 页面加载时立即获取结果
    fetchImageAnalysisResults();

    // Lightbox 相关逻辑 (现在使用 common.js 中的函数)
    // 移除重复的 Lightbox 元素获取和事件监听
    // const lightbox = document.getElementById('lightbox');
    // const lightboxClose = document.querySelector('.lightbox-close');
    // const lightboxImg = document.getElementById('lightbox-img');

    // if (lightboxClose) lightboxClose.addEventListener('click', () => {
    //     lightbox.classList.add('hidden');
    // });
    // if (lightbox) lightbox.addEventListener('click', (e) => { 
    //     if (e.target === lightbox) lightbox.classList.add('hidden'); 
    // });

    // 为裁剪图添加点击事件，以便在 lightbox 中显示大图，并启用导航
    personGrid.addEventListener('click', (event) => {
        const clickedImg = event.target.closest('.person-card img');
        if (clickedImg) {
            // Ensure the imageUrl is an absolute URL for consistent matching
            const imageUrl = new URL(clickedImg.getAttribute('src'), window.location.origin).href;
            console.log("点击的图片 URL:", imageUrl); // 新增日志
            console.log("所有人物图片 URL 数组:", allPersonImageUrls); // 新增日志
            const initialIndex = allPersonImageUrls.indexOf(imageUrl); // 找到当前图片的索引
            console.log("初始索引:", initialIndex); // 新增日志
            showLightboxWithNav(imageUrl, allPersonImageUrls, initialIndex);
        }
    });

     // 为原始图片添加点击事件，以便在 lightbox 中显示大图 (不启用导航)
     originalImageDisplay.addEventListener('click', () => {
        // Ensure the originalImageDisplay.src is an absolute URL
        const imageUrl = new URL(originalImageDisplay.src, window.location.origin).href;
        showLightboxWithNav(imageUrl, [imageUrl], 0);
    });
} 