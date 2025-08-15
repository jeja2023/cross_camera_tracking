import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js';
import { showLightboxWithNav, setupImageGalleryClickEvents } from './common.js'; // Import setupImageGalleryClickEvents

let currentPage = 1;
const itemsPerPage = 20; // 默认每页显示20条
let totalPersons = 0;
let allFeaturesImages = []; // 新增：存储当前页面的所有图片URL，用于Lightbox导航

export function initAllFeaturesPage() {
    console.log("Initializing All Features Page...");
    loadAllPersons(currentPage, itemsPerPage);

    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const firstPageButton = document.getElementById('first-page');
    const lastPageButton = document.getElementById('last-page');
    const featureSearchInput = document.getElementById('feature-search-input');
    const applyFilterButton = document.getElementById('apply-filter-button');
    const exportFeaturesButton = document.getElementById('export-features-button'); // 获取导出按钮引用

    if (firstPageButton) {
        firstPageButton.addEventListener('click', () => {
            if (currentPage !== 1) {
                currentPage = 1;
                loadAllPersons(currentPage, itemsPerPage, featureSearchInput.value);
            }
        });
    }

    if (prevPageButton) {
        prevPageButton.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                loadAllPersons(currentPage, itemsPerPage, featureSearchInput.value);
            }
        });
    }

    if (nextPageButton) {
        nextPageButton.addEventListener('click', () => {
            const totalPages = Math.ceil(totalPersons / itemsPerPage);
            if (currentPage < totalPages) {
                currentPage++;
                loadAllPersons(currentPage, itemsPerPage, featureSearchInput.value);
            }
        });
    }

    if (lastPageButton) {
        lastPageButton.addEventListener('click', () => {
            const totalPages = Math.ceil(totalPersons / itemsPerPage);
            if (currentPage !== totalPages) {
                currentPage = totalPages;
                loadAllPersons(currentPage, itemsPerPage, featureSearchInput.value);
            }
        });
    }

    if (applyFilterButton) {
        applyFilterButton.addEventListener('click', () => {
            currentPage = 1; // 筛选时重置到第一页
            loadAllPersons(currentPage, itemsPerPage, featureSearchInput.value);
        });
    }

    // 也允许在输入框内按回车键触发筛选
    if (featureSearchInput) {
        featureSearchInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                currentPage = 1;
                loadAllPersons(currentPage, itemsPerPage, featureSearchInput.value);
            }
        });
    }

    // 导出按钮事件监听器
    if (exportFeaturesButton) {
        exportFeaturesButton.addEventListener('click', () => {
            exportAllPersons(featureSearchInput.value); // 导出时使用当前的筛选查询
        });
    }
}

async function loadAllPersons(page, limit, filterQuery = '') {
    const allPersonsContainer = document.getElementById('all-persons-container');
    const allPersonsLoading = document.getElementById('all-persons-loading');
    const noAllPersonsMessage = document.getElementById('no-all-persons-message');
    const paginationControls = document.getElementById('pagination-controls');
    const currentPageSpan = document.getElementById('current-page');
    const totalPagesSpan = document.getElementById('total-pages');

    if (allPersonsContainer) allPersonsContainer.innerHTML = '';
    if (allPersonsLoading) allPersonsLoading.classList.remove('hidden');
    if (noAllPersonsMessage) noAllPersonsMessage.classList.add('hidden');
    if (paginationControls) paginationControls.classList.add('hidden'); // 默认隐藏分页控件

    const skip = (page - 1) * limit;
    let url = `/api/persons/all?skip=${skip}&limit=${limit}`;
    if (filterQuery) {
        url += `&query=${encodeURIComponent(filterQuery)}`;
    }

    try {
        const response = await fetchWithAuth(url);
        if (!response.ok) {
            throw new Error('无法加载所有人脸数据');
        }
        const data = await response.json(); // 后端现在返回PaginatedPersonsResponse
        const persons = data.items;
        totalPersons = data.total;
        console.log(`[loadAllPersons] totalPersons: ${totalPersons}`); // Add this line

        if (allPersonsContainer) {
            if (persons.length === 0) {
                if (noAllPersonsMessage) noAllPersonsMessage.classList.remove('hidden');
                // Removed: allFeaturesImages = []; // 清空图片列表
            } else {
                // Removed: 收集当前页面的所有图片URL，用于Lightbox导航
                // Removed: allFeaturesImages = persons.map(person => {
                // Removed:     // 处理图片路径，确保添加正确的前缀
                // Removed:     let fullFramePath = person.full_frame_image_path;
                // Removed:     let cropPath = person.crop_image_path;
                // Removed:     
                // Removed:     if (fullFramePath && !fullFramePath.startsWith('/full_frames/')) {
                // Removed:         fullFramePath = `/full_frames/${fullFramePath}`;
                // Removed:     }
                // Removed:     if (cropPath && !cropPath.startsWith('/crops/')) {
                // Removed:         cropPath = `/crops/${cropPath}`;
                // Removed:     }
                // Removed:     
                // Removed:     return fullFramePath || cropPath;
                // Removed: });

                persons.forEach(person => {
                    const personItem = document.createElement('div');
                    personItem.classList.add('result-item');
                    const displayTime = new Intl.DateTimeFormat('zh-CN', {
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: false,
                        timeZone: 'Asia/Shanghai'
                    }).format(new Date(person.created_at));

                    let cropImagePath = person.crop_image_path ? person.crop_image_path.replace(/\\/g, '/') : '';
                    let fullFrameImagePath = person.full_frame_image_path ? person.full_frame_image_path.replace(/\\/g, '/') : '';

                    personItem.innerHTML = `
                        <img src="${cropImagePath}" alt="人物裁剪" data-full-size="${fullFrameImagePath || cropImagePath}">
                        <p><strong>人物ID:</strong> ${person.uuid}</p>
                        ${person.stream_uuid ? `<p><strong>视频流名称:</strong> ${person.stream_name || 'N/A'}</p><p><strong>视频流UUID:</strong> ${person.stream_uuid}</p>` :
                         person.video_uuid ? `<p><strong>视频名称:</strong> ${person.video_name || 'N/A'}</p><p><strong>视频UUID:</strong> ${person.video_uuid}</p>` :
                         person.upload_image_uuid ? `<p><strong>上传图片UUID:</strong> ${person.upload_image_uuid}</p>` :
                         person.video_id ? `<p><strong>视频ID:</strong> ${person.video_id}</p>` : ''}
                        <p class="time-info"><strong>入库时间:</strong> ${displayTime}</p>
                    `;
                    // Removed: personItem.querySelector('img').addEventListener('click', (e) => {
                    // Removed:     const clickedImageUrl = e.target.dataset.fullSize;
                    // Removed:     const imageIndex = allFeaturesImages.indexOf(clickedImageUrl);
                    // Removed:     showLightboxWithNav(clickedImageUrl, allFeaturesImages, imageIndex);
                    // Removed: });
                    allPersonsContainer.appendChild(personItem);
                });
                // 在数据加载和DOM更新完成后，设置图片库点击事件
                setupImageGalleryClickEvents('#all-persons-container', '.result-item img', (img) => img.dataset.fullSize, true);

                // 更新分页信息和显示分页控件
                const totalPages = Math.ceil(totalPersons / itemsPerPage);
                if (currentPageSpan) currentPageSpan.textContent = currentPage;
                if (totalPagesSpan) totalPagesSpan.textContent = totalPages;
                if (paginationControls) paginationControls.classList.remove('hidden');

                // 更新分页按钮状态
                const prevPageButton = document.getElementById('prev-page');
                const nextPageButton = document.getElementById('next-page');
                const firstPageButton = document.getElementById('first-page');
                const lastPageButton = document.getElementById('last-page');

                if (firstPageButton) firstPageButton.disabled = (currentPage === 1);
                if (prevPageButton) prevPageButton.disabled = (currentPage === 1);
                if (nextPageButton) nextPageButton.disabled = (currentPage === totalPages);
                if (lastPageButton) lastPageButton.disabled = (currentPage === totalPages);
            }
        }
    } catch (error) {
        console.error('加载所有人脸数据失败:', error);
        if (allPersonsContainer) allPersonsContainer.innerHTML = `<p style="color: red; text-align: center;">加载所有人脸数据失败: ${error.message}</p>`;
    } finally {
        if (allPersonsLoading) allPersonsLoading.classList.add('hidden');
    }
} 

// 新增函数：导出所有人物特征
async function exportAllPersons(filterQuery = '') {
    console.log(`Attempting to export all persons with filter: ${filterQuery}`);

    const exportButton = document.getElementById('export-features-button');
    if (exportButton) {
        exportButton.disabled = true;
        exportButton.textContent = '正在导出...';
    }

    let url = `/export/all_persons`;
    if (filterQuery) {
        url += `?query=${encodeURIComponent(filterQuery)}`;
    }

    try {
        const response = await fetchWithAuth(url, {
            method: 'GET',
        });

        if (response.status === 204) {
            alert('没有符合筛选条件的人物特征可导出。');
            return;
        }

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '导出失败');
        }

        const blob = await response.blob();
        const urlBlob = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = urlBlob;

        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `all_persons_features_${new Date().toISOString().slice(0,10).replace(/-/g,"")}.xlsx`; // Fallback filename

        if (contentDisposition) {
            const filenameRegex = /filename\*?=(?:UTF-8'')?"?([^;\n]*?)"?$/i;
            const matches = contentDisposition.match(filenameRegex);
            if (matches && matches[1]) {
                try {
                    filename = decodeURIComponent(matches[1].replace(/%22/g, '').replace(/^"|"$/g, ''));
                } catch (e) {
                    console.error("Error decoding filename from Content-Disposition", e);
                }
            }
        }

        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(urlBlob);
        alert('导出成功！');

    } catch (error) {
        console.error('导出人物特征失败:', error);
        alert(`导出失败: ${error.message}`);
    } finally {
        if (exportButton) {
            exportButton.disabled = false;
            exportButton.textContent = '导出特征图片';
        }
    }
} 