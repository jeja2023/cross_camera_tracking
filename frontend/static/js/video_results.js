import { fetchWithAuth, API_BASE_URL } from './auth.js';
import { showLightbox, showLightboxWithNav } from './common.js';

let currentVideoId = null;
let currentPage = 1;
const itemsPerPage = 20; // 默认每页显示20条
let totalPersons = 0;
let currentImages = []; // 用于存储当前页面的所有图片URL

export function initVideoResultsPage() {
    console.log("Initializing Video Results Page...");
    const urlParams = new URLSearchParams(window.location.search);
    const videoId = urlParams.get('videoId');
    console.log('从 URL 获取到的 videoId:', videoId); // Debug log
    currentVideoId = videoId; // 保存当前视频ID

    const backToVideosButton = document.getElementById('back-to-videos-button');
    const prevPageButton = document.getElementById('prev-page-video'); // 新增：视频结果分页按钮
    const nextPageButton = document.getElementById('next-page-video'); // 新增：视频结果分页按钮
    const firstPageButton = document.getElementById('first-page-video');
    const lastPageButton = document.getElementById('last-page-video');

    if (backToVideosButton) {
        backToVideosButton.addEventListener('click', () => {
            window.location.href = '/video_analysis'; // 返回视频列表页面
        });
    }

    if (firstPageButton) {
        firstPageButton.addEventListener('click', () => {
            if (currentPage !== 1) {
                currentPage = 1;
                loadVideoPersons(currentVideoId, currentPage, itemsPerPage);
            }
        });
    }

    if (prevPageButton) {
        prevPageButton.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                loadVideoPersons(currentVideoId, currentPage, itemsPerPage);
            }
        });
    }

    if (nextPageButton) {
        nextPageButton.addEventListener('click', () => {
            const totalPages = Math.ceil(totalPersons / itemsPerPage);
            if (currentPage < totalPages) {
                currentPage++;
                loadVideoPersons(currentVideoId, currentPage, itemsPerPage);
            }
        });
    }

    if (lastPageButton) {
        lastPageButton.addEventListener('click', () => {
            const totalPages = Math.ceil(totalPersons / itemsPerPage);
            if (currentPage !== totalPages) {
                currentPage = totalPages;
                loadVideoPersons(currentVideoId, currentPage, itemsPerPage);
            }
        });
    }

    if (!isNaN(videoId) && videoId > 0) {
        console.log("Valid videoId, attempting to load persons and attach export listener.");
        loadVideoPersons(videoId, currentPage, itemsPerPage);
        const exportVideoResultsButton = document.getElementById('export-results-button'); // 确保ID正确
        if (exportVideoResultsButton) {
            console.log("Export button found, attaching listener.");
            exportVideoResultsButton.addEventListener('click', () => {
                console.log("Export button clicked.");
                exportVideoResults(videoId);
            });
        } else {
            console.log("Export button not found.");
        }
    } else {
        // 当video_id无效时，不加载数据，直接显示提示信息并隐藏相关元素
        document.getElementById('video-persons-loading').classList.add('hidden');
        document.getElementById('no-video-persons-message').classList.remove('hidden');
        document.getElementById('video-results-title').textContent = '请从视频解析页面选择一个视频查看结果';
        const exportVideoResultsButton = document.getElementById('export-video-results-button');
        if (exportVideoResultsButton) exportVideoResultsButton.classList.add('hidden');
        // 隐藏分页控件
        const paginationControls = document.getElementById('pagination-controls-video');
        if (paginationControls) paginationControls.classList.add('hidden');
    }
}

async function loadVideoPersons(videoId, page, limit) {
    const videoPersonsContainer = document.getElementById('video-persons-container');
    const videoPersonsLoading = document.getElementById('video-persons-loading');
    const noVideoPersonsMessage = document.getElementById('no-video-persons-message');
    const videoResultsTitle = document.getElementById('video-results-title');
    const exportVideoResultsButton = document.getElementById('export-video-results-button');
    const paginationControls = document.getElementById('pagination-controls-video'); // 新增：视频结果分页控件
    const currentPageSpan = document.getElementById('current-page-video'); // 新增：视频结果当前页码
    const totalPagesSpan = document.getElementById('total-pages-video'); // 新增：视频结果总页码

    if (videoPersonsContainer) videoPersonsContainer.innerHTML = '';
    if (videoPersonsLoading) videoPersonsLoading.classList.remove('hidden');
    if (noVideoPersonsMessage) noVideoPersonsMessage.classList.add('hidden');
    if (videoResultsTitle) videoResultsTitle.textContent = `视频 ID: ${videoId} 的人物结果`;
    if (exportVideoResultsButton) exportVideoResultsButton.classList.add('hidden');
    if (paginationControls) paginationControls.classList.add('hidden'); // 默认隐藏分页控件

    const skip = (page - 1) * limit;

    try {
        console.log(`正在请求视频人物数据: /videos/${videoId}/persons?skip=${skip}&limit=${limit}`); // Debug log
        const response = await fetchWithAuth(`/videos/${videoId}/persons?skip=${skip}&limit=${limit}`);
        if (!response.ok) {
            throw new Error('无法加载视频人物数据');
        }
        const data = await response.json(); // 后端现在返回PaginatedPersonsResponse
        console.log('后端返回的视频人物数据:', data); // Debug log
        const persons = data.items;
        totalPersons = data.total;

        if (videoPersonsContainer) {
            if (persons.length === 0) {
                if (noVideoPersonsMessage) noVideoPersonsMessage.classList.remove('hidden');
                if (exportVideoResultsButton) exportVideoResultsButton.classList.add('hidden');
                currentImages = []; // 清空图片列表
            } else {
                // 收集当前页面的所有图片URL，用于Lightbox导航
                currentImages = persons.map(person => {
                    let fullFramePath = person.full_frame_image_path ? API_BASE_URL + '/' + person.full_frame_image_path.replace(/\\/g, '/') : '';
                    let cropPath = person.crop_image_path ? API_BASE_URL + '/' + person.crop_image_path.replace(/\\/g, '/') : '';
                    return fullFramePath || cropPath;
                });

                persons.forEach(person => {
                    const personItem = document.createElement('div');
                    personItem.classList.add('result-item');
                    
                    let cropImagePath = person.crop_image_path ? API_BASE_URL + '/' + person.crop_image_path.replace(/\\/g, '/') : '';
                    let fullFrameImagePath = person.full_frame_image_path ? API_BASE_URL + '/' + person.full_frame_image_path.replace(/\\/g, '/') : '';
                    
                    personItem.innerHTML = `
                        <img src="${cropImagePath}" alt="人物裁剪" data-full-size="${fullFrameImagePath || cropImagePath}">
                        <p><strong>人物ID:</strong> ${person.uuid}</p>
                        <p><strong>视频ID:</strong> ${person.video_id}</p>
                        <p><strong>入库时间:</strong> ${
                            person.created_at ? new Intl.DateTimeFormat('zh-CN', {
                                year: 'numeric',
                                month: '2-digit',
                                day: '2-digit',
                                hour: '2-digit',
                                minute: '2-digit',
                                second: '2-digit',
                                hour12: false,
                                timeZone: 'Asia/Shanghai'
                            }).format(new Date(person.created_at)) : 'N/A'
                          }</p>
                    `;
                    personItem.querySelector('img').addEventListener('click', (e) => {
                        const clickedImageUrl = e.target.dataset.fullSize;
                        const imageIndex = currentImages.indexOf(clickedImageUrl);
                        showLightboxWithNav(clickedImageUrl, currentImages, imageIndex);
                    });
                    videoPersonsContainer.appendChild(personItem);
                });

                // 显示导出按钮
                if (exportVideoResultsButton) exportVideoResultsButton.classList.remove('hidden');
                // 更新分页信息和显示分页控件
                const totalPages = Math.ceil(totalPersons / itemsPerPage);
                if (currentPageSpan) currentPageSpan.textContent = currentPage;
                if (totalPagesSpan) totalPagesSpan.textContent = totalPages;
                if (paginationControls) paginationControls.classList.remove('hidden');

                // 更新分页按钮状态
                const prevPageButton = document.getElementById('prev-page-video');
                const nextPageButton = document.getElementById('next-page-video');
                const firstPageButton = document.getElementById('first-page-video');
                const lastPageButton = document.getElementById('last-page-video');

                if (firstPageButton) firstPageButton.disabled = (currentPage === 1);
                if (prevPageButton) prevPageButton.disabled = (currentPage === 1);
                if (nextPageButton) nextPageButton.disabled = (currentPage === totalPages);
                if (lastPageButton) lastPageButton.disabled = (currentPage === totalPages);
            }
        }
    } catch (error) {
        console.error('加载视频人物失败:', error);
        if (videoPersonsContainer) videoPersonsContainer.innerHTML = `<p style="color: red; text-align: center;">加载人物数据失败: ${error.message}</p>`;
        if (exportVideoResultsButton) exportVideoResultsButton.classList.add('hidden');
    } finally {
        if (videoPersonsLoading) videoPersonsLoading.classList.add('hidden');
    }
}

async function exportVideoResults(videoId) {
    console.log(`Attempting to export results for videoId: ${videoId}`);
    const videoPersonsContainer = document.getElementById('video-persons-container');
    if (!videoPersonsContainer || videoPersonsContainer.children.length === 0) {
        alert('没有视频人物结果可导出。');
        console.log("No video persons results to export.");
        return;
    }

    const results = Array.from(videoPersonsContainer.children).map(item => {
        const img = item.querySelector('img');
        const pTags = item.querySelectorAll('p');
        return {
            crop_url: img ? img.src : '',
            person_id: pTags[0] ? pTags[0].textContent.replace('人物ID: ', '') : '',
            time_range: pTags[1] ? pTags[1].textContent.replace('出现时间: ', '') : '',
        };
    });

    try {
        console.log(`Sending export request to /export/video_results/${videoId}`);
        const response = await fetchWithAuth(`/export/video_results/${videoId}`, {
            method: 'GET',
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error("Export failed on backend:", errorData);
            throw new Error(errorData.detail || '导出失败');
        }

        const blob = await response.blob();
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `视频_${videoId}_结果_${new Date().toISOString().slice(0,10).replace(/-/g,"")}.xlsx`; // Fallback filename

        if (contentDisposition) {
            const filenameRegex = /filename\*=UTF-8\'\'(.*)/;
            const matches = contentDisposition.match(filenameRegex);
            if (matches && matches[1]) {
                try {
                    filename = decodeURIComponent(matches[1]);
                } catch (e) {
                    console.warn("Failed to decode URI component from filename*", e);
                }
            } else {
                // Fallback for older browsers or if filename* is not present
                const asciiFilenameRegex = /filename="([^"]*)"/;
                const asciiMatches = contentDisposition.match(asciiFilenameRegex);
                if (asciiMatches && asciiMatches[1]) {
                    filename = asciiMatches[1];
                }
            }
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename; // Use the dynamically determined filename
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        alert('视频人物结果导出成功！');
        console.log("Export successful, file download initiated.");

    } catch (error) {
        console.error('导出错误:', error);
        alert(`导出视频人物结果失败: ${error.message}`);
    }
}