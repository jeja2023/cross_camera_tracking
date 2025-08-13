import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js';
import { showLightboxWithNav } from './common.js';

let currentSearchResults = [];
let currentSearchPage = 1;
const searchItemsPerPage = 10; // 每页显示的项目数
let totalSearchResults = 0; // 这个变量现在表示当前显示查询人物下的总匹配结果数
let currentQueryImage = null; // 用于存储当前查询图片的URL
let currentSearchResultsImages = []; // 存储当前搜索结果图片URL，用于Lightbox导航

// 新增：用于管理查询人物切换的状态变量
let allGroupedSearchResults = []; // 存储所有查询人物的分组结果
let currentQueryPersonIndex = 0; // 当前显示的查询人物的索引
const maxQueryPersonsToShow = 5; // 最多显示5个检索人物的结果

export function initImageSearchPage() {
    console.log("Initializing Image Search Page...");
    const queryImageInput = document.getElementById('query-image-input');
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdValueSpan = document.getElementById('threshold-value');
    const searchButton = document.getElementById('search-button');
    const exportSearchResultsButton = document.getElementById('export-search-results-button');
    const searchScopeSelect = document.getElementById('search-scope-select'); // 获取新的统一搜索范围下拉框

    if (queryImageInput) queryImageInput.addEventListener('change', handleImagePreview);
    if (thresholdSlider) thresholdSlider.addEventListener('input', () => { if (thresholdValueSpan) thresholdValueSpan.textContent = thresholdSlider.value; });
    if (searchButton) searchButton.addEventListener('click', () => {
        currentSearchPage = 1; // 每次新搜索时重置页码
        currentQueryPersonIndex = 0; // 每次新搜索时重置查询人物索引
        performSearch(currentSearchPage, searchItemsPerPage);
    });
    if (exportSearchResultsButton) {
        exportSearchResultsButton.addEventListener('click', exportSearchResults);
        exportSearchResultsButton.classList.add('hidden'); // 页面初始化时默认隐藏
    }
    // populateCombinedSearchScopeSelect(); // 移除调用，现在只支持全局搜索

    // 以图搜人分页按钮事件监听 (针对单个查询人物内部的结果分页)
    const prevSearchPageButton = document.getElementById('prev-page-search');
    const nextSearchPageButton = document.getElementById('next-page-search');
    const firstSearchPageButton = document.getElementById('first-page-search');
    const lastSearchPageButton = document.getElementById('last-page-search');

    if (prevSearchPageButton) {
        prevSearchPageButton.addEventListener('click', () => {
            if (currentSearchPage > 1) {
                currentSearchPage--;
                renderQueryPersonResults(); // 重新渲染当前查询人物的结果
            }
        });
    }

    if (nextSearchPageButton) {
        nextSearchPageButton.addEventListener('click', () => {
            // totalSearchResults 现在是当前查询人物下的总匹配结果数
            const totalPages = Math.ceil(totalSearchResults / searchItemsPerPage);
            if (currentSearchPage < totalPages) {
                currentSearchPage++;
                renderQueryPersonResults(); // 重新渲染当前查询人物的结果
            }
        });
    }

    if (firstSearchPageButton) {
        firstSearchPageButton.addEventListener('click', () => {
            if (currentSearchPage !== 1) {
                currentSearchPage = 1;
                renderQueryPersonResults();
            }
        });
    }
    if (lastSearchPageButton) {
        lastSearchPageButton.addEventListener('click', () => {
            const totalPages = Math.ceil(totalSearchResults / searchItemsPerPage);
            if (currentSearchPage !== totalPages) {
                currentSearchPage = totalPages;
                renderQueryPersonResults();
            }
        });
    }

    // 新增：查询人物切换按钮事件监听
    const prevQueryPersonButton = document.getElementById('prev-query-person');
    const nextQueryPersonButton = document.getElementById('next-query-person');

    if (prevQueryPersonButton) {
        prevQueryPersonButton.addEventListener('click', () => {
            if (currentQueryPersonIndex > 0) {
                currentQueryPersonIndex--;
                currentSearchPage = 1; // 切换查询人物时，重置其内部结果分页为第一页
                renderQueryPersonResults();
            }
        });
    }

    if (nextQueryPersonButton) {
        nextQueryPersonButton.addEventListener('click', () => {
            if (currentQueryPersonIndex < allGroupedSearchResults.length - 1) {
                currentQueryPersonIndex++;
                currentSearchPage = 1; // 切换查询人物时，重置其内部结果分页为第一页
                renderQueryPersonResults();
            }
        });
    }
}

function handleImagePreview() {
    const queryImageInput = document.getElementById('query-image-input');
    const imagePreview = document.getElementById('image-preview');
    if (queryImageInput && imagePreview) {
        const file = queryImageInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        } else {
            imagePreview.src = '';
            imagePreview.classList.add('hidden');
        }
    }
}

// 新增：上传查询图片函数
async function uploadQueryImage(queryImageFile) {
    const uploadFormData = new FormData();
    uploadFormData.append('file', queryImageFile);

    try {
        const response = await fetchWithAuth('/api/image_analysis/upload_image', { // 修改为图片解析的上传接口
            method: 'POST',
            body: uploadFormData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '图片上传失败');
        }

        const data = await response.json();
        return data.task_id; // 返回任务 ID
    } catch (error) {
        console.error('上传图片错误:', error);
        throw error;
    }
}

async function performSearch(page, limit) {
    const queryImageInput = document.getElementById('query-image-input');
    const queryImageFile = queryImageInput ? queryImageInput.files[0] : null;
    const thresholdSlider = document.getElementById('threshold-slider');
    // const searchScopeSelect = document.getElementById('search-scope-select'); // 移除对 searchScopeSelect 的引用

    const threshold = parseFloat(thresholdSlider.value) / 100.0;
    
    let videoUuid = null; // 强制为 null，只进行全局搜索
    let streamUuid = null; // 强制为 null，只进行全局搜索

    // 移除根据 searchScopeSelect.value 设置 videoUuid 和 streamUuid 的逻辑
    // const selectedScope = searchScopeSelect.value;
    // if (selectedScope !== 'all') {
    //     if (selectedScope.startsWith('video_')) {
    //         videoUuid = selectedScope.substring(6);
    //     } else if (selectedScope.startsWith('stream_')) {
    //         streamUuid = selectedScope.substring(7);
    //     }
    // }

    const searchResultsContainer = document.getElementById('results-container');
    const loadingIndicator = document.getElementById('loading-indicator');
    const noSearchResultsMessage = document.getElementById('no-results-message');
    const paginationControls = document.getElementById('pagination-controls-search');
    const queryPersonNav = document.getElementById('query-person-nav'); // 获取查询人物导航

    if (!queryImageFile) {
        alert('请选择一张图片进行搜索。');
        return;
    }

    showLoading('正在上传并解析图片...');
    let task_id = null;
    let query_person_uuids_list = []; // 用于存储解析出的人物UUID列表
    let query_image_path_for_search = null; // 用于存储原始图片路径以备日志或显示

    try {
        task_id = await uploadQueryImage(queryImageFile);
        const analysisResult = await pollImageAnalysisTask(task_id);

        if (analysisResult && analysisResult.original_image_info) {
            query_image_path_for_search = analysisResult.original_image_info.full_frame_image_path;
            if (analysisResult.analyzed_persons && analysisResult.analyzed_persons.length > 0) {
                query_person_uuids_list = analysisResult.analyzed_persons.map(person => person.uuid);
            } else {
                alert('图片中未检测到有效人物特征，无法进行搜索。请确保上传的图片包含清晰的人物图像。');
                hideLoading();
                return;
            }
        } else {
            alert('图片解析失败或未返回有效结果。');
            hideLoading();
            return;
        }
    } catch (error) {
        console.error('图片上传或解析失败:', error);
        alert('图片上传或解析失败，请重试。');
        hideLoading();
        return;
    }

    const searchRequest = {
        threshold: threshold,
        skip: 0, // 首次搜索不进行分页，后端返回所有分组结果
        limit: 100, // 首次搜索获取足够多的结果，后续前端再处理 (设置为足够大的值，以便获取所有分组)
        query_person_uuid: query_person_uuids_list, 
        query_image_path: query_image_path_for_search
    };

    if (videoUuid) {
        searchRequest.video_uuid = videoUuid;
    }
    if (streamUuid) {
        searchRequest.stream_uuid = streamUuid;
    }

    if (loadingIndicator) loadingIndicator.classList.remove('hidden');
    if (searchResultsContainer) searchResultsContainer.innerHTML = '';
    if (noSearchResultsMessage) noSearchResultsMessage.classList.add('hidden');
    if (paginationControls) paginationControls.classList.add('hidden');
    if (queryPersonNav) queryPersonNav.classList.add('hidden'); // 搜索开始时隐藏人物导航

    console.log("DEBUG: Sending search request to backend.");
    console.log("DEBUG: URL: /api/persons/new_person_search");
    console.log("DEBUG: Method: POST");
    console.log("DEBUG: Body:", JSON.stringify(searchRequest, null, 2));

    try {
        const response = await fetchWithAuth(`/api/persons/new_person_search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(searchRequest),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '图片搜索失败');
        }

        const data = await response.json();
        
        // 存储所有分组结果，并限制数量
        allGroupedSearchResults = data.items.slice(0, maxQueryPersonsToShow);
        
        if (allGroupedSearchResults.length === 0) {
            if (noSearchResultsMessage) noSearchResultsMessage.classList.remove('hidden');
            const exportSearchResultsButton = document.getElementById('export-search-results-button');
            if (exportSearchResultsButton) exportSearchResultsButton.classList.add('hidden');
            if (paginationControls) paginationControls.classList.add('hidden');
            if (queryPersonNav) queryPersonNav.classList.add('hidden');
            return;
        }

        // 显示第一个查询人物的结果
        currentQueryPersonIndex = 0; // 确保从第一个人物开始显示
        currentSearchPage = 1; // 确保从第一页开始显示
        renderQueryPersonResults();

        // 显示查询人物导航
        if (queryPersonNav) queryPersonNav.classList.remove('hidden');

    } catch (error) {
        console.error('搜索错误:', error);
        if (searchResultsContainer) searchResultsContainer.innerHTML = `<p style="color: red; text-align: center;">搜索失败: ${error.message}</p>`;
    } finally {
        if (loadingIndicator) loadingIndicator.classList.add('hidden');
    }
}

// 替换原有的 displayResultsGrouped 函数
function renderQueryPersonResults() {
    const searchResultsContainer = document.getElementById('results-container');
    const noResultsMessageElement = document.getElementById('no-results-message');
    const paginationControls = document.getElementById('pagination-controls-search');
    const currentPageSpan = document.getElementById('current-page-search');
    const totalPagesSpan = document.getElementById('total-pages-search');
    const prevQueryPersonButton = document.getElementById('prev-query-person');
    const nextQueryPersonButton = document.getElementById('next-query-person');
    const currentQueryPersonInfoSpan = document.getElementById('current-query-person-info');
    const queryPersonNav = document.getElementById('query-person-nav');

    if (searchResultsContainer) searchResultsContainer.innerHTML = '';
    if (noResultsMessageElement) noResultsMessageElement.classList.add('hidden');
    if (paginationControls) paginationControls.classList.add('hidden');

    if (!allGroupedSearchResults || allGroupedSearchResults.length === 0) {
        if (noResultsMessageElement) noResultsMessageElement.classList.remove('hidden');
        const exportSearchResultsButton = document.getElementById('export-search-results-button');
        if (exportSearchResultsButton) exportSearchResultsButton.classList.add('hidden');
        if (queryPersonNav) queryPersonNav.classList.add('hidden');
        return;
    }

    // 获取当前要显示的查询人物的分组结果
    const currentGroupedItem = allGroupedSearchResults[currentQueryPersonIndex];
    if (!currentGroupedItem) {
        console.warn(`renderQueryPersonResults: 索引 ${currentQueryPersonIndex} 处的查询人物数据不存在。`);
        if (noResultsMessageElement) noResultsMessageElement.classList.remove('hidden');
        if (queryPersonNav) queryPersonNav.classList.add('hidden');
        return;
    }

    // 更新查询人物导航信息
    if (currentQueryPersonInfoSpan) {
        currentQueryPersonInfoSpan.textContent = `查询人物 ${currentQueryPersonIndex + 1} / ${allGroupedSearchResults.length}`;
    }

    // 控制查询人物切换按钮的禁用状态
    if (prevQueryPersonButton) prevQueryPersonButton.disabled = (currentQueryPersonIndex === 0);
    if (nextQueryPersonButton) nextQueryPersonButton.disabled = (currentQueryPersonIndex === allGroupedSearchResults.length - 1);
    if (queryPersonNav) queryPersonNav.classList.remove('hidden');

    const queryPersonSection = document.createElement('div');
    queryPersonSection.classList.add('query-person-section');

    let queryPersonImageHtml = '';
    if (currentGroupedItem.query_crop_image_path) {
        queryPersonImageHtml = `<img src="${currentGroupedItem.query_crop_image_path.replace(/\\/g, '/')}" alt="查询人物裁剪图" class="query-person-img">`;
    } else if (currentGroupedItem.query_full_frame_image_path) {
        queryPersonImageHtml = `<img src="${currentGroupedItem.query_full_frame_image_path.replace(/\\/g, '/')}" alt="查询人物全帧图" class="query-person-img">`;
    }

    queryPersonSection.innerHTML = `
        <h3>查询人物 (UUID: ${currentGroupedItem.query_person_uuid})</h3>
        ${queryPersonImageHtml}
        <p>为此人物找到 <strong>${currentGroupedItem.total_results_for_query_person}</strong> 个结果。</p>
        <div class="results-grid"></div>
    `;
    searchResultsContainer.appendChild(queryPersonSection);

    const resultsGrid = queryPersonSection.querySelector('.results-grid');
    if (resultsGrid) {
        // 对当前查询人物的结果进行分页显示
        const startIndex = (currentSearchPage - 1) * searchItemsPerPage;
        const endIndex = startIndex + searchItemsPerPage;
        const paginatedResults = currentGroupedItem.results.slice(startIndex, endIndex);

        totalSearchResults = currentGroupedItem.results.length; // 更新总结果数，用于内部结果分页

        paginatedResults.forEach(item => {
            const resultItem = document.createElement('div');
            resultItem.classList.add('result-item');
            
            let sourceInfo = '';
            if (item.stream_uuid) {
                sourceInfo = `<p><strong>视频流:</strong> ${item.stream_name || 'N/A'} (UUID: ${item.stream_uuid})</p>`;
            } else if (item.video_uuid) {
                sourceInfo = `<p><strong>视频:</strong> ${item.video_filename || 'N/A'} (UUID: ${item.video_uuid})</p>`;
            } else if (item.upload_image_uuid) {
                sourceInfo = `<p><strong>上传图片UUID:</strong> ${item.upload_image_uuid}</p>`;
            }

            let cropImagePath = item.crop_image_path ? item.crop_image_path.replace(/\\/g, '/') : '';
            let fullFrameImagePath = item.full_frame_image_path ? item.full_frame_image_path.replace(/\\/g, '/') : '';

            resultItem.innerHTML = `
                <img src="${cropImagePath}" alt="查询结果" class="feature-img" data-full-image="${fullFrameImagePath || cropImagePath}">
                <p><strong>人物ID:</strong> ${item.uuid}</p>
                <p><strong>相似度:</strong>&nbsp;${item.score.toFixed(2)}%</p>
                ${sourceInfo}
                <p><strong>入库时间:</strong> ${
                    item.timestamp ? new Intl.DateTimeFormat('zh-CN', {
                        year: 'numeric',
                        month: '2-digit',
                        day: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                        hour12: false,
                        timeZone: 'Asia/Shanghai'
                    }).format(new Date(item.timestamp)) : 'N/A'
                }</p>
            `;
            resultsGrid.appendChild(resultItem);
        });

        // 更新单个查询人物内部的结果分页信息
        const totalPages = Math.ceil(totalSearchResults / searchItemsPerPage);
        if (currentPageSpan) currentPageSpan.textContent = currentSearchPage;
        if (totalPagesSpan) totalPagesSpan.textContent = totalPages;
        if (paginationControls) paginationControls.classList.remove('hidden');

        const prevSearchPageButton = document.getElementById('prev-page-search');
        const nextPageButton = document.getElementById('next-page-search');
        const firstSearchPageButton = document.getElementById('first-page-search');
        const lastSearchPageButton = document.getElementById('last-page-search');

        if (firstSearchPageButton) firstSearchPageButton.disabled = (currentSearchPage === 1);
        if (prevSearchPageButton) prevSearchPageButton.disabled = (currentSearchPage === 1);
        if (nextPageButton) nextPageButton.disabled = (currentSearchPage === totalPages || totalSearchResults === 0);
        if (lastSearchPageButton) lastSearchPageButton.disabled = (currentSearchPage === totalPages || totalSearchResults === 0);

    }

    // 收集所有图片的URL，用于Lightbox导航 (现在从所有分组结果中收集)
    currentSearchResultsImages = Array.from(searchResultsContainer.querySelectorAll('.feature-img')).map(img => img.dataset.fullImage);

    // 添加点击事件监听器，显示大图
    searchResultsContainer.addEventListener('click', (event) => {
        const clickedImg = event.target.closest('.feature-img');
        if (clickedImg) {
            const imageUrl = clickedImg.dataset.fullImage;
            const imageIndex = currentSearchResultsImages.indexOf(imageUrl);
            showLightboxWithNav(imageUrl, currentSearchResultsImages, imageIndex);
        }
    });

    const exportSearchResultsButton = document.getElementById('export-search-results-button');
    if (exportSearchResultsButton) exportSearchResultsButton.classList.remove('hidden');
    // paginationControls 的显示与否由 renderQueryPersonResults 内部逻辑控制
}

// Add new function to populate combined search scope dropdown
// async function populateCombinedSearchScopeSelect() {
//     const searchScopeSelect = document.getElementById('search-scope-select');
//     if (!searchScopeSelect) return;

//     // 获取当前用户信息
//     const currentUser = Auth.getUserInfo();

//     try {
//         // 获取上传视频列表（所有用户都可以查看自己的视频列表）
//         const videosResponse = await fetchWithAuth('/videos/?status=completed');
//         if (!videosResponse.ok) {
//             throw new Error('无法加载上传视频列表');
//         }
//         let videosData = {};
//         try {
//             videosData = await videosResponse.json();
//         } catch (e) {
//             console.error("解析视频列表 JSON 失败:", e);
//         }
//         const videos = videosData.items || []; // 修正：从 videosData.items 中获取数组

//         // 清空现有选项并添加默认的“全局搜索”
//         searchScopeSelect.innerHTML = '<option value="all">全局搜索 (所有视频和视频流)</option>';

//         // 添加上传视频选项
//         videos.forEach(video => {
//             if (video.uuid) {
//                 const option = document.createElement('option');
//                 option.value = `video_${video.uuid}`;
//                 option.textContent = `上传视频: ${video.filename || '无名称'} (UUID: ${video.uuid})`;
//                 searchScopeSelect.appendChild(option);
//             }
//         });

//         // 只有管理员和高级用户才加载视频流列表
//         if (currentUser && (currentUser.role === 'admin' || currentUser.role === 'advanced')) {
//             const streamsResponse = await fetchWithAuth('/streams/saved');
//             if (!streamsResponse.ok) {
//                 throw new Error('无法加载视频流列表');
//             }
//             let streamsData = {};
//             try {
//                 streamsData = await streamsResponse.json();
//             } catch (e) {
//                 console.error("解析视频流列表 JSON 失败:", e);
//             }
//             const streams = streamsData.items || []; // 修正：从 streamsData.items 中获取数组

//             // 添加视频流选项
//             streams.forEach(stream => {
//                 if (stream.stream_uuid) {
//                     const option = document.createElement('option');
//                     option.value = `stream_${stream.stream_uuid}`;
//                     option.textContent = `视频流: ${stream.name || '无名称'} (UUID: ${stream.stream_uuid})`;
//                     searchScopeSelect.appendChild(option);
//                 }
//             });
//         } else {
//             console.log("populateCombinedSearchScopeSelect: 非管理员/高级用户，跳过加载视频流列表。");
//         }

//     } catch (error) {
//         console.error('populateCombinedSearchScopeSelect: 加载搜索范围选项失败:', error);
//         // 如果加载失败，仍然保留全局搜索选项
//     }
// }

async function exportSearchResults() {
    const queryImageInput = document.getElementById('query-image-input');
    const thresholdSlider = document.getElementById('threshold-slider');
    // const searchScopeSelect = document.getElementById('search-scope-select'); // 移除对 searchScopeSelect 的引用

    if (!queryImageInput || !queryImageInput.files || queryImageInput.files.length === 0) {
        alert('请选择一张查询图片才能导出。');
        return;
    }

    const file = queryImageInput.files[0];
    const threshold = parseFloat(thresholdSlider.value) / 100.0;
    // const selectedScope = searchScopeSelect.value; // 移除 selectedScope 的使用

    let videoUuid = null; // 强制为 null，只进行全局搜索
    let streamUuid = null; // 强制为 null，只进行全局搜索

    // 移除根据 selectedScope 设置 videoUuid 和 streamUuid 的逻辑
    // if (selectedScope !== 'all') {
    //     if (selectedScope.startsWith('video_')) {
    //         videoUuid = selectedScope.substring(6);
    //     } else if (selectedScope.startsWith('stream_')) {
    //         streamUuid = selectedScope.substring(7);
    //     }
    // }

    const formData = new FormData();
    formData.append('query_image', file);
    formData.append('threshold', threshold);

    if (videoUuid) {
        formData.append('video_uuid', videoUuid);
    }
    if (streamUuid) {
        formData.append('stream_uuid', streamUuid);
    }

    try {
        const response = await fetchWithAuth(`/export/query_results`, {
            method: 'POST',
            body: formData, // 直接发送 FormData
            // Content-Type header is automatically set by FormData
        });

        if (response.status === 204) {
            alert('没有搜索结果可导出。');
            return;
        }

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '导出失败');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;

        // 从响应头中解析文件名并设置给 a.download
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = 'image_search_results.xlsx'; // 默认回退文件名
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename\*?=(?:UTF-8'')?"?([^;\n]*?)"?$/i);
            if (filenameMatch && filenameMatch[1]) {
                filename = decodeURIComponent(filenameMatch[1].replace(/%22/g, '').replace(/^"|"$/g, ''));
            }
        }
        a.download = filename;
        
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        alert('搜索结果导出成功！');

    } catch (error) {
        console.error('导出错误:', error);
        alert(`导出搜索结果失败: ${error.message}`);
    }
} 

async function pollImageAnalysisTask(taskId) {
    const statusElement = document.getElementById('loading-message');
    if (statusElement) statusElement.textContent = '图片解析中...';

    return new Promise((resolve, reject) => {
        const pollInterval = setInterval(async () => {
            try {
                const response = await fetchWithAuth(`/api/image_analysis/tasks/${taskId}`);
                if (!response.ok) {
                    throw new Error(`Failed to fetch task status: ${response.statusText}`);
                }
                const data = await response.json();

                if (statusElement) {
                    statusElement.textContent = `图片解析进度: ${data.progress}% - ${data.message || data.status}`;
                }

                if (data.status === 'SUCCESS') {
                    clearInterval(pollInterval);
                    resolve(data.result); // 返回解析结果
                } else if (data.status === 'FAILED' || data.status === 'REVOKED') {
                    clearInterval(pollInterval);
                    reject(new Error(data.message || '图片解析失败。'));
                }
            } catch (error) {
                clearInterval(pollInterval);
                reject(error);
            }
        }, 2000); // 每2秒轮询一次
    });
}

function showLoading(message) {
    const loadingIndicator = document.getElementById('loading-indicator');
    const loadingMessage = document.getElementById('loading-message');
    if (loadingIndicator) loadingIndicator.classList.remove('hidden');
    if (loadingMessage) loadingMessage.textContent = message;
}

function hideLoading() {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (loadingIndicator) loadingIndicator.classList.add('hidden');
} 