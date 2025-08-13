import { Auth, API_BASE_URL } from './auth.js';
import { initializeAuthenticatedPage } from './common.js';

document.addEventListener('DOMContentLoaded', async () => {
    initializeAuthenticatedPage();

    const historyResultsSection = document.getElementById('history-results-section');
    const historyMessage = document.getElementById('history-message');
    const historyResultsContainer = document.getElementById('history-results-container');
    const queryHistoryButton = document.getElementById('query-history-button');
    const individualNameDisplay = document.getElementById('individual-name-display');
    const lastQueryTimeDisplay = document.getElementById('last-query-time-display'); // 新增

    // Get individual ID and name from URL query parameters
    const urlParams = new URLSearchParams(window.location.search);
    const individualId = urlParams.get('individual_id');
    const individualName = urlParams.get('individual_name');

    if (individualName) {
        individualNameDisplay.textContent = individualName;
    } else {
        individualNameDisplay.textContent = '未知人物';
    }

    let currentSelectedIndividualId = individualId; // Initialize with ID from URL
    let lastQueryTime = localStorage.getItem(`lastQueryTime_${individualId}`) ? new Date(localStorage.getItem(`lastQueryTime_${individualId}`)) : null; // 为每个 individualId 存储独立的时间戳

    let allDisplayedResults = []; // Store all currently displayed search results
    let currentModalImageIndex = 0; // Track the index of the image currently displayed in the modal

    // Pagination variables
    let currentPage = 1;
    const itemsPerPage = 20; // You can adjust this value as needed
    let totalResults = 0; // Total count of results from the backend
    let totalPages = 1;

    // Function to update the last query time display
    function updateLastQueryTimeDisplay() {
        if (lastQueryTime) {
            lastQueryTimeDisplay.textContent = `最近查询时间: ${lastQueryTime.toLocaleString()}`;
        } else {
            lastQueryTimeDisplay.textContent = `最近查询时间: 无`;
        }
    }

    async function fetchGlobalSearchResults(individualId, isInitialSearch = null, queryTime = null, skip = 0, limit = itemsPerPage) {
        const token = Auth.getToken();
        if (!token) {
            console.error("Auth token not found.");
            alert("认证失败，请重新登录。");
            return null;
        }

        try {
            let url = `/followed_persons/${individualId}/global_search_results?min_confidence=0.9&skip=${skip}&limit=${limit}`;
            if (isInitialSearch !== null) {
                url += `&is_initial_search=${isInitialSearch}`;
            }
            if (queryTime) {
                url += `&last_query_time=${queryTime.toISOString()}`; // 将时间戳作为 ISO 格式传递
            }

            const response = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                if (response.status === 401) {
                    alert("会话过期，请重新登录。");
                    window.location.href = '/login';
                    return null;
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            console.log(`Fetched global search results for individual ${individualId}:`, data);
            
            return data;

        } catch (error) {
            console.error("Error fetching global search results:", error);
            historyMessage.textContent = `获取历史轨迹失败: ${error.message}`;
            historyResultsSection.classList.remove('hidden');
            return null;
        }
    }

    function renderGlobalSearchResults(individualId) {
        historyResultsContainer.innerHTML = ''; // Always clear and re-render all displayed results
        historyMessage.textContent = '';
        historyResultsSection.classList.remove('hidden'); // Ensure section is visible

        if (allDisplayedResults.length === 0) {
            historyMessage.textContent = `人物 ${individualNameDisplay.textContent} 没有找到相关历史轨迹结果。`;
            return;
        }

        historyMessage.textContent = `人物 ${individualNameDisplay.textContent} 找到 ${totalResults} 条历史轨迹结果 (置信度 > 90%):`;
        
        // Update pagination totals
        totalPages = Math.ceil(totalResults / itemsPerPage);
        updatePaginationControls();

        // Render only the items for the current page from allDisplayedResults
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = Math.min(startIndex + itemsPerPage, allDisplayedResults.length);
        const itemsToRender = allDisplayedResults.slice(startIndex, endIndex);

        itemsToRender.forEach((item, index) => {
            const imageGridItem = document.createElement('div');
            imageGridItem.classList.add('image-grid-item');

            const img = document.createElement('img');
            img.src = item.matched_image_path;
            img.alt = `匹配图片 (置信度: ${(item.confidence * 100).toFixed(2)}%)`;
            img.title = `人物UUID: ${item.matched_person_uuid}\n置信度: ${(item.confidence * 100).toFixed(2)}%\n搜索时间: ${new Date(item.search_time).toLocaleString()}`;
            // Use full_frame_image_path if available, otherwise fallback to matched_image_path (cropped)
            img.dataset.fullFramePath = (item.person && item.person.full_frame_image_path) ? item.person.full_frame_image_path : item.matched_image_path; 
            img.dataset.index = index; // Store its index in the allDisplayedResults array

            img.addEventListener('click', () => {
                openImageModal(index);
            });

            const confidenceSpan = document.createElement('span');
            confidenceSpan.classList.add('confidence-score');
            confidenceSpan.textContent = `${(item.confidence * 100).toFixed(2)}%`;

            imageGridItem.appendChild(img);
            imageGridItem.appendChild(confidenceSpan);
            historyResultsContainer.appendChild(imageGridItem);
        });
    }

    const imageModal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');
    const closeButton = document.querySelector('.close-button');
    const prevButton = document.getElementById('prev-button');
    const nextButton = document.getElementById('next-button');
    const imageInfo = document.getElementById('image-info');

    function openImageModal(index) {
        currentModalImageIndex = index;
        updateModalImage();
        imageModal.style.display = 'block';
    }

    function closeImageModal() {
        imageModal.style.display = 'none';
    }

    function updateModalImage() {
        if (allDisplayedResults.length === 0) return;

        const item = allDisplayedResults[currentModalImageIndex];
        // Use full_frame_image_path if available, otherwise fallback to matched_image_path
        modalImage.src = (item.person && item.person.full_frame_image_path) ? item.person.full_frame_image_path : item.matched_image_path;
        imageInfo.textContent = `人物UUID: ${item.matched_person_uuid} | 置信度: ${(item.confidence * 100).toFixed(2)}% | 搜索时间: ${new Date(item.search_time).toLocaleString()}`;

        prevButton.disabled = currentModalImageIndex === 0;
        nextButton.disabled = currentModalImageIndex === allDisplayedResults.length - 1;
    }

    // Modal event listeners
    closeButton.addEventListener('click', closeImageModal);
    prevButton.addEventListener('click', () => {
        if (currentModalImageIndex > 0) {
            currentModalImageIndex--;
            updateModalImage();
        }
    });
    nextButton.addEventListener('click', () => {
        if (currentModalImageIndex < allDisplayedResults.length - 1) {
            currentModalImageIndex++;
            updateModalImage();
        }
    });

    window.addEventListener('click', (event) => {
        if (event.target === imageModal) {
            closeImageModal();
        }
    });

    const prevPageButton = document.getElementById('prev-page-button');
    const nextPageButton = document.getElementById('next-page-button');
    const firstPageButton = document.getElementById('first-page-button'); // 新增
    const lastPageButton = document.getElementById('last-page-button');   // 新增
    const pageInfoSpan = document.getElementById('page-info');

    function updatePaginationControls() {
        pageInfoSpan.textContent = `页数: ${currentPage} / ${totalPages}`;
        prevPageButton.disabled = currentPage === 1;
        nextPageButton.disabled = currentPage === totalPages;
        firstPageButton.disabled = currentPage === 1; // 新增
        lastPageButton.disabled = currentPage === totalPages; // 新增
    }

    prevPageButton.addEventListener('click', async () => {
        if (currentPage > 1) {
            currentPage--;
            await performSearchAndRender(false); // Only re-render from existing data
        }
    });

    nextPageButton.addEventListener('click', async () => {
        if (currentPage < totalPages) {
            currentPage++;
            await performSearchAndRender(false); // Only re-render from existing data
        }
    });

    // 新增首页和尾页按钮事件监听器
    firstPageButton.addEventListener('click', async () => {
        if (currentPage !== 1) {
            currentPage = 1;
            await performSearchAndRender(false); // Only re-render from existing data
        }
    });

    lastPageButton.addEventListener('click', async () => {
        if (currentPage !== totalPages) {
            currentPage = totalPages;
            await performSearchAndRender(false); // Only re-render from existing data
        }
    });

    async function performSearchAndRender(triggerNewFetch = false, isInitialLoad = false, append = false) {
        if (triggerNewFetch) {
            // Only clear and fetch if a new search is explicitly triggered
            if (isInitialLoad || !append) { // For initial load or full re-query
                allDisplayedResults = []; 
                currentPage = 1;
            } else if (append) { // For incremental search, just reset page to 1 to see new data first
                currentPage = 1;
            }

            historyMessage.textContent = '正在获取历史轨迹...';
            
            const globalSearchResults = await fetchGlobalSearchResults(
                currentSelectedIndividualId, 
                isInitialLoad ? null : false, // is_initial_search parameter for backend (null for initial load to get all data, false for subsequent)
                isInitialLoad ? null : lastQueryTime, // lastQueryTime only for incremental
                0, // Always fetch from start (skip=0) when a new fetch is triggered, as we's managing overall data
                Number.MAX_SAFE_INTEGER // Fetch all available new/initial data
            );

            if (globalSearchResults) {
                // Concatenate new results with existing ones
                let combinedResults = allDisplayedResults.concat(globalSearchResults.items);

                // Deduplicate based on matched_person_uuid (keep the latest if duplicates exist)
                const seenUuids = new Set();
                const uniqueResults = [];
                // Sort by search_time descending to ensure the latest duplicate is kept
                combinedResults.sort((a, b) => new Date(b.search_time).getTime() - new Date(a.search_time).getTime());

                for (const item of combinedResults) {
                    if (!seenUuids.has(item.matched_person_uuid)) {
                        seenUuids.add(item.matched_person_uuid);
                        uniqueResults.push(item);
                    }
                }
                
                allDisplayedResults = uniqueResults;
                totalResults = uniqueResults.length; // Total results after deduplication
            } else {
                totalResults = 0; // If fetch fails, reset total
            }
        }
        
        // Always re-render based on current page and accumulated results
        renderGlobalSearchResults(currentSelectedIndividualId);
    }

    queryHistoryButton.addEventListener('click', async () => {
        if (currentSelectedIndividualId) {
            historyMessage.textContent = '正在触发全局搜索比对任务...';
            queryHistoryButton.disabled = true; // Disable button while task is being triggered

            const token = Auth.getToken();
            if (!token) {
                console.error("Auth token not found.");
                alert("认证失败，请重新登录。");
                queryHistoryButton.disabled = false;
                return;
            }

            try {
                const response = await fetch(`/followed_persons/${currentSelectedIndividualId}/trigger_global_search`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    }
                });

                if (!response.ok) {
                    if (response.status === 401) {
                        alert("会话过期，请重新登录。");
                        window.location.href = '/login';
                        return;
                    }
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();
                alert(result.message);

                // Determine the last search time from currently displayed results
                // This ensures incremental fetches are based on what's already loaded
                let latestExistingSearchTime = null;
                if (allDisplayedResults.length > 0) {
                    // Iterate to find the truly latest search time from all loaded results
                    latestExistingSearchTime = allDisplayedResults.reduce((maxTime, item) => {
                        const itemTime = new Date(item.search_time);
                        return maxTime === null || itemTime > maxTime ? itemTime : maxTime;
                    }, null);
                }

                // Store this for the *next* incremental query.
                // For the *current* fetch, we will pass this to the backend implicitly via `lastQueryTime` var.
                lastQueryTime = latestExistingSearchTime ? new Date(latestExistingSearchTime) : null;
                localStorage.setItem(`lastQueryTime_${currentSelectedIndividualId}`, lastQueryTime ? lastQueryTime.toISOString() : null);
                updateLastQueryTimeDisplay(); // Update display with this derived time

                // Now fetch and append the new incremental results
                currentPage = 1; // Reset to first page to show new results from top
                await performSearchAndRender(true, false, true); // Trigger new fetch, not initial, append

            } catch (error) {
                console.error("Error triggering global search:", error);
                historyMessage.textContent = `触发全局搜索失败: ${error.message}`;
                alert(`触发全局搜索失败: ${error.message}`);
            } finally {
                queryHistoryButton.disabled = false; // Re-enable button
            }
        } else {
            alert("无法获取人物档案ID，请从关注人员列表页进入。无需关注的也可以手动查询");
        }
    });

    // Initial fetch when the page loads
    if (individualId) {
        // 页面首次加载时，执行全量查询，不使用 lastQueryTime
        // 调用 performSearchAndRender，以初始加载模式获取所有数据，不清空 allDisplayedResults
        await performSearchAndRender(true, true, false); // Trigger new fetch, is initial, no append
        updateLastQueryTimeDisplay(); // 页面加载时更新最近查询时间显示

        // After initial load, if there are results, set lastQueryTime to the latest search_time among them
        if (allDisplayedResults.length > 0) {
            const latestTime = allDisplayedResults.reduce((maxTime, item) => {
                const itemTime = new Date(item.search_time);
                return maxTime === null || itemTime > maxTime ? itemTime : maxTime;
            }, null);
            lastQueryTime = latestTime; // Update the global lastQueryTime variable
            localStorage.setItem(`lastQueryTime_${currentSelectedIndividualId}`, lastQueryTime ? lastQueryTime.toISOString() : null);
            updateLastQueryTimeDisplay(); // Update display again to reflect this derived time
        }
    } else {
        historyMessage.textContent = '请从关注人员列表页面选择人物查看历史轨迹。';
        updateLastQueryTimeDisplay(); // 如果无ID，也显示无最近查询时间
    }
});