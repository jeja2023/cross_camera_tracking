import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js';
import { showLightboxWithNav, initializeAuthenticatedPage, setupImageGalleryClickEvents } from './common.js'; // Import setupImageGalleryClickEvents

let newFeaturesInterval = null;
let streamPingInterval = null;
let currentStreamId = null;
let currentPageLive = 1; // 新增：实时流结果当前页码
const itemsPerPageLive = 20; // 新增：实时流结果每页显示数量
let totalFeaturesLive = 0; // 新增：实时流结果总特征数

let liveFeedImgElement = null; // 用于存储<img>元素引用 (实时流)
let savedFeedVideoElement = null; // 用于存储<video>元素引用 (已保存视频)
let stopStreamButton = null; // 新增：用于停止流按钮引用
let exportResultsButton = null; // 新增：用于导出结果按钮引用
let isStoppingStream = false; // 新增：防止重复点击停止按钮
let displayedFeatureUUIDs = new Set(); // 新增：存储已显示的特征UUID
// allFeaturesImages 不再是全局存储所有特征，而是在每次加载/更新时基于当前DOM重建，以支持Lightbox导航
let allFeaturesImages = []; // 用于Lightbox导航，每次更新时重新构建

// 初始化实时流结果页面的主函数
export async function initLiveStreamResultsPage() {
    // alert("Alert from initLiveStreamResultsPage!"); // 移除弹窗
    console.log("[initLiveStreamResultsPage] Initializing Live Stream Results Page...");
    console.log(`[initLiveStreamResultsPage] API_BASE_URL: ${API_BASE_URL}`);

    // 1. 权限检查：确保只有管理员或高级用户可以访问此页面
    const userInfo = Auth.getUserInfo();
    if (!userInfo || (userInfo.role !== 'admin' && userInfo.role !== 'advanced')) {
        document.body.innerHTML = '<h1>权限不足</h1><p>只有管理员或高级用户才能访问此页面。</p><a href="/">返回主页</a>';
        return;
    }

    // 2. 从URL中获取stream_uuid
    // 这是关键一步，实时流的ID通过URL参数传递过来
    const urlParams = new URLSearchParams(window.location.search);
    currentStreamId = urlParams.get('streamId');

    console.log(`[initLiveStreamResultsPage] URL查询字符串: ${window.location.search}`);
    console.log(`[initLiveStreamResultsPage] 提取到的 Stream ID: ${currentStreamId}`);

    // 获取视频元素和停止按钮
    liveFeedImgElement = document.getElementById('live-video-feed');
    savedFeedVideoElement = document.getElementById('saved-video-feed');
    stopStreamButton = document.getElementById('stop-stream-button');
    exportResultsButton = document.getElementById('export-results-button'); // 获取导出按钮引用

    // 3. 检查stream_id是否存在
    // 如果URL中没有stream_id，则无法加载实时流结果，并给出提示
    if (!currentStreamId) {
        console.warn("[initLiveStreamResultsPage] URL中缺少stream_id。无法显示实时流结果。");
        // 确保live-stream-results div存在，因为它在HTML中被添加了
        const resultsDiv = document.getElementById('live-stream-results');
        if (resultsDiv) {
            resultsDiv.innerHTML = '<p>URL中缺少实时流ID。请通过视频流页面启动流。</p>';
        } else {
            console.error('[initLiveStreamResultsPage] HTML中缺少 ID 为 live-stream-results 的元素。');
        }
        // 如果没有stream_id，则隐藏视频播放器和停止按钮
        if (liveFeedImgElement) liveFeedImgElement.style.display = 'none';
        if (savedFeedVideoElement) savedFeedVideoElement.style.display = 'none';
        if (stopStreamButton) stopStreamButton.style.display = 'none';
        return;
    }

    // 4. 初始化页面并加载实时流结果
    // 获取流的当前状态以决定播放哪个视频源
    console.log("[initLiveStreamResultsPage] Calling updateVideoStreamSource...");
    const isStreamActive = await updateVideoStreamSource(); // 新增函数调用，负责设置视频源
    console.log(`[initLiveStreamResultsPage] updateVideoStreamSource returned isStreamActive: ${isStreamActive}`);
    loadLiveStreamResults(currentPageLive, itemsPerPageLive); // 首次加载全部特征
    
    // 5. 定时刷新和Ping
    if (isStreamActive) {
        newFeaturesInterval = setInterval(() => { // 重新启用定时刷新
            // 只有在用户查看第一页时才自动刷新，以显示最新特征
            if (currentPageLive === 1) { 
                loadLiveStreamResults(1, itemsPerPageLive, true); // 传递 true 表示是自动刷新
            }
        }, 5000); // 每5秒刷新一次，以获取新特征
        streamPingInterval = setInterval(pingStream, 10000); 
    } else {
        console.log("[initLiveStreamResultsPage] Stream is not active, not starting auto-refresh or pingStream interval.");
    }
    // 6. 停止流按钮事件监听器
    if (stopStreamButton) {
        stopStreamButton.addEventListener('click', async () => {
            if (isStoppingStream) {
                console.log("[initLiveStreamResultsPage] 停止流操作正在进行中，忽略重复点击。");
                return; // 如果已经在处理中，则忽略本次调用
            }
            isStoppingStream = true; // 设置标志为true，表示正在处理

            console.log(`[initLiveStreamResultsPage] Stopping stream ${currentStreamId}...`);
            stopStreamButton.disabled = true; // 点击后立即禁用按钮
            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/streams/stop/${currentStreamId}`, {
                    method: 'POST',
                });
                if (!response.ok) {
                    const errorText = await response.text();
                    console.error("[initLiveStreamResultsPage] 停止流请求失败:", response.status, errorText);
                    // 检查是否是后端返回的，指示流已处于非处理状态的 400 错误
                    if (response.status === 400) {
                        const statusMatch = errorText.match(/视频流当前状态为 '(\w+)'/); // 提取状态信息
                        if (statusMatch && ['stopped', 'completed', 'failed', 'inactive', 'terminated'].includes(statusMatch[1])) {
                            console.log(`[initLiveStreamResultsPage] Stream was already in a non-processing state (${statusMatch[1]}). Redirecting to stream management page.`);
                            alert(`视频流已处于 ${statusMatch[1] === 'stopped' ? '停止' : '非活跃'} 状态。`); // 更合适的提示
                            window.location.href = '/video_stream'; // 即使已停止或非活跃，也重定向
                            return; // 退出，不再抛出错误
                        }
                    }
                    throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
                }

                // SUCCESS PATH: only execute if response.ok is true
                const data = await response.json();
                console.log("[initLiveStreamResultsPage] Stream stopped successfully:", data);
                alert("视频流已成功停止解析！"); // 添加成功提示
                // 停止视频播放，清除定时器，并重定向到视频流管理页面
                if (liveFeedImgElement) liveFeedImgElement.src = "";
                if (savedFeedVideoElement) savedFeedVideoElement.pause(); savedFeedVideoElement.src = "";
                if (newFeaturesInterval) clearInterval(newFeaturesInterval);
                if (streamPingInterval) clearInterval(streamPingInterval);
                window.location.href = '/video_stream'; // 重定向到流管理页面
                return; // EXIT FUNCTION HERE FOR SUCCESS PATH TOO

            } catch (error) {
                console.error("[initLiveStreamResultsPage] Error stopping stream:", error);
                alert(`停止视频流失败: ${error.message}`);
                stopStreamButton.disabled = false; // 出错时重新启用按钮
            } finally {
                isStoppingStream = false; // 无论成功或失败，都重置标志
                // 无论成功或失败，都重新启用按钮 (如果它因为错误而被禁用)
                if (stopStreamButton && stopStreamButton.disabled) {
                    stopStreamButton.disabled = false;
                }
            }
        });
    }

    // 7. 分页按钮事件监听器
    const firstPageButton = document.getElementById('first-page-live');
    const prevPageButton = document.getElementById('prev-page-live');
    const nextPageButton = document.getElementById('next-page-live');
    const lastPageButton = document.getElementById('last-page-live');

    if (firstPageButton) {
        firstPageButton.addEventListener('click', () => {
            if (currentPageLive !== 1) {
                currentPageLive = 1;
                console.log(`[Pagination Click] First Page Button Clicked. currentPageLive: ${currentPageLive}`);
                loadLiveStreamResults(currentPageLive, itemsPerPageLive);
            }
        });
    }

    if (prevPageButton) {
        prevPageButton.addEventListener('click', () => {
            if (currentPageLive > 1) {
                currentPageLive--;
                console.log(`[Pagination Click] Previous Page Button Clicked. currentPageLive: ${currentPageLive}`);
                loadLiveStreamResults(currentPageLive, itemsPerPageLive);
            }
        });
    }

    if (nextPageButton) {
        nextPageButton.addEventListener('click', () => {
            const totalPages = Math.ceil(totalFeaturesLive / itemsPerPageLive);
            if (currentPageLive < totalPages) {
                currentPageLive++;
                console.log(`[Pagination Click] Next Page Button Clicked. currentPageLive: ${currentPageLive}`);
                loadLiveStreamResults(currentPageLive, itemsPerPageLive);
            }
        });
    }

    if (lastPageButton) {
        lastPageButton.addEventListener('click', () => {
            const totalPages = Math.ceil(totalFeaturesLive / itemsPerPageLive);
            if (currentPageLive !== totalPages) {
                currentPageLive = totalPages;
                console.log(`[Pagination Click] Last Page Button Clicked. currentPageLive: ${currentPageLive}`);
                loadLiveStreamResults(currentPageLive, itemsPerPageLive);
            }
        });
    }

    // 8. 导出结果按钮事件监听器
    if (exportResultsButton) {
        // 移除所有现有的点击事件监听器，防止重复绑定
        // 注意: removeEventListener 需要与 addEventListener 时的参数完全一致才能移除。
        // 对于匿名函数，这通常不可行。改为直接设置 onclick 属性来确保唯一性。
        exportResultsButton.onclick = async () => {
            // 显示确认对话框
            const confirmExport = confirm("确定要导出当前视频流的解析结果吗？这可能需要一些时间。");
            if (confirmExport) {
                await exportLiveStreamResults();
            }
        };
    }

    // 9. 页面卸载时清除定时器，防止内存泄漏
    window.addEventListener('beforeunload', () => {
        if (newFeaturesInterval) clearInterval(newFeaturesInterval);
        if (streamPingInterval) clearInterval(streamPingInterval);
        if (liveFeedImgElement) liveFeedImgElement.src = ""; // 清空src来停止显示
        if (savedFeedVideoElement) savedFeedVideoElement.pause(); savedFeedVideoElement.src = ""; // 暂停并清空视频源
    });
}

// 异步函数：加载并显示实时流结果
async function loadLiveStreamResults(page, limit, isAutoRefresh = false) {
    console.log(`[loadLiveStreamResults] 拉取实时流结果, page: ${page}, limit: ${limit}, isAutoRefresh: ${isAutoRefresh}`);

    const resultsDiv = document.getElementById('live-stream-results');
    const newFeaturesContainer = document.getElementById('new-features-container');
    const noNewFeaturesMessage = document.getElementById('no-new-features-message');
    const paginationControls = document.getElementById('pagination-controls-live');
    const currentPageSpan = document.getElementById('current-page-live');
    const totalPagesSpan = document.getElementById('total-pages-live');

    if (!resultsDiv) {
        console.error('[loadLiveStreamResults] Live stream results div not found.');
        return;
    }

    // 每次加载新页面时，如果不是第一页的自动刷新，则清空容器和已显示的UUIDs
    if (!isAutoRefresh || page !== 1) {
        console.log(`[loadLiveStreamResults] 条件满足，清空容器和已显示UUIDs。当前isAutoRefresh: ${isAutoRefresh}, page: ${page}`);
        if (newFeaturesContainer) newFeaturesContainer.innerHTML = '';
        displayedFeatureUUIDs.clear(); // 清空Set
        console.log("[loadLiveStreamResults] displayedFeatureUUIDs cleared.");
    }
    // allFeaturesImages 将在数据加载和DOM更新后重新构建

    const skip = (page - 1) * limit;
    console.log(`[loadLiveStreamResults] Debug - page: ${page}, limit: ${limit}, calculated skip: ${skip}`);

    try {
        let apiUrl = `${API_BASE_URL}/streams/results?stream_uuid=${currentStreamId}&skip=${skip}&limit=${limit}`;
        console.log(`[loadLiveStreamResults] Fetching from: ${apiUrl}`);
        const response = await fetchWithAuth(apiUrl);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log("[loadLiveStreamResults] Data from /streams/results:", data);

        const persons = data.items;
        totalFeaturesLive = data.total;
        console.log(`[loadLiveStreamResults] totalFeaturesLive: ${totalFeaturesLive}`); 
        const totalPages = Math.ceil(totalFeaturesLive / itemsPerPageLive);
        console.log(`[loadLiveStreamResults] Calculated totalPages: ${totalPages}`); 

        if (!newFeaturesContainer || !noNewFeaturesMessage) {
            console.error("[loadLiveStreamResults] Missing feature container or message elements.");
            return;
        }

        if (!persons || persons.length === 0) {
            noNewFeaturesMessage.classList.remove('hidden');
            // 如果没有特征，并且不是第一页的自动刷新模式（为了避免清除已显示内容），则清空Lightbox图片
            if (!isAutoRefresh || page !== 1) {
                allFeaturesImages = []; 
            }
        } else {
            noNewFeaturesMessage.classList.add('hidden');
            let newFeaturesAddedToDOM = false;

            // 新增：按 created_at 时间倒序排列
            persons.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());

            // 遍历获取到的特征
            for (const result of persons) {
                console.log(`[loadLiveStreamResults] 尝试添加特征卡片, UUID: ${result.uuid}, 已显示: ${displayedFeatureUUIDs.has(result.uuid)}`);
                // 只有当特征UUID未被显示时才添加
                if (!displayedFeatureUUIDs.has(result.uuid)) {
                    const featureCard = document.createElement('div');
                    featureCard.className = 'feature-card';
                    
                    let cropImagePath = result.crop_image_path.replace(/\\/g, '/');
                    let fullFrameImagePath = result.full_frame_image_path ? result.full_frame_image_path.replace(/\\/g, '/') : '';
                    
                    const fullImagePathForLightbox = fullFrameImagePath || cropImagePath; // 用于Lightbox的完整图片路径
                    
                    featureCard.innerHTML = `
                        <img src="${API_BASE_URL}/${cropImagePath}" alt="特征图像" class="feature-img" data-full-image="${API_BASE_URL}/${fullImagePathForLightbox}">
                        <p>UUID: ${result.uuid}</p>
                        <p>时间: ${
                            result.created_at ? new Intl.DateTimeFormat('zh-CN', {
                                year: 'numeric',
                                month: '2-digit',
                                day: '2-digit',
                                hour: '2-digit',
                                minute: '2-digit',
                                second: '2-digit',
                                hour12: false,
                                timeZone: 'Asia/Shanghai'
                            }).format(new Date(result.created_at)) : 'N/A'
                          }</p>
                    `;
                    
                    // 如果是第一页的自动刷新，则在容器顶部插入新特征
                    if (isAutoRefresh && page === 1) {
                        newFeaturesContainer.prepend(featureCard);
                        console.log(`[loadLiveStreamResults] 特征卡片添加到顶部, UUID: ${result.uuid}`);
                    } else {
                        newFeaturesContainer.appendChild(featureCard);
                        console.log(`[loadLiveStreamResults] 特征卡片添加到尾部, UUID: ${result.uuid}`);
                    }
                    
                    displayedFeatureUUIDs.add(result.uuid);
                    console.log(`[loadLiveStreamResults] UUID ${result.uuid} 已添加到 displayedFeatureUUIDs. 当前数量: ${displayedFeatureUUIDs.size}`);
                    newFeaturesAddedToDOM = true;
                }
            }

            // 如果是第一页的自动刷新，并且添加了新特征，需要清理旧特征以保持数量
            if (isAutoRefresh && page === 1 && newFeaturesAddedToDOM) {
                console.log(`[loadLiveStreamResults] 执行旧特征清理。当前子元素数量: ${newFeaturesContainer.children.length}, 限制: ${limit}`);
                while (newFeaturesContainer.children.length > limit) {
                    const lastChild = newFeaturesContainer.lastChild;
                    if (lastChild) {
                        const uuidToRemove = lastChild.querySelector('p:nth-child(2)').textContent.replace('UUID: ', '');
                        displayedFeatureUUIDs.delete(uuidToRemove);
                        newFeaturesContainer.removeChild(lastChild);
                        console.log(`[loadLiveStreamResults] 移除旧特征卡片, UUID: ${uuidToRemove}. 剩余数量: ${newFeaturesContainer.children.length}`);
                    }
                }
            }
            
            // 使用通用的图片库点击事件设置函数
            // 在每次更新后重新设置，确保所有新旧图片都能触发Lightbox
            // setupImageGalleryClickEvents 调用不需要每次都重复，因为它会重新绑定。此处添加日志来确认。
            console.log("[loadLiveStreamResults] 调用 setupImageGalleryClickEvents.");
            setupImageGalleryClickEvents('#new-features-container', '.feature-img', (img) => {
                const imageUrl = img.dataset.fullImage;
                console.log("[Lightbox Debug] Image URL from data-full-image:", imageUrl); // Add this line for debug
                return imageUrl;
            }, true); // true 表示内容是动态变化的

            // 更新分页信息和显示分页控件
            // const totalPages = Math.ceil(totalFeaturesLive / itemsPerPageLive); // 这一行是重复的，因为上面已经计算过了，所以注释掉
            currentPageSpan.textContent = page;
            totalPagesSpan.textContent = totalPages;
            paginationControls.classList.remove('hidden');

            // 更新分页按钮状态
            const firstPageButton = document.getElementById('first-page-live');
            const prevPageButton = document.getElementById('prev-page-live');
            const nextPageButton = document.getElementById('next-page-live');
            const lastPageButton = document.getElementById('last-page-live');

            if (firstPageButton) {
                firstPageButton.disabled = (page === 1);
                console.log(`[loadLiveStreamResults] firstPageButton disabled: ${firstPageButton.disabled}`);
            }
            if (prevPageButton) {
                prevPageButton.disabled = (page === 1);
                console.log(`[loadLiveStreamResults] prevPageButton disabled: ${prevPageButton.disabled}`);
            }
            if (nextPageButton) {
                nextPageButton.disabled = (page === totalPages || totalPages === 0);
                console.log(`[loadLiveStreamResults] nextPageButton disabled: ${nextPageButton.disabled}`);
            }
            if (lastPageButton) {
                lastPageButton.disabled = (page === totalPages || totalPages === 0);
                console.log(`[loadLiveStreamResults] lastPageButton disabled: ${lastPageButton.disabled}`);
            }
        }
    } catch (error) {
        console.error('[loadLiveStreamResults] 获取实时流结果失败:', error);
        if (newFeaturesContainer) newFeaturesContainer.innerHTML = `<p style="color: red; text-align: center;">加载实时特征失败: ${error.message}</p>`;
    } finally {
        // Hide loading indicator if any
    }
}

// 新增函数：导出实时流结果
async function exportLiveStreamResults() {
    if (!currentStreamId) {
        alert("没有可导出的视频流ID。");
        return;
    }

    // 禁用导出按钮，防止重复点击
    if (exportResultsButton) {
        exportResultsButton.disabled = true;
        exportResultsButton.textContent = '正在导出...';
    }

    try {
        console.log(`[exportLiveStreamResults] 导出实时流 ${currentStreamId} 的结果...`);
        const response = await fetchWithAuth(`${API_BASE_URL}/export/streams/export_results/${currentStreamId}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        // 获取Blob数据
        const blob = await response.blob();

        // 从Content-Disposition头中获取文件名
        let filename = `stream_results_export.xlsx`; // 默认文件名，更通用
        const contentDisposition = response.headers.get('Content-Disposition');
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename\*?=(?:UTF-8'')?"?([^;\n]*?)"?$/i);
            if (filenameMatch && filenameMatch[1]) {
                try {
                    // 解码URI编码的文件名，并移除可能存在的引号
                    filename = decodeURIComponent(filenameMatch[1].replace(/%22/g, '').replace(/^"|"$/g, ''));
                } catch (e) {
                    console.warn("[exportLiveStreamResults] 无法解码文件名，使用默认值。", e);
                }
            }
        }

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        // alert("视频流解析结果已成功导出！"); // 移除此行，避免重复触发下载

    } catch (error) {
        console.error("[exportLiveStreamResults] 导出实时流结果失败:", error);
        alert(`导出失败: ${error.message}`);
    } finally {
        // 无论成功或失败，都重新启用按钮
        if (exportResultsButton) {
            exportResultsButton.disabled = false;
            exportResultsButton.textContent = '导出解析结果';
        }
    }
}

// pingStream函数
async function pingStream() {
    if (!currentStreamId) return; // 没有stream_id则不发送ping

    try {
        console.log(`[pingStream] Pinging stream ${currentStreamId}...`);
        const response = await fetchWithAuth(`${API_BASE_URL}/streams/ping/${currentStreamId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        });
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }
        const data = await response.json();
        console.log(`[pingStream] Stream ping successful for ID ${currentStreamId}:`, data.status);
        const streamStatusMessage = document.getElementById('stream-status-message');
        if (streamStatusMessage) {
            streamStatusMessage.textContent = `视频流正在解析中... (状态: ${data.status || '活跃'})`;
            streamStatusMessage.style.color = 'green';
        }
        // 根据流状态更新停止按钮的可用性
        if (stopStreamButton) {
            stopStreamButton.disabled = !(data.status === 'processing' || data.status === 'active');
        }
    } catch (error) {
        console.error(`[pingStream] Error pinging stream ${currentStreamId}:`, error);
        // 如果ping失败，可以考虑停止刷新，并提示用户流已中断
        if (streamPingInterval) clearInterval(streamPingInterval);
        if (newFeaturesInterval) clearInterval(newFeaturesInterval);
        if (liveFeedImgElement) liveFeedImgElement.src = ""; // 清空src来停止显示
        if (savedFeedVideoElement) savedFeedVideoElement.pause(); savedFeedVideoElement.src = ""; // 暂停并清空视频源
        const streamStatusMessage = document.getElementById('stream-status-message');
        if (streamStatusMessage) {
            streamStatusMessage.textContent = `视频流已中断或出现错误: ${error.message}`;
            streamStatusMessage.style.color = 'red';
        }
    }
}

// 新增函数：更新视频流源（实时或已保存）
async function updateVideoStreamSource() {
    try {
        console.log("[updateVideoStreamSource] Fetching saved streams for status check...");
        // 首先获取所有已保存的视频流信息，以确定当前流的状态
        const response = await fetchWithAuth(`${API_BASE_URL}/streams/saved`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const responseData = await response.json();
        console.log("[updateVideoStreamSource] Response Data from /streams/saved:", responseData); // Add this line
        const streams = responseData.items; // 从 items 字段中获取流数组

        const currentStream = streams.find(s => s.stream_uuid === currentStreamId);

        if (!currentStream) {
            console.error(`[updateVideoStreamSource] 未找到 Stream ID 为 ${currentStreamId} 的流。`);
            // 显示错误信息或重定向
            const resultsDiv = document.getElementById('live-stream-results');
            if (resultsDiv) {
                resultsDiv.innerHTML = `<p>未找到 ID 为 <strong>${currentStreamId}</strong> 的实时流。</p>`;
            }
            if (liveFeedImgElement) liveFeedImgElement.style.display = 'none';
            if (savedFeedVideoElement) savedFeedVideoElement.style.display = 'none';
            if (stopStreamButton) stopStreamButton.style.display = 'none';
            if (exportResultsButton) exportResultsButton.style.display = 'none';

            // 停止所有定时刷新，因为它不再有效
            if (newFeaturesInterval) clearInterval(newFeaturesInterval);
            if (streamPingInterval) clearInterval(streamPingInterval);

            return false; // 返回 false 表示流不活跃
        }

        // 根据流状态设置视频源
        console.log(`[updateVideoStreamSource] Current stream status from backend: ${currentStream.status}, is_active: ${currentStream.is_active}`);
        if (currentStream.is_active) {
            // 如果流是活跃的，显示实时视频流
            liveFeedImgElement.src = `${API_BASE_URL}/streams/feed/${currentStreamId}`;
            liveFeedImgElement.style.display = 'block';
            savedFeedVideoElement.style.display = 'none';
            stopStreamButton.style.display = 'block'; // 显示停止按钮
            exportResultsButton.style.display = 'block'; // 显示导出按钮
            console.log(`[updateVideoStreamSource] 实时流 ${currentStreamId} 状态活跃，显示实时视频。`);
            return true; // 返回 true 表示流活跃
        } else {
            // 如果流不活跃，检查是否有保存的视频文件
            if (currentStream.output_video_path) {
                // output_video_path 现在应该已经是相对于 /saved_streams/ 的路径
                // 例如: "/saved_streams/uuid/uuid.mp4"
                // 直接构建视频URL，移除之前不必要的 replace('backend/', '')
                const videoUrl = `${API_BASE_URL}${currentStream.output_video_path}`;
                savedFeedVideoElement.src = videoUrl;
                savedFeedVideoElement.style.display = 'block';
                liveFeedImgElement.style.display = 'none';
                stopStreamButton.style.display = 'none'; // 隐藏停止按钮
                exportResultsButton.style.display = 'block'; // 显示导出按钮
                console.log(`[updateVideoStreamSource] 实时流 ${currentStreamId} 状态不活跃，显示保存的视频: ${videoUrl}`);
                const streamStatusMessage = document.getElementById('stream-status-message');
                if (streamStatusMessage) {
                    streamStatusMessage.textContent = '正在播放本地视频...';
                    streamStatusMessage.style.color = 'green'; // 或者你想要的颜色
                }
                return false; // 返回 false 表示流不活跃
            } else {
                // 没有实时流也没有保存的视频
                const streamStatusMessage = document.getElementById('stream-status-message');
                if (streamStatusMessage) {
                    streamStatusMessage.textContent = `实时流 ${currentStreamId} 未激活且没有保存的视频文件。状态: ${currentStream.status}`;
                    streamStatusMessage.style.color = 'orange';
                }
                if (liveFeedImgElement) liveFeedImgElement.style.display = 'none';
                if (savedFeedVideoElement) savedFeedVideoElement.style.display = 'none';
                if (stopStreamButton) stopStreamButton.style.display = 'none';
                if (exportResultsButton) exportResultsButton.style.display = 'none';
                console.log(`[updateVideoStreamSource] 实时流 ${currentStreamId} 状态不活跃且无保存视频。`);
                return false; // 返回 false 表示流不活跃
            }
        }
    } catch (error) {
        console.error('[updateVideoStreamSource] Error fetching stream status or setting video source:', error);
        // 显示错误信息给用户
        const streamStatusMessage = document.getElementById('stream-status-message');
        if (streamStatusMessage) {
            streamStatusMessage.textContent = `加载视频源失败: ${error.message}`;
            streamStatusMessage.style.color = 'red';
        }
        if (liveFeedImgElement) liveFeedImgElement.style.display = 'none';
        if (savedFeedVideoElement) savedFeedVideoElement.style.display = 'none';
        if (stopStreamButton) stopStreamButton.style.display = 'none';
        if (exportResultsButton) exportResultsButton.style.display = 'none';
        return false; // 出现错误时返回 false
    }
}