import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js';
import { showLightbox } from './common.js';

const activeEventSources = new Map(); // 新增：用于存储活跃的EventSource实例
let usersMap = new Map(); // 新增：存储用户ID到用户名的映射

let currentPage = 1;
const itemsPerPage = 10; // 每页显示的视频数量
let totalPages = 1;
let totalVideos = 0;

// 新增：获取并缓存所有用户列表
async function fetchAndCacheUsers() {
    try {
        console.log("fetchAndCacheUsers: 正在获取用户列表...");
        const response = await fetchWithAuth(`${API_BASE_URL}/admin/users`);
        if (response.ok) {
            const users = await response.json();
            users.forEach(user => {
                usersMap.set(user.id, user.username);
            });
            console.log('fetchAndCacheUsers: 用户列表已缓存:', usersMap);
        } else {
            console.error('fetchAndCacheUsers: 获取用户列表失败:', response.statusText);
        }
    } catch (error) {
        console.error('fetchAndCacheUsers: 获取用户列表时发生错误:', error);
    }
}

export function initVideoAnalysisPage() {
    console.log("Initializing Video Analysis Page...");
    console.log("initVideoAnalysisPage called!"); // Added for debugging
    const videoUploadInput = document.getElementById('video-upload-input');
    const uploadVideoButton = document.getElementById('upload-video-button');
    const prevPageButton = document.getElementById('prevPageVideos');
    const nextPageButton = document.getElementById('nextPageVideos');
    const currentPageSpan = document.getElementById('currentPageVideos');
    const totalPagesSpan = document.getElementById('totalPagesVideos');
    const firstPageButton = document.getElementById('firstPageVideos'); // 新增
    const lastPageButton = document.getElementById('lastPageVideos'); // 新增

    if (videoUploadInput) videoUploadInput.addEventListener('change', handleVideoFileSelection);
    if (uploadVideoButton) uploadVideoButton.addEventListener('click', handleVideoUpload);
    if (firstPageButton) firstPageButton.addEventListener('click', () => {
        if (currentPage !== 1) {
            currentPage = 1;
            loadVideos();
        }
    });
    if (prevPageButton) prevPageButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            loadVideos();
        }
    });
    if (nextPageButton) nextPageButton.addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            loadVideos();
        }
    });
    if (lastPageButton) lastPageButton.addEventListener('click', () => {
        if (currentPage !== totalPages) {
            currentPage = totalPages;
            loadVideos();
        }
    });
    
    // 根据用户角色隐藏上传人列和视频上传功能
    const currentUser = Auth.getUserInfo();
    const uploaderHeader = document.getElementById('uploader-header');
    const videoUploadSection = document.getElementById('video-upload-section'); // 获取视频上传区域

    if (currentUser && currentUser.role !== 'admin') {
        if (uploaderHeader) {
            uploaderHeader.style.display = 'none';
        }
    }
    
    // 根据用户角色隐藏视频上传功能
    // 普通用户和高级用户应该能看到上传功能
    if (currentUser && (currentUser.role === 'user' || currentUser.role === 'advanced' || currentUser.role === 'admin')) {
        // 用户有权限，确保上传区域是可见的
        if (videoUploadSection) {
            videoUploadSection.style.display = ''; // 或者 'block', 'flex' 根据原始样式恢复
            const videoUploadStatus = document.getElementById('upload-status');
            if (videoUploadStatus) videoUploadStatus.style.display = '';
        }
    } else { // 其他角色（如果存在且无权限）则隐藏
        if (videoUploadSection) {
            videoUploadSection.style.display = 'none';
            const videoUploadStatus = document.getElementById('upload-status');
            if (videoUploadStatus) videoUploadStatus.style.display = 'none';
        }
    }

    // 确保在加载视频前，只有管理员用户才获取用户列表
    if (currentUser && currentUser.role === 'admin') {
        fetchAndCacheUsers().then(() => {
            console.log("initVideoAnalysisPage: 用户列表获取完成，加载视频...");
            loadVideos();
        });
    } else {
        // 非管理员用户直接加载视频，不获取用户列表
        console.log("initVideoAnalysisPage: 非管理员用户，跳过获取用户列表，直接加载视频...");
        loadVideos();
    }

    // 使用事件委托处理按钮点击，确保只绑定一次
    const videoListTableBody = document.getElementById('video-list-table-body');
    if (videoListTableBody) {
        videoListTableBody.addEventListener('click', (e) => {
            const target = e.target;
            const videoId = target.dataset.videoId;
            if (target.classList.contains('delete-button')) {
                deleteVideo(videoId);
            } else if (target.classList.contains('view-results-button')) {
                viewResults(videoId);
            } else if (target.classList.contains('terminate-button')) {
                terminateVideo(videoId);
            }
        });
    }
}

// 创建并返回一个视频表格行 (此函数未修改)
function createVideoRow(video, index) { // 添加 index 参数
    const currentUser = Auth.getUserInfo(); // 将声明移动到函数顶部
    console.log("createVideoRow: 正在创建视频行，视频信息:", video);
    const row = document.createElement('tr');
    row.setAttribute('data-video-id', video.id);
    row.setAttribute('data-video-uuid', video.uuid); // 添加 video_uuid 属性

    // 序号列
    row.insertCell(0).textContent = (currentPage - 1) * itemsPerPage + index + 1; // 修正为序号列

    row.insertCell(1).textContent = video.filename;
    row.insertCell(2).textContent = video.status === 'processing' ? '处理中' :
                                    (video.status === 'completed' ? '已完成' :
                                    (video.status === 'failed' ? '失败' :
                                    (video.status === 'terminated' ? '已终止' : '等待处理')));

    const progressCell = row.insertCell(3);
    if (video.status === 'processing') {
        progressCell.innerHTML = `
            <div class="progress-container">
                <progress value="${video.progress}" max="100"></progress>
                <span>${video.progress || 0}%</span>
            </div>
        `;
        listenToProgress(video.id); // 监听处理进度
    } else if (video.status === 'completed') {
        progressCell.textContent = '100%';
    } else if (video.status === 'terminated') {
        progressCell.innerHTML = `
            <div class="progress-container">
                <progress value="${video.progress}" max="100"></progress>
                <span>已终止 (${video.progress || 0}%)</span>
            </div>
        `;
    } else {
        progressCell.textContent = '-';
    }

    row.insertCell(4).textContent = video.processed_at ? new Date(video.processed_at).toLocaleString() : 'N/A';
    
    // 新增：显示上传人信息，但非管理员用户不显示整列
    // const currentUser = Auth.getUserInfo(); // 已经从函数顶部获取
    const ownerUsername = usersMap.get(video.owner_id) || '未知用户';
    console.log(`createVideoRow: 视频ID: ${video.id}, owner_id: ${video.owner_id}, 上传人: ${ownerUsername}`);
    
    let actionsCellIndex = 5; // 默认情况下，操作列的索引是 5 (如果没有上传人列)
    if (currentUser && currentUser.role === 'admin') {
        row.insertCell(5).textContent = ownerUsername;
        actionsCellIndex = 6; // 如果是管理员，上传人列存在，操作列索引是 6
    }

    const actionsCell = row.insertCell(actionsCellIndex);

    const buttonsContainer = document.createElement('div');
    buttonsContainer.classList.add('actions-buttons-container');

    // Terminate button (for processing, paused, or pending videos)
    if (video.status === 'processing' || video.status === 'paused' || video.status === 'pending') {
        const terminateButton = document.createElement('button');
        terminateButton.textContent = '终止';
        terminateButton.classList.add('terminate-button');
        terminateButton.setAttribute('data-video-id', video.id);
        if (video.status === 'pending') {
            terminateButton.disabled = true; // Disable if pending
            terminateButton.title = '视频正在等待处理，无法终止。'; // Add tooltip
        }
        buttonsContainer.appendChild(terminateButton);
    }

    // 按钮不再有 inline onclick，通过事件委托处理
    if (video.status === 'completed') {
        const viewButton = document.createElement('button');
        viewButton.textContent = '查看结果';
        viewButton.classList.add('view-results-button'); // 添加类名用于事件委托
        viewButton.setAttribute('data-video-id', video.id);
        buttonsContainer.appendChild(viewButton);
    }

    if (currentUser && currentUser.role === 'admin' || (currentUser && currentUser.role === 'advanced' && video.owner_id === currentUser.id)) {
        const deleteButton = document.createElement('button');
        deleteButton.textContent = '删除';
        deleteButton.classList.add('delete-button');
        deleteButton.setAttribute('data-video-id', video.id);
        buttonsContainer.appendChild(deleteButton);
    }

    // Display status text for failed or terminated videos (where no action button is provided)
    else if (video.status === 'failed' || video.status === 'terminated') {
        const statusText = document.createElement('span');
        statusText.textContent = video.status === 'failed' ? '处理失败' : '已终止';
        buttonsContainer.appendChild(statusText);
    }

    actionsCell.appendChild(buttonsContainer);

    return row;
}

// 新增：处理删除视频
async function deleteVideo(videoId) {
    if (confirm(`确定要删除视频 ID: ${videoId} 吗？`)) {
        try {
            const deleteResponse = await fetchWithAuth(`${API_BASE_URL}/videos/${videoId}`, { method: 'DELETE' });
            if (!deleteResponse.ok) {
                const errorData = await deleteResponse.json();
                throw new Error(errorData.detail || '删除失败');
            }
            alert('视频删除成功');
            // 明确关闭并移除EventSource
            if (activeEventSources.has(videoId)) {
                activeEventSources.get(videoId).close();
                activeEventSources.delete(videoId);
                console.log(`视频 ${videoId} 的 EventSource 已关闭 (删除)。`);
            }
            loadVideos(); // 重新加载视频列表
        } catch (error) {
            alert(`删除视频失败: ${error.message}`);
            console.error('删除错误:', error);
        }
    }
}

// 新增：处理终止视频处理
async function terminateVideo(videoId) {
    if (confirm(`确定要终止视频 ID: ${videoId} 的处理吗？`)) {
        try {
            const terminateResponse = await fetchWithAuth(`${API_BASE_URL}/videos/${videoId}/terminate`, {
                method: 'POST'
            });
            if (!terminateResponse.ok) {
                const errorData = await terminateResponse.json();
                throw new Error(errorData.detail || '终止失败');
            }

            console.log(`视频 ${videoId} 已成功请求终止。`);

            // 查找视频行并立即更新其状态
            const row = document.querySelector(`tr[data-video-id="${videoId}"]`);
            if (row) {
                const statusCell = row.cells[2];
                const progressCell = row.cells[3];
                const actionsCell = row.cells[5];

                if (statusCell) {
                    statusCell.textContent = '已终止'; // 更新状态文本
                }
                if (progressCell) {
                    progressCell.innerHTML = `<span>已终止</span>`;
                }
                // 禁用或移除终止按钮
                if (actionsCell) {
                    const terminateButton = actionsCell.querySelector('.terminate-button');
                    if (terminateButton) {
                        terminateButton.remove(); // 移除按钮
                    }
                }
            }

            // 明确关闭该视频的 EventSource
            if (activeEventSources.has(videoId)) {
                activeEventSources.get(videoId).close();
                activeEventSources.delete(videoId);
                console.log(`视频 ${videoId} 的 EventSource 已在终止成功后关闭。`);
            }

            // 不再调用 loadVideos()。让定期刷新来处理更新。
            loadVideos(); // 重新加载视频列表以显示最新的状态
            location.reload(); // 强制刷新整个页面

        } catch (error) {
            alert(`终止视频处理失败: ${error.message}`);
            console.error('终止错误:', error);
        }
    }
}

// 新增：处理查看结果
function viewResults(videoId) {
    // 打开新的标签页并传递 videoId
    window.open(`/video_results?videoId=${videoId}`, '_blank');
}

async function loadVideos() {
    console.log("loadVideos: 正在加载视频列表...");
    const videoListTableBody = document.getElementById('video-list-table-body');
    const paginationControls = document.getElementById('videoPaginationControls');
    const currentPageSpan = document.getElementById('currentPageVideos');
    const totalPagesSpan = document.getElementById('totalPagesVideos');
    const prevPageButton = document.getElementById('prevPageVideos');
    const nextPageButton = document.getElementById('nextPageVideos');
    const firstPageButton = document.getElementById('firstPageVideos'); // 新增
    const lastPageButton = document.getElementById('lastPageVideos'); // 新增

    videoListTableBody.innerHTML = ''; // 清空现有内容
    if (paginationControls) paginationControls.style.display = 'none'; // 默认隐藏分页控件

    // 清除所有活跃的 EventSource，避免旧的进度更新影响新的列表
    activeEventSources.forEach(es => es.close());
    activeEventSources.clear();

    const currentUser = Auth.getUserInfo();
    const skip = (currentPage - 1) * itemsPerPage;
    const limit = itemsPerPage;

    let url = `${API_BASE_URL}/videos/?skip=${skip}&limit=${limit}`;

    try {
        const response = await fetchWithAuth(url);
        if (response.ok) {
            const data = await response.json();
            const videos = data.items; // 获取视频列表
            totalVideos = data.total_count; // 获取总数
            totalPages = Math.ceil(totalVideos / itemsPerPage); // 计算总页数

            console.log("loadVideos: 获取到视频列表:", videos);
            console.log(`总视频数: ${totalVideos}, 总页数: ${totalPages}, 当前页: ${currentPage}`);

            // 更新分页控件状态
            if (currentPageSpan) currentPageSpan.textContent = currentPage;
            if (totalPagesSpan) totalPagesSpan.textContent = totalPages;
            if (prevPageButton) prevPageButton.disabled = (currentPage === 1);
            if (nextPageButton) nextPageButton.disabled = (currentPage === totalPages || totalVideos === 0);
            if (firstPageButton) firstPageButton.disabled = (currentPage === 1);
            if (lastPageButton) lastPageButton.disabled = (currentPage === totalPages || totalVideos === 0);

            if (videos.length === 0) {
                const noRecordsRow = videoListTableBody.insertRow();
                noRecordsRow.className = 'no-records-row'; // 添加一个类名，方便CSS样式
                const cell = noRecordsRow.insertCell(0);
                cell.colSpan = 7; // 跨越所有列
                cell.textContent = '暂无视频记录。';
                cell.style.textAlign = 'center';
                cell.style.padding = '20px';
                cell.style.color = '#888';
            } else {
                videos.forEach((video, index) => { // 添加 index 参数
                    const row = createVideoRow(video, index); // 将 index 传递给 createVideoRow
                    videoListTableBody.appendChild(row);
                });
                if (paginationControls) paginationControls.style.display = 'flex'; // 显示分页控件
            }
        } else {
            const errorData = await response.json();
            alert('加载视频列表失败：' + (errorData.detail || response.statusText));
            const noRecordsRow = videoListTableBody.insertRow();
            noRecordsRow.className = 'no-records-row';
            const cell = noRecordsRow.insertCell(0);
            cell.colSpan = 7;
            cell.textContent = '无法加载视频记录。 ';
            cell.style.textAlign = 'center';
            cell.style.padding = '20px';
            cell.style.color = 'red';
        }
    } catch (error) {
        console.error('加载视频列表时发生错误:', error);
        if (error.message !== 'Unauthorized') {
            alert('加载视频列表时发生网络错误。');
        }
        const noRecordsRow = videoListTableBody.insertRow();
        noRecordsRow.className = 'no-records-row';
        const cell = noRecordsRow.insertCell(0);
        cell.colSpan = 7;
        cell.textContent = '无法加载视频记录，请检查网络或刷新页面。';
        cell.style.textAlign = 'center';
        cell.style.padding = '20px';
        cell.style.color = 'red';
    }
}

/**
 * [已修改] 监听视频处理进度
 * 修复了在视频处理被终止(terminated)后，监听不会停止的BUG。
 */
function listenToProgress(videoId) {
    const token = Auth.getToken();
    if (!token) {
        console.error("未找到认证令牌，无法监听视频进度。");
        return;
    }

    // 关闭并移除当前视频ID的现有EventSource
    if (activeEventSources.has(videoId)) {
        activeEventSources.get(videoId).close();
        activeEventSources.delete(videoId);
        console.log(`视频 ${videoId} 的旧 EventSource 已关闭。`);
    }

    console.log(`正在为视频 ${videoId} 启动 EventSource 监听。`);
    const eventSource = new EventSource(`/videos/${videoId}/progress?token=${token}`);
    activeEventSources.set(videoId, eventSource); // 存储新的EventSource实例

    const handleError = async function(err) {
        console.error(`EventSource 错误 for video ${videoId}:`, err);
        const row = document.querySelector(`tr[data-video-id="${videoId}"]`);
        if (row) {
            const progressText = row.querySelector('.progress-container span');
            if (progressText) {
                try {
                    const response = await fetchWithAuth(`/videos/${videoId}`);
                    console.log(`获取视频 ${videoId} 状态响应:`, response.status);
                    const videoData = await response.json();
                    console.log(`获取视频 ${videoId} 状态数据:`, videoData);

                    if (videoData.status === 'terminated') {
                        progressText.textContent = `视频处理已终止。`;
                        progressText.style.color = 'orange';
                    } else if (videoData.status === 'completed') {
                        progressText.textContent = `视频处理已完成。`;
                        progressText.style.color = 'green';
                    } else if (videoData.status === 'failed') {
                        progressText.textContent = `视频处理失败。`;
                        progressText.style.color = 'red';
                    } else {
                        // 其他错误情况，显示连接中断
                        progressText.textContent = `连接中断 (错误)。请刷新页面。`;
                        progressText.style.color = 'red';
                    }
                } catch (fetchError) {
                    console.error(`获取视频 ${videoId} 状态失败:`, fetchError);
                    progressText.textContent = `连接中断 (错误)。请刷新页面。`;
                    progressText.style.color = 'red';
                }
            }
        }
        eventSource.close();
        activeEventSources.delete(videoId); // 从映射中移除
        console.log(`视频 ${videoId} 的 EventSource 已关闭 (由于错误)。`);
    };

    const handleMessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            if (data.status === 'error') {
                console.error(`服务器报告视频 ${videoId} 错误:`, data.message);
                handleError(new Error(data.message)); // 触发错误处理逻辑
            } else if (data.status === 'completed' || data.status === 'terminated' || data.status === 'failed') {
                eventSource.close();
                activeEventSources.delete(videoId); // 从映射中移除
                console.log(`视频 ${videoId} 的 EventSource 已关闭 (状态: ${data.status})。`);
                // 立即更新UI，因为EventSource已关闭，不会再有新的进度消息
                const row = document.querySelector(`tr[data-video-id="${videoId}"]`);
                if (row) {
                    const statusCell = row.cells[2];
                    const progressCell = row.cells[3];
                    if (statusCell) {
                        statusCell.textContent = data.status === 'completed' ? '已完成' : (data.status === 'terminated' ? '已终止' : '失败');
                    }
                    if (progressCell) {
                        progressCell.innerHTML = `
                            <div class="progress-container">
                                <progress value="${data.progress || 0}" max="100"></progress>
                                <span>${statusCell.textContent} (${data.progress || 0}%)</span>
                            </div>
                        `;
                    }
                    // 禁用或移除终止按钮
                    const actionsCell = row.cells[5];
                    if (actionsCell) {
                        const terminateButton = actionsCell.querySelector('.terminate-button');
                        if (terminateButton) {
                            terminateButton.remove();
                        }
                    }
                }

                // 这是关键部分：如果完成、终止或失败，重新获取并重新渲染行
                if (data.status === 'completed' || data.status === 'terminated' || data.status === 'failed') {
                    console.log(`视频 ${videoId} 状态变为 ${data.status}，重新获取最新信息并刷新行。`);
                    // 关闭该视频的 EventSource
                    if (activeEventSources.has(videoId)) {
                        activeEventSources.get(videoId).close();
                        activeEventSources.delete(videoId);
                        console.log(`视频 ${videoId} 的 EventSource 已关闭。`);
                    }

                    // 从 API 重新获取更新后的视频数据
                    fetchWithAuth(`${API_BASE_URL}/videos/${videoId}`)
                        .then(response => {
                            if (!response.ok) {
                                throw new Error(`无法获取视频 ${videoId} 的最新状态: ${response.statusText}`);
                            }
                            return response.json();
                        })
                        .then(updatedVideo => {
                            console.log(`已获取更新后的视频 ${videoId}:`, updatedVideo);
                            // 找到并移除旧的行
                            const oldRow = document.querySelector(`tr[data-video-id="${videoId}"]`);
                            if (oldRow) {
                                oldRow.remove();
                            }
                            // 使用更新后的数据创建并追加新的行
                            const newRow = createVideoRow(updatedVideo);
                            document.getElementById('video-list-table-body').appendChild(newRow);
                        })
                        .catch(error => {
                            console.error(`刷新视频 ${videoId} 行时出错:`, error);
                            // 可选：向用户显示一个警告框
                            showLightbox(`更新视频 ${videoId} 状态失败: ${error.message}`, 'error');
                        });
                }
            } else {
                const row = document.querySelector(`tr[data-video-id="${data.id}"]`);
                if (!row) return;

                const progressBar = row.querySelector('progress');
                const progressText = row.querySelector('span');
                
                if (progressBar && progressText) {
                    progressBar.value = data.progress;
                    let statusText = data.status;
                    if(data.status === 'processing') statusText = '处理中';
                    else if(data.status === 'terminated') statusText = '已终止';
                    else if(data.status === 'completed') statusText = '已完成';
                    else if(data.status === 'failed') statusText = '失败';

                    progressText.textContent = `${statusText} (${data.progress}%)`;
                }
            }
        } catch (e) {
            console.error(`解析视频 ${videoId} 的 EventSource 消息失败:`, e);
            handleError(e); // 传递错误对象
        }
    };

    eventSource.addEventListener('message', handleMessage);
    eventSource.addEventListener('error', handleError);

    // 添加 EventSource 的 open 和 close 事件监听，用于调试和清理
    eventSource.addEventListener('open', () => {
        console.log(`EventSource for video ${videoId} opened.`);
    });
    eventSource.addEventListener('close', () => {
        console.log(`EventSource for video ${videoId} closed.`);
        activeEventSources.delete(videoId); // 确保在外部关闭时也从映射中移除
    });
}

async function handleVideoFileSelection() {
    const videoUploadInput = document.getElementById('video-upload-input');
    const videoFileNameDisplay = document.getElementById('selected-file-name'); // 修正：匹配HTML中的ID

    if (videoUploadInput && videoFileNameDisplay) {
        if (videoUploadInput.files.length > 0) {
            videoFileNameDisplay.textContent = `已选择文件: ${videoUploadInput.files[0].name}`;
        } else {
            videoFileNameDisplay.textContent = '未选择文件';
        }
    }
}

async function handleVideoUpload() {
    const currentUser = Auth.getUserInfo();
    // 如果不是管理员、高级用户或普通用户，则阻止上传
    if (!currentUser || (currentUser.role !== 'admin' && currentUser.role !== 'advanced' && currentUser.role !== 'user')) {
        alert('您没有权限上传视频。');
        return;
    }

    const videoUploadInput = document.getElementById('video-upload-input');
    const uploadVideoMessage = document.getElementById('upload-status'); // 修正：匹配HTML中的ID
    const uploadVideoButton = document.getElementById('upload-video-button');

    if (!videoUploadInput || !uploadVideoMessage || !uploadVideoButton) {
        console.error("缺少必要的DOM元素，无法处理视频上传。");
        return;
    }

    if (uploadVideoButton) {
        uploadVideoButton.disabled = true; // 禁用按钮防止重复提交
        uploadVideoButton.textContent = "正在上传并解析...";
    }

    if (videoUploadInput.files.length === 0) {
        uploadVideoMessage.textContent = "请选择一个视频文件上传。";
        uploadVideoMessage.classList.remove('hidden', 'success', 'info');
        uploadVideoMessage.classList.add('error');
        if (uploadVideoButton) {
            uploadVideoButton.disabled = false;
            uploadVideoButton.textContent = "上传并开始解析";
        }
        return;
    }

    const file = videoUploadInput.files[0];
    const formData = new FormData();
    formData.append('video', file); // 修正：将字段名从 'file' 更改为 'video'

    uploadVideoMessage.textContent = `正在上传视频 "${file.name}"，请稍候...`;
    uploadVideoMessage.classList.remove('hidden', 'error');
    uploadVideoMessage.classList.add('info');

    try {
        const response = await fetchWithAuth(`${API_BASE_URL}/videos/extract-features`, { // 修正：添加 /videos 前缀
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `HTTP error! status: ${response.status}`);
        }

        uploadVideoMessage.textContent = `视频 "${file.name}" 已成功上传并开始解析。`;
        uploadVideoMessage.classList.remove('error', 'info');
        uploadVideoMessage.classList.add('success');

        // 清空文件输入框，防止重复上传
        videoUploadInput.value = '';
        const videoFileNameDisplay = document.getElementById('selected-file-name'); // 修正：重新获取元素
        if (videoFileNameDisplay) {
            videoFileNameDisplay.textContent = '未选择文件';
        }

        loadVideos(); // 重新加载视频列表以显示新上传的视频及其状态

    } catch (error) {
        console.error('视频上传失败:', error);
        uploadVideoMessage.textContent = `视频上传和解析失败: ${error.message}`;
        uploadVideoMessage.classList.remove('hidden', 'success', 'info');
        uploadVideoMessage.classList.add('error');
    } finally {
        if (uploadVideoButton) {
            uploadVideoButton.disabled = false;
            uploadVideoButton.textContent = "上传并开始解析";
        }
    }
}