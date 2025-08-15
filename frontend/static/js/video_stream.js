import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js';
import { initializeAuthenticatedPage } from './common.js';

let currentPage = 1;
const itemsPerPage = 10; // 每页显示的视频流数量
let totalPages = 1;
let totalStreams = 0;

let addStreamForm;
let streamNameInput;
let rtspUrlInput; // 修改变量名
let apiStreamUrlInput; // 新增变量
let monitorPointIdInput; // 新增变量
let addStreamMessage;
let savedStreamsTable;
let savedStreamsTableBody;
let noStreamsMessage;
let refreshInterval;
let isAddingStream = false; // 新增标志，防止addStream重复执行
let isDeletingStream = false; // 新增标志，防止deleteStream重复执行
let userInfo; // 将 userInfo 定义为模块级别的全局变量
let usersMap = new Map(); // 新增：存储用户ID到用户名的映射

// 新增：获取并缓存所有用户列表
async function fetchAndCacheUsers() {
    try {
        const response = await fetchWithAuth(`${API_BASE_URL}/admin/users`);
        if (response.ok) {
            const users = await response.json();
            users.forEach(user => {
                usersMap.set(user.id, user.username);
            });
            console.log('用户列表已缓存:', usersMap);
        } else {
            console.error('获取用户列表失败:', response.statusText);
        }
    } catch (error) {
        console.error('获取用户列表时发生错误:', error);
    }
}

// 初始化视频流管理页面的主函数
export async function initVideoStreamPage() {
    console.log("Initializing Video Stream Page...");

    // 1. 权限检查：确保只有管理员或高级用户可以访问此页面
    userInfo = Auth.getUserInfo(); // 在这里赋值给全局 userInfo
    if (!userInfo || (userInfo.role !== 'admin' && userInfo.role !== 'advanced')) { // 将 advanced_user 更改为 advanced
        document.body.innerHTML = '<h1>权限不足</h1><p>只有管理员或高级用户才能访问此页面。</p><a href="/">返回主页</a>';
        return;
    }

    // 根据用户角色隐藏“添加人”列的表头
    const adderHeader = document.getElementById('adder-header');
    if (userInfo.role !== 'admin') {
        if (adderHeader) {
            adderHeader.style.display = 'none';
        }
    }

    // 2. 获取DOM元素引用
    addStreamForm = document.getElementById('add-stream-form');
    streamNameInput = document.getElementById('stream-name');
    rtspUrlInput = document.getElementById('rtsp-url'); // 修改获取ID
    apiStreamUrlInput = document.getElementById('api-stream-url'); // 获取新增的输入框
    monitorPointIdInput = document.getElementById('monitor-point-id'); // 获取监控点编号输入框
    addStreamMessage = document.getElementById('add-stream-message');
    savedStreamsTable = document.getElementById('saved-streams-table');
    savedStreamsTableBody = savedStreamsTable ? savedStreamsTable.querySelector('tbody') : null;
    noStreamsMessage = document.getElementById('no-streams-message');
    const addStreamSection = document.getElementById('add-stream-section'); // 获取添加视频流区域
    const prevPageButton = document.getElementById('prevPageStreams');
    const nextPageButton = document.getElementById('nextPageStreams');
    const currentPageSpan = document.getElementById('currentPageStreams');
    const totalPagesSpan = document.getElementById('totalPagesStreams');
    const firstPageButton = document.getElementById('firstPageStreams'); // 新增
    const lastPageButton = document.getElementById('lastPageStreams'); // 新增

    // 健壮性检查：确保所有关键DOM元素都已加载
    if (!addStreamForm || !streamNameInput || !addStreamMessage || !savedStreamsTable || !savedStreamsTableBody || !noStreamsMessage) {
        console.error("缺少必要的DOM元素，无法初始化视频流页面。检查：addStreamForm, streamNameInput, addStreamMessage, savedStreamsTable, savedStreamsTableBody, noStreamsMessage");
        return;
    }

    // 额外检查URL输入框是否存在，它们是可选的，但需要被正确引用
    if (!rtspUrlInput || !apiStreamUrlInput || !monitorPointIdInput) {
        console.error("缺少RTSP/HTTP、API视频流或监控点编号输入框。检查：rtsp-url, api-stream-url, monitor-point-id");
        return;
    }

    // 3. 绑定添加流表单的提交事件 (始终绑定，权限在 addStream 函数中检查)
    addStreamForm.addEventListener('submit', addStream);
    if (firstPageButton) firstPageButton.addEventListener('click', () => {
        if (currentPage !== 1) {
            currentPage = 1;
            loadSavedStreams();
        }
    });
    if (prevPageButton) prevPageButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            loadSavedStreams();
        }
    });
    if (nextPageButton) nextPageButton.addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            loadSavedStreams();
        }
    });
    if (lastPageButton) lastPageButton.addEventListener('click', () => {
        if (currentPage !== totalPages) {
            currentPage = totalPages;
            loadSavedStreams();
        }
    });

    // 如果不是管理员或高级用户，隐藏整个添加视频流区域
    if (userInfo.role !== 'admin' && userInfo.role !== 'advanced') {
        if (addStreamSection) {
            addStreamSection.style.display = 'none';
        }
    }

    // 确保在加载视频流前，只有管理员用户才获取用户列表
    if (userInfo && userInfo.role === 'admin') {
        await fetchAndCacheUsers(); // 使用 await 确保在加载视频流前用户列表已缓存
    }

    // 4. 加载并显示已保存的视频流列表 (首次加载)
    loadSavedStreams(); 

    // 5. 设置定时刷新
    refreshInterval = setInterval(loadSavedStreams, 5000);

    // 6. 页面卸载时清除定时器
    window.addEventListener('beforeunload', () => {
        if (refreshInterval) {
            clearInterval(refreshInterval);
        }
    });

    // 7. 使用事件委托为动态生成的按钮绑定事件监听器，确保只绑定一次
    savedStreamsTableBody.addEventListener('click', (e) => {
        const target = e.target;
        // 检查点击的元素是否是按钮，并且是否被禁用
        if (target.tagName === 'BUTTON' && target.disabled) {
            console.log(`点击了已禁用的按钮: ${target.textContent}`);
            return; // 如果按钮被禁用，则不执行任何操作
        }

        if (target.classList.contains('start-resume-stream')) {
            // 获取当前行的 streamUrl 和 streamName
            const streamUrlToRestart = target.dataset.url;
            const streamNameTorestart = target.dataset.name;

            // 将这些值填充到添加视频流的表单中
            streamNameInput.value = streamNameTorestart;
            // 根据URL类型填充到对应的输入框
            if (streamUrlToRestart.startsWith('rtsp://') || streamUrlToRestart.startsWith('http://') && !streamUrlToRestart.includes('/api/video/v1/cameras/previewURLs')) { // 简单判断是否为RTSP/HTTP
                rtspUrlInput.value = streamUrlToRestart;
                apiStreamUrlInput.value = '';
                monitorPointIdInput.value = '';
            } else { // 否则认为是API直接URL或其他类型
                rtspUrlInput.value = '';
                apiStreamUrlInput.value = streamUrlToRestart;
                monitorPointIdInput.value = '';
            }

            // 模拟表单提交，触发 addStream 函数来创建新条目
            // addStream 函数期望一个 event 对象，这里提供一个模拟对象
            addStream({ preventDefault: () => {} });

        } else if (target.classList.contains('stop-stream')) {
            stopStream(target.dataset.streamUuid);
        } else if (target.classList.contains('view-results')) {
            viewResults(target.dataset.streamUuid);
        } else if (target.classList.contains('delete-stream')) {
            deleteStream(target.dataset.streamId);
        }
    });
}

// 异步函数：加载并显示已保存的视频流列表
async function loadSavedStreams() {
    try {
        console.log("loadSavedStreams: 正在尝试加载已保存的视频流。");
        console.log("loadSavedStreams: savedStreamsTableBody 在 innerHTML 赋值前:", savedStreamsTableBody); // 诊断日志

        const paginationControls = document.getElementById('streamPaginationControls');
        const currentPageSpan = document.getElementById('currentPageStreams');
        const totalPagesSpan = document.getElementById('totalPagesStreams');
        const prevPageButton = document.getElementById('prevPageStreams');
        const nextPageButton = document.getElementById('nextPageStreams');
        const firstPageButton = document.getElementById('firstPageStreams'); // 新增
        const lastPageButton = document.getElementById('lastPageStreams'); // 新增

        // 默认隐藏分页控件
        if (paginationControls) paginationControls.style.display = 'none';

        const skip = (currentPage - 1) * itemsPerPage;
        const limit = itemsPerPage;
        
        const response = await fetchWithAuth(`${API_BASE_URL}/streams/saved?skip=${skip}&limit=${limit}`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const responseData = await response.json(); // 获取整个响应对象
        console.log("[loadSavedStreams] Received response data from /streams/saved:", responseData); // 新增日志
        const streams = responseData.items; // 从 items 字段中获取流数组
        totalStreams = responseData.total; // 获取总数
        totalPages = Math.ceil(totalStreams / itemsPerPage); // 计算总页数

        // 防御性检查：确保 savedStreamsTableBody 不为 null 或 undefined
        if (!savedStreamsTableBody) {
            console.error("严重错误: savedStreamsTableBody 在尝试清除 innerHTML 时为 null 或 undefined。请确保 video_stream.html 中存在 'saved-streams-table' 表格及其 'tbody' 元素。");
            // 如果关键 DOM 元素缺失，则停止进一步执行
            return;
        }

        // 清空现有表格内容
        savedStreamsTableBody.innerHTML = '';

        // 更新分页控件状态
        if (currentPageSpan) currentPageSpan.textContent = currentPage;
        if (totalPagesSpan) totalPagesSpan.textContent = totalPages;
        if (prevPageButton) prevPageButton.disabled = (currentPage === 1);
        if (nextPageButton) nextPageButton.disabled = (currentPage === totalPages || totalStreams === 0);
        if (firstPageButton) firstPageButton.disabled = (currentPage === 1);
        if (lastPageButton) lastPageButton.disabled = (currentPage === totalPages || totalStreams === 0);

        if (streams.length === 0) {
            noStreamsMessage.classList.remove('hidden');
            savedStreamsTable.classList.add('hidden');
            if (paginationControls) paginationControls.style.display = 'none'; // 隐藏分页控件
            // 如果没有流，则停止轮询
            if (refreshInterval) {
                clearInterval(refreshInterval);
                refreshInterval = null;
                console.log("所有视频流已处理，停止刷新。");
            }
        } else {
            noStreamsMessage.classList.add('hidden');
            savedStreamsTable.classList.remove('hidden');
            if (paginationControls) paginationControls.style.display = 'flex'; // 显示分页控件

            let allStreamsCompletedOrTerminated = true;
            streams.forEach((stream, index) => {
                console.log("Processing stream:", stream); // Add this line to log the full stream object
                const row = document.createElement('tr');
                row.id = `stream-row-${stream.id}`; // Add ID for easier removal
                const statusInfo = getStatusTagClass(stream.status, stream.is_active);
                const entryTime = stream.created_at ? new Intl.DateTimeFormat('zh-CN', {
                    year: 'numeric',
                    month: '2-digit',
                    day: '2-digit',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false,
                    timeZone: 'Asia/Shanghai'
                }).format(new Date(stream.created_at)) : 'N/A';

                // 检查是否有未完成或活跃的流
                if (stream.is_active || (stream.status !== 'completed' && stream.status !== 'terminated' && stream.status !== 'failed' && stream.status !== 'stopped')) {
                    allStreamsCompletedOrTerminated = false;
                }

                // 确定按钮的文本和禁用状态
                let startResumeButtonText = '开始解析';
                let startResumeButtonDisabled = '';
                let startResumeButtonClass = 'primary'; // 默认是primary
                if (stream.status === 'processing' || stream.status === 'active') {
                    startResumeButtonText = '正在解析';
                    startResumeButtonDisabled = 'disabled';
                    startResumeButtonClass = 'processing-state-button'; // 使用新类
                } else if (stream.status === 'stopped' || stream.status === 'failed' || stream.status === 'terminated') {
                    startResumeButtonText = '开始解析'; // 明确改为“开始解析”
                }

                row.innerHTML = `
                    <td>${(currentPage - 1) * itemsPerPage + index + 1}</td> <!-- 序号 -->
                    <td>${stream.name}</td>
                    <td>${stream.stream_url}</td>
                    <td><span class="status-tag ${statusInfo.class}">${statusInfo.text}</span></td>
                    <td>${entryTime}</td>
                    ${userInfo.role === 'admin' ? `<td>${usersMap.get(stream.owner_id) || '未知用户'}</td>` : ''}
                    <td>
                        <div class="actions-buttons-container">
                            <button class="button ${startResumeButtonClass} start-resume-stream" data-stream-uuid="${stream.stream_uuid}" data-url="${stream.stream_url}" data-name="${stream.name}" ${startResumeButtonDisabled}>${startResumeButtonText}</button>
                            <button class="button secondary stop-stream" data-stream-uuid="${stream.stream_uuid}" ${!stream.is_active ? 'disabled' : ''}>停止解析</button>
                            <button class="button info view-results" data-stream-uuid="${stream.stream_uuid}">查看结果</button>
                            <button class="button danger delete-stream" data-stream-id="${stream.id}" ${userInfo.role === 'admin' || (userInfo.role === 'advanced' && stream.owner_id === userInfo.id) ? '' : 'disabled'}>删除</button>
                        </div>
                    </td>
                `;
                savedStreamsTableBody.appendChild(row);
            });

            // 如果所有流都已完成或终止，并且轮询还在进行中，则停止轮询
            if (allStreamsCompletedOrTerminated && refreshInterval) {
                clearInterval(refreshInterval);
                refreshInterval = null;
                console.log("所有视频流已完成解析，停止刷新。");
            }   
        }

    } catch (error) {
        console.error('加载已保存视频流失败:', error);
        if (paginationControls) paginationControls.style.display = 'none'; // 隐藏分页控件
        // 可以显示一个错误消息给用户
    }
}

// 处理添加新视频流
async function addStream(event) {
    // 权限检查：只有管理员和高级用户可以添加视频流
    if (!userInfo || (userInfo.role !== 'admin' && userInfo.role !== 'advanced')) {
        alert('您没有权限添加视频流。');
        if (event) event.preventDefault(); // 阻止表单默认提交行为
        return;
    }

    console.log("addStream function called."); // 添加日志以调试重复调用
    if (isAddingStream) {
        console.log("addStream is already in progress, ignoring duplicate call.");
        return; // 如果已经在处理中，则忽略本次调用
    }
    isAddingStream = true; // 设置标志为true，表示正在处理

    event.preventDefault();

    const streamName = streamNameInput.value.trim();
    const rtspUrl = rtspUrlInput.value.trim(); // 获取RTSP URL
    const apiStreamUrl = apiStreamUrlInput.value.trim(); // 获取API URL
    const monitorPointId = monitorPointIdInput.value.trim(); // 获取监控点编号

    // 确保只提供一种URL来源
    const providedUrls = [rtspUrl, apiStreamUrl, monitorPointId].filter(Boolean);

    if (providedUrls.length === 0) {
        addStreamMessage.textContent = "必须提供RTSP/HTTP视频流地址、API视频流地址或监控点编号其中一个。";
        addStreamMessage.classList.remove('hidden');
        const submitButton = addStreamForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = "保存并开始解析";
        }
        isAddingStream = false; // 重置标志
        return;
    }

    if (providedUrls.length > 1) {
        addStreamMessage.textContent = "只能提供RTSP/HTTP视频流地址、API视频流地址或监控点编号其中一个，不能同时提供。";
        addStreamMessage.classList.remove('hidden');
        const submitButton = addStreamForm.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = "保存并开始解析";
        }
        isAddingStream = false; // 重置标志
        return;
    }

    let finalStreamUrl = '';
    let requestBody = {
        stream_name: streamName,
        camera_id: "", // 假设为空，如果需要可以从UI获取
    };

    if (rtspUrl) {
        finalStreamUrl = rtspUrl;
        requestBody.rtsp_url = finalStreamUrl;
    } else if (apiStreamUrl) {
        finalStreamUrl = apiStreamUrl;
        requestBody.api_stream_url = finalStreamUrl;
    } else if (monitorPointId) {
        // 调用外部API获取视频流URL
        addStreamMessage.textContent = "正在通过监控点编号获取视频流地址...";
        addStreamMessage.classList.remove('hidden');
        addStreamMessage.classList.add('message-info'); // 添加信息样式

        try {
            const externalApiUrl = '/api/video/v1/cameras/previewURLs'; // 外部API地址
            const externalRequestBody = {
                cameraIndexCode: monitorPointId,
                streamType: 0, // 主码流
                protocol: "rtsp", // RTSP协议
                transmode: 1, // TCP传输
                artemisConfig: { 
                    host: "YOUR_ARTEMIS_HOST", // 替换为实际的Artemis Host，例如 "10.0.0.1:443"
                    appKey: "YOUR_ACCESS_KEY",     // 替换为实际的Access Key
                    appSecret: "YOUR_SECRET_KEY",     // 替换为实际的Secret Key
                }
            };
            console.log(`调用外部API: ${externalApiUrl}，请求体:`, externalRequestBody); // 调试日志

            const externalApiResponse = await fetch(externalApiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    // 如果外部API需要认证，这里需要添加认证头部
                    // 'Authorization': 'Bearer YOUR_EXTERNAL_API_TOKEN'
                },
                body: JSON.stringify(externalRequestBody)
            });

            const externalApiData = await externalApiResponse.json();

            if (externalApiResponse.ok && externalApiData.code === "0" && externalApiData.data && externalApiData.data.url) {
                finalStreamUrl = externalApiData.data.url;
                requestBody.api_stream_url = finalStreamUrl; // 将获取到的URL作为api_stream_url发送给后端
                addStreamMessage.classList.add('hidden'); // 隐藏信息消息
            } else {
                const errorMsg = externalApiData.msg || "未知错误";
                addStreamMessage.textContent = `通过监控点编号获取视频流地址失败: ${errorMsg}。`;
                addStreamMessage.classList.remove('message-info');
                addStreamMessage.classList.add('message-error');
                addStreamMessage.classList.remove('hidden');
                const submitButton = addStreamForm.querySelector('button[type="submit"]');
                if (submitButton) {
                    submitButton.disabled = false;
                    submitButton.textContent = "保存并开始解析";
                }
                isAddingStream = false; // 重置标志
                return;
            }
        } catch (error) {
            console.error('调用外部API时发生网络错误:', error);
            addStreamMessage.textContent = "调用外部API时发生网络错误，请检查网络连接。";
            addStreamMessage.classList.remove('message-info');
            addStreamMessage.classList.add('message-error');
            addStreamMessage.classList.remove('hidden');
            const submitButton = addStreamForm.querySelector('button[type="submit"]');
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = "保存并开始解析";
            }
            isAddingStream = false; // 重置标志
            return;
        }
    }

    const submitButton = addStreamForm.querySelector('button[type="submit"]');
    if (submitButton) {
        submitButton.disabled = true; // 禁用按钮防止重复提交
        submitButton.textContent = "正在解析..."; // 更改按钮文本
    }

    addStreamMessage.classList.add('hidden'); // 隐藏之前的消息

    try {
        console.log("Sending request to /streams/start with body:", requestBody); // 调试日志
        const response = await fetchWithAuth(`${API_BASE_URL}/streams/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody) // 发送构建好的数据体
        });

        const data = await response.json();

        if (response.ok) {
            addStreamMessage.textContent = data.message || "视频流解析已成功启动！";
            addStreamMessage.classList.remove('message-error');
            addStreamMessage.classList.add('message-success');
            addStreamMessage.classList.remove('hidden');
            addStreamForm.reset();
            loadSavedStreams(); // 重新加载列表
        } else {
            const errorMessage = data.detail || "启动视频流解析失败。";
            addStreamMessage.textContent = errorMessage;
            addStreamMessage.classList.remove('message-success');
            addStreamMessage.classList.add('message-error');
            addStreamMessage.classList.remove('hidden');
        }
    } catch (error) {
        console.error('启动视频流解析时发生错误:', error);
        if (error.message !== 'Unauthorized') {
            addStreamMessage.textContent = "启动视频流解析时发生网络错误。";
            addStreamMessage.classList.remove('message-success');
            addStreamMessage.classList.add('message-error');
            addStreamMessage.classList.remove('hidden');
        }
    } finally {
        isAddingStream = false; // 重置标志
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = "保存并开始解析";
        }
    }
}

// 启动或恢复视频流解析
async function startStream(streamUuid, streamUrl, streamName) {
    // 找到对应的表格行
    const targetRow = document.querySelector(`button[data-stream-uuid="${streamUuid}"]`).closest('tr');
    if (!targetRow) {
        console.error(`未找到 Stream UUID 为 ${streamUuid} 的行。`);
        alert(`操作失败：未找到视频流。`);
        return;
    }

    const startButton = targetRow.querySelector('.start-resume-stream');
    const stopButton = targetRow.querySelector('.stop-stream');
    const statusSpan = targetRow.querySelector('.status-tag');

    // 立即更新前端状态为"启动中"
    if (startButton) startButton.disabled = true;
    if (stopButton) stopButton.disabled = true; // 暂时禁用停止，直到确认后端启动成功
    if (statusSpan) {
        const statusInfo = getStatusTagClass("starting", true);
        statusSpan.className = `status-tag ${statusInfo.class}`;
        statusSpan.textContent = `${statusInfo.text}`;
    }

    // Temporarily clear refresh interval to prevent immediate overwrite
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }

    try {
        let response;
        let apiUrl;
        let requestBody = {};

        // 根据是否提供了 streamUuid 来判断是恢复现有流还是启动新流
        // 注意：这里的 startStream 函数预计只被用于恢复现有流，新流的启动由 addStream 函数处理
        if (streamUuid) {
            apiUrl = `${API_BASE_URL}/streams/resume/${streamUuid}`;
            response = await fetchWithAuth(apiUrl, {
                method: 'POST',
            });
        } else {
            // 理论上这部分代码不会被执行，因为 startStream 应该只处理已有流的恢复
            // 但作为备用，如果意外调用，则按原逻辑启动新流
            apiUrl = `${API_BASE_URL}/streams/start`;
            requestBody = { rtsp_url: streamUrl, stream_name: streamName, camera_id: null };
            response = await fetchWithAuth(apiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });
        }

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `HTTP error! status: ${response.status}`);
        }

        // 成功，更新为活跃状态
        if (statusSpan) {
            const statusInfo = getStatusTagClass("active", true);
            statusSpan.className = `status-tag ${statusInfo.class}`;
            statusSpan.textContent = `${statusInfo.text} (活跃)`;
        }
        if (startButton) startButton.disabled = true;
        if (stopButton) stopButton.disabled = false;
        alert(`视频流 (ID: ${streamUuid}) 已成功启动/恢复解析。`);

    } catch (error) {
        console.error(`启动/恢复视频流 (ID: ${streamUuid}) 失败:`, error);
        // 失败，更新为失败状态，并重新启用开始按钮
        if (statusSpan) {
            const statusInfo = getStatusTagClass("failed", false);
            statusSpan.className = `status-tag ${statusInfo.class}`;
            statusSpan.textContent = `${statusInfo.text}`;
        }
        if (startButton) startButton.disabled = false;
        if (stopButton) stopButton.disabled = true;
        alert(`启动/恢复视频流失败: ${error.message}`);
    } finally { // Ensure interval is restarted even on error
        if (!refreshInterval) {
            refreshInterval = setInterval(loadSavedStreams, 5000);
        }
    }
}

// 异步函数：处理停止视频流解析
async function stopStream(streamUuid) {
    // 找到对应的表格行
    const targetRow = document.querySelector(`button[data-stream-uuid="${streamUuid}"]`).closest('tr');
    if (!targetRow) {
        console.error(`未找到 Stream UUID 为 ${streamUuid} 的行。`);
        alert(`操作失败：未找到视频流。`);
        return;
    }

    const startButton = targetRow.querySelector('.start-resume-stream');
    const stopButton = targetRow.querySelector('.stop-stream');
    const statusSpan = targetRow.querySelector('.status-tag');

    // 立即更新前端状态为"停止中"
    if (startButton) startButton.disabled = true; // 暂时禁用开始按钮
    if (stopButton) stopButton.disabled = true;
    if (statusSpan) {
        const statusInfo = getStatusTagClass("stopped", false); // Assuming "stopped" is the interim status
        statusSpan.className = `status-tag ${statusInfo.class}`;
        statusSpan.textContent = `停止中`; // Custom text for "stopping"
    }

    // Temporarily clear refresh interval to prevent immediate overwrite
    if (refreshInterval) {
        clearInterval(refreshInterval);
        refreshInterval = null;
    }

    try {
        const response = await fetchWithAuth(`${API_BASE_URL}/streams/stop/${streamUuid}`, {
            method: 'POST',
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || `HTTP error! status: ${response.status}`);
        }

        // 成功，更新为停止状态
        if (statusSpan) {
            const statusInfo = getStatusTagClass("stopped", false);
            statusSpan.className = `status-tag ${statusInfo.class}`;
            statusSpan.textContent = `${statusInfo.text}`;
        }
        if (startButton) startButton.disabled = false;
        if (stopButton) stopButton.disabled = true;
        alert(`视频流 (ID: ${streamUuid}) 已成功停止解析。`);

    } catch (error) {
        console.error(`停止视频流 (ID: ${streamUuid}) 失败:`, error);
        // 失败，更新为失败状态，并重新启用停止按钮
        if (statusSpan) {
            const statusInfo = getStatusTagClass("failed", false);
            statusSpan.className = `status-tag ${statusInfo.class}`;
            statusSpan.textContent = `${statusInfo.text}`;
        }
        if (startButton) startButton.disabled = false; // Re-enable start if it was disabled for stopping
        if (stopButton) stopButton.disabled = false; // Re-enable stop if it failed to stop
        alert(`停止视频流失败: ${error.message}`);
    } finally {
        if (!refreshInterval) {
            refreshInterval = setInterval(loadSavedStreams, 5000);
        }
    }
}

// 查看解析结果
function viewResults(streamUuid) {
    // 打开新的标签页并传递 streamUuid
    window.open(`/live_stream_results?streamId=${streamUuid}`, '_blank');
}

// 获取状态标签的CSS类
function getStatusTagClass(status, isActive) {
    let text = '';
    let className = '';

    switch (status) {
        case 'processing':
        case 'active': // Both mean active processing
            text = '正在解析';
            className = 'processing';
            break;
        case 'completed':
            text = '已完成';
            className = 'completed';
            break;
        case 'failed':
            text = '失败';
            className = 'failed';
            break;
        case 'stopped':
            text = '已停止';
            className = 'stopped';
            break;
        case 'inactive':
            text = '未启动';
            className = 'inactive';
            break;
        case 'terminated':
            text = '已终止';
            className = 'terminated';
            break;
        case 'starting': // Frontend internal state for starting
            text = '启动中';
            className = 'processing';
            break;
        default:
            text = `未知 (${status})`; // Fallback for unhandled statuses
            className = '';
    }

    // if (isActive && status !== 'starting') { // Only append (活跃) if actually active from backend, and not a local 'starting' state
    //     text += ' (活跃)';
    // }

    return { text: text, class: className };
}

// 删除视频流
async function deleteStream(streamId) {
    if (isDeletingStream) {
        console.log("Delete stream is already in progress, ignoring duplicate call.");
        return; // 如果已经在处理中，则忽略本次调用
    }
    isDeletingStream = true; // 设置标志为true，表示正在处理

    console.log(`尝试删除视频流 ID: ${streamId}`);
    if (!confirm(`确定要删除视频流 ID: ${streamId} 吗？这将同时删除所有相关人物特征和保存的视频文件！`)) {
        isDeletingStream = false; // 用户取消时重置标志
        return;
    }

    try {
        const response = await fetchWithAuth(`${API_BASE_URL}/streams/delete/${streamId}`, {
            method: 'DELETE',
        });

        console.log("删除请求响应:", response);

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        // 删除成功后，从表格中移除该行
        const rowToRemove = document.getElementById(`stream-row-${streamId}`);
        if (rowToRemove) {
            rowToRemove.remove();
        }

        // 重新加载列表以确保状态最新
        loadSavedStreams();

        console.log(`视频流 (ID: ${streamId}) 已成功删除。`);
        alert(`视频流 (ID: ${streamId}) 已成功删除。`);

    } catch (error) {
        console.error(`删除视频流 (ID: ${streamId}) 失败:`, error);
        alert(`删除视频流 (ID: ${streamId}) 失败: ${error.message}`);
    } finally {
        isDeletingStream = false; // 重置标志
    }
}