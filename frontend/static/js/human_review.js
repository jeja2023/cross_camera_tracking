import { initializeAuthenticatedPage } from '/static/js/common.js';
import { Auth, API_BASE_URL } from '/static/js/auth.js';

document.addEventListener('DOMContentLoaded', () => {
    let pollingInterval = null; // 用于存储 setInterval 的 ID

    const updateProgressModal = (status, progress, message) => {
        retrainProgressBar.style.width = `${progress}%`;
        
        let displayMessage = message; // 默认显示后端传来的 message

        // 根据 Celery 状态和后端自定义的状态消息来更新显示文本
        switch (status) {
            case 'PENDING':
                displayMessage = '任务等待中...';
                break;
            case 'STARTED':
                displayMessage = '任务已启动...';
                break;
            case 'PROGRESS': // 处理 Celery 的通用 PROGRESS 状态
                // 如果消息是通用或空，根据进度值推断阶段
                if (!message || message === '任务正在进行中...') {
                    if (progress <= 30) { // 假设数据准备阶段的进度在 0% 到 30%
                        displayMessage = `数据准备中: ${progress}%`;
                    } else if (progress > 30 && progress <= 70) { // 假设模型训练阶段的进度在 30% 到 70%
                        displayMessage = `模型训练中: ${progress}%`;
                    } else if (progress > 70 && progress < 100) { // 假设后续阶段的进度在 70% 到 99%
                        displayMessage = `任务正在进行中: ${progress}%`; // 更通用的描述，以防有其他未定义的 PROGRESS 阶段
                    } else { // 进度为 100% 但状态仍为 PROGRESS
                        displayMessage = `任务完成中: ${progress}%`;
                    }
                } else {
                    displayMessage = message; // 否则，使用后端提供的具体消息
                }
                break;
            case 'PREPARING_DATA':
                displayMessage = `模型训练中: ${message}`;
                break;
            case 'TRAINING_MODEL':
                displayMessage = `数据准备中: ${message}`;
                break;
            case 'UPDATING_CONFIG':
                displayMessage = `更新配置中: ${message}`;
                break;
            case 'RELOADING_FAISS':
                displayMessage = `重新加载索引中: ${message}`;
                break;
            case 'SUCCESS':
                displayMessage = '任务已成功完成！';
                break;
            case 'FAILED':
                displayMessage = '任务失败！';
                break;
            case 'SKIPPED':
                displayMessage = '任务已跳过。';
                break;
            default:
                displayMessage = message; // 未知状态，直接显示 message
        }

        retrainProgressBar.textContent = `${progress}% - ${displayMessage}`;
        retrainStatusText.textContent = `状态: ${displayMessage} (进度: ${progress}%)`;

        // 根据状态调整进度条颜色（可选）
        if (status === 'FAILED') {
            retrainProgressBar.style.backgroundColor = '#dc3545'; // 红色
        } else if (status === 'SUCCESS') {
            retrainProgressBar.style.backgroundColor = '#28a745'; // 绿色
        } else {
            retrainProgressBar.style.backgroundColor = '#007bff'; // 蓝色
        }
    };

    const pollTaskStatus = async (taskId) => {
        if (pollingInterval) {
            clearInterval(pollingInterval);
        }

        pollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/admin/task-status/${taskId}`, {
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                    }
                });
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                console.log("Polling Task Status Data:", data); // 添加调试日志
                
                // 确保从 data.meta 中获取自定义的 progress 和 status (message)
                const currentProgress = data.meta && data.meta.progress !== undefined ? data.meta.progress : data.progress;
                const currentMessage = data.meta && data.meta.status !== undefined ? data.meta.status : data.message;

                updateProgressModal(data.status, currentProgress, currentMessage);

                if (data.status === 'SUCCESS') {
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                    loadPersons(); // 任务完成后重新加载人物列表
                    if (data.result) { // 检查 result 是否存在且非空
                        alert('模型再训练任务已成功完成！');
                    } else { // status 是 SUCCESS 但 result 是 false 或其他值
                        alert(`模型再训练任务失败: ${data.message || '未知错误'}`);
                    }
                    retrainProgressModal.style.display = 'none'; // 隐藏进度模态框
                } else if (data.status === 'FAILED' || data.status === 'SKIPPED' || data.status === 'REVOKED' || data.status === 'TERMINATED') {
                    clearInterval(pollingInterval);
                    pollingInterval = null;
                    loadPersons(); // 任务完成后重新加载人物列表
                    alert(`模型再训练任务失败或被取消: ${data.message || '未知错误'}`);
                    retrainProgressModal.style.display = 'none'; // 隐藏进度模态框
                }
            } catch (error) {
                console.error('查询任务状态失败:', error);
                clearInterval(pollingInterval);
                pollingInterval = null;
                updateProgressModal('FAILED', 0, '查询任务状态失败。');
                alert('查询模型训练任务状态失败。');
            }
        }, 3000); // 每3秒查询一次
    };

    // 从 Auth.js 导入
    // import { initializeAuthenticatedPage } from '/static/js/common.js'; // 已经在文件顶部导入
    // import { Auth } from '/static/js/auth.js'; // 已经在文件顶部导入

    console.log("human_review.js: DOMContentLoaded event fired. (Unified)");

    // 确保通用认证页面初始化
    initializeAuthenticatedPage(); 

    // 用户信息显示逻辑
    const userInfo = Auth.getUserInfo();
    console.log('human_review.js: User Info after init', userInfo);
    const usernameDisplay = document.getElementById('username-display');
    const roleDisplay = document.getElementById('role-display');

    if (userInfo) {
        if (usernameDisplay) usernameDisplay.textContent = userInfo.username;
        if (roleDisplay) {
            let displayRole = '未知角色';
            switch (userInfo.role) {
                case 'admin':
                    displayRole = '管理员';
                    break;
                case 'advanced':
                    displayRole = '高级用户';
                    break;
                case 'user':
                    displayRole = '普通用户';
                    break;
            }
            roleDisplay.textContent = displayRole;
        }
        console.log("human_review.js: Username and role displayed.");
    } else {
        console.warn("human_review.js: User info not available, or not logged in. Redirecting should be handled by common.js.");
    }

    // 原始的页面加载逻辑
    const personListDiv = document.getElementById('person-list');
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const firstPageButton = document.getElementById('first-page');
    const lastPageButton = document.getElementById('last-page');
    const pageInfoSpan = document.getElementById('page-info');
    const filterStatusSelect = document.getElementById('filter-status');
    const applyFilterButton = document.getElementById('apply-filter-button');

    // 模态框元素
    const correctionModal = document.getElementById('correction-modal');
    const closeModalButton = correctionModal.querySelector('.close-button');
    const correctionTypeSelect = document.getElementById('correction-type');
    const correctionDetailsTextarea = document.getElementById('correction-details');
    const submitCorrectionButton = document.getElementById('submit-correction-button');
    let currentPersonIdToCorrect = null; // 用于存储当前需要纠正的人物ID
    let currentPersonUuidToCorrect = null; // 用于存储当前需要纠正的人物UUID
    let selectedTargetPersonUuid = null; // 用于存储选定的合并目标人物UUID

    // 新增：模型训练进度模态框元素
    const retrainProgressModal = document.getElementById('retrain-progress-modal');
    const retrainProgressBar = document.getElementById('retrain-progress-bar');
    const retrainStatusText = document.getElementById('retrain-status-text');
    const closeRetrainModalButton = document.getElementById('close-retrain-modal-button');

    // 新增：关闭模型训练进度模态框按钮事件
    if (closeRetrainModalButton) {
        closeRetrainModalButton.addEventListener('click', () => {
            retrainProgressModal.style.display = 'none';
            // 如果有正在进行的轮询，也应该停止
            if (pollingInterval) {
                clearInterval(pollingInterval);
                pollingInterval = null;
            }
        });
    }

    // 模态框元素（这些应该在 DOMContentLoaded 顶部声明，以便在整个作用域内可访问）
    const mergePersonSection = document.getElementById('merge-person-section');
    const targetPersonUuidInput = document.getElementById('target-person-uuid');
    const searchTargetPersonUuidButton = document.getElementById('search-target-person-uuid'); 
    const targetPersonIdCardInput = document.getElementById('target-person-idcard'); 
    const searchTargetPersonIdCardButton = document.getElementById('search-target-person-idcard'); 
    const targetPersonInfoDiv = document.getElementById('target-person-info');
    const targetInfoUuidSpan = document.getElementById('target-info-uuid');
    const targetInfoNameSpan = document.getElementById('target-info-name');
    const targetInfoIdCardSpan = document.getElementById('target-info-idcard'); 
    const targetInfoImageImg = document.getElementById('target-info-image');
    const selectTargetPersonButton = document.getElementById('select-target-person');

    let currentPage = 1;
    const itemsPerPage = 20; // 每页显示的人物数量
    let currentFilterStatus = 'unverified';
    let totalPages = 1;
    const pageTitleElement = document.querySelector('#human-review-section h2'); // 获取页面标题元素

    async function fetchPersons(page, status) {
        const skip = (page - 1) * itemsPerPage;
        try {
            let url = `${API_BASE_URL}/hitl/persons/unverified?skip=${skip}&limit=${itemsPerPage}`;
            if (status === 'all') {
                url = `${API_BASE_URL}/api/persons/all?skip=${skip}&limit=${itemsPerPage}`;
            } else if (status === 'verified') {
                url = `${API_BASE_URL}/api/persons/all?skip=${skip}&limit=${itemsPerPage}&is_verified=true&marked_for_retrain=false`;
            } else if (status === 'marked_for_retrain') {
                url = `${API_BASE_URL}/api/persons/all?skip=${skip}&limit=${itemsPerPage}&marked_for_retrain=true&is_verified=false`;
            } else if (status === 'trained') {
                url = `${API_BASE_URL}/api/persons/all?skip=${skip}&limit=${itemsPerPage}&is_trained=true`;
            }
            
            const response = await fetch(url, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            });

            if (!response.ok) {
                if (response.status === 401 || response.status === 403) {
                    alert('会话过期或无权限，请重新登录。');
                    window.location.href = '/login';
                    return;
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('获取人物列表失败:', error);
            alert('获取人物列表失败。');
            return { total: 0, items: [] };
        }
    }

    function renderPersons(persons) {
        personListDiv.innerHTML = '';
        if (persons.length === 0) {
            personListDiv.innerHTML = '<p>没有需要审核的人物。</p>';
            return;
        }

        persons.forEach(person => {
            console.log("Rendering person:", person); 
            console.log("Confidence score for person:", person.confidence_score); 
            console.log("DEBUG: Person UUID is", person.uuid); // 新增日志
            console.log("DEBUG: Person is_trained is", person.is_trained); // 新增日志
            console.log("DEBUG: Full person object:", person); // 新增：打印完整的 person 对象

            const personCard = document.createElement('div');
            personCard.className = 'person-card';

            const imageUrl = person.crop_image_path ? `${API_BASE_URL}/${person.crop_image_path.replace(/\\/g, '/')}` : `${API_BASE_URL}/static/images/default_avatar.png`;
            const fullFrameUrl = person.full_frame_image_path ? `${API_BASE_URL}/${person.full_frame_image_path.replace(/\\/g, '/')}` : '';

            console.debug("DEBUG: Final imageUrl for person card:", imageUrl);
            console.debug("DEBUG: Final fullFrameUrl for person card:", fullFrameUrl);

            const finalCropImagePath = imageUrl;
            const finalFullFrameImagePath = fullFrameUrl;

            let statusText;
            let statusClass;

            if (person.marked_for_retrain) {
                statusText = '待再训练';
                statusClass = 'marked-for-retrain';
            } else if (person.is_trained) { // 新增：检查是否已训练
                statusText = '已训练';
                statusClass = 'trained';
            } else if (person.is_verified) {
                // 检查 correction_type_display 来决定最终显示的状态和样式
                if (person.correction_type_display === "身份已合并确认") {
                    statusText = '已合并确认';
                    statusClass = 'merged-confirmed'; // 使用新的类
                } else if (person.correction_type_display === "身份已注册确认") { // 新增：主动注册状态
                    statusText = '身份已确认'; // 显示为“身份已确认”
                    statusClass = 'verified'; // 使用已确认的绿色样式
                } else if (person.correction_type_display) { // 如果有其他具体的纠正类型显示
                    statusText = person.correction_type_display; 
                    statusClass = 'corrected'; // 仍使用 corrected 类
                } else if (person.correction_details && person.correction_details !== '') {
                    statusText = '已纠正';
                    statusClass = 'corrected';
                } else {
                    statusText = '已确认无误'; 
                    statusClass = 'verified';
                }
            } else {
                statusText = '未审核';
                statusClass = 'unverified';
            }
            
            personCard.innerHTML = `
                <img src="${finalCropImagePath}" alt="人物裁剪图" data-full-frame-src="${finalFullFrameImagePath || finalCropImagePath}">
                <div class="person-info">
                    <h3>人物 ID: ${person.id}</h3>
                    <p>姓名: ${person.name || 'N/A'}</p>
                    <p>身份证号/ID: ${person.id_card || 'N/A'}</p>
                    <p>UUID: ${person.uuid}</p>
                    <p>创建时间: ${new Date(person.created_at).toLocaleString()}</p>
                    <p>状态: <span class="status ${statusClass}">${statusText}</span></p>
                    <p>置信度: ${person.confidence_score !== undefined && person.confidence_score !== null ? (person.confidence_score * 100).toFixed(2) + '%' : 'N/A'}</p>
                    ${person.video_uuid ? `<p>视频: ${person.video_uuid || person.video_name}</p>` : ''}
                    ${person.stream_uuid ? `<p>视频流: ${person.stream_uuid || person.stream_name}</p>` : ''}
                    ${person.upload_image_uuid ? `<p>图片: ${person.upload_image_uuid || person.upload_image_filename}</p>` : ''}
                </div>
                <div class="person-actions">
                    <button class="btn-verify" data-id="${person.id}" data-uuid="${person.uuid}">确认无误</button>
                    <button class="btn-correct" data-id="${person.id}" data-uuid="${person.uuid}">纠正</button>
                    <button class="btn-retrain" data-id="${person.id}" data-uuid="${person.uuid}" data-marked="${person.marked_for_retrain}">${person.marked_for_retrain ? '取消再训练' : '标记再训练'}</button>
                </div>
            `;
            personListDiv.appendChild(personCard);
        });
    }

    function updatePaginationControls(total) {
        totalPages = Math.ceil(total / itemsPerPage);
        pageInfoSpan.textContent = `页码 ${currentPage} / ${totalPages}`;
        firstPageButton.disabled = currentPage === 1;
        prevPageButton.disabled = currentPage === 1;
        nextPageButton.disabled = currentPage === totalPages || totalPages === 0;
        lastPageButton.disabled = currentPage === totalPages || totalPages === 0;
    }

    async function loadPersons() {
        const data = await fetchPersons(currentPage, currentFilterStatus);
        renderPersons(data.items);
        updatePaginationControls(data.total);
        updatePageTitle(currentFilterStatus); 

        // 根据当前过滤状态显示/隐藏模型训练按钮
        const retrainModelButton = document.getElementById('retrain-model-button');
        if (retrainModelButton) {
            if (currentFilterStatus === 'marked_for_retrain') {
                retrainModelButton.style.display = 'inline-block'; // 显示按钮
            } else {
                retrainModelButton.style.display = 'none'; // 隐藏按钮
            }
        }
    }

    function updatePageTitle(status) {
        let title = '人物列表'; 
        switch (status) {
            case 'unverified':
                title = '待审核人物列表';
                break;
            case 'verified':
                title = '已审核人物列表';
                break;
            case 'marked_for_retrain':
                title = '待再训练人物列表';
                break;
            case 'all':
                title = '所有人物列表';
                break;
            case 'trained':
                title = '已训练人物列表';
                break;
        }
        if (pageTitleElement) {
            pageTitleElement.textContent = title;
        }
    }

    // --- Start: Static element event listeners (defined once) ---

    // 当纠正类型改变时，显示/隐藏合并人员区域
    correctionTypeSelect.addEventListener('change', () => {
        if (correctionTypeSelect.value === 'merge') {
            mergePersonSection.style.display = 'block';
        } else {
            mergePersonSection.style.display = 'none';
            selectedTargetPersonUuid = null; // 清除已选择的目标人物
            targetPersonUuidInput.value = '';
            targetPersonIdCardInput.value = ''; // 清除身份证输入框
            targetPersonInfoDiv.style.display = 'none';
        }
    });

    // 纠正按钮点击事件，打开模态框 
    document.querySelectorAll('.btn-correct').forEach(button => {
        button.onclick = (event) => {
            currentPersonIdToCorrect = event.target.dataset.id;
            currentPersonUuidToCorrect = event.target.dataset.uuid; // 获取被纠正人物的 UUID
            // 清空模态框内容和合并相关字段
            correctionTypeSelect.value = 'misdetection';
            correctionDetailsTextarea.value = '';
            mergePersonSection.style.display = 'none';
            targetPersonUuidInput.value = '';
            targetPersonIdCardInput.value = ''; // 清空身份证输入框
            targetPersonInfoDiv.style.display = 'none';
            selectedTargetPersonUuid = null;
            correctionModal.style.display = 'flex'; 
        };
    });

    // 搜索目标人物 (UUID)
    searchTargetPersonUuidButton.addEventListener('click', async () => {
        const uuid = targetPersonUuidInput.value.trim();
        if (!uuid) {
            alert('请输入要搜索的目标人物 UUID。');
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/api/persons/${uuid}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            });

            if (response.ok) {
                const person = await response.json();
                targetInfoUuidSpan.textContent = person.uuid;
                targetInfoNameSpan.textContent = person.name || 'N/A';
                targetInfoIdCardSpan.textContent = person.id_card || 'N/A'; // 更新身份证号显示
                if (person.crop_image_path) {
                    targetInfoImageImg.src = `${API_BASE_URL}/${person.crop_image_path.replace(/\\/g, '/')}`;
                    targetInfoImageImg.style.display = 'block';
                } else {
                    targetInfoImageImg.style.display = 'none';
                }
                targetPersonInfoDiv.style.display = 'block';
            } else if (response.status === 404) {
                alert('未找到指定 UUID 的人物。');
                targetPersonInfoDiv.style.display = 'none';
            } else {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        } catch (error) {
            console.error('搜索目标人物失败:', error);
            alert('搜索目标人物失败。');
            targetPersonInfoDiv.style.display = 'none';
        }
    });

    // 搜索目标人物 (身份证号) - 新增功能
    searchTargetPersonIdCardButton.addEventListener('click', async () => {
        const idCard = targetPersonIdCardInput.value.trim();
        if (!idCard) {
            alert('请输入要搜索的目标人物身份证号。');
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/api/persons/by_id_card/${idCard}`, {
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                }
            });

            if (response.ok) {
                const data = await response.json(); // 这是一个 PaginatedPersonsResponse
                if (data.items && data.items.length > 0) {
                    const person = data.items[0]; // 默认取第一个匹配的人物
                    targetInfoUuidSpan.textContent = person.uuid;
                    targetInfoNameSpan.textContent = person.name || 'N/A';
                    targetInfoIdCardSpan.textContent = person.id_card || 'N/A';
                    if (person.crop_image_path) {
                        targetInfoImageImg.src = `${API_BASE_URL}/${person.crop_image_path.replace(/\\/g, '/')}`;
                        targetInfoImageImg.style.display = 'block';
                    } else {
                        targetInfoImageImg.style.display = 'none';
                    }
                    targetPersonInfoDiv.style.display = 'block';
                    alert(`找到人物：${person.name || person.uuid}。`);
                } else {
                    alert('未找到指定身份证号的人物。');
                    targetPersonInfoDiv.style.display = 'none';
                }
            } else {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
        } catch (error) {
            console.error('搜索目标人物失败:', error);
            alert('搜索目标人物失败。');
            targetPersonInfoDiv.style.display = 'none';
        }
    });

    // 选择目标人物
    selectTargetPersonButton.addEventListener('click', () => {
        selectedTargetPersonUuid = targetInfoUuidSpan.textContent;
        alert(`已选择人物 UUID: ${selectedTargetPersonUuid} 作为合并目标。`);
        // 可以在这里隐藏搜索结果，但保留输入框的值
    });

    // 关闭模态框
    closeModalButton.onclick = () => {
        correctionModal.style.display = 'none';
    };

    // 点击模态框外部关闭
    window.onclick = (event) => {
        if (event.target === correctionModal) {
            correctionModal.style.display = 'none';
        }
    };

    // 提交纠正按钮点击事件
    submitCorrectionButton.onclick = async () => {
        const personId = currentPersonIdToCorrect;
        const personUuid = currentPersonUuidToCorrect; // 被纠正的人物 UUID
        const correctionType = correctionTypeSelect.value;
        const details = correctionDetailsTextarea.value;
        const userInfo = Auth.getUserInfo(); // 获取用户信息

        if (!details.trim()) {
            alert("请填写详细说明。");
            return;
        }

        // 根据纠正类型生成消息，尽管后端现在可能不直接使用 message
        let message = `人物 ${personUuid} 已被 ${userInfo.username} 纠正为类型: ${correctionType}。`;
        if (details) {
            message += ` 详细说明: ${details}`; // 如果有详细说明，则附加
        }

        let payload = {
            person_id: parseInt(personId), 
            correction_type: correctionType,
            details: details, // 详细信息
            message: message, // 仍传递 message，但后端可能只用 details
            username: userInfo.username, // 仍传递 username，后端可能会用这个而不是 user_id 查找
            corrected_by_user_id: userInfo.id // 新增：传递用户 ID
            // ip_address 将在后端路由层获取
        };

        if (correctionType === 'merge') {
            if (!selectedTargetPersonUuid) {
                alert("请搜索并选择一个目标人物进行合并。");
                return;
            }
            if (selectedTargetPersonUuid === personUuid) {
                alert("不能将人物合并到其自身。");
                return;
            }
            payload.target_person_uuid = selectedTargetPersonUuid; // 添加目标人物 UUID
        }

        try {
            const response = await fetch(`${API_BASE_URL}/hitl/persons/${personUuid}/correct`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            alert(`人物 ${personUuid} 已提交纠正 (${correctionType})。`);
            correctionModal.style.display = 'none'; 
            loadPersons(); 
        } catch (error) {
            console.error('提交纠正失败:', error);
            alert('提交纠正失败。');
        }
    };

    // Pagination and filter event listeners (static elements)
    if (firstPageButton) {
        firstPageButton.addEventListener('click', () => {
            if (currentPage !== 1) {
                currentPage = 1;
                loadPersons();
            }
        });
    }

    if (prevPageButton) {
        prevPageButton.addEventListener('click', () => {
            if (currentPage > 1) {
                currentPage--;
                loadPersons();
            }
        });
    }

    if (nextPageButton) {
        nextPageButton.addEventListener('click', () => {
            if (currentPage < totalPages) {
                currentPage++;
                loadPersons();
            }
        });
    }

    if (lastPageButton) {
        lastPageButton.addEventListener('click', () => {
            if (currentPage !== totalPages) {
                currentPage = totalPages;
                loadPersons();
            }
        });
    }

    if (applyFilterButton) {
        applyFilterButton.addEventListener('click', () => {
            currentPage = 1; 
            currentFilterStatus = filterStatusSelect.value;
            loadPersons();
        });
    }

    // 模型训练按钮事件监听器
    const retrainModelButton = document.getElementById('retrain-model-button');
    if (retrainModelButton) {
        retrainModelButton.addEventListener('click', async () => {
            if (confirm('确定要开始模型再训练吗？此操作可能需要一段时间。')) {
                // 重置进度条状态
                retrainProgressBar.style.width = '0%';
                retrainProgressBar.textContent = '0%';
                retrainProgressBar.style.backgroundColor = '#007bff'; // 恢复默认蓝色
                retrainStatusText.textContent = '状态: 任务已调度...';

                try {
                    const response = await fetch(`${API_BASE_URL}/admin/trigger-retrain-reid`, {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                            'Content-Type': 'application/json'
                        }
                    });
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    const data = await response.json();
                    alert(`模型训练任务已调度: ${data.message}`);
                    // 启动进度查询
                    retrainProgressModal.style.display = 'flex'; // 显示模态框
                    pollTaskStatus(data.task_id); // 开始轮询任务状态

                    // 训练完成后，重新加载待再训练列表，这些人物应该被清除
                    // 或者直接切换到已训练列表，或者回到未审核列表
                    // 这里我们选择回到“待再训练”列表，以便看到清空的效果
                    currentPage = 1;
                    currentFilterStatus = 'marked_for_retrain'; // 切换到待再训练列表
                    filterStatusSelect.value = 'marked_for_retrain'; // 更新下拉菜单
                    loadPersons(); 
                } catch (error) {
                    console.error('调度模型训练失败:', error);
                    alert('调度模型训练失败。');
                }
            }
        });
    }

    // 将 attachEventListeners 函数的定义移动到这里
    function attachEventListeners() {
        // 使用事件委托处理 .btn-verify, .btn-correct, .btn-retrain 按钮的点击事件
        personListDiv.addEventListener('click', async (event) => {
            if (event.target.classList.contains('btn-verify')) {
                const button = event.target;
                const personUuid = button.dataset.uuid; // 获取人物 UUID
                if (confirm(`确定要将人物 ${personUuid} 标记为“确认无误”吗？`)) {
                    try {
                        const response = await fetch(`${API_BASE_URL}/hitl/persons/${personUuid}/verify`, {
                            method: 'POST',
                            headers: {
                                'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                                'Content-Type': 'application/json'
                            }
                        });
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        alert(`人物 ${personUuid} 已标记为“确认无误”。`);
                        loadPersons();
                    } catch (error) {
                        console.error('验证人物失败:', error);
                        alert('验证人物失败。');
                    }
                }
            } else if (event.target.classList.contains('btn-correct')) {
                const button = event.target;
                currentPersonIdToCorrect = button.dataset.id;
                currentPersonUuidToCorrect = button.dataset.uuid; // 获取被纠正人物的 UUID
                // 清空模态框内容和合并相关字段
                correctionTypeSelect.value = 'misdetection';
                correctionDetailsTextarea.value = '';
                mergePersonSection.style.display = 'none';
                targetPersonUuidInput.value = '';
                targetPersonIdCardInput.value = ''; // 清空身份证输入框
                targetPersonInfoDiv.style.display = 'none';
                selectedTargetPersonUuid = null;
                correctionModal.style.display = 'flex'; 
            } else if (event.target.classList.contains('btn-retrain')) {
                const button = event.target;
                console.log("Retrain button clicked. Event target:", button); 
                console.log("Retrain button clicked. Event target dataset:", button.dataset); 
                const personId = button.dataset.id;
                const personUuid = button.dataset.uuid; // 获取人物 UUID
                const currentMarkedStatus = button.dataset.marked === 'true';
                const newMarkedStatus = !currentMarkedStatus;
                const actionText = newMarkedStatus ? '标记为“待再训练”' : '取消“待再训练”标记';

                if (!personUuid) {
                    alert("无法获取人物 UUID，请刷新页面重试。");
                    console.error("人物 UUID 为 undefined，无法提交再训练标记。");
                    return;
                }

                if (confirm(`确定要将人物 ${personUuid} ${actionText}吗？`)) {
                    try {
                        const response = await fetch(`${API_BASE_URL}/hitl/persons/${personUuid}/mark_for_retrain?mark=${newMarkedStatus}`, {
                            method: 'POST',
                            headers: {
                                'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                                'Content-Type': 'application/json'
                            }
                        });
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        alert(`人物 ${personUuid} 已${actionText}。`);
                        loadPersons();
                    } catch (error) {
                        console.error('更新再训练标记失败:', error);
                        alert('更新再训练标记失败。');
                    }
                }
            }
        });

        // 使用事件委托处理 .person-card img 的点击事件 (大图预览)
        personListDiv.addEventListener('click', (event) => {
            if (event.target.tagName === 'IMG' && event.target.closest('.person-card')) {
                const img = event.target;
                img.style.cursor = 'pointer'; 
                const imageUrl = img.dataset.fullFrameSrc || img.src; 
                const largeImageModal = document.createElement('div');
                largeImageModal.className = 'large-image-modal';
                largeImageModal.innerHTML = `
                    <span class="close-large-image">&times;</span>
                    <img src="${imageUrl}" alt="大图">
                `;
                document.body.appendChild(largeImageModal);

                largeImageModal.style.display = 'flex'; 

                largeImageModal.querySelector('.close-large-image').onclick = () => {
                    largeImageModal.style.display = 'none';
                    largeImageModal.remove();
                };

                largeImageModal.onclick = (e) => {
                    if (e.target === largeImageModal) {
                        largeImageModal.style.display = 'none';
                        largeImageModal.remove();
                    }
                };
            }
        });
    }

    // 将 attachEventListeners 函数的调用移到这里，确保它只在 DOMContentLoaded 触发时执行一次
    attachEventListeners();

    loadPersons();
}); 