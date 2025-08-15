// frontend/static/js/admin.js

import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js'; // 导入 Auth 和 fetchWithAuth

document.addEventListener('DOMContentLoaded', () => {
    // 获取切换按钮和内容区域
    const showLogsButton = document.getElementById('show-logs-button');
    const showUsersButton = document.getElementById('show-users-button');
    const systemLogsSection = document.getElementById('system-logs');
    const userManagementSection = document.getElementById('user-management');
    const showModelParametersButton = document.getElementById('show-model-parameters-button');
    const systemParametersSection = document.getElementById('system-parameters'); // 已修改为 system-parameters

    // 初始显示系统日志，隐藏用户管理和系统参数
    systemLogsSection.classList.remove('hidden');
    userManagementSection.classList.add('hidden');
    systemParametersSection.classList.add('hidden');
    showLogsButton.classList.add('active');
    showUsersButton.classList.remove('active');
    showModelParametersButton.classList.remove('active');

    // 切换页面函数
    function showSection(sectionToShow) {
        // 隐藏所有 sections
        systemLogsSection.classList.add('hidden');
        userManagementSection.classList.add('hidden');
        systemParametersSection.classList.add('hidden');

        // 移除所有按钮的 active 状态
        showLogsButton.classList.remove('active');
        showUsersButton.classList.remove('active');
        showModelParametersButton.classList.remove('active');

        if (sectionToShow === 'logs') {
            systemLogsSection.classList.remove('hidden');
            showLogsButton.classList.add('active');
            fetchSystemLogs(getLogFilters(), 1); // 重新加载日志
        } else if (sectionToShow === 'users') {
            userManagementSection.classList.remove('hidden');
            showUsersButton.classList.add('active');
            fetchUsers(); // 重新加载用户
        } else if (sectionToShow === 'model-parameters') { // 注意: admin.html 中的 section id 仍是 'model-parameters'
            systemParametersSection.classList.remove('hidden');
            showModelParametersButton.classList.add('active');
            fetchSystemConfigs(); // 加载系统参数
        }
    }

    // 按钮点击事件监听
    showLogsButton.addEventListener('click', () => showSection('logs'));
    showUsersButton.addEventListener('click', () => showSection('users'));
    showModelParametersButton.addEventListener('click', () => showSection('model-parameters'));

    // System Config Elements
    const systemConfigForm = document.getElementById('system-config-form'); // 已修改为 system-config-form
    const systemConfigFields = document.getElementById('system-config-fields'); // 已修改为 system-config-fields
    const systemConfigMessage = document.getElementById('system-config-message'); // 已修改为 system-config-message
    const systemConfigLoading = document.getElementById('system-config-loading'); // 已修改为 system-config-loading

    const systemConfigSidebar = document.getElementById('system-config-sidebar'); // 已修改为 system-config-sidebar
    const systemConfigFieldsContainer = document.getElementById('system-config-fields-container'); // 已修改为 system-config-fields-container

    // Function to fetch and display system configurations
    async function fetchSystemConfigs() {
        systemConfigLoading.classList.remove('hidden');
        systemConfigMessage.classList.add('hidden');
        systemConfigFieldsContainer.innerHTML = ''; // Clear previous content
        systemConfigSidebar.innerHTML = '';

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/admin/system-configs`);
            if (response.ok) {
                const configs = await response.json();
                renderSystemConfigForm(configs);
            } else {
                const errorData = await response.json();
                systemConfigMessage.textContent = '获取系统参数失败：' + (errorData.detail || response.statusText);
                systemConfigMessage.classList.remove('hidden');
            }
        } catch (error) {
            console.error('获取系统参数时发生错误:', error);
            systemConfigMessage.textContent = '获取系统参数时发生网络错误。';
            systemConfigMessage.classList.remove('hidden');
        } finally {
            systemConfigLoading.classList.add('hidden');
        }
    }

    // Function to render the system config form with sidebar
    function renderSystemConfigForm(configs) {
        systemConfigSidebar.innerHTML = '';
        systemConfigFieldsContainer.innerHTML = '';

        // 中文翻译字典
        const parameterTranslations = {
            "DETECTION_MODEL_FILENAME": "目标检测模型文件名",
            "REID_MODEL_FILENAME": "Re-ID 模型文件名",
            "ACTIVE_REID_MODEL_PATH": "当前激活Re-ID模型路径",
            "POSE_MODEL_FILENAME": "姿态估计模型文件名",
            "FACE_DETECTION_MODEL_FILENAME": "人脸检测模型文件名",
            "FACE_RECOGNITION_MODEL_FILENAME": "人脸识别模型文件名",
            "GAIT_RECOGNITION_MODEL_FILENAME": "步态识别模型文件名",
            "CLOTHING_ATTRIBUTE_MODEL_FILENAME": "衣着属性模型文件名",

            "REID_INPUT_WIDTH": "Re-ID输入宽度",
            "REID_INPUT_HEIGHT": "Re-ID输入高度",
            "FEATURE_DIM": "特征维度",
            "FACE_FEATURE_DIM": "人脸特征维度",
            "GAIT_FEATURE_DIM": "步态特征维度",

            "DEVICE_TYPE": "设备类型",

            "REID_WEIGHT": "Re-ID权重",
            "FACE_WEIGHT": "人脸权重",
            "GAIT_WEIGHT": "步态权重",

            "K1": "Re-Ranking K1",
            "K2": "Re-Ranking K2",
            "LAMBDA_VALUE": "Re-Ranking Lambda值",

            "GAIT_SEQUENCE_LENGTH": "步态序列长度",

            "HUMAN_REVIEW_CONFIDENCE_THRESHOLD": "人机回环置信度阈值",
            "IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE": "图片分析人物最低置信度",
            "ENROLLMENT_MIN_PERSON_CONFIDENCE": "主动注册人物最低置信度",

            "FACE_DETECTION_CONFIDENCE_THRESHOLD": "人脸检测置信度阈值",
            "MIN_FACE_WIDTH": "人脸最小宽度",
            "MIN_FACE_HEIGHT": "人脸最小高度",

            "TRACKER_PROXIMITY_THRESH": "追踪器接近度阈值",
            "TRACKER_APPEARANCE_THRESH": "追踪器外观相似度阈值",
            "TRACKER_HIGH_THRESH": "追踪器高阈值",
            "TRACKER_LOW_THRESH": "追踪器低阈值",
            "TRACKER_NEW_TRACK_THRESH": "追踪器新轨迹阈值",
            "TRACKER_MIN_HITS": "追踪器最低检测次数",
            "TRACKER_TRACK_BUFFER": "追踪器轨迹保留帧数",

            "VIDEO_PROCESSING_FRAME_RATE": "视频处理帧率",
            "STREAM_PROCESSING_FRAME_RATE": "视频流处理帧率",
            "VIDEO_COMMIT_BATCH_SIZE": "视频提交批处理大小",
            "MJPEG_STREAM_FPS": "MJPEG视频流帧率",
            "DETECTION_CONFIDENCE_THRESHOLD": "检测置信度阈值",
            "PERSON_CLASS_ID": "人物类别ID",

            "EXCEL_EXPORT_MAX_IMAGES": "Excel导出最大图片数量",
            "EXCEL_EXPORT_IMAGE_SIZE_PX": "Excel导出图片大小(像素)",
            "EXCEL_EXPORT_ROW_HEIGHT_PT": "Excel导出行高(点)",

            "FAISS_METRIC": "Faiss距离度量",
            "FAISS_SEARCH_K": "Faiss搜索K值",

            "REID_TRAIN_BATCH_SIZE": "Re-ID训练批量大小",
            "REID_TRAIN_LEARNING_RATE": "Re-ID训练学习率",
            "REALTIME_COMPARISON_THRESHOLD": "实时比对阈值",
            "REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS": "实时比对最大关注人数",
            "GLOBAL_SEARCH_MIN_CONFIDENCE": "全局搜索最小置信度",
            "STREAM_LOG_INTERVAL_FRAMES": "视频流日志间隔帧数" // 新增：流日志间隔帧数
        };

        const categories = {
            "模型文件配置": ["DETECTION_MODEL_FILENAME", "REID_MODEL_FILENAME", "ACTIVE_REID_MODEL_PATH", 
                       "POSE_MODEL_FILENAME", "FACE_DETECTION_MODEL_FILENAME", 
                       "FACE_RECOGNITION_MODEL_FILENAME", "GAIT_RECOGNITION_MODEL_FILENAME",
                       "CLOTHING_ATTRIBUTE_MODEL_FILENAME"],
            "模型维度配置": ["REID_INPUT_WIDTH", "REID_INPUT_HEIGHT", "FEATURE_DIM", 
                          "FACE_FEATURE_DIM", "GAIT_FEATURE_DIM"],
            "模型权重与参数": ["REID_WEIGHT", "FACE_WEIGHT", "GAIT_WEIGHT", 
                         "K1", "K2", "LAMBDA_VALUE", "GAIT_SEQUENCE_LENGTH"],
            "设备与阈值": ["DEVICE_TYPE", "HUMAN_REVIEW_CONFIDENCE_THRESHOLD", 
                       "IMAGE_ANALYSIS_MIN_PERSON_CONFIDENCE", "ENROLLMENT_MIN_PERSON_CONFIDENCE",
                       "FACE_DETECTION_CONFIDENCE_THRESHOLD", "MIN_FACE_WIDTH", "MIN_FACE_HEIGHT",
                       "DETECTION_CONFIDENCE_THRESHOLD", "PERSON_CLASS_ID",
                       "GLOBAL_SEARCH_MIN_CONFIDENCE"],
            "追踪器配置": ["TRACKER_PROXIMITY_THRESH", "TRACKER_APPEARANCE_THRESH", 
                       "TRACKER_HIGH_THRESH", "TRACKER_LOW_THRESH", 
                       "TRACKER_NEW_TRACK_THRESH", "TRACKER_MIN_HITS", "TRACKER_TRACK_BUFFER"],
            "视频处理配置": ["VIDEO_PROCESSING_FRAME_RATE", "STREAM_PROCESSING_FRAME_RATE",
                           "VIDEO_COMMIT_BATCH_SIZE", "MJPEG_STREAM_FPS", "STREAM_LOG_INTERVAL_FRAMES"], // 添加 STREAM_LOG_INTERVAL_FRAMES
            "Faiss配置": ["FAISS_METRIC", "FAISS_SEARCH_K"],
            "Re-ID训练参数": ["REID_TRAIN_BATCH_SIZE", "REID_TRAIN_LEARNING_RATE"],
            "实时比对配置": ["REALTIME_COMPARISON_THRESHOLD", "REALTIME_COMPARISON_MAX_FOLLOWED_PERSONS"],
            "Excel导出配置": ["EXCEL_EXPORT_MAX_IMAGES", "EXCEL_EXPORT_IMAGE_SIZE_PX",
                         "EXCEL_EXPORT_ROW_HEIGHT_PT"]
        };

        let firstCategory = true;

        for (const categoryName in categories) {
            // 创建侧边栏菜单项
            const listItem = document.createElement('li');
            listItem.textContent = categoryName;
            listItem.classList.add('sidebar-item');
            listItem.dataset.category = categoryName.replace(/\s/g, ''); // 用于匹配内容的ID
            systemConfigSidebar.appendChild(listItem);

            // 创建右侧内容区域的表单
            const categoryForm = document.createElement('form');
            categoryForm.id = `form-${listItem.dataset.category}`;
            categoryForm.classList.add('system-config-category-form'); // 修改类名
            if (!firstCategory) {
                categoryForm.classList.add('hidden'); // 默认隐藏除第一个外的所有表单
            } else {
                listItem.classList.add('active'); // 默认激活第一个侧边栏项
            }

            const formFieldsDiv = document.createElement('div');
            formFieldsDiv.classList.add('category-content');

            categories[categoryName].forEach(key => {
                const value = configs[key];
                const fieldDiv = document.createElement('div');
                fieldDiv.classList.add('config-field');

                let inputElement;
                let inputType = 'text';

                // 根据值类型判断输入框类型
                if (typeof value === 'number') {
                    inputType = 'number';
                    if (!Number.isInteger(value)) {
                        inputType = 'text'; 
                    }
                } else if (typeof value === 'boolean') {
                    inputElement = document.createElement('select');
                    inputElement.innerHTML = `
                        <option value="true" ${value ? 'selected' : ''}>是</option>
                        <option value="false" ${!value ? 'selected' : ''}>否</option>
                    `;
                } else if (key === "FAISS_METRIC") {
                    inputElement = document.createElement('select');
                    inputElement.innerHTML = `
                        <option value="IP" ${value === 'IP' ? 'selected' : ''}>内积 (IP)</option>
                        <option value="L2" ${value === 'L2' ? 'selected' : ''}>欧氏距离 (L2)</option>
                    `;
                } else if (key === "DEVICE_TYPE") {
                    inputElement = document.createElement('select');
                    inputElement.innerHTML = `
                        <option value="cpu" ${value === 'cpu' ? 'selected' : ''}>CPU</option>
                        <option value="cuda" ${value === 'cuda' ? 'selected' : ''}>CUDA (GPU)</option>
                    `;
                }

                if (!inputElement) {
                    inputElement = document.createElement('input');
                    inputElement.type = inputType;
                    if (inputType === 'text' && typeof value === 'number' && !Number.isInteger(value)) {
                        inputElement.step = "any"; 
                    }
                    inputElement.value = value;
                }

                inputElement.id = `${categoryForm.id}-${key}`;
                inputElement.name = key;

                const labelElement = document.createElement('label');
                labelElement.setAttribute('for', inputElement.id);
                labelElement.textContent = `${parameterTranslations[key] || key} (${key}):`; 

                fieldDiv.appendChild(labelElement);
                fieldDiv.appendChild(inputElement);
                formFieldsDiv.appendChild(fieldDiv);
            });

            categoryForm.appendChild(formFieldsDiv);

            const saveButton = document.createElement('button');
            saveButton.type = 'submit';
            saveButton.classList.add('button', 'save-category-button');
            saveButton.textContent = '保存修改';
            categoryForm.appendChild(saveButton);

            systemConfigFieldsContainer.appendChild(categoryForm);

            firstCategory = false;
        }
        // Add event listeners for sidebar items
        const sidebarItems = document.querySelectorAll('.sidebar-item');
        sidebarItems.forEach(item => {
            item.addEventListener('click', (event) => {
                // Remove active class from all sidebar items
                sidebarItems.forEach(i => i.classList.remove('active'));
                // Add active class to the clicked item
                event.target.classList.add('active');

                // 隐藏之前的消息提示
                systemConfigMessage.classList.add('hidden');

                const selectedCategory = event.target.dataset.category;
                const allForms = document.querySelectorAll('.system-config-category-form'); // 修改类名
                allForms.forEach(form => {
                    if (form.id === `form-${selectedCategory}`) {
                        form.classList.remove('hidden');
                    } else {
                        form.classList.add('hidden');
                    }
                });
            });
        });
        // Handle form submission for each category form
        document.querySelectorAll('.system-config-category-form').forEach(form => {
            form.addEventListener('submit', async (event) => {
                event.preventDefault();
                systemConfigLoading.classList.remove('hidden');
                systemConfigMessage.classList.add('hidden');

                const formData = new FormData(event.target);
                const updatedConfigs = {};
                for (const [key, value] of formData.entries()) {
                    let processedValue = value;
                    if (!isNaN(parseFloat(value)) && isFinite(value)) {
                        if (value.includes('.')) {
                            processedValue = parseFloat(value);
                        } else {
                            processedValue = parseInt(value, 10);
                        }
                    } else if (value.toLowerCase() === 'true') {
                        processedValue = true;
                    } else if (value.toLowerCase() === 'false') {
                        processedValue = false;
                    }
                    updatedConfigs[key] = processedValue;
                }
                
                try {
                    const response = await fetchWithAuth(`${API_BASE_URL}/admin/system-configs`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(updatedConfigs)
                    });

                    if (response.ok) {
                        systemConfigMessage.textContent = '系统参数更新成功！部分更改可能需要重启服务才能完全生效。 ';
                        systemConfigMessage.classList.remove('hidden');
                        systemConfigMessage.style.color = 'green';
                    } else {
                        const errorData = await response.json();
                        systemConfigMessage.textContent = '更新系统参数失败：' + (errorData.detail || response.statusText);
                        systemConfigMessage.classList.remove('hidden');
                        systemConfigMessage.style.color = 'red';
                    }
                } catch (error) {
                    console.error('更新系统参数时发生错误:', error);
                    systemConfigMessage.textContent = '更新系统参数时发生网络错误。 ';
                    systemConfigMessage.classList.remove('hidden');
                    systemConfigMessage.style.color = 'red';
                } finally {
                    systemConfigLoading.classList.add('hidden');
                }
            });
        });
    }

    // Log Management Elements
    const logLevelFilter = document.getElementById('log-level-filter');
    const logStartTime = document.getElementById('log-start-time');
    const logEndTime = document.getElementById('log-end-time');
    const logKeyword = document.getElementById('log-keyword');
    const filterLogsButton = document.getElementById('filter-logs-button');
    const refreshLogsButton = document.getElementById('refresh-logs-button');
    const logTableBody = document.getElementById('log-table-body');
    const noLogsMessage = document.getElementById('no-logs-message');
    const logLoading = document.getElementById('log-loading');

    // Log Pagination Elements
    const prevLogPageButton = document.getElementById('prev-log-page');
    const nextLogPageButton = document.getElementById('next-log-page');
    const firstLogPageButton = document.getElementById('first-log-page');
    const lastLogPageButton = document.getElementById('last-log-page');
    const logPageInfo = document.getElementById('log-page-info');

    let currentLogPage = 1;
    const logsPerPage = 10; // Display 10 logs per page
    let totalLogs = 0; // Declare totalLogs here, accessible to all event listeners

    // Helper function to get current log filter values
    function getLogFilters() {
        return {
            level: logLevelFilter.value,
            start_time: logStartTime.value,
            end_time: logEndTime.value,
            keyword: logKeyword.value
        };
    }

    // User Management Elements
    const userTableBody = document.querySelector('#user-table tbody');
    const refreshUsersButton = document.getElementById('refresh-users-button');
    const addUserForm = document.getElementById('add-user-form');
    const userStatusFilter = document.getElementById('user-status-filter');
    const filterUsersButton = document.getElementById('filter-users-button');

    // Function to fetch and display system logs
    async function fetchSystemLogs(filters = {}, page = 1) {
        currentLogPage = page;
        logLoading.classList.remove('hidden');
        logTableBody.innerHTML = '';
        noLogsMessage.classList.add('hidden');

        let queryParams = new URLSearchParams();
        if (filters.level) queryParams.append('level', filters.level);
        if (filters.start_time) queryParams.append('start_time', filters.start_time);
        if (filters.end_time) queryParams.append('end_time', filters.end_time);
        if (filters.keyword) queryParams.append('keyword', filters.keyword);
        queryParams.append('skip', (currentLogPage - 1) * logsPerPage);
        queryParams.append('limit', logsPerPage);

        const url = `${API_BASE_URL}/admin/logs?${queryParams.toString()}`;

        try {
            // 使用 fetchWithAuth 替换手动 fetch
            const response = await fetchWithAuth(url);

            if (response.ok) {
                const data = await response.json(); // Backend returns {total: ..., logs: [...]}
                const logs = data.logs;
                totalLogs = data.total; // Update the global totalLogs variable
                const totalPages = Math.ceil(totalLogs / logsPerPage);

                logPageInfo.textContent = `第 ${currentLogPage} 页 / 共 ${totalPages} 页`;
                firstLogPageButton.disabled = currentLogPage === 1;
                prevLogPageButton.disabled = currentLogPage === 1;
                nextLogPageButton.disabled = currentLogPage === totalPages || totalPages === 0;
                lastLogPageButton.disabled = currentLogPage === totalPages || totalPages === 0;

                if (logs.length > 0) {
                    logs.forEach(log => {
                        const row = logTableBody.insertRow();
                        const timestamp = new Date(log.timestamp).toLocaleString();
                        row.insertCell().textContent = timestamp;
                        row.insertCell().textContent = log.logger;
                        row.insertCell().textContent = log.level;
                        row.insertCell().textContent = log.message;
                        row.insertCell().textContent = log.username || 'N/A'; // 显示用户名，如果没有则显示 'N/A'
                        row.insertCell().textContent = log.ip_address || 'N/A'; // 显示 IP 地址，如果没有则显示 'N/A'
                    });
                } else {
                    noLogsMessage.classList.remove('hidden');
                }
            } else { // fetchWithAuth 已经处理了 401，这里处理其他错误
                alert('获取日志失败：' + response.statusText);
                noLogsMessage.classList.remove('hidden');
                noLogsMessage.textContent = '无法加载日志。'
            }
        } catch (error) {
            console.error('获取日志时发生错误:', error);
            // fetchWithAuth 在 401 时会抛出 Error，无需额外处理
            if (error.message !== 'Unauthorized') { // 避免重复提示
                alert('获取日志时发生网络错误。');
            }
            noLogsMessage.classList.remove('hidden');
            noLogsMessage.textContent = '无法加载日志。'
        } finally {
            logLoading.classList.add('hidden');
        }
    }

    // Function to fetch and display user list
    async function fetchUsers(filters = {}) {
        // 移除手动获取 token 的代码，由 fetchWithAuth 处理
        // const token = localStorage.getItem('accessToken');
        // if (!token) {
        //     alert('未授权，请登录。');
        //     window.location.href = '/login';
        //     return;
        // }

        let queryParams = new URLSearchParams();
        if (filters.is_active !== undefined && filters.is_active !== null) {
            queryParams.append('is_active', filters.is_active);
        }
        const url = `${API_BASE_URL}/admin/users?${queryParams.toString()}`;

        try {
            // 使用 fetchWithAuth 替换手动 fetch
            const response = await fetchWithAuth(url);

            if (response.ok) {
                const users = await response.json();
                userTableBody.innerHTML = ''; // Clear existing content
                if (users.length > 0) {
                    users.forEach(user => {
                        const row = userTableBody.insertRow();
                        row.insertCell().textContent = user.id;
                        row.insertCell().textContent = user.username;

                        const roleCell = row.insertCell();
                        const roleSelect = document.createElement('select');
                        roleSelect.dataset.userId = user.id;
                        // 确保根据后端返回的枚举值进行选择和显示
                        roleSelect.innerHTML = `
                            <option value="user" ${user.role === 'user' ? 'selected' : ''}>普通用户</option>
                            <option value="advanced" ${user.role === 'advanced' ? 'selected' : ''}>高级用户</option>
                            <option value="admin" ${user.role === 'admin' ? 'selected' : ''}>管理员</option>
                        `;
                        roleSelect.addEventListener('change', handleRoleChange);
                        roleCell.appendChild(roleSelect);

                        row.insertCell().textContent = user.unit;
                        row.insertCell().textContent = user.phone_number || 'N/A'; // Display phone number
                        row.insertCell().textContent = user.is_active ? '活跃' : '非活跃'; // Display status

                        const actionsCell = row.insertCell();
                        // Activate/Deactivate button
                        const statusButton = document.createElement('button');
                        statusButton.dataset.userId = user.id;
                        statusButton.className = user.is_active ? 'deactivate-button' : 'activate-button';
                        statusButton.textContent = user.is_active ? '停用' : '激活';
                        statusButton.addEventListener('click', user.is_active ? handleDeactivateUser : handleActivateUser);
                        actionsCell.appendChild(statusButton);

                        // Delete button
                        const deleteButton = document.createElement('button');
                        deleteButton.textContent = '删除';
                        deleteButton.className = 'delete-button';
                        deleteButton.dataset.userId = user.id;
                        deleteButton.addEventListener('click', handleDeleteUser);
                        actionsCell.appendChild(deleteButton);
                    });
                } else {
                    userTableBody.innerHTML = '<tr><td colspan="6">未找到用户。</td></tr>'; // colspan changed to 6
                }
            } else { // fetchWithAuth 已经处理了 401，这里处理其他错误
                alert('获取用户列表失败：' + response.statusText);
                userTableBody.innerHTML = '<tr><td colspan="6">无法加载用户列表。</td></tr>'; // colspan changed to 6
            }
        } catch (error) {
            console.error('获取用户列表时发生错误:', error);
            if (error.message !== 'Unauthorized') { // 避免重复提示
                alert('获取用户列表时发生网络错误。');
            }
            userTableBody.innerHTML = '<tr><td colspan="6">无法加载用户列表。</td></tr>'; // colspan changed to 6
        }
    }

    // Handle role change
    async function handleRoleChange(event) {
        const userId = event.target.dataset.userId;
        const newRole = event.target.value;
        // 移除手动获取 token 的代码
        // const token = localStorage.getItem('accessToken');

        if (confirm(`确定要将用户 ID: ${userId} 的角色更改为 ${newRole} 吗？`)) {
            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/admin/users/${userId}/role`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        // 'Authorization': `Bearer ${token}` // 由 fetchWithAuth 添加
                    },
                    body: JSON.stringify({ role: newRole })
                });

                if (response.ok) {
                    alert('用户角色更新成功！');
                    fetchUsers(); // Refresh list
                } else { // fetchWithAuth 已经处理了 401
                    const errorData = await response.json();
                    alert('更新用户角色失败：' + (errorData.detail || response.statusText));
                }
            } catch (error) {
                console.error('更新用户角色时发生错误:', error);
                if (error.message !== 'Unauthorized') { // 避免重复提示
                    alert('更新用户角色时发生网络错误。');
                }
            }
        }
    }

    // Handle activate user
    async function handleActivateUser(event) {
        const userId = event.target.dataset.userId;
        // 移除手动获取 token 的代码
        // const token = localStorage.getItem('accessToken');

        if (confirm(`确定要激活用户 ID: ${userId} 吗？`)) {
            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/admin/users/${userId}/status`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        // 'Authorization': `Bearer ${token}` // 由 fetchWithAuth 添加
                    },
                    body: JSON.stringify({ id: parseInt(userId), is_active: true })
                });

                if (response.ok) {
                    alert('用户激活成功！');
                    fetchUsers(); // Refresh list
                } else { // fetchWithAuth 已经处理了 401
                    const errorData = await response.json();
                    alert('激活用户失败：' + (errorData.detail || response.statusText));
                }
            } catch (error) {
                console.error('激活用户时发生错误:', error);
                if (error.message !== 'Unauthorized') { // 避免重复提示
                    alert('激活用户时发生网络错误。');
                }
            }
        }
    }

    // Handle deactivate user
    async function handleDeactivateUser(event) {
        const userId = event.target.dataset.userId;
        // 移除手动获取 token 的代码
        // const token = localStorage.getItem('accessToken');

        if (confirm(`确定要停用用户 ID: ${userId} 吗？`)) {
            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/admin/users/${userId}/status`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        // 'Authorization': `Bearer ${token}` // 由 fetchWithAuth 添加
                    },
                    body: JSON.stringify({ id: parseInt(userId), is_active: false })
                });

                if (response.ok) {
                    alert('用户停用成功！');
                    fetchUsers(); // Refresh list
                } else { // fetchWithAuth 已经处理了 401
                    const errorData = await response.json();
                    alert('停用用户失败：' + (errorData.detail || response.statusText));
                }
            } catch (error) {
                console.error('停用用户时发生错误:', error);
                if (error.message !== 'Unauthorized') { // 避免重复提示
                    alert('停用用户时发生网络错误。');
                }
            }
        }
    }

    // Handle delete user
    async function handleDeleteUser(event) {
        const userId = event.target.dataset.userId;
        // 移除手动获取 token 的代码
        // const token = localStorage.getItem('accessToken');

        if (confirm(`确定要删除用户 ID: ${userId} 吗？此操作不可逆！`)) {
            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/admin/users/${userId}`, {
                    method: 'DELETE',
                    // headers: {
                    //     'Authorization': `Bearer ${token}` // 由 fetchWithAuth 添加
                    // }
                });

                if (response.ok) {
                    alert('用户删除成功！');
                    fetchUsers(); // Refresh list
                } else { // fetchWithAuth 已经处理了 401
                    const errorData = await response.json();
                    alert('删除用户失败：' + (errorData.detail || response.statusText));
                }
            } catch (error) {
                console.error('删除用户时发生错误:', error);
                if (error.message !== 'Unauthorized') { // 避免重复提示
                    alert('删除用户时发生网络错误。');
                }
            }
        }
    }

    // Handle Add User Form Submission
    addUserForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const username = document.getElementById('add-username').value;
        const password = document.getElementById('add-password').value;
        const unit = document.getElementById('add-unit').value;
        const role = document.getElementById('add-role').value;
        const phone_number = document.getElementById('add-phone-number').value; // 获取手机号码

        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/admin/users/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, password, unit, role, phone_number })
            });

            if (response.ok) {
                alert('用户添加成功！');
                addUserForm.reset(); // Clear the form
                fetchUsers(); // Refresh the user list
            } else {
                const errorData = await response.json();
                alert('添加用户失败：' + (errorData.detail || response.statusText));
            }
        } catch (error) {
            console.error('添加用户时发生错误:', error);
            if (error.message !== 'Unauthorized') {
                alert('添加用户时发生网络错误。');
            }
        }
    });

    // Event Listeners for Log Management
    filterLogsButton.addEventListener('click', () => fetchSystemLogs(getLogFilters()));
    refreshLogsButton.addEventListener('click', () => {
        logLevelFilter.value = '';
        logStartTime.value = '';
        logEndTime.value = '';
        logKeyword.value = '';
        fetchSystemLogs(getLogFilters(), 1); // 刷新时也带上过滤器，并回到第一页
    });

    prevLogPageButton.addEventListener('click', () => {
        if (currentLogPage > 1) {
            fetchSystemLogs(getLogFilters(), currentLogPage - 1);
        }
    });

    nextLogPageButton.addEventListener('click', () => {
        const totalPages = Math.ceil(totalLogs / logsPerPage); // Now totalLogs is accessible
        if (currentLogPage < totalPages) {
            fetchSystemLogs(getLogFilters(), currentLogPage + 1);
        }
    });

    firstLogPageButton.addEventListener('click', () => {
        if (currentLogPage !== 1) {
            fetchSystemLogs(getLogFilters(), 1);
        }
    });

    lastLogPageButton.addEventListener('click', () => {
        const totalPages = Math.ceil(totalLogs / logsPerPage); // Now totalLogs is accessible
        if (currentLogPage !== totalPages) {
            fetchSystemLogs(getLogFilters(), totalPages);
        }
    });

    // Bind user filter button event
    filterUsersButton.addEventListener('click', () => {
        const is_active = userStatusFilter.value === 'true' ? true : (userStatusFilter.value === 'false' ? false : undefined);
        fetchUsers({ is_active });
    });

    // Bind refresh users button event
    refreshUsersButton.addEventListener('click', () => {
        userStatusFilter.value = ''; // Clear filter
        fetchUsers({});
    });

    // Initialize logs and users on page load
    fetchSystemLogs({}, 1); // 初始加载第一页日志
    fetchUsers();
});