import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js';

console.log("image_analysis.js: 文件已加载。");

let currentPage = 1;
const itemsPerPage = 10; // 每页显示的图片数量
let totalPages = 1;
let totalImages = 0;

// 轮询任务状态的函数
async function pollTaskStatus(taskId, messageElement) {
    let attempts = 0;
    const maxAttempts = 360; // 最多等待360次，每次0.5秒，总计180秒
    const pollInterval = 500; // 轮询间隔 500 毫秒
    
    while (attempts < maxAttempts) {
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/api/image_analysis/tasks/${taskId}`);
            const taskStatus = await response.json();
            
            if (taskStatus.status === 'SUCCESS') {
                // 正确获取人物数量的路径
                const detectedPersons = taskStatus.result?.analyzed_persons || [];
                const personCount = taskStatus.result?.original_image_info?.person_count || detectedPersons.length;

                messageElement.textContent = `图片解析成功，检测到 ${personCount} 个人物。`;
                messageElement.classList.remove('info', 'error');
                messageElement.classList.add('success');
                await loadImagesHistory(); // 刷新图片列表
                return;
            } else if (taskStatus.status === 'FAILED') {
                throw new Error(taskStatus.message || '解析失败');
            } else if (taskStatus.status === 'PENDING' || taskStatus.status === 'PROGRESS') {
                messageElement.textContent = `正在处理...${taskStatus.progress || 0}%`;
                await new Promise(resolve => setTimeout(resolve, pollInterval)); // 使用新的轮询间隔
                attempts++;
            }
        } catch (error) {
            throw new Error(`任务状态查询失败: ${error.message}`);
        }
    }
    throw new Error('任务处理超时');
}

export function initImageAnalysisPage() {
    const imageUpload = document.getElementById('imageUpload');
    const uploadButton = document.getElementById('uploadButton');
    const selectedFileNameSpan = document.getElementById('selected-file-name');
    const uploadMessage = document.getElementById('uploadMessage');
    const imageListTableBody = document.getElementById('image-list-table-body');
    const noResultsMessage = document.getElementById('noResultsMessage');
    const uploaderHeader = document.getElementById('uploader-header'); // 获取上传人列头
    const prevPageButton = document.getElementById('prevPage');
    const nextPageButton = document.getElementById('nextPage');
    const currentPageSpan = document.getElementById('currentPage');
    const totalPagesSpan = document.getElementById('totalPages');
    const paginationControls = document.getElementById('paginationControls'); // 获取分页容器
    const firstPageButton = document.getElementById('firstPage'); // 新增
    const lastPageButton = document.getElementById('lastPage'); // 新增

    // 根据用户角色隐藏上传功能和显示上传人列头
    const currentUser = Auth.getUserInfo();
    const imageUploadSection = document.getElementById('image-upload-section');
    if (currentUser && (currentUser.role === 'user' || currentUser.role === 'advanced' || currentUser.role === 'admin')) {
        if (imageUploadSection) imageUploadSection.style.display = '';
        if (currentUser.role === 'admin' && uploaderHeader) { // 管理员显示上传人列头
            uploaderHeader.style.display = '';
        }
    } else {
        if (imageUploadSection) imageUploadSection.style.display = 'none';
        if (uploaderHeader) uploaderHeader.style.display = 'none'; // 非管理员隐藏上传人列头
    }

    // 事件监听器
    if (imageUpload) imageUpload.addEventListener('change', handleImageFileSelection);
    if (uploadButton) uploadButton.addEventListener('click', handleImageUpload);
    if (firstPageButton) firstPageButton.addEventListener('click', () => {
        if (currentPage !== 1) {
            currentPage = 1;
            loadImagesHistory();
        }
    });
    if (prevPageButton) prevPageButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            loadImagesHistory();
        }
    });
    if (nextPageButton) nextPageButton.addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            loadImagesHistory();
        }
    });
    if (lastPageButton) lastPageButton.addEventListener('click', () => {
        if (currentPage !== totalPages) {
            currentPage = totalPages;
            loadImagesHistory();
        }
    });

    // 初始化时加载图片历史列表
    loadImagesHistory();

    // 使用事件委托处理按钮点击，确保只绑定一次
    if (imageListTableBody) {
        imageListTableBody.addEventListener('click', (e) => {
            const target = e.target;
            const imageUUID = target.dataset.imageUuid;
            if (target.classList.contains('delete-button')) {
                deleteImage(imageUUID);
            } else if (target.classList.contains('view-results-button')) {
                viewResults(imageUUID);
            }
        });
    }
}

// 处理文件选择
function handleImageFileSelection() {
    const imageUpload = document.getElementById('imageUpload'); // 获取元素，因为它们在函数外部定义
    const selectedFileNameSpan = document.getElementById('selected-file-name');
    const uploadMessage = document.getElementById('uploadMessage');

    if (imageUpload.files.length > 0) {
        selectedFileNameSpan.textContent = imageUpload.files[0].name;
        uploadMessage.textContent = '文件已选择，等待上传。';
    } else {
        selectedFileNameSpan.textContent = '未选择文件';
        uploadMessage.textContent = '请选择一个图片文件进行解析...';
    }
}

// 处理图片上传
async function handleImageUpload() {
    const imageUpload = document.getElementById('imageUpload');
    const uploadMessage = document.getElementById('uploadMessage');
    const uploadButton = document.getElementById('uploadButton');
    const selectedFileNameSpan = document.getElementById('selected-file-name');

    if (!imageUpload.files || imageUpload.files.length === 0) {
        uploadMessage.textContent = '请先选择一个图片文件。';
        return;
    }

    const file = imageUpload.files[0];
    const formData = new FormData();
    formData.append('file', file);

    uploadMessage.textContent = '正在上传和解析图片，请稍候...';
    uploadMessage.classList.remove('success', 'error');
    uploadMessage.classList.add('info');
    uploadButton.disabled = true;

    try {
        const response = await fetchWithAuth(`${API_BASE_URL}/api/image_analysis/upload_image`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '图片解析失败');
        }

        const result = await response.json();
        console.log('图片上传和解析结果:', result);

        if (result && result.task_id) {
            await pollTaskStatus(result.task_id, uploadMessage);
            imageUpload.value = ''; // 清空文件输入，以便再次上传同一文件
            selectedFileNameSpan.textContent = '未选择文件';
        } else {
            throw new Error('服务器响应格式错误');
        }
    } catch (error) {
        console.error('图片上传错误:', error);
        uploadMessage.textContent = `错误：${error.message}`;
        uploadMessage.classList.remove('info', 'success'); // 清除信息和成功状态类
        uploadMessage.classList.add('error'); // 添加错误状态类
    } finally {
        uploadButton.disabled = false; // 启用按钮
    }
}

// 创建并返回一个图片历史表格行
function createImageRow(imageData, index) {
    const row = document.createElement('tr');
    row.setAttribute('data-image-uuid', imageData.uuid);

    // 序号列
    row.insertCell(0).textContent = (currentPage - 1) * itemsPerPage + index + 1; // 计算正确序号

    // 图片UUID列，显示部分，鼠标悬停显示完整
    const uuidCell = row.insertCell(1);
    uuidCell.textContent = imageData.uuid; // 直接显示完整的UUID
    uuidCell.title = imageData.uuid; // 鼠标悬停时显示完整UUID

    // 上传时间列
    row.insertCell(2).textContent = new Date(imageData.created_at).toLocaleString();
    // 检测到人物列
    row.insertCell(3).textContent = imageData.person_count;

    const currentUser = Auth.getUserInfo();
    let actionsCellIndex = 4; // 默认操作列的索引

    if (currentUser && currentUser.role === 'admin') {
        // 管理员显示上传人信息
        const uploaderCell = row.insertCell(4); // 上传人列
        uploaderCell.textContent = imageData.uploader_username || '未知';
        actionsCellIndex = 5; // 如果有上传人列，操作列索引后移
    }

    const actionsCell = row.insertCell(actionsCellIndex);
    const buttonsContainer = document.createElement('div');
    buttonsContainer.classList.add('actions-buttons-container');

    const viewButton = document.createElement('button');
    viewButton.textContent = '查看结果';
    viewButton.classList.add('button', 'view-results-button');
    viewButton.setAttribute('data-image-uuid', imageData.uuid);
    buttonsContainer.appendChild(viewButton);

    if (currentUser && currentUser.role === 'admin') {
        const deleteButton = document.createElement('button');
        deleteButton.textContent = '删除';
        deleteButton.classList.add('button', 'delete-button');
        deleteButton.setAttribute('data-image-uuid', imageData.uuid);
        buttonsContainer.appendChild(deleteButton);
    }

    actionsCell.appendChild(buttonsContainer);
    return row;
}

// 加载图片历史列表
async function loadImagesHistory() {
    const imageListTableBody = document.getElementById('image-list-table-body');
    const noResultsMessage = document.getElementById('noResultsMessage');
    const uploaderHeader = document.getElementById('uploader-header'); // 获取上传人列头
    const prevPageButton = document.getElementById('prevPage');
    const nextPageButton = document.getElementById('nextPage');
    const currentPageSpan = document.getElementById('currentPage');
    const totalPagesSpan = document.getElementById('totalPages');
    const paginationControls = document.getElementById('paginationControls'); // 获取分页容器
    const firstPageButton = document.getElementById('firstPage'); // 新增
    const lastPageButton = document.getElementById('lastPage'); // 新增

    imageListTableBody.innerHTML = ''; // 清空现有列表
    noResultsMessage.style.display = 'none';
    if (paginationControls) paginationControls.style.display = 'none'; // 默认隐藏分页控件

    const currentUser = Auth.getUserInfo();

    // 根据用户角色显示或隐藏上传人列头
    if (currentUser && currentUser.role === 'admin' && uploaderHeader) {
        uploaderHeader.style.display = '';
    } else if (uploaderHeader) {
        uploaderHeader.style.display = 'none';
    }

    try {
        // 获取总数
        const countResponse = await fetchWithAuth(`${API_BASE_URL}/api/image_analysis/history/count`);
        if (!countResponse.ok) {
            throw new Error('无法加载图片历史总数');
        }
        const countData = await countResponse.json();
        totalImages = countData.total_count;
        totalPages = Math.ceil(totalImages / itemsPerPage);

        // 根据总数和当前页更新分页按钮状态
        if (currentPageSpan) currentPageSpan.textContent = currentPage;
        if (totalPagesSpan) totalPagesSpan.textContent = totalPages;
        if (prevPageButton) prevPageButton.disabled = (currentPage === 1);
        if (nextPageButton) nextPageButton.disabled = (currentPage === totalPages || totalImages === 0);
        if (firstPageButton) firstPageButton.disabled = (currentPage === 1);
        if (lastPageButton) lastPageButton.disabled = (currentPage === totalPages || totalImages === 0);
        
        // 计算 skip 和 limit
        const skip = (currentPage - 1) * itemsPerPage;
        const limit = itemsPerPage;

        const response = await fetchWithAuth(`${API_BASE_URL}/api/image_analysis/history?skip=${skip}&limit=${limit}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '无法加载图片历史记录');
        }

        const history = await response.json();
        console.log('图片历史记录:', history);

        if (history.length > 0) {
            history.forEach((imageData, index) => { // 传递 index
                const row = createImageRow(imageData, index); // 传递 index
                imageListTableBody.appendChild(row);
            });
            if (paginationControls) paginationControls.style.display = 'flex'; // 显示分页控件
        } else {
            noResultsMessage.style.display = 'block';
            noResultsMessage.textContent = '没有图片解析历史记录。';
            if (paginationControls) paginationControls.style.display = 'none';
        }
    } catch (error) {
        console.error('加载图片历史记录失败:', error);
        noResultsMessage.style.display = 'block';
        noResultsMessage.textContent = `加载历史记录失败：${error.message}`;
        if (paginationControls) paginationControls.style.display = 'none';
    }
}

// 查看结果
function viewResults(imageUUID) {
    // 在新标签页中打开结果页面
    window.open(`/image_analysis_results?uuid=${imageUUID}`, '_blank');
}

// 删除图片
async function deleteImage(imageUUID) {
    if (confirm(`确定要删除图片 UUID: ${imageUUID} 吗？这将同时删除所有相关的分析结果和文件。`)) {
        try {
            const deleteResponse = await fetchWithAuth(`${API_BASE_URL}/api/image_analysis/${imageUUID}`, { method: 'DELETE' }); // 添加 /api 前缀
            if (!deleteResponse.ok) {
                const errorData = await deleteResponse.json();
                throw new Error(errorData.detail || '删除失败');
            }
            alert('图片及相关结果删除成功！');
            loadImagesHistory(); // 重新加载列表
        } catch (error) {
            alert(`删除图片失败: ${error.message}`);
            console.error('删除图片错误:', error);
        }
    }
}

// Lightbox 相关的公共函数 (如果 common.js 中没有)
// 这里暂时不移除，确保 lightbox 功能在 image_analysis.html 页面仍然可用
// const lightbox = document.getElementById('lightbox');
// const lightboxClose = document.querySelector('.lightbox-close');
// if (lightboxClose) lightboxClose.addEventListener('click', () => {
//     lightbox.classList.add('hidden');
// });
// if (lightbox) lightbox.addEventListener('click', (e) => { 
//     if (e.target === lightbox) lightbox.classList.add('hidden'); 
// }); 