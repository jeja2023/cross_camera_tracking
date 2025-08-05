import { API_BASE_URL } from './auth.js';
import { setupImageGalleryClickEvents, showNotification } from './common.js';

let currentPage = 1;
const limit = 20; // Number of items per page
let totalPersons = 0; // Total number of persons
let currentPersonsData = []; // Store current page's person data for modal

export function initPersonListPage() {
    const personTableBody = document.getElementById('personTableBody');
    const prevPageBtn = document.getElementById('prevPageBtn');
    const nextPageBtn = document.getElementById('nextPageBtn');
    const firstPageBtn = document.getElementById('firstPageBtn');
    const lastPageBtn = document.getElementById('lastPageBtn');
    const pageInfoSpan = document.getElementById('pageInfo');
    const personSearchInput = document.getElementById('person-search-input');
    const applyPersonFilterButton = document.getElementById('apply-person-filter-button');
    // const filterIsVerifiedSelect = document.getElementById('filter-is-verified'); // 已移除，无需引用
    const filterMarkedForRetrainSelect = document.getElementById('filter-marked-for-retrain'); // 新增
    const exportPersonsBtn = document.getElementById('export-persons-btn'); // 新增

    // 新增一个隐藏的输入框来存储 has_id_card 的值
    // const filterHasIdCard = document.getElementById('filter-has-id-card'); // 暂时不需要UI，直接硬编码

    async function fetchPersons(searchQuery = '', markedForRetrain = '') { // 移除 isVerified 参数
        try {
            const skip = (currentPage - 1) * limit;
            let url = `/api/persons/all?skip=${skip}&limit=${limit}&has_id_card=true&is_verified=true`; // 默认只显示有身份证号/ID且已审核的数据
            
            if (searchQuery) {
                url += `&query=${encodeURIComponent(searchQuery)}`;
            }
            // if (isVerified !== '') { // 只在选择了具体值时才添加
            //     url += `&is_verified=${isVerified}`;
            // }
            if (markedForRetrain !== '') { // 只在选择了具体值时才添加
                url += `&marked_for_retrain=${markedForRetrain}`;
            }
            // if (filterHasIdCard && filterHasIdCard.value !== '') { // 如果需要UI筛选，取消注释
            //     url += `&has_id_card=${filterHasIdCard.value}`;
            // }

            const response = await fetch(url, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '获取人员信息失败');
            }

            const data = await response.json(); 
            currentPersonsData = data.items; 
            totalPersons = data.total; 
            personTableBody.innerHTML = ''; 

            if (currentPersonsData.length === 0) {
                personTableBody.innerHTML = '<tr><td colspan="11">暂无人员信息。</td></tr>'; // 更新 colspan 为 11 (原10+1)
                prevPageBtn.disabled = true;
                nextPageBtn.disabled = true;
                firstPageBtn.disabled = true;
                lastPageBtn.disabled = true;
                pageInfoSpan.textContent = `页码 0 / 0`;
                return;
            }

            currentPersonsData.forEach(async (person, index) => { 
                const row = document.createElement('tr');
                // const truncatedUuid = `${person.uuid.substring(0, 4)}...${person.uuid.substring(person.uuid.length - 4)}`;
                const createdAt = new Date(person.created_at).toLocaleString();
                const sourceInfo = person.upload_image_uuid || person.video_uuid || person.stream_uuid || '';

                let cropImagePath = person.crop_image_path ? person.crop_image_path.replace(/\\/g, '/') : '';

                // 审核状态和再训练标记的显示文本
                const verifiedStatusText = person.is_verified ? '已审核' : '未审核';
                const retrainStatusText = person.marked_for_retrain ? '是' : '否';
                const confidenceText = person.confidence_score !== undefined && person.confidence_score !== null ? (person.confidence_score * 100).toFixed(2) + '%' : '';

                let actionButtonHtml = ''
                const individualId = person.individual_id; // 获取 individual_id
                const personName = person.name || '未知';

                if (individualId) { // 只有当 individual_id 存在时才显示关注/取消关注按钮
                    const isFollowed = await checkFollowStatus(individualId);
                    if (isFollowed) {
                        actionButtonHtml = `<button class="btn btn-sm unfollow-btn unfollow-button" data-individual-id="${individualId}" data-person-name="${personName}">取消关注</button>`;
                    } else {
                        actionButtonHtml = `<button class="btn btn-sm follow-btn follow-button" data-individual-id="${individualId}" data-person-name="${personName}">关注</button>`;
                    }
                } else {
                    actionButtonHtml = 'N/A'; // 如果没有 individual_id，显示 N/A
                }

                row.innerHTML = `
                    <td>${(currentPage - 1) * limit + index + 1}</td> 
                    <td>${person.name || ''}</td>
                    <td>${person.id_card || ''}</td>
                    <td class="uuid-column" title="${person.uuid}">${person.uuid}</td> 
                    <td><img src="${cropImagePath.replace(/\\/g, '/')}" alt="人物裁剪图" class="person-crop-img"></td>
                    <td class="source-column" title="${sourceInfo}">${sourceInfo}</td> 
                    <td>${person.uploaded_by_username || ''}</td> <!-- 新增的上传用户列 -->
                    <td class="created-at-cell">${createdAt}</td> 
                    <td>${verifiedStatusText}</td>
                    <td>${retrainStatusText}</td>
                    <td>${confidenceText}</td>
                    <td>${actionButtonHtml}</td> <!-- 操作列 -->
                `;
                personTableBody.appendChild(row);
            });

            // 移除对 addFollowButtonsToTable 的调用，改为事件委托
            // addFollowButtonsToTable(currentPersonsData);

            const totalPages = Math.ceil(totalPersons / limit);
            pageInfoSpan.textContent = `页码 ${currentPage} / ${totalPages}`; 
            prevPageBtn.disabled = currentPage === 1;
            nextPageBtn.disabled = currentPage === totalPages || totalPages === 0;
            firstPageBtn.disabled = currentPage === 1;
            lastPageBtn.disabled = currentPage === totalPages || totalPages === 0;

            setupImageClickEvents();

        } catch (error) {
            console.error('获取人员信息错误:', error);
            showNotification(`获取人员信息错误: ${error.message}`, 'error');
            personTableBody.innerHTML = `<tr><td colspan="11" style="color: red;">错误：${error.message}</td></tr>`; // 更新 colspan 为 11
        }
    }

    // 使用事件委托处理关注/取消关注按钮的点击事件
    personTableBody.addEventListener('click', async (event) => {
        if (event.target.classList.contains('follow-btn')) {
            const individualId = parseInt(event.target.dataset.individualId);
            const personName = event.target.dataset.personName;
            await followPerson(individualId, personName);
        } else if (event.target.classList.contains('unfollow-btn')) {
            const individualId = parseInt(event.target.dataset.individualId);
            const personName = event.target.dataset.personName;
            await unfollowPerson(individualId, personName);
        }
    });

    // 新增：检查人物是否已被关注的函数
    async function checkFollowStatus(individualId) {
        try {
            const response = await fetch(`${API_BASE_URL}/followed_persons/${individualId}/is_followed`, {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                }
            });
            if (!response.ok) {
                // 如果是 404，可能是 Individual 不存在，也当作未关注处理
                if (response.status === 404) {
                    return false;
                }
                throw new Error('检查关注状态失败');
            }
            const isFollowed = await response.json();
            return isFollowed;
        } catch (error) {
            console.error(`检查 Individual ${individualId} 关注状态时出错:`, error);
            return false; // 默认返回未关注
        }
    }

    // 新增：关注人物的函数
    async function followPerson(individualId, personName) {
        try {
            const response = await fetch(`${API_BASE_URL}/followed_persons/follow/${individualId}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '关注人物失败');
            }
            showNotification(`成功关注 ${personName || '该人物'}！`, 'success');
            // 重新加载人物列表以更新按钮状态
            fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value);
        } catch (error) {
            console.error('关注人物错误:', error);
            showNotification(`关注人物失败: ${error.message}`, 'error');
        }
    }

    // 新增：取消关注人物的函数
    async function unfollowPerson(individualId, personName) {
        try {
            const response = await fetch(`${API_BASE_URL}/followed_persons/unfollow/${individualId}`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || '取消关注人物失败');
            }
            showNotification(`已取消关注 ${personName || '该人物'}！`, 'info');
            // 重新加载人物列表以更新按钮状态
            fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value);
        } catch (error) {
            console.error('取消关注人物错误:', error);
            showNotification(`取消关注人物失败: ${error.message}`, 'error');
        }
    }

    // Initial load (without filters)
    fetchPersons();

    firstPageBtn.addEventListener('click', () => {
        if (currentPage !== 1) {
            currentPage = 1;
            fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value);
        }
    });

    prevPageBtn.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value);
        }
    });

    nextPageBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(totalPersons / limit);
        if (currentPage < totalPages) {
            currentPage++;
            fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value);
        }
    });

    lastPageBtn.addEventListener('click', () => {
        const totalPages = Math.ceil(totalPersons / limit);
        if (currentPage !== totalPages) {
            currentPage = totalPages;
            fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value);
        }
    });

    if (applyPersonFilterButton) {
        applyPersonFilterButton.addEventListener('click', () => {
            currentPage = 1; 
            fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value);
        });
    }

    if (personSearchInput) {
        personSearchInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                currentPage = 1;
                fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value);
            }
        });
    }

    // 新增导出按钮的事件监听器
    if (exportPersonsBtn) {
        exportPersonsBtn.addEventListener('click', async () => {
            const searchQuery = personSearchInput.value;
            // const isVerified = filterIsVerifiedSelect.value; // 已移除，无需引用
            const markedForRetrain = filterMarkedForRetrainSelect.value;

            let exportUrl = `/export/person_archives?has_id_card=true&is_verified=true`; // 更改为新的人员档案导出路由
            if (searchQuery) {
                exportUrl += `&query=${encodeURIComponent(searchQuery)}`;
            }
            if (markedForRetrain !== '') {
                exportUrl += `&marked_for_retrain=${markedForRetrain}`;
            }
            // 移除末尾的 & 或 ?
            exportUrl = exportUrl.endsWith('&') ? exportUrl.slice(0, -1) : exportUrl;
            exportUrl = exportUrl.endsWith('?') ? exportUrl.slice(0, -1) : exportUrl;

            try {
                const response = await fetch(exportUrl, {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${localStorage.getItem('accessToken')}`,
                    },
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`导出失败: ${response.status} ${response.statusText} - ${errorText}`);
                }

                // 获取文件名
                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = 'persons_export.xlsx'; // Default filename
                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/filename\*?=\S*?''(.*?)(?:;|$)/i);
                    if (filenameMatch && filenameMatch[1]) {
                        // 解码URL编码的文件名，并替换可能存在的双引号
                        filename = decodeURIComponent(filenameMatch[1].replace(/"/g, ''));
                    } else {
                        const fallbackMatch = contentDisposition.match(/filename="(.*?)"/i);
                        if (fallbackMatch && fallbackMatch[1]) {
                            filename = fallbackMatch[1];
                        }
                    }
                }
                
                // 创建Blob对象并下载文件
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
                alert('人员档案已成功导出！');

            } catch (error) {
                console.error('导出人员档案时出错:', error);
                alert(error.message || '导出人员档案失败，请检查控制台了解详情。');
            }
        });
    }
}

// Function to add event listeners for the image modal
// Removed custom modal logic. Using common.js for lightbox.
function setupImageClickEvents() {
    const personTableBody = document.getElementById('personTableBody');
    if (!personTableBody) {
        console.warn('personTableBody not found, cannot set up image click events.');
        return;
    }
    console.log('person_list.js: 调用 setupImageGalleryClickEvents...');
    // Use common.js's setupImageGalleryClickEvents to handle clicks on cropped images
    // The getImageUrlFn should return the full_frame_image_path for the clicked person
    setupImageGalleryClickEvents(
        '#personTableBody', // Selector for the gallery container
        '.person-crop-img',  // Selector for the image elements within the gallery
        (imgElement) => {
            console.log('person_list.js: 图片点击事件被触发!', imgElement);
            // Find the parent row of the clicked image
            const row = imgElement.closest('tr');
            if (row) {
                // rowIndex is the index of the row within the current table display
                const rowIndex = Array.from(row.parentNode.children).indexOf(row);
                const person = currentPersonsData[rowIndex]; // Access directly by rowIndex from currentPersonsData
                if (person) {
                    let fullPath = person.full_frame_image_path || person.crop_image_path;
                    if (fullPath) {
                        // Remove leading slash if present, then prepend API_BASE_URL
                        fullPath = fullPath.replace(/\\/g, '/');
                        if (fullPath.startsWith('/')) {
                            fullPath = fullPath.substring(1);
                        }
                        const finalImageUrl = API_BASE_URL + '/' + fullPath;
                        console.log('person_list.js: 构造的最终图片URL:', finalImageUrl);
                        return finalImageUrl;
                    }
                }
            }
            console.log('person_list.js: 无法获取图片URL。');
            return ''; // Return empty string if path not found
        },
        true // dynamicContent is true because table content changes with pagination/filters
    );
}

// Call this function after the table content is loaded/updated
// Or use MutationObserver if content is frequently added/removed, as done in common.js's setupImageGalleryClickEvents itself
// The initial call for person_list page
document.addEventListener('DOMContentLoaded', () => {
    // Make sure initPersonListPage is called first, which fetches initial data
    // Then call setupImageClickEvents
    // No, setupImageGalleryClickEvents in common.js already handles MutationObserver,
    // so we just need to call it once after the page is loaded and `personTableBody` exists.
}); 