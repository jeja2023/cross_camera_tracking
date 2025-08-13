import { setupImageGalleryClickEvents } from './common.js';
import { Auth, API_BASE_URL } from './auth.js'; // Import Auth object and API_BASE_URL

let currentPage = 1;
const limit = 20; // Number of items per page
let totalPersons = 0; // Total number of persons
let currentPersonsData = []; // Store current page's person data for modal

// 将 DOM 元素变量声明移到全局作用域
let personSearchInput;
let filterMarkedForRetrainSelect;
let exportPersonsBtn;
let pageInfoSpan;
let prevPageBtn;
let nextPageBtn;
let firstPageBtn;
let lastPageBtn;

// 将 fetchPersons 函数移到全局作用域
async function fetchPersons(searchQuery = '', markedForRetrain = '') { // 移除 isVerified 参数
    try {
        const skip = (currentPage - 1) * limit;
        let url = `/api/persons/all?skip=${skip}&limit=${limit}&has_id_card=true&is_verified=true`; // 默认只显示有身份证号/ID且已审核的数据
        
        if (searchQuery) {
            url += `&query=${encodeURIComponent(searchQuery)}`;
        }
        if (markedForRetrain !== '') { // 只在选择了具体值时才添加
            url += `&marked_for_retrain=${markedForRetrain}`;
        }

        const response = await fetch(url, {
            method: 'GET',
            headers: {
                'Authorization': `Bearer ${Auth.getToken()}`, // Use Auth.getToken()
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
        const personTableBody = document.getElementById('personTableBody'); // 在这里获取，确保每次都有
        personTableBody.innerHTML = ''; 

        if (currentPersonsData.length === 0) {
            personTableBody.innerHTML = '<tr><td colspan="12">暂无人员信息。</td></tr>'; // 更新 colspan 为 12
            prevPageBtn.disabled = true;
            nextPageBtn.disabled = true;
            firstPageBtn.disabled = true;
            lastPageBtn.disabled = true;
            pageInfoSpan.textContent = `页码 0 / 0`;
            return;
        }

        currentPersonsData.forEach((person, index) => { 
            const row = document.createElement('tr');
            const createdAt = new Date(person.created_at).toLocaleString();
            const sourceInfo = person.upload_image_uuid || person.video_uuid || person.stream_uuid || '';
            let cropImagePath = person.crop_image_path ? person.crop_image_path.replace(/\\/g, '/') : '';
            const verifiedStatusText = person.is_verified ? '已审核' : '未审核';
            const retrainStatusText = person.marked_for_retrain ? '是' : '否';
            const confidenceText = person.confidence_score !== undefined && person.confidence_score !== null ? (person.confidence_score * 100).toFixed(2) + '%' : '';

            row.innerHTML = `
                <td>${(currentPage - 1) * limit + index + 1}</td> 
                <td class="name-column">${person.name || ''}</td>
                <td>${person.id_card || ''}</td>
                <td class="uuid-column" title="${person.uuid}">${person.uuid}</td> 
                <td><img src="${cropImagePath.replace(/\\/g, '/')}" alt="人物裁剪图" class="person-crop-img"></td>
                <td class="source-column" title="${sourceInfo}">${sourceInfo}</td> 
                <td>${person.uploaded_by_username || ''}</td> <!-- 新增的上传用户列 -->
                <td class="created-at-cell created-at-column">${createdAt}</td> 
                <td>${verifiedStatusText}</td>
                <td>${retrainStatusText}</td>
                <td>${confidenceText}</td>
                <td class="actions-column">
                    <button class="btn-delete-person" data-uuid="${person.uuid}">删除</button>
                    <button class="btn-toggle-follow${person.is_followed ? ' btn-unfollow' : ''}" data-uuid="${person.uuid}">
                        ${person.is_followed ? '取消关注' : '关注'}
                    </button>
                </td>
            `;
            personTableBody.appendChild(row);
        });

        const totalPages = Math.ceil(totalPersons / limit);
        pageInfoSpan.textContent = `页码 ${currentPage} / ${totalPages}`; 
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = currentPage === totalPages || totalPages === 0;
        firstPageBtn.disabled = currentPage === 1;
        lastPageBtn.disabled = currentPage === totalPages || totalPages === 0;

        setupImageClickEvents();

    } catch (error) {
        console.error('获取人员信息错误:', error);
        const personTableBody = document.getElementById('personTableBody'); // 确保能获取到
        if (personTableBody) {
            personTableBody.innerHTML = `<tr><td colspan="12" style="color: red;">错误：${error.message}</td></tr>`; // 更新 colspan 为 12
        }
    }
}

export function initPersonListPage() {
    // 在这里获取 DOM 元素并赋值给全局变量
    personSearchInput = document.getElementById('person-search-input');
    filterMarkedForRetrainSelect = document.getElementById('filter-marked-for-retrain');
    exportPersonsBtn = document.getElementById('export-persons-btn');
    pageInfoSpan = document.getElementById('pageInfo');
    prevPageBtn = document.getElementById('prevPageBtn');
    nextPageBtn = document.getElementById('nextPageBtn');
    firstPageBtn = document.getElementById('firstPageBtn');
    lastPageBtn = document.getElementById('lastPageBtn');

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

    const applyPersonFilterButton = document.getElementById('apply-person-filter-button'); // 确保在这里获取
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
            const markedForRetrain = filterMarkedForRetrainSelect.value;

            let exportUrl = `/export/person_archives?has_id_card=true&is_verified=true`; // 更改为新的人员档案导出路由
            if (searchQuery) {
                exportUrl += `&query=${encodeURIComponent(searchQuery)}`;
            }
            if (markedForRetrain !== '') {
                exportUrl += `&marked_for_retrain=${markedForRetrain}`;
            }
            exportUrl = exportUrl.endsWith('&') ? exportUrl.slice(0, -1) : exportUrl;
            exportUrl = exportUrl.endsWith('?') ? exportUrl.slice(0, -1) : exportUrl;

            try {
                const response = await fetch(exportUrl, {
                    method: 'GET',
                    headers: {
                        'Authorization': `Bearer ${Auth.getToken()}`, // Use Auth.getToken()
                    },
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`导出失败: ${response.status} ${response.statusText} - ${errorText}`);
                }

                const contentDisposition = response.headers.get('Content-Disposition');
                let filename = 'persons_export.xlsx'; 
                if (contentDisposition) {
                    const filenameMatch = contentDisposition.match(/filename\*?=\S*?''(.*?)(?:;|$)/i);
                    if (filenameMatch && filenameMatch[1]) {
                        filename = decodeURIComponent(filenameMatch[1].replace(/"/g, ''));
                    } else {
                        const fallbackMatch = contentDisposition.match(/filename="(.*?)"/i);
                        if (fallbackMatch && fallbackMatch[1]) {
                            filename = fallbackMatch[1];
                        }
                    }
                }
                
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

// 新增：处理删除人物的函数
async function handleDeletePerson(personUuid) {
    if (!confirm(`确定要删除人物 ${personUuid} 吗？此操作不可逆！`)) {
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/api/persons/${personUuid}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${Auth.getToken()}`, // Use Auth.getToken()
            },
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || '删除人物失败');
        }

        alert(`人物 ${personUuid} 已成功删除。`);
        fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value); // 重新加载列表
    } catch (error) {
        console.error('删除人物时出错:', error);
        alert(error.message || '删除人物失败，请检查控制台了解详情。');
    }
}

// 新增：处理切换关注状态的函数
async function handleToggleFollowStatus(personUuid, currentStatus) {
    const actionText = currentStatus ? '取消关注' : '关注';
    if (!confirm(`确定要${actionText}人物 ${personUuid} 吗？`)) {
        return;
    }

    try {
        // 获取当前人物的完整数据，以便拿到 individual_id
        const personToToggle = currentPersonsData.find(p => p.uuid === personUuid);
        if (!personToToggle || !personToToggle.individual_id) {
            alert("无法获取人物档案ID，操作失败。");
            return;
        }
        const individualId = personToToggle.individual_id;

        const response = await fetch(`/followed_persons/toggle_follow/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${Auth.getToken()}` // Use Auth.getToken()
            },
            body: JSON.stringify({
                individual_id: individualId,
                is_followed: !currentStatus // 切换关注状态
            })
        });

        if (!response.ok) {
            if (response.status === 401) {
                alert("会话过期，请重新登录。");
                window.location.href = '/login';
                return;
            }
            const errorData = await response.json();
            throw new Error(errorData.detail || `切换关注状态失败`);
        }

        const result = await response.json();
        alert(result.message);
        // 重新加载人物列表以反映最新状态
        fetchPersons(personSearchInput.value, filterMarkedForRetrainSelect.value);
    } catch (error) {
        console.error(`切换关注状态时出错:`, error);
        alert(error.message || `切换关注状态失败，请检查控制台了解详情。`);
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

// 新增：为操作按钮添加事件委托
function setupActionButtonsEvents() {
    const personTableBody = document.getElementById('personTableBody');
    if (!personTableBody) {
        console.warn('personTableBody not found, cannot set up action button events.');
        return;
    }

    personTableBody.addEventListener('click', async (event) => {
        if (event.target.classList.contains('btn-delete-person')) {
            const personUuid = event.target.dataset.uuid;
            await handleDeletePerson(personUuid);
        } else if (event.target.classList.contains('btn-toggle-follow')) {
            const personUuid = event.target.dataset.uuid;
            // 获取当前关注状态，以便传递给后端和提示用户
            const row = event.target.closest('tr');
            if (row) {
                const rowIndex = Array.from(row.parentNode.children).indexOf(row);
                const person = currentPersonsData[rowIndex];
                if (person) {
                    await handleToggleFollowStatus(personUuid, person.is_followed);
                }
            }
        }
    });
}

// Call this function after the table content is loaded/updated
// Or use MutationObserver if content is frequently added/removed, as done in common.js's setupImageGalleryClickEvents itself
// The initial call for person_list page
document.addEventListener('DOMContentLoaded', () => {
    // Make sure initPersonListPage is called first, which fetches initial data
    // Then call setupImageClickEvents
    // No, setupImageGalleryClickEvents in common.js already handles MutationObserver,
    // so we just need to call it once after the page is loaded and `personTableBody` exists.

    // 在页面加载和数据获取后调用 setupActionButtonsEvents
    initPersonListPage(); 
    setupActionButtonsEvents(); // 新增：调用操作按钮事件设置函数
}); 