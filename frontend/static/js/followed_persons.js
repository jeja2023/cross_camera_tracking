import { Auth, API_BASE_URL } from './auth.js';
import { initializeAuthenticatedPage } from './common.js';

document.addEventListener('DOMContentLoaded', async () => {
    console.log("followed_persons.js: DOMContentLoaded event fired.");
    initializeAuthenticatedPage(); // Call common function for auth check and UI update

    // Removed redundant authentication check, as initializeAuthenticatedPage already handles redirection
    // if (!Auth.isAuthenticated()) {
    //     console.log("followed_persons.js: User not authenticated, redirecting to login.");
    //     return; // Redirection handled by initializeAuthenticatedPage
    // }

    const tbody = document.querySelector('#followed-persons-table tbody');
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const pageInfoSpan = document.getElementById('page-info');
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');

    let currentPage = 1;
    const itemsPerPage = 10; // 可以根据需要调整每页显示的数量
    let totalPages = 1;

    async function fetchFollowedPersons() {
        const token = Auth.getToken(); // Corrected: use Auth.getToken()
        if (!token) {
            console.error("Auth token not found.");
            return;
        }

        try {
            // 构建查询参数
            const skip = (currentPage - 1) * itemsPerPage;
            const limit = itemsPerPage;
            const url = `/followed_persons/?skip=${skip}&limit=${limit}`;

            const response = await fetch(url, {
                headers: {
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

            const data = await response.json();
            console.log("Fetched followed persons data:", data);
            renderTable(data.items);

            totalPages = Math.ceil(data.total / itemsPerPage);
            updatePaginationControls();

        } catch (error) {
            console.error("Error fetching followed persons:", error);
            alert("获取关注人员列表失败: " + error.message);
        }
    }

    function renderTable(persons) {
        tbody.innerHTML = ''; // Clear existing rows
        if (persons.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5">没有关注人员。</td></tr>'; // Update colspan to 5
            return;
        }

        const isAdmin = Auth.currentUserIsAdmin(); // 假设 Auth 模块提供了此方法

        persons.forEach((person, index) => {
            const row = tbody.insertRow();
            row.insertCell().textContent = (currentPage - 1) * itemsPerPage + index + 1; // Add row number
            row.insertCell().textContent = person.individual ? person.individual.name : '未知';
            row.insertCell().textContent = person.individual ? person.individual.id_card : '未知';
            row.insertCell().textContent = new Date(person.follow_time).toLocaleString();
            // 新增实时比对状态列
            const actionCell = row.insertCell();
            actionCell.classList.add('action-buttons'); // Add a class for styling

            // 1. 取消关注按钮
            const unfollowButton = document.createElement('button');
            unfollowButton.textContent = '取消关注';
            unfollowButton.classList.add('unfollow-button');
            unfollowButton.onclick = () => handleToggleFollow(person.individual.id, false); // Pass individual ID
            actionCell.appendChild(unfollowButton);

            // 2. 查看注册图片按钮
            const viewRegisteredImagesButton = document.createElement('button');
            viewRegisteredImagesButton.textContent = '注册图片'; // 文本修改为“注册图片”
            viewRegisteredImagesButton.classList.add('view-registered-images-button');
            viewRegisteredImagesButton.onclick = () => {
                if (person.individual && person.individual.id) {
                    // 重定向到新的图片查看页面，并传递 individual_id, name, id_card
                    window.location.href = `/enrollment_images_viewer?individual_id=${person.individual.id}&individual_name=${encodeURIComponent(person.individual.name)}&individual_id_card=${encodeURIComponent(person.individual.id_card)}`;
                } else {
                    alert("无法获取人物档案ID。");
                }
            };
            actionCell.appendChild(viewRegisteredImagesButton);

            // 3. 历史轨迹按钮 (新增)
            const historyButton = document.createElement('button');
            historyButton.textContent = '历史轨迹';
            historyButton.classList.add('history-button');
            historyButton.onclick = () => {
                if (person.individual && person.individual.id) {
                    window.location.href = `/followed_person_history.html?individual_id=${person.individual.id}&individual_name=${encodeURIComponent(person.individual.name)}`;
                } else {
                    alert("无法获取人物档案ID。");
                }
            };
            actionCell.appendChild(historyButton);

            // 4. 查看预警按钮
            const viewWarningButton = document.createElement('button');
            viewWarningButton.textContent = '预警信息'; // 文本修改为“预警信息”
            viewWarningButton.classList.add('view-warning-button');
            viewWarningButton.onclick = () => {
                if (person.individual && person.individual.id) {
                    // 将 individual.id 作为 id 参数传递，将 individual.name 作为 name 参数传递
                    window.location.href = `/followed_person_alerts.html?id=${person.individual.id}&name=${encodeURIComponent(person.individual.name)}`;
                } else {
                    alert("无法获取人物档案ID。");
                }
            };
            actionCell.appendChild(viewWarningButton);

            // 5. 实时比对切换按钮 (仅管理员可见)
            if (isAdmin) {
                const toggleComparisonButton = document.createElement('button');
                const isRealtimeEnabled = person.individual && person.individual.is_realtime_comparison_enabled;
                toggleComparisonButton.textContent = isRealtimeEnabled ? '关闭实时比对' : '开启实时比对';
                toggleComparisonButton.classList.add(isRealtimeEnabled ? 'toggle-off-button' : 'toggle-on-button');
                toggleComparisonButton.onclick = () => handleToggleRealtimeComparison(person.individual.id, !isRealtimeEnabled);
                actionCell.appendChild(toggleComparisonButton);
            }
        });
    }

    async function handleToggleRealtimeComparison(individualId, enable) {
        const actionText = enable ? '开启' : '关闭';
        if (!confirm(`确定要${actionText}此人物的实时比对吗？`)) {
            return;
        }

        const token = Auth.getToken();
        if (!token) {
            console.error("Auth token not found.");
            alert("认证失败，请重新登录。");
            return;
        }

        try {
            const response = await fetch('/followed_persons/toggle_realtime_comparison/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({
                    individual_id: individualId,
                    is_enabled: enable
                })
            });

            if (!response.ok) {
                if (response.status === 401) {
                    alert("会话过期，请重新登录。");
                    window.location.href = '/login';
                    return;
                }
                if (response.status === 403) {
                    alert("权限不足，只有管理员才能执行此操作。");
                    return;
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            alert(result.message);
            fetchFollowedPersons(); // Refresh the list to reflect status change

        } catch (error) {
            console.error("Error toggling realtime comparison:", error);
            alert(`${actionText}实时比对失败: ` + error.message);
        }
    }

    // Renamed from handleUnfollow to handleToggleFollow
    async function handleToggleFollow(individualId, isFollowed, performGlobalSearch = false) {
        const actionText = isFollowed ? '关注' : '取消关注';
        let confirmed = true;

        if (isFollowed) {
            confirmed = confirm(`确定要${actionText}此人物吗？`);
            if (confirmed && performGlobalSearch) {
                confirmed = confirm(`是否以该人物注册图片进行一次全局搜索比对？`);
            }
        } else {
            confirmed = confirm(`确定要${actionText}此人物吗？`);
        }

        if (!confirmed) {
            return;
        }

        const token = Auth.getToken();
        if (!token) {
            console.error("Auth token not found.");
            alert("认证失败，请重新登录。");
            return;
        }

        try {
            const bodyData = {
                individual_id: individualId,
                is_followed: isFollowed,
            };

            if (isFollowed && performGlobalSearch) {
                bodyData.perform_global_search = true;
            }

            const response = await fetch('/followed_persons/toggle_follow/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify(bodyData)
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
            fetchFollowedPersons(); // Refresh the list

        } catch (error) {
            console.error("Error toggling follow status:", error);
            alert(`${actionText}失败: ` + error.message);
        }
    }

    // Function to show enrollment images in a modal
    // 此函数已被废弃，因为我们现在使用单独的页面来显示图片。
    // 请确保此函数的所有调用已被移除或修改。
    // async function showEnrollmentImagesModal(individualId, individualName, idCard) {
    //     const token = Auth.getToken();
    //     if (!token) {
    //         console.error("Auth token not found.");
    //         alert("认证失败，请重新登录。");
    //         return;
    //     }
    //     try {
    //         const url = `/followed_persons/${individualId}/enrollment_images`;
    //         const response = await fetch(url, {
    //             headers: {
    //                 'Authorization': `Bearer ${token}`
    //             }
    //         });
    //         if (!response.ok) {
    //             if (response.status === 401) {
    //                 alert("会话过期，请重新登录。");
    //                 window.location.href = '/login';
    //                 return;
    //             }
    //             throw new Error(`HTTP error! status: ${response.status}`);
    //         }
    //         const data = await response.json();
    //         const images = data.items;
    //         const modal = document.getElementById('enrollmentImagesModal');
    //         const modalTitle = document.getElementById('enrollmentImagesModalTitle');
    //         const modalBody = document.getElementById('enrollmentImagesModalBody');
    //         modalTitle.textContent = `人物档案: ${individualName} (${idCard}) 的注册图片`;
    //         modalBody.innerHTML = ''; // Clear previous images
    //         if (images.length === 0) {
    //             modalBody.innerHTML = '<p>没有找到注册图片。</p>';
    //         } else {
    //             images.forEach(image => {
    //                 const imgContainer = document.createElement('div');
    //                 imgContainer.classList.add('enrollment-image-item');
    //                 const img = document.createElement('img');
    //                 img.src = image.image_path;
    //                 img.alt = `注册图片 ${image.image_path}`;
    //                 img.classList.add('img-thumbnail'); // Add Bootstrap-like styling
    //                 const imgPath = document.createElement('p');
    //                 imgPath.textContent = image.image_path.split('/').pop(); // Display filename only
    //                 imgPath.classList.add('image-filename');
    //                 imgContainer.appendChild(img);
    //                 imgContainer.appendChild(imgPath);
    //                 modalBody.appendChild(imgContainer);
    //             });
    //         }
    //         modal.style.display = 'block'; // Show the modal
    //         // Close button functionality
    //         const closeButton = modal.querySelector('.close-button');
    //         closeButton.onclick = () => {
    //             modal.style.display = 'none';
    //         };
    //         // Close when clicking outside the modal content
    //         window.onclick = (event) => {
    //             if (event.target == modal) {
    //                 modal.style.display = 'none';
    //             }
    //         };
    //     } catch (error) {
    //         console.error("Error fetching enrollment images:", error);
    //         alert("获取注册图片失败: " + error.message);
    //     }
    // }

    function updatePaginationControls() {
        pageInfoSpan.textContent = `页数：${currentPage} / ${totalPages}`;
        prevPageButton.disabled = currentPage === 1;
        nextPageButton.disabled = currentPage === totalPages;
    }

    prevPageButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            fetchFollowedPersons();
        }
    });

    nextPageButton.addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            fetchFollowedPersons();
        }
    });

    // Initial fetch
    fetchFollowedPersons();
});