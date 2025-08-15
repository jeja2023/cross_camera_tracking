import { initializeAuthenticatedPage } from '/static/js/common.js';
import { Auth } from '/static/js/auth.js';

document.addEventListener('DOMContentLoaded', () => {
    initializeAuthenticatedPage();

    const personNameInput = document.getElementById('person-name');
    const idCardInputField = document.getElementById('id-card'); // 获取身份证号输入框
    const imageUploadInput = document.getElementById('image-upload');
    const imagePreviewDiv = document.getElementById('image-preview');
    // 移除旧的 UUID 相关元素引用
    // const existingPersonUuidInput = document.getElementById('existing-person-uuid');
    // const searchExistingPersonButton = document.getElementById('search-existing-person');
    // const existingPersonInfoDiv = document.getElementById('existing-person-info');
    // const infoUuidSpan = document.getElementById('info-uuid');
    // const infoNameSpan = document.getElementById('info-name');
    // const infoImageImg = document.getElementById('info-image');
    // const selectExistingPersonButton = document.getElementById('select-existing-person');
    const submitEnrollmentButton = document.getElementById('submit-enrollment');
    const statusMessageDiv = document.getElementById('status-message');

    let selectedFiles = [];
    // 移除 selectedPersonUuid = null; // 存储用户选择的现有 UUID

    // 移除动态添加身份证号输入框的代码，因为它已在HTML中硬编码
    // const idCardInput = document.createElement('div');
    // idCardInput.className = 'form-group';
    // idCardInput.innerHTML = `
    //     <label for="id-card">身份证号/其他ID (可选):</label>
    //     <input type="text" id="id-card" placeholder="请输入身份证号或MUID等标识">
    // `;
    // personNameInput.parentNode.insertBefore(idCardInput, personNameInput.nextSibling); // 插入到人物姓名下方
    // const idCardInputField = document.getElementById('id-card');

    // 获取图片上传区域元素
    const imageUploadArea = document.querySelector('.image-upload-area');

    // 阻止默认的拖放行为
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        imageUploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false); // 阻止全局拖放
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // 突出显示拖放区域
    ['dragenter', 'dragover'].forEach(eventName => {
        imageUploadArea.addEventListener(eventName, () => imageUploadArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        imageUploadArea.addEventListener(eventName, () => imageUploadArea.classList.remove('highlight'), false);
    });

    // 处理文件拖放
    imageUploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // 点击上传区域时触发文件选择
    imageUploadArea.addEventListener('click', () => {
        imageUploadInput.click();
    });

    // 处理文件选择
    imageUploadInput.addEventListener('change', (event) => {
        handleFiles(event.target.files);
    });

    function handleFiles(files) {
        imagePreviewDiv.innerHTML = ''; // 清空现有预览
        selectedFiles = Array.from(files);

        if (selectedFiles.length === 0) {
            setStatusMessage('请选择至少一张图片进行注册。', 'error');
            return;
        }

        selectedFiles.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const imgContainer = document.createElement('div');
                imgContainer.classList.add('image-preview-item');

                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = file.name;

                // const deleteButton = document.createElement('button'); // 旧的删除按钮
                // deleteButton.classList.add('delete-preview-button');
                // deleteButton.textContent = '删除';
                // deleteButton.title = '从待上传列表中删除此图片';
                // deleteButton.onclick = () => {
                //     removeImageFromPreview(index);
                // };

                // 新增：删除图标
                const deleteIcon = document.createElement('div');
                deleteIcon.classList.add('delete-icon');
                deleteIcon.innerHTML = '&times;'; // 使用乘号作为删除图标
                deleteIcon.title = '从待上传列表中删除此图片';
                deleteIcon.onclick = () => {
                    removeImageFromPreview(index);
                };

                imgContainer.appendChild(img);
                imgContainer.appendChild(deleteIcon); // 添加删除图标
                imagePreviewDiv.appendChild(imgContainer);
            };
            reader.readAsDataURL(file);
        });
        setStatusMessage('', ''); // 清除之前的状态消息
    }

    // 新增：从预览中移除图片的函数
    function removeImageFromPreview(indexToRemove) {
        selectedFiles.splice(indexToRemove, 1); // 移除指定索引的文件
        displayImagesForPreview(); // 重新显示图片，更新索引和删除按钮
    }

    // 新增：重新渲染图片预览的函数
    function displayImagesForPreview() {
        imagePreviewDiv.innerHTML = ''; // 清空现有预览
        if (selectedFiles.length === 0) {
            setStatusMessage('请选择至少一张图片进行注册。', 'error');
            return;
        }

        selectedFiles.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const imgContainer = document.createElement('div');
                imgContainer.classList.add('image-preview-item');

                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = file.name;

                // const deleteButton = document.createElement('button'); // 旧的删除按钮
                // deleteButton.classList.add('delete-preview-button');
                // deleteButton.textContent = '删除';
                // deleteButton.title = '从待上传列表中删除此图片';
                // deleteButton.onclick = () => {
                //     removeImageFromPreview(index); // 再次绑定索引
                // };

                // 新增：删除图标
                const deleteIcon = document.createElement('div');
                deleteIcon.classList.add('delete-icon');
                deleteIcon.innerHTML = '&times;'; // 使用乘号作为删除图标
                deleteIcon.title = '从待上传列表中删除此图片';
                deleteIcon.onclick = () => {
                    removeImageFromPreview(index); // 再次绑定索引
                };

                imgContainer.appendChild(img);
                imgContainer.appendChild(deleteIcon); // 添加删除图标
                imagePreviewDiv.appendChild(imgContainer);
            };
            reader.readAsDataURL(file);
        });
        setStatusMessage('', ''); // 清除之前的状态消息
    }


    // 移除搜索已有人员功能
    // searchExistingPersonButton.addEventListener('click', async () => {
    //     const uuid = existingPersonUuidInput.value.trim();
    //     if (!uuid) {
    //         alert('请输入要搜索的人物 UUID。');
    //         return;
    //     }

    //     try {
    //         const response = await fetch(`/persons/${uuid}`, {
    //             headers: {
    //                 'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
    //             }
    //         });

    //         if (response.ok) {
    //             const person = await response.json();
    //             infoUuidSpan.textContent = person.uuid;
    //             infoNameSpan.textContent = person.name || 'N/A'; 
    //             if (person.crop_image_path) {
    //                 infoImageImg.src = `/crops/${person.crop_image_path}`;
    //                 infoImageImg.style.display = 'block';
    //             } else {
    //                 infoImageImg.style.display = 'none';
    //             }
    //             existingPersonInfoDiv.style.display = 'block';
    //             selectedPersonUuid = null; // 重置已选择的人物
    //         } else if (response.status === 404) {
    //             alert('未找到指定 UUID 的人物。');
    //             existingPersonInfoDiv.style.display = 'none';
    //             selectedPersonUuid = null;
    //         } else {
    //             throw new Error(`HTTP error! status: ${response.status}`);
    //         }
    //     } catch (error) {
    //         console.error('搜索人物失败:', error);
    //         alert('搜索人物失败。');
    //         existingPersonInfoDiv.style.display = 'none';
    //         selectedPersonUuid = null;
    //     }
    // });

    // 移除选择已有人员按钮
    // selectExistingPersonButton.addEventListener('click', () => {
    //     selectedPersonUuid = infoUuidSpan.textContent;
    //     existingPersonUuidInput.value = ''; // 清空输入框
    //     alert(`已选择人物 UUID: ${selectedPersonUuid}`);
    //     existingPersonInfoDiv.style.display = 'none'; // 隐藏信息框
    // });

    // 提交注册
    submitEnrollmentButton.addEventListener('click', async () => {
        const personName = personNameInput.value.trim();
        const idCard = idCardInputField.value.trim(); // 获取身份证号
        // 移除 let targetPersonUuid = null; 和相关逻辑

        if (selectedFiles.length === 0) {
            setStatusMessage('请选择至少一张图片进行注册。', 'error');
            return;
        }

        // 更新表单验证逻辑
        if (!personName && !idCard) { 
            setStatusMessage('请输入人物姓名或身份证号。', 'error'); // 移除旧的提示，更明确
            return;
        }

        const formData = new FormData();
        formData.append('person_name', personName); 
        formData.append('id_card', idCard); // 添加身份证号
        // 移除 if (targetPersonUuid) { formData.append('target_person_uuid', targetPersonUuid); }

        selectedFiles.forEach(file => {
            formData.append('images', file);
        });

        setStatusMessage('正在提交注册，请稍候...', '');
        try {
            const response = await fetch('/api/persons/enroll', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('accessToken')}`
                },
                body: formData 
            });

            const result = await response.json();

            if (response.ok) {
                // 更新成功消息，显示 individual_uuid 和 individual_id_card
                setStatusMessage(`注册成功！特征 UUID: ${result.person_uuid || 'N/A'}, 逻辑人物 UUID: ${result.individual_uuid || 'N/A'}, 身份证号: ${result.individual_id_card || 'N/A'}`, 'success');
                // 清空表单
                personNameInput.value = '';
                idCardInputField.value = ''; 
                imageUploadInput.value = '';
                imagePreviewDiv.innerHTML = '';
                // 移除 existingPersonUuidInput.value = ''; 和 existingPersonInfoDiv.style.display = 'none';
                selectedFiles = [];
                // 移除 selectedPersonUuid = null;
            } else {
                throw new Error(result.detail || '注册失败');
            }
        } catch (error) {
            console.error('注册失败:', error);
            setStatusMessage(`注册失败: ${error.message}`, 'error');
        }
    });

    function setStatusMessage(message, type) {
        statusMessageDiv.textContent = message;
        statusMessageDiv.className = `status-message ${type}`;
        statusMessageDiv.style.display = 'block';
    }
}); 