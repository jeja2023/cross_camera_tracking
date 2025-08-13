import { fetchWithAuth, Auth } from './auth.js';
import { API_BASE_URL } from './auth.js';

export async function initProfilePage() {
    console.log("Initializing Profile Page...");

    const profileButton = document.getElementById('profile-button');
    const profileForm = document.getElementById('profile-form');
    const passwordForm = document.getElementById('password-form');
    const usernameInput = document.getElementById('username');
    const unitInput = document.getElementById('unit');
    const phoneNumberInput = document.getElementById('phone-number');
    const currentPasswordInput = document.getElementById('current-password');
    const newPasswordInput = document.getElementById('new-password');
    const confirmNewPasswordInput = document.getElementById('confirm-new-password');

    // 获取用户个人信息
    async function fetchUserProfile() {
        try {
            const response = await fetchWithAuth(`${API_BASE_URL}/auth/users/me`);
            if (response.ok) {
                const user = await response.json();
                usernameInput.value = user.username;
                unitInput.value = user.unit || '';
                phoneNumberInput.value = user.phone_number || '';
                usernameInput.readOnly = true;
            } else {
                const errorData = await response.json();
                alert('获取用户资料失败: ' + (errorData.detail || response.statusText));
            }
        } catch (error) {
            console.error('获取用户资料时发生网络错误:', error);
            alert('获取用户资料时发生网络错误。');
        }
    }

    // 页面加载时立即获取用户个人信息
    await fetchUserProfile();

    // 更新个人信息
    if (profileForm) {
        profileForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            
            const passwordVerification = prompt('更新个人信息需要验证您的当前密码。请输入：');
            if (!passwordVerification) {
                alert('密码验证取消。');
                return;
            }

            const updatedProfile = {
                unit: unitInput.value,
                phone_number: phoneNumberInput.value,
                current_password: passwordVerification
            };

            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/auth/users/me`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(updatedProfile)
                });

                if (response.ok) {
                    alert('个人信息更新成功！');
                    // 在页面形式下，无需关闭模态框
                } else if (response.status === 401 || response.status === 403) {
                    alert('密码验证失败或无权更新。');
                } else {
                    const errorData = await response.json();
                    alert('更新个人信息失败: ' + (errorData.detail || response.statusText));
                }
            } catch (error) {
                console.error('更新个人信息时发生网络错误:', error);
                alert('更新个人信息时发生网络错误。');
            }
        });
    }

    // 修改密码
    if (passwordForm) {
        passwordForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            const currentPassword = currentPasswordInput.value;
            const newPassword = newPasswordInput.value;
            const confirmNewPassword = confirmNewPasswordInput.value;

            if (newPassword !== confirmNewPassword) {
                alert('新密码和确认密码不匹配！');
                return;
            }

            if (newPassword.length < 6) {
                alert('新密码长度不能少于6位。');
                return;
            }

            const passwordData = {
                current_password: currentPassword,
                new_password: newPassword
            };

            try {
                const response = await fetchWithAuth(`${API_BASE_URL}/auth/users/change-password`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(passwordData)
                });

                if (response.ok) {
                    alert('密码修改成功！请使用新密码重新登录。');
                    Auth.removeToken();
                    window.location.href = '/'; // 重定向到根路径
                } else if (response.status === 401 || response.status === 403) {
                    alert('当前密码不正确或无权修改密码。');
                } else {
                    const errorData = await response.json();
                    alert('修改密码失败: ' + (errorData.detail || response.statusText));
                }
            } catch (error) {
                console.error('修改密码时发生网络错误:', error);
                alert('修改密码时发生网络错误。');
            } finally {
                currentPasswordInput.value = '';
                newPasswordInput.value = '';
                confirmNewPasswordInput.value = '';
            }
        });
    }
} 