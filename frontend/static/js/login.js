// static/login.js
import { Auth, fetchWithAuth, API_BASE_URL } from './auth.js';

document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('login-form');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const errorMessageDiv = document.getElementById('login-message');

    const registerLink = document.getElementById('register-link');
    const registerModal = document.getElementById('register-modal');
    const closeButton = document.querySelector('.modal .close-button');

    const registerForm = document.getElementById('register-form');
    const regUsernameInput = document.getElementById('reg-username');
    const regPasswordInput = document.getElementById('reg-password');
    const regConfirmPasswordInput = document.getElementById('reg-confirm-password');
    const regUnitInput = document.getElementById('reg-unit');
    const regPhoneNumberInput = document.getElementById('reg-phone-number');
    const registerMessageDiv = document.getElementById('register-message');

    const showLoginBtn = document.getElementById('show-login-btn');
    const showRegisterBtn = document.getElementById('show-register-btn');
    const loginSection = document.getElementById('login-section');
    const registerSection = document.getElementById('register-section');

    // 切换表单显示
    if (showLoginBtn) {
        showLoginBtn.addEventListener('click', () => {
            if (loginSection) loginSection.classList.add('active');
            if (registerSection) registerSection.classList.remove('active');
            if (showLoginBtn) showLoginBtn.classList.add('active');
            if (showRegisterBtn) showRegisterBtn.classList.remove('active');
            if (errorMessageDiv) errorMessageDiv.textContent = ''; // 清空错误信息
            if (registerMessageDiv) registerMessageDiv.textContent = ''; // 清空注册错误信息
        });
    }

    if (showRegisterBtn) {
        showRegisterBtn.addEventListener('click', () => {
            if (loginSection) loginSection.classList.remove('active');
            if (registerSection) registerSection.classList.add('active');
            if (showLoginBtn) showLoginBtn.classList.remove('active');
            if (showRegisterBtn) showRegisterBtn.classList.add('active');
            if (errorMessageDiv) errorMessageDiv.textContent = ''; // 清空错误信息
            if (registerMessageDiv) registerMessageDiv.textContent = ''; // 清空注册错误信息
        });
    }

    // 密码可见性切换
    document.querySelectorAll('.toggle-password').forEach(button => {
        button.addEventListener('click', () => {
            const targetId = button.dataset.target;
            const passwordInput = document.getElementById(targetId);
            if (passwordInput) {
                if (passwordInput.type === 'password') {
                    passwordInput.type = 'text';
                    button.textContent = '🙈'; // 更改图标为隐藏状态
                } else {
                    passwordInput.type = 'password';
                    button.textContent = '👁️'; // 更改图标为可见状态
                }
            }
        });
    });

    // 新增模态框显示/隐藏逻辑
    if (registerLink) {
        registerLink.addEventListener('click', (e) => {
            e.preventDefault();
            if (registerModal) registerModal.style.display = 'flex';
        });
    }

    if (closeButton) {
        closeButton.addEventListener('click', () => {
            if (registerModal) registerModal.style.display = 'none';
        });
    }

    if (registerModal) {
        registerModal.addEventListener('click', (e) => {
            if (e.target === registerModal) {
                registerModal.style.display = 'none';
            }
        });
    }

    if (loginForm) {
        loginForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            const username = usernameInput.value;
            const password = passwordInput.value;
            if (errorMessageDiv) errorMessageDiv.textContent = '';

            const formData = new URLSearchParams();
            formData.append('username', username);
            formData.append('password', password);

            try {
                const response = await fetch(`${API_BASE_URL}/auth/token`, { // 确保使用反引号
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: formData,
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `登录失败: ${response.statusText}`);
                }

                const data = await response.json();
                console.log('登录成功，收到后端数据:', data);
                Auth.setToken(data.access_token);
                console.log('Token已设置。');
                // Parse user info from token and store it
                // 移除冗余的 localStorage.setItem('user')，Auth.getUserInfo() 足以提供信息
                // const userInfo = Auth.parseJwt(data.access_token);
                // if (userInfo) {
                //     localStorage.setItem('user', JSON.stringify({ username: userInfo.sub, is_admin: userInfo.role === 'admin' }));
                // }
                console.log('即将重定向...');
                window.location.href = '/index'; // 重定向到 /index

            } catch (error) {
                if (errorMessageDiv) errorMessageDiv.textContent = error.message;
                console.error('登录错误:', error);
            }
        });
    }

    if (registerForm) {
        registerForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            const username = regUsernameInput.value;
            const password = regPasswordInput.value;
            const confirmPassword = regConfirmPasswordInput.value;
            const unit = regUnitInput.value;
            const phoneNumber = regPhoneNumberInput.value;
            const role = document.getElementById('reg-role') ? document.getElementById('reg-role').value : 'user'; // 从下拉菜单获取角色，如果不存在则默认为user
            if (registerMessageDiv) registerMessageDiv.textContent = '';

            if (password !== confirmPassword) {
                if (registerMessageDiv) {
                    registerMessageDiv.textContent = '两次输入的密码不一致！';
                    registerMessageDiv.style.color = 'red';
                }
                return;
            }

            try {
                const response = await fetchWithAuth('/auth/users/', { // 使用 fetchWithAuth
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username, password, role, unit, phone_number: phoneNumber }),
                });

                const data = await response.json();

                if (!response.ok) {
                    let errorMessage = '注册失败';
                    if (data.detail && Array.isArray(data.detail)) {
                        errorMessage = data.detail.map(err => {
                            if (err.loc && err.loc.length > 1) {
                                return `${err.loc[1]}: ${err.msg}`;
                            }
                            return err.msg;
                        }).join('; ');
                    } else if (data.detail) {
                        errorMessage = data.detail;
                    }
                    throw new Error(errorMessage);
                }

                if (registerMessageDiv) {
                    registerMessageDiv.textContent = `用户 ${data.username} 注册成功！请等待管理员审核激活。\n在管理员激活您的账户之前，您将无法登录。`;
                    registerMessageDiv.style.color = 'green';
                }
                if (registerForm) registerForm.reset();
                setTimeout(() => {
                    if (registerModal) registerModal.style.display = 'none';
                    if (errorMessageDiv) {
                        errorMessageDiv.textContent = '注册成功，请登录。';
                        errorMessageDiv.style.color = 'green';
                    }
                }, 1500);

            } catch (error) {
                if (registerMessageDiv) {
                    registerMessageDiv.textContent = `注册失败: ${error.message}`;
                    registerMessageDiv.style.color = 'red';
                }
                console.error('注册错误:', error);
            }
        });
    }

    // script.js 中的 DOMContentLoaded 逻辑，现在由 common.js 处理认证页面的初始化
    // 登录页面自己的初始化逻辑保持在 login.js 中
    const path = window.location.pathname;
    if (path === '/login' || path === '/') {
        // If on the login page or root, hide the logout button
        const logoutButton = document.getElementById('logout-button');
        if (logoutButton) { // 添加空值检查
            logoutButton.classList.add('hidden');
        }
    }
});