const API_BASE_URL = window.location.origin;

/**
 * 存储和获取 JWT token
 */
const Auth = {
    getToken: () => localStorage.getItem('accessToken'),
    setToken: (token) => {
        localStorage.setItem('accessToken', token);
    },
    removeToken: () => {
        localStorage.removeItem('accessToken');
        localStorage.removeItem('accessTokenExpiresAt');
    },
    
    // 解析JWT以获取用户信息和角色
    parseJwt: (token) => {
        try {
            console.log("Auth.parseJwt: 正在解析令牌...");
            const base64Url = token.split('.')[1];
            console.log("Auth.parseJwt: base64Url =", base64Url);
            // Base64Url 转换为标准 Base64
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            console.log("Auth.parseJwt: base64 =", base64);
            const jsonPayload = decodeURIComponent(atob(base64).split('').map(function(c) {
                return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
            }).join(''));
            console.log("Auth.parseJwt: jsonPayload =", jsonPayload);
            const parsedPayload = JSON.parse(jsonPayload);
            console.log("Auth.parseJwt: 解析成功，负载为:", parsedPayload);
            return parsedPayload;
        } catch (e) {
            console.error("解析JWT失败:", e); // 确保错误被记录
            return null;
        }
    },

    getUserInfo: () => {
        const token = Auth.getToken();
        if (!token) return null;
        const payload = Auth.parseJwt(token);
        console.log("Auth.getUserInfo: payload =", payload); // 添加此行来检查 payload 内容
        if (payload && payload.sub && payload.role && payload.id) {
            return { username: payload.sub, role: payload.role, id: payload.id };
        }
        return null;
    },

    currentUserIsAdmin: () => {
        const userInfo = Auth.getUserInfo();
        return userInfo && userInfo.role === 'admin';
    }
};

/**
 * 封装带认证的fetch函数
 */
async function fetchWithAuth(url, options = {}) {
    const token = Auth.getToken();
    const headers = { ...options.headers };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    const requestOptions = { ...options, headers };
    console.log(`[fetchWithAuth] Fetching: ${url}`); // Log the URL being fetched
    const response = await fetch(url, requestOptions);

    // Log response details for debugging, especially for non-OK responses
    console.log(`[fetchWithAuth] Response Status for ${url}: ${response.status}`);
    console.log(`[fetchWithAuth] Response Headers for ${url}:`, response.headers);

    if (response.status === 401 && !url.includes('/auth/token')) {
        Auth.removeToken();
        alert('认证已过期或无效，请重新登录。');
        window.location.href = '/login';
        throw new Error('Unauthorized');
    }

    if (!response.ok) {
        // Attempt to clone response for logging, as body can only be read once
        const errorResponse = response.clone();
        try {
            const errorBody = await errorResponse.text();
            console.error(`[fetchWithAuth] Error Response Body for ${url}:`, errorBody);
        } catch (bodyError) {
            console.error(`[fetchWithAuth] Could not read error response body for ${url}:`, bodyError);
        }
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response;
}

/**
 * 登出函数
 */
function handleLogout() {
    Auth.removeToken();
    alert('您已成功登出。');
    window.location.href = '/login';
}

export { Auth, fetchWithAuth, handleLogout, API_BASE_URL };