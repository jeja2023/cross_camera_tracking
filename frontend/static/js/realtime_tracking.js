import { Auth } from './auth.js';
import { initializeAuthenticatedPage } from './common.js';

// 移除 DOMContentLoaded 监听器，因为其内容将由 auto_tracking.js 控制显示和隐藏
// document.addEventListener('DOMContentLoaded', async () => {
//     console.log("realtime_tracking.js: DOMContentLoaded event fired.");
//     initializeAuthenticatedPage();

const startComparisonButton = document.getElementById('start-realtime-comparison-button');
if (startComparisonButton) {
    startComparisonButton.addEventListener('click', () => {
        alert('开始实时比对功能即将推出！');
        // TODO: Implement real-time comparison logic
    });
}
// });