import { Auth } from './auth.js';
import { initializeAuthenticatedPage } from './common.js';

// 移除 DOMContentLoaded 监听器，因为其内容将由 auto_tracking.js 控制显示和隐藏
// document.addEventListener('DOMContentLoaded', async () => {
//     console.log("historical_tracking.js: DOMContentLoaded event fired.");
//     initializeAuthenticatedPage();

const viewHistoryButton = document.getElementById('view-history-button');
if (viewHistoryButton) {
    viewHistoryButton.addEventListener('click', () => {
        alert('查看历史追踪记录功能即将推出！');
        // TODO: Implement fetching and displaying historical tracking data
    });
}
// });