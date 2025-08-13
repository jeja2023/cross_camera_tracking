document.addEventListener('DOMContentLoaded', () => {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    function showTab(targetId) {
        tabContents.forEach(content => {
            if (content.id === targetId) {
                content.classList.remove('hidden');
            } else {
                content.classList.add('hidden');
            }
        });

        tabButtons.forEach(button => {
            if (button.dataset.target === targetId) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });
    }

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            showTab(button.dataset.target);
        });
    });

    // 默认显示第一个标签页
    if (tabButtons.length > 0) {
        showTab(tabButtons[0].dataset.target);
    }
});