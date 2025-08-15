import { Auth, API_BASE_URL } from './auth.js';
import { initializeAuthenticatedPage } from './common.js';

let allImages = []; // Global array to store all fetched images
let currentImageIndex = 0; // Global variable to track current image in lightbox

// Pagination variables
let currentPage = 1;
const itemsPerPage = 10; // Adjust as needed
let totalPages = 1;

document.addEventListener('DOMContentLoaded', async () => {
    initializeAuthenticatedPage();

    const urlParams = new URLSearchParams(window.location.search);
    const individualId = urlParams.get('individual_id');
    const individualName = urlParams.get('individual_name') || '未知人物';
    const individualIdCard = urlParams.get('individual_id_card') || '未知ID';

    const viewerTitle = document.getElementById('viewerTitle');
    const imageGrid = document.getElementById('image-grid');
    const noImagesMessage = document.getElementById('no-images-message');

    // Pagination elements
    const prevPageButton = document.getElementById('prev-page');
    const nextPageButton = document.getElementById('next-page');
    const pageInfoSpan = document.getElementById('page-info');
    const firstPageButton = document.getElementById('first-page'); // New
    const lastPageButton = document.getElementById('last-page'); // New

    // Lightbox elements
    const imageLightbox = document.getElementById('imageLightbox');
    const lightboxImage = document.getElementById('lightboxImage');
    const closeLightboxBtn = document.querySelector('.close-lightbox');
    const lightboxPrev = document.getElementById('lightboxPrev'); // Get prev button
    const lightboxNext = document.getElementById('lightboxNext'); // Get next button

    // Function to show image in lightbox
    const showImageInLightbox = (index) => {
        if (index >= 0 && index < allImages.length) {
            currentImageIndex = index;
            lightboxImage.src = allImages[currentImageIndex].image_path;
            imageLightbox.style.display = "flex";
            // Update prev/next button visibility
            lightboxPrev.style.display = (currentImageIndex === 0) ? "none" : "block";
            lightboxNext.style.display = (currentImageIndex === allImages.length - 1) ? "none" : "block";
        }
    };

    // Close lightbox when click on close button
    closeLightboxBtn.onclick = () => {
        imageLightbox.style.display = "none";
    };

    // Close lightbox when click outside the image
    imageLightbox.onclick = (event) => {
        if (event.target === imageLightbox) {
            imageLightbox.style.display = "none";
        }
    };

    // Lightbox navigation
    lightboxPrev.onclick = (event) => {
        event.stopPropagation(); // Prevent closing lightbox when clicking button
        showImageInLightbox(currentImageIndex - 1);
    };

    lightboxNext.onclick = (event) => {
        event.stopPropagation(); // Prevent closing lightbox when clicking button
        showImageInLightbox(currentImageIndex + 1);
    };

    if (!individualId) {
        viewerTitle.textContent = "错误：未提供人物档案ID。";
        noImagesMessage.classList.remove('hidden');
        return;
    }

    viewerTitle.textContent = `${individualName} (${individualIdCard}) 的注册图片`;

    const token = Auth.getToken();
    if (!token) {
        console.error("Auth token not found.");
        alert("认证失败，请重新登录。");
        window.location.href = '/login';
        return;
    }

    // Function to fetch images from backend and render them
    async function fetchImagesAndRenderGrid() {
        try {
            const skip = (currentPage - 1) * itemsPerPage;
            const limit = itemsPerPage;
            // Modify the URL to include pagination parameters
            const url = `/followed_persons/${individualId}/enrollment_images?skip=${skip}&limit=${limit}`;
            
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
            allImages = data.items; // Update allImages with current page items
            const totalImages = data.total; // Get total count from backend
            totalPages = Math.ceil(totalImages / itemsPerPage); // Calculate total pages

            renderImageGrid(allImages); // Render images for the current page
            updatePaginationControls(); // Update pagination buttons and info

            if (totalImages === 0) {
                noImagesMessage.classList.remove('hidden');
                imageGrid.innerHTML = ''; // Clear any potential content
            } else {
                noImagesMessage.classList.add('hidden');
            }

        } catch (error) {
            console.error("Error fetching enrollment images:", error);
            alert("获取注册图片失败: " + error.message);
            noImagesMessage.textContent = "获取图片失败，请稍后再试或联系管理员。";
            noImagesMessage.classList.remove('hidden');
        }
    }

    // Function to render images in the grid
    function renderImageGrid(imagesToRender) {
        imageGrid.innerHTML = ''; // Clear existing images
        if (imagesToRender.length === 0) {
            noImagesMessage.classList.remove('hidden');
            return;
        }
        noImagesMessage.classList.add('hidden');

        imagesToRender.forEach((image, index) => {
            const imgContainer = document.createElement('div');
            imgContainer.classList.add('image-item');

            const img = document.createElement('img');
            img.src = image.image_path;
            img.alt = `注册图片 ${image.image_path}`;

            // Add click listener to open lightbox
            // Note: Lightbox will now show images from the 'allImages' array, not just the current page's subset.
            // So, pass the original index in 'allImages' (adjusted for pagination)
            img.onclick = () => {
                // Calculate the actual index in the global allImages array
                // For a separate page, we should re-fetch all images for lightbox if we really want to switch all images.
                // For now, it will only switch images on the current page.
                // However, the user request specifically asked to switch *all* images, so this requires fetching all initially.
                // Let's modify fetchImagesAndRenderGrid to *only* fetch current page images, and keep allImages for lightbox.
                // This means the 'limit=1000' in fetch for allImages is correct for lightbox, but then we need to manually paginate `allImages` for rendering.
                // Let's re-evaluate this.

                // Re-eval: The backend already handles pagination. So, allImages will only contain the current page's images.
                // If we want to switch ALL images in lightbox, we need to fetch all images (limit=1000000 etc.) ONCE, then handle pagination purely on frontend.
                // OR, keep backend pagination, and adjust showImageInLightbox to only cycle images on current page.
                // The request says "可以参考人员档案页面的切换查看和分页处理", and the latter typically means backend handles pagination.
                // So, let's assume `allImages` holds only the current page's images. For lightbox, we can ONLY cycle images of the current page.
                // If user wants to cycle ALL images, that's a different UX.

                // Given the current setup, allImages contains *only* the images for the current page.
                // So, the index passed here refers to the index within the *current page's* images.
                // The lightboxPrev/Next logic needs to be aware of this, or we refactor to fetch all images for lightbox separately.
                // For simplicity for now, let's assume lightbox only cycles through images on the current page.
                showImageInLightbox(index); // Index is relative to the current page's images
            };

            const imgPath = document.createElement('p');
            const fullFileName = image.filename; // Use image.filename directly
            const maxLength = 25; // Define max length for displayed filename
            imgPath.textContent = fullFileName.length > maxLength ? fullFileName.substring(0, maxLength) + '...' : fullFileName;
            imgPath.title = fullFileName; // Full name on hover

            imgContainer.appendChild(img);
            imgContainer.appendChild(imgPath);
            imageGrid.appendChild(imgContainer);

            // 新增：删除按钮
            const deleteButton = document.createElement('button');
            deleteButton.classList.add('delete-image-button');
            deleteButton.textContent = '删除';
            deleteButton.title = '删除此注册图片';
            deleteButton.onclick = async (event) => {
                event.stopPropagation(); // 阻止事件冒泡到图片点击事件
                if (confirm('您确定要删除这张注册图片吗？此操作不可撤销！')) {
                    await deleteImage(individualId, image.image_db_uuid); // 将 image.uuid 改为 image.image_db_uuid
                }
            };
            imgContainer.appendChild(deleteButton);
        });
    }

    // 新增：删除图片的函数
    async function deleteImage(individualId, imageUuid) {
        try {
            const url = `${API_BASE_URL}/followed_persons/${individualId}/enrollment_images/${imageUuid}`;
            const response = await fetch(url, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${token}`
                }
            });

            if (!response.ok) {
                if (response.status === 401) {
                    alert("会话过期，请重新登录。");
                    window.location.href = '/login';
                    return false;
                }
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            alert(result.message);
            fetchImagesAndRenderGrid(); // 重新加载图片以更新视图
            return true;

        } catch (error) {
            console.error("Error deleting image:", error);
            alert("删除图片失败: " + error.message);
            return false;
        }
    }

    // Update pagination controls visibility and text
    function updatePaginationControls() {
        pageInfoSpan.textContent = `页数：${currentPage} / ${totalPages}`;
        firstPageButton.disabled = currentPage === 1; // Disable first page if on page 1
        prevPageButton.disabled = currentPage === 1;
        nextPageButton.disabled = currentPage === totalPages;
        lastPageButton.disabled = currentPage === totalPages; // Disable last page if on last page
    }

    // Pagination button event listeners
    firstPageButton.addEventListener('click', () => {
        if (currentPage !== 1) {
            currentPage = 1;
            fetchImagesAndRenderGrid();
        }
    });

    prevPageButton.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            fetchImagesAndRenderGrid();
        }
    });

    nextPageButton.addEventListener('click', () => {
        if (currentPage < totalPages) {
            currentPage++;
            fetchImagesAndRenderGrid();
        }
    });

    lastPageButton.addEventListener('click', () => {
        if (currentPage !== totalPages) {
            currentPage = totalPages;
            fetchImagesAndRenderGrid();
        }
    });

    // Initial fetch of images
    fetchImagesAndRenderGrid();

});