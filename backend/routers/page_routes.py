from fastapi import APIRouter
from fastapi.responses import FileResponse
from backend.config import settings # 导入 settings

router = APIRouter(
    tags=["Pages"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", summary="Root - Redirect to Login", include_in_schema=False)
async def root():
    return FileResponse(settings.LOGIN_PAGE_PATH)

@router.get("/video_analysis", summary="Video Analysis Page", include_in_schema=False)
async def video_analysis_page():
    return FileResponse(settings.VIDEO_ANALYSIS_PAGE_PATH)

@router.get("/image_analysis", summary="Image Analysis Page", include_in_schema=False)
async def image_analysis_page():
    return FileResponse(settings.IMAGE_ANALYSIS_PAGE_PATH)

@router.get("/image_analysis_results", summary="Image Analysis Results Page", include_in_schema=False)
async def image_analysis_results_page():
    return FileResponse(settings.IMAGE_ANALYSIS_RESULTS_PAGE_PATH)

@router.get("/image_search", summary="Image Search Page", include_in_schema=False)
async def image_search_page():
    return FileResponse(settings.IMAGE_SEARCH_PAGE_PATH)

@router.get("/video_results/{video_id}", summary="Video Results Page", include_in_schema=False)
async def video_results_page(video_id: int):
    return FileResponse(settings.VIDEO_RESULTS_PAGE_PATH)

@router.get("/all_features", summary="All Features Page (Admin Only)", include_in_schema=False)
async def all_features_page():
    return FileResponse(settings.ALL_FEATURES_PAGE_PATH)

@router.get("/person_list", summary="Person List Page", include_in_schema=False)
async def person_list_page():
    return FileResponse(settings.PERSON_LIST_PAGE_PATH)

@router.get("/video_stream", summary="Video Stream Page", include_in_schema=False)
async def video_stream_page():
    return FileResponse(settings.VIDEO_STREAM_PAGE_PATH)

@router.get("/live_stream_results", summary="Live Stream Results Page", include_in_schema=False)
async def live_stream_results_page():
    return FileResponse(settings.LIVE_STREAM_RESULTS_PAGE_PATH)





