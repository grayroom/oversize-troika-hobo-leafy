from io import BytesIO

from fastapi import APIRouter, BackgroundTasks
from fastapi import File, UploadFile, Depends

from admin.service import AdminService

router = APIRouter()


@router.post("/document")
async def add_document_on_library(
    background_tasks: BackgroundTasks,
    document_file: UploadFile = File(...),
    admin_service: AdminService = Depends(),
):
    """문서를 벡터화하여 VectorDB에 추가합니다.

    1. 업로드한 문서를 BytesIO로 변환하여, 응답을 생성한 이후에도 접근할 수 있도록 함
    2. 백그라운드 작업으로 VectorDB에 문서 추가

    Args:
        background_tasks: BackgroundTasks(업로드에 오랜 시간이 걸리므로 백그라운드 작업으로 처리)
        document_file: 업로드할 문서 파일

        admin_service: [의존성 주입] AdminService 인스턴스
    """

    # document_add_request.file을 background_task에서도 읽을 수 있도록, UploadFile을 BytesIO로 변환
    file_content = await document_file.read()
    filename = document_file.filename
    file_like_object = BytesIO(file_content)

    # background_task로 vectorize_document 함수 실행하여 VectorDB에 문서 추가
    background_tasks.add_task(
        admin_service.vectorize_document,
        document_file=file_like_object,
        filename=filename,
    )

    return {"message": f"{filename}에 대한 백그라운드 작업이 시작되었습니다."}
