from fastapi import APIRouter, Body, Depends
from fastapi import HTTPException
from fastapi import status
from starlette.responses import StreamingResponse, FileResponse

from agent.schema import UserQuerySchema
from agent.service import AgentService

router = APIRouter()


@router.post("/ask")
async def ask(
    question: UserQuerySchema = Body(..., description="질문"),
    agent_service: AgentService = Depends(),
):
    """사용자 질문에 대한 답변을 반환합니다.

    Args:
        question: 사용자 질문 + 세션 ID

        agent_service: [의존성 주입] ㄴAgentService 인스턴스

    Returns:
        dict: 답변
    """
    return StreamingResponse(
        agent_service.generate_answer(question=question),
        media_type="text/event-stream",
    )


@router.get("/document/{filename}/{page_number}")
async def get_pdf(
    filename: str,
    page_number: int,
):
    """문서의 특정 페이지 이미지를 반환합니다.

    Args:
        filename: 문서 파일 이름
        page_number: 페이지 번호
        config: AppConfig 인스턴스

    Returns:
        FileResponse: 이미지 파일
    """
    try:
        return FileResponse(
            f"./documents/page_chunk/{filename}_{page_number}.png",
            media_type="image/png",
        )
    except FileNotFoundError:
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND)
