from pydantic import BaseModel


class ChainQuerySchema(BaseModel):
    """Langchain 질문 스키마

    Langchain에 질문을 보낼 때 사용하는 스키마입니다.

    Attributes:
        query: 질문
        document: RAG를 통해 생성된 문서
    """

    query: list
    document: list


class UserQuerySchema(BaseModel):
    """사용자 질문 스키마

    API 엔드포인트에서 사용자 질문을 받을 때 사용하는 스키마입니다.
    - session_id가 None인 경우 새로운 세션을 생성하여 응답에 포함합니다.
    - session_id가 None이 아닌 경우 해당 세션을 사용하여 응답에 포함합니다.

    Attributes:
        query: 사용자 질문
        session_id: 세션 ID
    """

    query: str
    session_id: str | None
