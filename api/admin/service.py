from io import BytesIO

from fastapi import Depends

from util.dependency import VectorStore, VectorStorePair
from util.splitter import get_documents_from_file


class AdminService:
    def __init__(
        self,
        vector_store: VectorStorePair = Depends(VectorStore()),
    ):
        """AdminService 생성자

        싱글톤 객체인 vector_store를 주입받는다.

        Args:
            vector_store: 싱글톤 PGVector 인스턴스
        """
        self.vector_store = vector_store.sync_vector_store

    async def vectorize_document(
        self, *, document_file: BytesIO, filename: str
    ) -> dict:
        """문서를 벡터화하여 vector_store에 추가

        1. 문서를 Chunk로 분리
            - 만약 로컬 캐시에 존재한다면 로컬 캐시에서 문서를 가져온다.
            - 그렇지 않다면 직접 Splitter를 사용하여 문서를 Chunk로 분리한다.
        2. vector_store에 문서 추가

        Args:
            document_file: BytesIO 객체 (BackgroundTask에서 처리를 위함)
            filename: 파일 이름
        """

        # pickle 로컬 캐시 혹은 직접 Splitter를 사용하여 문서를 Chunk로 분리
        documents = get_documents_from_file(
            document_file=document_file, filename=filename
        )

        # vector_store에 문서 추가
        self.vector_store.add_documents(documents=documents)

        return {"filename": filename, "status": "vectorized"}
