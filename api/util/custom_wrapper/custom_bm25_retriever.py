from __future__ import annotations

from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document


class CustomBM25Retriever(BM25Retriever):
    def _get_relevant_documents(
        self, query: str | dict, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """
        Dict타입 쿼리에 대응할 수 있도록 커스텀된 BM25Retriever 클래스
        """
        if isinstance(query, dict):
            query = query["query"]

        return super()._get_relevant_documents(query, run_manager=run_manager)
