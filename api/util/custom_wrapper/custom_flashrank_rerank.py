from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document

if TYPE_CHECKING:
    pass
else:
    # Avoid pydantic annotation issues when actually instantiating
    # while keeping this import optional
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        pass

DEFAULT_MODEL_NAME = "ms-marco-MultiBERT-L-12"


class CustomFlashrankRerank(FlashrankRerank):
    """
    Dict타입 쿼리에 대응할 수 있도록 커스텀된 FlashrankRerank 클래스
    """

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str | dict,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if isinstance(query, dict):
            query = query["query"]

        return super().compress_documents(documents, query, callbacks)
