import os
from functools import lru_cache
from io import BytesIO
from typing import Any

import yaml
from dotenv import load_dotenv, find_dotenv
from fastapi import Depends
from langchain.embeddings import CacheBackedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import MultiQueryRetriever
from langchain.storage import InMemoryByteStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine

from util.config import AppConfig
from util.custom_wrapper.custom_bm25_retriever import CustomBM25Retriever
from util.custom_wrapper.custom_flashrank_rerank import CustomFlashrankRerank
from util.splitter import get_documents_from_file
from util.utils import singleton


@lru_cache
def get_app_config():
    """FastAPI 설정값을 불러오는 함수

    Pydantic Settings를 사용하여 .env 파일을 읽어온다
    """
    load_dotenv(find_dotenv())

    app_config = AppConfig()
    if os.getenv("env") == "test":
        app_config.POSTGRES_DB = "test_db"
        app_config.POSTGRES_USER = "test_db_user"
    return app_config


@singleton
class MemoryCache:
    """전역 싱글톤 객체로 InMemoryByteStore를 생성한다

    - Depends를 통해 __call__ 메서드를 호출할 때마다 InMemoryByteStore 객체를 반환한다
    - InMemoryByteStore는 ByteStore를 상속받아 메모리에 데이터를 저장하는 클래스
    """

    def __init__(self):
        self.store = InMemoryByteStore()

    async def __call__(self) -> InMemoryByteStore:
        return self.store


class DBEnginePair(BaseModel):
    sync_engine: Any
    async_engine: Any


@singleton
class DBEngineFactory:
    """데이터베이스 연결을 생성하는 클래스

    - AppConfig을 사용하여 데이터베이스 연결 정보를 불러온다
    - create_engine을 사용하여 동기 데이터베이스 연결을 생성한다

    - 동기, 비동기 객체를 모두 반환하는 이유는
        - 비동기 객체는 비동기 VectorStore 객체를 생성할 때 사용
        - 동기 객체는 동기 VectorStore 객체를 생성할 때 사용
    """

    def __init__(self, config: AppConfig = get_app_config()):
        self.sync_engine = create_engine(
            f"postgresql+psycopg2://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:"
            f"{config.POSTGRES_PORT}/{config.POSTGRES_DB}",
        )
        self.async_engine = create_async_engine(
            f"postgresql+asyncpg://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@{config.POSTGRES_HOST}:"
            f"{config.POSTGRES_PORT}/{config.POSTGRES_DB}",
        )

    async def __call__(self) -> DBEnginePair:
        return DBEnginePair(
            sync_engine=self.sync_engine, async_engine=self.async_engine
        )


@singleton
class LLM:
    """ChatOpenAI 객체를 생성하는 클래스

    - AppConfig을 사용하여 ChatOpenAI 모델 정보를 불러온다
    - ChatOpenAI를 사용하여 ChatOpenAI 객체를 생성한다
    """

    def __init__(self, config: AppConfig = get_app_config()):
        self.config = config
        self.llm = ChatOpenAI(
            temperature=0,
            model_name=self.config.CHAT_MODEL,
            model_kwargs={"stream": False},
        )

    async def __call__(self) -> ChatOpenAI:
        return self.llm


class VectorStorePair(BaseModel):
    sync_vector_store: Any
    async_vector_store: Any


@singleton
class VectorStore:
    """PGVector 객체를 생성하는 클래스

    - CacheBackedEmbeddings를 사용하여 Embeddings 객체를 생성한다
    - OpenAIEmbeddings를 사용하여 Embeddings 객체를 생성한다
    - PGVector를 사용하여 VectorStore 객체를 생성한다

    - 동기, 비동기 객체를 모두 반환하는 이유는
        - 비동기 객체는 벡터화된 문서를 추가할 때 사용
        - 동기 객체는 벡터화된 문서를 검색할 때 사용 (비동기 미지원)
    """

    def __init__(
        self,
        config: AppConfig = get_app_config(),
    ):
        self.config = config
        self.embedding_function = OpenAIEmbeddings(model=self.config.EMBEDDING_MODEL)

    async def __call__(
        self,
        db_engine: DBEnginePair = Depends(DBEngineFactory()),
        store: InMemoryByteStore = Depends(MemoryCache()),
    ) -> VectorStorePair:
        # MemoryCache를 사용하는 Embeddings 객체를 생성
        embedding = CacheBackedEmbeddings.from_bytes_store(
            self.embedding_function, store, namespace=self.embedding_function.model
        )

        sync_vector_store = PGVector(
            connection=db_engine.sync_engine,
            create_extension=True,
            embeddings=embedding,
            async_mode=False,
        )
        async_vector_store = PGVector(
            connection=db_engine.async_engine,
            create_extension=True,
            embeddings=embedding,
            async_mode=True,
        )
        return VectorStorePair(
            sync_vector_store=sync_vector_store, async_vector_store=async_vector_store
        )


@singleton
class BM25RetrieverFactory:
    """BM25Retriever 객체를 생성하는 클래스

    1. AppConfig을 사용하여 BM25Retriever의 top_k 값을 불러온다
    2. DOCUMENT_DIR 경로에 있는 모든 PDF 파일을 읽어온다
        - get_documents_from_file 함수를 사용하여 문서를 Document로 변환
    3. CustomBM25Retriever를 사용하여 BM25Retriever 객체를 생성한다

    - Depends를 통해 __call__ 메서드를 호출할 때마다 BM25Retriever 객체를 반환한다
    """

    def __init__(self, config: AppConfig = get_app_config()):
        self.config = config

        # DOCUMENT_DIR 경로 아래 있는 모든 PDF파일을 읽어온다
        document_list: list[Document] = []
        for document_file in os.listdir(self.config.DOCUMENT_DIR):
            # pdf 파일만 읽어온다 / 디렉토리도 무시
            if not document_file.endswith(".pdf") or os.path.isdir(document_file):
                continue

            with open(f"{self.config.DOCUMENT_DIR}/{document_file}", "rb") as f:
                file_like_object = BytesIO(f.read())
                document_list.extend(
                    # pickle 혹은 Splitter를 사용하여 문서를 Document로 변환
                    get_documents_from_file(
                        document_file=file_like_object, filename=document_file
                    )
                )

        if document_list:
            self.retriever = CustomBM25Retriever.from_documents(
                documents=document_list, k=self.config.RETRIEVER_TOP_K
            )
        else:
            self.retriever = None

    async def __call__(self) -> CustomBM25Retriever:
        return self.retriever


@singleton
class FileNameRetriever:
    """가장 유사한 파일명을 반환하는 Retriever 클래스

    1. DOCUMENT_DIR 경로에 있는 모든 PDF 파일을 읽어온다
        - Document 객체로 변환하여 리스트에 저장
    2. CustomBM25Retriever를 사용하여 BM25Retriever 객체를 생성한다

    - __call__ 메서드를 통해 가장 유사한 파일명을 반환한다
    """

    def __init__(self, config: AppConfig = get_app_config()):
        self.config = config

        filename_list = list()
        for document_file in os.listdir(self.config.DOCUMENT_DIR):
            # pdf 파일만 읽어온다 / 디렉토리도 무시
            if not document_file.endswith(".pdf") or os.path.isdir(document_file):
                continue

            filename_list.append(Document(page_content=document_file))

        self.retriever = CustomBM25Retriever.from_documents(
            documents=filename_list, k=2
        )

    def __call__(self, filename: str) -> list[str]:
        return [document.page_content for document in self.retriever.invoke(filename)]


@singleton
class RetrieverFactory:
    """PGVector, LLM, BM25Retriever를 사용하여 ContextualCompressionRetriever 객체를 생성하는 클래스

    1. PGVector와 LLM을 사용하여 MultiQueryRetriever 생성 -> [K개 Retrieve]
    2. 로컬파일로부터 Document를 읽어와 BM25Retriever 생성 -> [K개 Retrieve]
    3. PGVector와 BM25Retriever를 사용하여 EnsembleRetriever 생성
    4. CustomFlashrankRerank를 사용하여 ContextualCompressionRetriever 생성 -> [2*K -> N개 Rerank]
    5. ContextualCompressionRetriever 객체를 반환한다

    - Depends를 통해 __call__ 메서드를 호출할 때마다 ContextualCompressionRetriever 객체를 반환한다
    """

    def __init__(self, config: AppConfig = get_app_config()):
        self.config = config

    async def __call__(
        self,
        pg_vector: VectorStorePair = Depends(VectorStore()),
        llm: ChatOpenAI = Depends(LLM()),
        bm25_retriever: CustomBM25Retriever = Depends(BM25RetrieverFactory()),
    ) -> ContextualCompressionRetriever:
        self.llm = llm
        # 0. 프롬프트 로드
        with open("./prompt/prompt.yml", "r", encoding="utf-8") as f:
            self.prompt = yaml.safe_load(f)

        self.retrievers = []

        # 1-A. PGVector와 LLM을 사용하여 MultiQueryRetriever 생성
        if multi_query_retriever := MultiQueryRetriever.from_llm(
            prompt=PromptTemplate.from_template(
                self.prompt["multi_query"].format(
                    top_k=self.config.RETRIEVER_TOP_K, question="{question}"
                ),
            ),
            llm=self.llm,
            retriever=pg_vector.sync_vector_store.as_retriever(),
        ):
            self.retrievers.append(multi_query_retriever)

        # 1-B. PGVector와 BM25Retriever를 사용하여 EnsembleRetriever 생성
        if bm25_retriever:
            self.retrievers.append(bm25_retriever)

        # 2. PGVector와 BM25Retriever를 사용하여 EnsembleRetriever 생성
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=self.retrievers,
            # weights는 retrievers의 갯수에 맞게 1/n로 설정
            weights=[1 / len(self.retrievers) for _ in self.retrievers],
        )

        # 3. CustomFlashrankRerank를 사용하여 ContextualCompressionRetriever 생성: 2 * K -> N개를 추린다
        self.compressor = CustomFlashrankRerank(top_n=self.config.RETRIEVER_TOP_N)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, base_retriever=self.ensemble_retriever
        )
        return self.retriever
