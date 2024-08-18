from typing import AsyncGenerator

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from sqlalchemy.sql import text

from main import app
from util.dependency import DBEngineFactory, get_app_config


@pytest.fixture
def fastapi_app():
    return app


@pytest.fixture
async def client(
    fastapi_app: FastAPI,
) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(app=fastapi_app, base_url="http://127.0.0.1:8000") as ac:
        yield ac


@pytest.fixture
def test_pdf_file():
    file = open("documents/테스트.pdf", "rb").read()
    return file


@pytest.fixture
async def db_session():
    db_engine_factory = DBEngineFactory(config=get_app_config())
    db_engine = (await db_engine_factory()).sync_engine
    session = db_engine.connect()
    yield session
    session.close()


@pytest.mark.anyio
async def test_add_document_on_library(
    fastapi_app: FastAPI,
    client: AsyncClient,
    test_pdf_file: bytes,
    db_session,
):
    url = fastapi_app.url_path_for("add_document_on_library")

    before_count = db_session.execute(
        text("SELECT COUNT(*) FROM langchain_pg_embedding")
    ).fetchone()

    # multipart/form-data로 파일을 업로드
    response = await client.post(
        url, files={"document_file": ("테스트.pdf", test_pdf_file, "application/pdf")}
    )
    assert response.status_code == 200

    after_count = db_session.execute(
        text("SELECT COUNT(*) FROM langchain_pg_embedding")
    ).fetchone()
    assert before_count[0] + 1 == after_count[0]
