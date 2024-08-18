import os

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(find_dotenv())


class AppConfig(BaseSettings):
    REDIS_HOST: str = ""
    REDIS_PORT: str = ""
    REDIS_DB: str = ""

    POSTGRES_DB: str = "" if os.getenv("env") != "test" else "test-db"
    POSTGRES_USER: str = "" if os.getenv("env") != "test" else "test-db-user"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_HOST: str = ""
    POSTGRES_PORT: str = ""

    RETRIEVER_TOP_K: int = 10
    RETRIEVER_TOP_N: int = 8

    EMBEDDING_MODEL: str = "text-embedding-3-large"
    CHAT_MODEL: str = "gpt-4o"

    DOCUMENT_DIR: str = "documents"
    OPENAI_API_KEY: str = ""
