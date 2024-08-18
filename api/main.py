from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from admin.router import router as admin_router
from agent.router import router as agent_router

app = FastAPI(
    root_path="/api",
    title="Mydata Document RAG Agent",
    description="금융분야 마이데이터 가이드 문서를 바탕으로 정책과 기술사양을 답변하는 RAG-LLM Agent",
    version="0.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(admin_router, prefix="/admin", tags=["백오피스"])
app.include_router(agent_router, prefix="/agent", tags=["LLM Agent"])
