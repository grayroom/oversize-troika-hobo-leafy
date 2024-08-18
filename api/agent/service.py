import json
from typing import Annotated
from typing import Literal
from uuid import uuid4

import tiktoken
import yaml
from fastapi import Depends
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.messages import ToolMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.schema import UserQuerySchema, ChainQuerySchema
from agent.tools import (
    multiply,
    CalculatorInput,
    add,
    RAGInput,
    rag,
    GetPDFInput,
    get_pdf,
)
from util.chain_builder import ChainBuilder
from util.config import AppConfig
from util.custom_wrapper.asnyc_redis_saver import AsyncRedisSaver
from util.dependency import (
    LLM,
    RetrieverFactory,
    get_app_config,
)
from util.processor import ProcessDocuments


class State(TypedDict):
    """LangGraph에서 사용하는 State 스키마

    Attributes:
        messages: AIMessage / ToolMessage 리스트
    """

    messages: Annotated[list, add_messages]


class BasicToolNode:
    """Agent Tool을 실행하는 노드"""

    def __init__(self, tools: list) -> None:
        """BasicToolNode 생성자

        인자로 전달받은 tools를 이름을 key로 하는 dict로 변환하여, 이후 호출할 수 있도록 준비합니다.

        Args:
            tools: Agent Tool 리스트
        """
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State):
        """Agent가 Tool 사용을 요청한 경우, 해당 Tool을 실행

        1. 가장 마지막 메시지를 가져와서, 해당 메시지에 포함된 Tool을 실행합니다.
        2. Tool 실행 결과를 ToolMessage로 변환하여 반환합니다.

        Args:
            state: 현재 상태 정보 (메시지 리스트로 구성됨)
        """

        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class AgentService:
    def __init__(
        self,
        config: AppConfig = Depends(get_app_config),
        llm: ChatOpenAI = Depends(LLM()),
        retriever: ContextualCompressionRetriever = Depends(RetrieverFactory()),
    ):
        """AgentService 생성자

        1. 싱글톤 LLM, Retriever를 주입 받습니다.
        2. Agent가 사용할 Tool Set을 구성하고, LLM과 Tool을 바인딩합니다.

        Args:
            config: AppConfig 인스턴스
            llm: ChatOpenAI 인스턴스, asyncio.Queue 인스턴스 (입력), asyncio.Queue 인스턴스 (출력)
            retriever: EnsembleRetriever[PGVector, BM25] -> CustomFlashrankRerank 로 구성된 Retriever
        """
        self.config = config
        self.llm = llm
        self.retriever = retriever

        # 프롬프트 로드
        with open("./prompt/prompt.yml", "r", encoding="utf-8") as f:
            self.prompt = yaml.safe_load(f)

        # Tool 세트 생성 및 LLM 바인딩
        self.tool_set = [
            StructuredTool.from_function(
                func=rag(self.retriever),
                name="mydata-retriever",
                description="금융분야 마이데이터 표준 API 규격과 금융분야 마이데이터 기술 가이드라인을 검색합니다. 'API 스펙 중 aNS는 어떤 것을 "
                "뜻하나요?'같은 질문에 이 도구를 사용하세요.",
                args_schema=RAGInput,
            ),
            StructuredTool.from_function(
                func=multiply,
                name="multiply",
                description="두 정수를 곱한 결과를 반환합니다. '3와 4를 곱해줘'같은 질문에 이 도구를 사용하세요.",
                args_schema=CalculatorInput,
            ),
            StructuredTool.from_function(
                func=add,
                name="add",
                description="두 정수를 더한 결과를 반환합니다. '3와 4를 더해줘'같은 질문에 이 도구를 사용하세요.",
                args_schema=CalculatorInput,
            ),
            StructuredTool.from_function(
                func=get_pdf,
                name="get-pdf",
                description="파일 이름과 페이지 번호를 받아 해당 페이지의 이미지를 반환합니다. '가이드라인.pdf의 5페이지를 보여줘'같은 질문에 이 도구를 사용하세요.",
                args_schema=GetPDFInput,
            ),
        ]
        self.llm_with_tool = self.llm.bind_tools(
            tools=self.tool_set,
        )

    async def agent(self, state: State):
        """Tool 사용, 기본응답을 처리하는 에이전트 노드

        Args:
            state: 현재 상태 정보 (메시지 리스트로 구성됨)
        """

        async def _get_token_length(message) -> int:
            encoder = tiktoken.encoding_for_model(self.config.CHAT_MODEL)
            return len(encoder.encode(message))

        async def _filter_message_by_token_length(messages):
            # 메시지를 최근것부터 순회하면서, 토큰 길이 총 합이 1000을 넘지 않도록 합니다.
            filtered_messages = []
            token_length = 0
            for message in reversed(messages):
                token_length += await _get_token_length(message.content)
                if token_length > 3072:
                    break
                filtered_messages.append(message)

            return list(reversed(filtered_messages))

        try:
            # 최근 메시지를 가져와서, LLM에 전달합니다.
            target_messages = await _filter_message_by_token_length(state["messages"])
            return {"messages": [await self.llm_with_tool.ainvoke(target_messages)]}
        except Exception as e:
            return {"messages": [AIMessage(content=str(e))]}

    @staticmethod
    async def route_tools(
        state: State,
    ) -> Literal["tools", "__end__"]:
        """Agent가 Tool 사용을 요청한 경우, Tool 노드로 이동시키는 Router

        1. 가장 마지막 메시지를 가져와서, 해당 메시지에 tool_calls가 있는지 확인합니다.
        2. tool_calls가 있으면 "tools"로, 없으면 "__end__"로 라우팅합니다.

        Args:
            state: 현재 상태 정보 (메시지 리스트로 구성됨)
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return "__end__"

    @staticmethod
    async def route_after_tool(
        state: State,
    ) -> Literal["chatbot", "__complete__", "__end__"]:
        """Tool사용을 완료한 후, 다음 노드로 이동시키는 Router

        RAG:
            Query와 Document로 응답을 생성하도록 -> __complete__
        PDF파일 조회:
            파일 조회 URL 응답하도록 -> __end__
        그 외:
            Chatbot으로 이동하여 Agent 응답 생성 -> chatbot

        Args:
            state: 현재 상태 정보 (메시지 리스트로 구성됨)
        """
        if isinstance(state, list):
            tool_message = state[-1]
        elif messages := state.get("messages", []):
            tool_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")

        match tool_message.name:
            case "mydata-retriever":
                return "__complete__"
            case "get-pdf":
                if json.loads(tool_message.content)["image_url"]:
                    return "__end__"
                else:
                    # PDF 파일이 없는 경우, Chatbot으로 이동하여 실패에 대한 응답 생성
                    return "chatbot"
            case _:
                return "chatbot"

    async def complete_with_document(self, state: State):
        """tool을 통해 받아온 document를 활용하여 최종 답변을 생성하는 Node

        1. 대화 기록에서, 그동안 Retrieve한 [Document / Query]를 추출
        2. UserQuerySchema에 맞게 데이터를 변환
        3. [질문에 대한 답변 / 근거 Document]로 구성된 답변을 생성하는 프롬프트 로드

        - ProcessDocuments는 state에서 정제된 Document와 Query를 추출하여, ChainQuerySchema로 변환합니다.

        Args:
            state: 현재 상태 정보 (메시지 리스트로 구성됨)
        """
        chain_builder = ChainBuilder()
        chain_builder.build_chain(
            preprocessor=ProcessDocuments(),
            input_schema=ChainQuerySchema,
            prompt=self.prompt["system"],
            llm=self.llm,
            output_parser="agent",
        )

        chain = chain_builder.make_runnable()

        result = await chain.ainvoke(state)

        return {"messages": result.messages}

    async def get_agent_graph(self, checkpointer=None):
        """Agent LangGraph를 생성합니다.

        정확한 다이어그램은 README.md에 첨부되어 있습니다.

        Args:
            checkpointer: AsyncRedisSaver 인스턴스 (History 저장을 위함)
        """
        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", self.agent)
        graph_builder.add_node("tools", BasicToolNode(tools=self.tool_set))
        graph_builder.add_node("answer", self.complete_with_document)
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            self.route_tools,
            {
                "tools": "tools",
                "__end__": END,
            },
        )
        graph_builder.add_conditional_edges(
            "tools",
            self.route_after_tool,
            {
                "chatbot": "chatbot",
                "__complete__": "answer",
                "__end__": END,
            },
        )
        graph_builder.add_edge("answer", END)

        if checkpointer:
            graph = graph_builder.compile(checkpointer=checkpointer)
        else:
            graph = graph_builder.compile()

        return graph

    async def generate_answer(self, *, question: UserQuerySchema):
        """사용자 질문에 대한 답변을 생성합니다.

        1. Agent LangGraph를 생성합니다.
        2. 세션 ID가 없는 경우 새로운 세션을 생성하여 사용합니다.
        3. Agent LangGraph를 실행하고, 결과를 반환합니다.

        Args:
            question: 사용자 질문 + 세션 ID
        """
        async with AsyncRedisSaver.from_conn_info(
            host=self.config.REDIS_HOST,
            port=self.config.REDIS_PORT,
            db=self.config.REDIS_DB,
        ) as checkpointer:

            agent_graph = await self.get_agent_graph(checkpointer=checkpointer)

            new_session_id = str(uuid4().hex)

            async for event in agent_graph.astream(
                {"messages": ("user", question.query)},
                {
                    "configurable": {
                        "thread_id": (question.session_id or new_session_id)
                    }
                },
            ):
                for value in event.values():
                    last_message = value["messages"][-1]
                    content = last_message.content

                    # Dict 형태인 경우 JSON으로 변환, 그렇지 않은 경우 비어있지 않은 문자열인 경우에만 진행
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        if content:
                            pass
                        else:
                            continue
                    finally:
                        content = {"answer": content}

                    # 세션 ID를 반환값에 포함 -> 이후 요청에서 사용
                    session_field = {
                        "session_id": question.session_id or new_session_id
                    }
                    # 최종 답변과 이벤트 여부를 반환값에 포함
                    type_field = {
                        "type": (
                            "answer" if isinstance(last_message, AIMessage) else "event"
                        )
                    }

                    yield f"data: {json.dumps(session_field | type_field | content)}\n\n"
