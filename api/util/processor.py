from __future__ import annotations

import json
from typing import (
    Optional,
)

from langchain_core.messages import ToolMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import (
    RunnableConfig,
)
from langchain_core.runnables.utils import (
    Input,
    Output,
)


class ProcessDocuments(Runnable):
    """Message를 간략한 형태로 변환하는 Runnable

    Message타입에 포함된 불필요한 정보를 제거하고 필요한 정보만을 추출하여, LLM이 사용할 수 있는 형태로 변환한다.

    - 메시지 목록에서 HumanMessage의 Content만을 추출하여 query로 사용
    - 메시지 목록에서 ToolMessage의 Content만을 추출하여 document로 사용

    Args:
        state: Agent가 가지고 있는 메시지(HumanMessage, ToolMessage, AIMessage) 목록

    Returns:
        query: HumanMessage의 Content
        document: ToolMessage의 Content
    """

    def invoke(self, state: Input, config: Optional[RunnableConfig] = None) -> Output:
        query, document = list(), list()
        for message in state["messages"][::-1]:
            if isinstance(message, HumanMessage):
                query.append(message.content)
                break

        # state.messages에서 가장 마지막 ToolMessage를 가져와 content를 취한다
        for message in state["messages"][::-1]:
            if isinstance(message, ToolMessage):
                docs = message.content
                try:
                    docs_content = json.loads(docs)
                except json.JSONDecodeError:
                    docs_content = docs

                if isinstance(docs_content, list):
                    document.extend(docs_content)
                else:
                    document.append(docs_content)

        return {
            "query": query,
            "document": document,
        }
