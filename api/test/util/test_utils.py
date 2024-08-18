from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.messages.human import HumanMessage

from util.processor import ProcessDocuments


def test_process_document():
    document_processor = ProcessDocuments()

    state = {
        "messages": [
            AIMessage(content="이것은 AI 메시지입니다."),
            HumanMessage(content="이것은 사용자 메시지입니다."),
            ToolMessage(content="이것은 도구 메시지입니다.", tool_call_id="1"),
        ]
    }

    result = document_processor.invoke(state)

    assert result["query"] == ["이것은 사용자 메시지입니다."]
    assert result["document"] == ["이것은 도구 메시지입니다."]
