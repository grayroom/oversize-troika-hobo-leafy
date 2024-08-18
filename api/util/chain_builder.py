from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables.utils import AddableDict
from pydantic import BaseModel, ValidationError


class ChainBuilder:
    """동적으로 Langchain runnable을 생성하는 클래스

    runnable_registry에 등록된 함수를 이용하여 chain을 생성합니다.
    - key에 해당하는 함수들은, 미리 정의된 Langchain runnable으로 변환됩니다.
    """

    def __init__(
        self,
    ):
        self.chain = []
        self.runnable_registry = {
            "preprocessor": lambda runnable: runnable,
            "postprocessor": lambda runnable: runnable,
            "input_schema": lambda schema: self._validate_with_schema(schema),
            "prompt": self._prompt,
            "llm": lambda llm: llm,
            "output_parser": self._output_parser,
        }

    @staticmethod
    def _prompt(prompt_obj):
        """SystemPrompt를 전달받아, ChatPromptTemplate으로 변환합니다.

        - RAG Agent가 사용할 수 있도록, document와 query를 추가합니다.
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", prompt_obj),
                ("system", "- [문서]: {document}"),
                ("user", "- [사용자 쿼리]: {query}"),
            ]
        )

    @staticmethod
    def _output_parser(output_parser: str):
        """output_parser_type에 따라, 해당하는 OutputParser를 반환합니다."""
        match output_parser:
            case "str":
                return StrOutputParser()
            case "json":
                return JsonOutputParser()
            case "agent":
                return OpenAIToolsAgentOutputParser()
            case _:
                raise ValueError(
                    f"지원하지 않는 output_parser_type입니다. ({output_parser})"
                )

    @staticmethod
    def _validate_with_schema(input_schema: BaseModel) -> RunnableLambda:
        """
        Pydantic schema를 사용하여 입력 데이터를 검증합니다.
        """

        def validate(data: AddableDict) -> dict:
            try:
                # data를 AddableDict -> dict로 변환하여 Pydantic 모델에 맞게 검증합니다.
                data = {k: v for k, v in data.items()}

                validated_data = input_schema.model_validate(data)
                return validated_data.model_dump()
            except ValidationError as e:
                raise ValueError(f"입력 데이터가 스키마에 맞지 않습니다. ({e})")

        return RunnableLambda(validate)

    def register_runnable(self, key, conversion_func):
        """
        특정 key에 대한 runnable 변환 함수를 등록합니다.
        """
        self.runnable_registry[key] = conversion_func

    def build_chain(self, **kwargs):
        """
        kwargs로 전달받은 파라미터(preprocessor, postprocessor, schema, llm 등)를 이용하여 동적으로 chain을 생성합니다.
        - kwargs에 전달한 순서대로 chain이 생성됩니다.
        """
        self.chain = []
        for key, value in kwargs.items():
            if key in self.runnable_registry:
                self.chain.append((key, value))
            else:
                raise ValueError(
                    f"사용할 수 없는 Chain 구성요소({key})입니다. register_runnable을 통해 등록해주세요."
                )

    def make_runnable(self):
        """
        build_chain에서 생성한 chain을 이용하여 실행가능한 langchain runnable을 생성합니다.
        """

        runnable_list = []
        for key, value in self.chain:
            if conversion_func := self.runnable_registry.get(key):
                conversion_func = conversion_func(value)
            else:
                raise ValueError(
                    f"사용할 수 없는 Chain 구성요소({key})입니다. register_runnable을 통해 등록해주세요."
                )

            runnable_list.append(conversion_func)

        return RunnableSequence(*runnable_list)
