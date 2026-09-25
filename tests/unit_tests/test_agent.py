"""End to end with a LangChain agent (create_agent) and a fake chat model that calls the tool."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from langchain_typesearch import TypesearchNewsSearch

from ..fake_api import KEY, FakeApi


class ToolCallingFake(GenericFakeChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ToolCallingFake:
        return self


def agent_for(api: FakeApi, key: str = KEY) -> Any:
    model = ToolCallingFake(
        messages=iter(
            [
                AIMessage(
                    content="", tool_calls=[{"name": "typesearch_news_search", "args": {"query": "el dólar", "days": 1}, "id": "call_1"}]
                ),
                AIMessage(content="The peso held steady (Diario Ejemplo)."),
            ]
        )
    )
    return create_agent(model=model, tools=[TypesearchNewsSearch(typesearch_api_key=key, typesearch_base_url=api.url, max_retries=0)])


def test_the_agent_calls_the_tool_and_the_model_reads_the_results(api: FakeApi) -> None:
    result = agent_for(api).invoke({"messages": [{"role": "user", "content": "What happened with the peso today?"}]})
    assert api.last.body == {"query": "el dólar", "mode": "fast", "max_results": 10, "days": 1}
    tool = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert str(tool.content).startswith('10 results for "el dólar" · fast')
    assert len(tool.artifact["results"]) == 10
    assert result["messages"][-1].content == "The peso held steady (Diario Ejemplo)."


def test_an_api_error_reaches_the_model_as_the_tool_result(api: FakeApi) -> None:
    result = agent_for(api, key="ts_live_wrong").invoke({"messages": [{"role": "user", "content": "News about the peso?"}]})
    tool = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool.content == "typesearch error (invalid_api_key): The API key is not valid. [request req_fakeerr1]"
