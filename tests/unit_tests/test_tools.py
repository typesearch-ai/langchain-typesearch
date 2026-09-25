"""TypesearchNewsSearch against the fake API: what it sends, what the model reads and the artifact."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from typesearch import RateLimitError, Typesearch

from langchain_typesearch import TypesearchNewsSearch, __version__

from ..fake_api import KEY, FakeApi, Scripted, problem, search_response


def tool(api: FakeApi, **kwargs: Any) -> TypesearchNewsSearch:
    return TypesearchNewsSearch(typesearch_api_key=KEY, typesearch_base_url=api.url, max_retries=0, **kwargs)


def call(t: TypesearchNewsSearch, **args: Any) -> ToolMessage:
    msg = t.invoke({"args": args, "id": "call_1", "name": t.name, "type": "tool_call"})
    assert isinstance(msg, ToolMessage)
    return msg


def parameters(t: TypesearchNewsSearch) -> dict[str, Any]:
    return convert_to_openai_tool(t)["function"]  # type: ignore[no-any-return]


def test_the_model_sees_the_mcp_parameters(api: FakeApi) -> None:
    f = parameters(tool(api))
    assert f["name"] == "typesearch_news_search"
    assert list(f["parameters"]["properties"]) == [
        "query",
        "days",
        "published_after",
        "published_before",
        "include_domains",
        "exclude_domains",
        "countries",
        "languages",
    ]
    assert f["parameters"]["required"] == ["query"]
    assert len(f["description"]) > 200


def test_defaults_fast_and_10_results_with_the_key_and_user_agent(api: FakeApi) -> None:
    text = tool(api).invoke({"query": "el dólar"})
    assert api.last.path == "/v1/search"
    assert api.last.body == {"query": "el dólar", "mode": "fast", "max_results": 10}
    assert api.last.headers["authorization"] == f"Bearer {KEY}"
    assert api.last.headers["user-agent"].startswith(f"langchain-typesearch/{__version__} typesearch-python/")
    assert text.startswith('10 results for "el dólar" · fast · US$0.0014')
    assert (
        "1. El dólar cerró estable por 1ª rueda\nDiario Ejemplo · 2026-09-21 18:05 UTC · AR/es\nhttps://diarioejemplo.example/economia/nota-1"
        in text
    )
    assert "found beyond the index" in text
    assert "> The peso ended the session unchanged" in text


def test_tool_call_returns_the_results_as_artifact(api: FakeApi) -> None:
    msg = call(tool(api, max_results=2), query="el dólar", days=2)
    assert msg.tool_call_id == "call_1"
    assert isinstance(msg.content, str) and msg.content.startswith('2 results for "el dólar"')
    assert msg.artifact["query"] == "el dólar"
    assert msg.artifact["results"][0] == {
        "title": "El dólar cerró estable por 1ª rueda",
        "url": "https://diarioejemplo.example/economia/nota-1",
        "source": "Diario Ejemplo",
        "published_at": "2026-09-21T18:05Z",
        "country": "AR",
        "language": "es",
        "snippet": "La divisa se mantuvo sin cambios frente al cierre anterior.",
        "score": 0.95,
    }
    assert msg.artifact["results"][1] == {
        "title": "Peso holds steady for day 2",
        "url": "https://examplewire.example/markets/story-2",
        "source": "Example Wire",
        "published_at": "2026-09-21T18:05Z",
        "language": "en",
        "highlights": ["The peso ended the session unchanged"],
        "score": 0.94,
        "found_in": "discovery",
    }
    assert msg.artifact["cost_usd"] == 0.0014
    assert msg.artifact["request_id"] == "req_fakelc"


def test_content_only(api: FakeApi) -> None:
    msg = call(tool(api, response_format="content"), query="el dólar")
    assert msg.content.startswith('10 results for "el dólar"')  # type: ignore[union-attr]
    assert msg.artifact is None


def test_passes_what_the_model_asks_for(api: FakeApi) -> None:
    tool(api).invoke(
        {
            "query": "lithium royalties",
            "days": 3,
            "include_domains": ["diarioejemplo.example"],
            "exclude_domains": ["examplewire.example"],
            "countries": ["AR", "CL"],
            "languages": ["es"],
        }
    )
    assert api.last.body == {
        "query": "lithium royalties",
        "mode": "fast",
        "max_results": 10,
        "days": 3,
        "include_domains": ["diarioejemplo.example"],
        "exclude_domains": ["examplewire.example"],
        "countries": ["AR", "CL"],
        "languages": ["es"],
    }
    tool(api).invoke({"query": "lithium royalties", "published_after": "2026-09-01", "published_before": "2026-09-20T12:00:00Z"})
    assert api.last.body["published_after"] == "2026-09-01"
    assert api.last.body["published_before"] == "2026-09-20T12:00:00Z"


def test_fixed_filters_are_hidden_from_the_model_and_always_applied(api: FakeApi) -> None:
    t = tool(
        api, mode="normal", max_results=5, countries=["AR"], languages=["es"], highlights=True, timezone="America/Argentina/Buenos_Aires"
    )
    f = parameters(t)
    assert list(f["parameters"]["properties"]) == [
        "query",
        "days",
        "published_after",
        "published_before",
        "include_domains",
        "exclude_domains",
    ]
    assert f["description"].endswith("Searches are limited to sources from AR; sources in es.")
    t.invoke({"query": "inflación", "countries": ["US"]})
    assert api.last.body == {
        "query": "inflación",
        "mode": "normal",
        "max_results": 5,
        "countries": ["AR"],
        "languages": ["es"],
        "highlights": True,
        "timezone": "America/Argentina/Buenos_Aires",
    }


def test_days_is_the_window_when_the_model_asks_for_none(api: FakeApi) -> None:
    t = tool(api, days=1)
    t.invoke({"query": "el dólar"})
    assert api.last.body["days"] == 1
    t.invoke({"query": "el dólar", "days": 30})
    assert api.last.body["days"] == 30
    t.invoke({"query": "el dólar", "published_after": "2026-09-01"})
    assert "days" not in api.last.body


@pytest.mark.parametrize(
    "args",
    [
        {"query": "x"},
        {"query": "  x "},
        {"query": "el dólar", "days": 0},
        {"query": "el dólar", "published_after": "yesterday"},
        {"query": "el dólar", "include_domains": ["a.example"] * 21},
    ],
)
def test_invalid_input_never_reaches_the_api(api: FakeApi, args: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        tool(api).invoke(args)
    assert api.requests == []


def test_invalid_settings() -> None:
    with pytest.raises(ValueError):
        TypesearchNewsSearch(max_results=51)
    with pytest.raises(ValueError):
        TypesearchNewsSearch(mode="turbo")  # type: ignore[arg-type]


def test_nothing_found_closest_articles_warnings_cached_incomplete(api: FakeApi) -> None:
    body = search_response(
        0,
        near_misses=[search_response(1)["results"][0]],
        cached_at="2026-09-25T10:00:00Z",
        incomplete=True,
        warnings=[{"code": "country_not_indexed", "message": "No source from XX."}],
    )
    api.next(Scripted(200, body))
    msg = call(tool(api), query="el dólar")
    assert isinstance(msg.content, str)
    assert msg.content.startswith('No results for "el dólar" · fast · cached, free')
    assert "Closest articles, which may not be about it:" in msg.content
    assert "Incomplete:" in msg.content
    assert "Note (country_not_indexed): No source from XX." in msg.content
    assert msg.artifact["results"] == []
    assert msg.artifact["cached"] is True and msg.artifact["incomplete"] is True


def test_api_errors_reach_the_agent_readable_and_without_the_key(api: FakeApi) -> None:
    msg = call(TypesearchNewsSearch(typesearch_api_key="ts_live_wrong", typesearch_base_url=api.url), query="el dólar")
    assert msg.status == "error"
    assert msg.content == "typesearch error (invalid_api_key): The API key is not valid. [request req_fakeerr1]"
    api.next(Scripted(402, problem(402, "insufficient_credits", "No credit left.")))
    assert call(tool(api), query="el dólar").content == "typesearch error (insufficient_credits): No credit left. [request req_fakeerr1]"
    api.next(Scripted(429, problem(429, "rate_limited", "Too many requests."), {"Retry-After": "9"}))
    assert "Retry after 9 s." in str(call(tool(api), query="el dólar").content)


def test_without_handling_the_error_keeps_the_sdk_error_as_cause(api: FakeApi) -> None:
    api.next(Scripted(429, problem(429, "rate_limited", "Too many requests."), {"Retry-After": "9"}))
    with pytest.raises(Exception, match=r"typesearch error \(rate_limited\)") as e:
        tool(api, handle_tool_error=False).invoke({"query": "el dólar"})
    assert isinstance(e.value.__cause__, RateLimitError)


def test_a_missing_key_fails_the_call_not_the_constructor(api: FakeApi) -> None:
    t = TypesearchNewsSearch(typesearch_base_url=api.url)
    assert str(t.invoke({"query": "el dólar"})).startswith("typesearch error: Missing API key")
    assert api.requests == []


def test_the_key_comes_from_the_environment(api: FakeApi, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESEARCH_API_KEY", KEY)
    monkeypatch.setenv("TYPESEARCH_BASE_URL", api.url)
    TypesearchNewsSearch().invoke({"query": "el dólar"})
    assert api.last.headers["authorization"] == f"Bearer {KEY}"
    assert "ts_test" not in repr(TypesearchNewsSearch())


def test_a_client_of_your_own(api: FakeApi) -> None:
    client = Typesearch(KEY, base_url=api.url, default_headers={"X-Trace": "abc"})
    TypesearchNewsSearch(client=client).invoke({"query": "el dólar"})
    assert api.last.headers["x-trace"] == "abc"


async def test_async(api: FakeApi) -> None:
    msg = await tool(api, max_results=3).ainvoke(
        {"args": {"query": "el dólar"}, "id": "c", "name": "typesearch_news_search", "type": "tool_call"}
    )
    assert isinstance(msg, ToolMessage)
    assert len(msg.artifact["results"]) == 3
    assert api.last.body == {"query": "el dólar", "mode": "fast", "max_results": 3}
