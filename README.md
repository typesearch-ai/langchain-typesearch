# langchain-typesearch

[typesearch](https://typesearch.ai) news search for [LangChain](https://docs.langchain.com/oss/python): a
tool for agents and a retriever for chains. Recent news on any topic from outlets worldwide, by country and
language, with a calibrated relevance score on every result.

```bash
pip install -U langchain-typesearch
```

Python 3.10+, `langchain-core` 1.x. Create a key in the [dashboard](https://app.typesearch.ai) and set it
as `TYPESEARCH_API_KEY`.

## Tool: TypesearchNewsSearch

```python
from langchain_typesearch import TypesearchNewsSearch

search = TypesearchNewsSearch(max_results=5)  # reads TYPESEARCH_API_KEY

search.invoke({"query": "EU AI Act enforcement", "days": 7})
# '5 results for "EU AI Act enforcement" · fast · US$0.0014\n\n1. …'
```

### In an agent

```python
from langchain.agents import create_agent
from langchain_typesearch import TypesearchNewsSearch

agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    tools=[TypesearchNewsSearch(max_results=8)],
    system_prompt="Answer with recent news. Cite the source and the link of every fact.",
)

agent.invoke({"messages": [{"role": "user", "content": "What changed in EU AI Act enforcement this week?"}]})
```

The tool is named `typesearch_news_search`. The model can set `query`, `days`, `published_after`,
`published_before`, `include_domains`, `exclude_domains`, `countries` (ISO 3166-1 alpha-2) and `languages`
(ISO 639-1) — the same parameters, limits and descriptions as `search_news` in the
[typesearch MCP server](https://typesearch.ai/docs/integrations/mcp).

The model reads a compact text list: title, outlet, date, country and language, link, standfirst and
excerpts. Invoked with a tool call, the `ToolMessage` also carries the structured results as its `artifact`:

```python
msg = search.invoke({"args": {"query": "el dólar"}, "id": "1", "name": search.name, "type": "tool_call"})
msg.artifact["results"]  # [{"title", "url", "source", "published_at", "country", "language", "snippet", "highlights", "score"}]
```

Pass `response_format="content"` to get only the text.

| Parameter | Default | |
| --- | --- | --- |
| `typesearch_api_key` | `TYPESEARCH_API_KEY` | Your key. |
| `typesearch_base_url` | `TYPESEARCH_BASE_URL` | Defaults to `https://api.typesearch.ai`. |
| `mode` | `"fast"` | `ultra`, `fast`, `normal` or `deep`: how much is read before ranking. See [modes](https://typesearch.ai/docs/modes). |
| `max_results` | `10` | 1 to 50. |
| `days` | — | The window when the model asks for none. The API's default is the last 7 days. |
| `include_domains` · `exclude_domains` | — | Fixed domain filters. |
| `countries` · `languages` | — | Fixed country and language filters of the sources. |
| `highlights` | — | Verbatim excerpts from the articles read (`normal` and `deep`). |
| `timezone` | — | IANA time zone that decides what day "today" is. |
| `timeout` · `max_retries` | `70` · `2` | Per request, as in the [`typesearch`](https://pypi.org/project/typesearch/) SDK. |
| `client` · `async_client` | — | `Typesearch` / `AsyncTypesearch` clients you already have. |

A filter you set in the constructor is always applied and is no longer offered to the model; the
description tells the model about it. `ainvoke` works too, with the async client.

## Retriever: TypesearchRetriever

```python
from langchain_typesearch import TypesearchRetriever

retriever = TypesearchRetriever(k=5, days=30)
docs = retriever.invoke("lithium royalties in Chile")
docs = retriever.invoke("lithium royalties in Chile", k=3)  # k per call
# Document(page_content="headline\n\nstandfirst\n\nexcerpt…", metadata={"title", "url", "source", "published_at", "country", "language", "score"})
```

`page_content` is the headline, the standfirst and, in `normal` and `deep` modes, short verbatim excerpts —
never the full article: cite `metadata["url"]`. `metadata["source"]` is the outlet and `metadata["score"]`
the calibrated probability that the article is about the query. It takes the same parameters as the tool,
with `k` (1–50, default 10) instead of `max_results`.

## Errors

API errors become a `ToolException` with a message the agent can read and act on —
`typesearch error (rate_limited): … Retry after 12 s.` — without the key. The tool handles it
(`handle_tool_error=True`, as in `langchain-tavily`), so the model gets it as the tool result; set
`handle_tool_error=False` to have it raised. The original [`typesearch`](https://pypi.org/project/typesearch/)
error is its `__cause__`. The key is read on the first call, so building the tool never fails without it.

## Pricing

Each search is billed to your key like the API request it makes, by its mode. Identical searches within
10 minutes come from the cache and cost nothing. Prices: [typesearch.ai/pricing](https://typesearch.ai/pricing).

## Examples

[`examples/agent.py`](examples/agent.py) (an agent that cites its sources) and
[`examples/retriever.py`](examples/retriever.py).

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv run pytest                     # LangChain's standard tests and the package's own, against a fake API
                                  # that validates every request against the API schema, and create_agent end to end
TYPESEARCH_LIVE=1 TYPESEARCH_API_KEY=ts_live_… uv run pytest tests/integration_tests   # the standard integration
                                  # tests against the real API (less than a cent)
```

## License

[MIT](LICENSE)
