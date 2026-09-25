# Changelog

All notable changes to `langchain-typesearch` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

First release.

- `TypesearchNewsSearch`: a LangChain tool (`typesearch_news_search`) with the parameters, limits and
  description of the typesearch MCP `search_news` tool. The model reads a compact text list; the
  `ToolMessage` carries the structured results as its `artifact`. Sync and async.
- `TypesearchRetriever`: news articles as documents, with the link, outlet, date, country, language and
  relevance score as metadata; `k` in the constructor or per call. Sync and async.
- Filters fixed in the constructor are always applied and hidden from the model.
- Errors become readable `ToolException`s, handled by default, with the SDK error as `__cause__`.
- Passes LangChain's standard unit and integration tests for tools and retrievers.
