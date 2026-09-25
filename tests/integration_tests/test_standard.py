"""LangChain's standard integration tests for the tool and the retriever.

By default they run against the fake API (tests/fake_api.py), offline. With ``TYPESEARCH_LIVE=1`` and
``TYPESEARCH_API_KEY`` they run against the real API (a few ``fast`` searches: less than a cent).
"""

from __future__ import annotations

import os
from typing import Any

from langchain_tests.integration_tests import RetrieversIntegrationTests, ToolsIntegrationTests

from langchain_typesearch import TypesearchNewsSearch, TypesearchRetriever

from ..fake_api import KEY, shared

LIVE = os.environ.get("TYPESEARCH_LIVE") == "1"


def connection() -> dict[str, Any]:
    if LIVE:
        return {"typesearch_api_key": os.environ["TYPESEARCH_API_KEY"]}
    return {"typesearch_api_key": KEY, "typesearch_base_url": shared().url}


class TestTypesearchNewsSearchIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[TypesearchNewsSearch]:
        return TypesearchNewsSearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {**connection(), "max_results": 3}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "inflation", "days": 7}


class TestTypesearchRetrieverIntegration(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> type[TypesearchRetriever]:
        return TypesearchRetriever

    @property
    def retriever_constructor_params(self) -> dict[str, Any]:
        # 30 days so that the live index has at least 3 articles for the query.
        return {**connection(), "k": 3, "days": 30}

    @property
    def retriever_query_example(self) -> str:
        return "inflation"
