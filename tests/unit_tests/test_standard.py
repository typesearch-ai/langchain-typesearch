"""LangChain's standard unit tests for tools."""

from __future__ import annotations

from typing import Any

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_typesearch import TypesearchNewsSearch


class TestTypesearchNewsSearchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[TypesearchNewsSearch]:
        return TypesearchNewsSearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"typesearch_api_key": "ts_test_unit", "max_results": 3}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "inflation in Argentina", "days": 7}

    @property
    def init_from_env_params(self) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        return (
            {"TYPESEARCH_API_KEY": "ts_test_env", "TYPESEARCH_BASE_URL": "http://127.0.0.1:9"},
            {},
            {"typesearch_api_key": "ts_test_env", "typesearch_base_url": "http://127.0.0.1:9"},
        )
