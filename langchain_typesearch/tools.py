"""TypesearchNewsSearch: la búsqueda de noticias de typesearch como herramienta de LangChain."""

from __future__ import annotations

import re
from typing import Any, Literal

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model, field_validator, model_validator
from typing_extensions import Self

from ._client import TypesearchSettings, readable_error
from ._format import news_search_output, news_search_text

# Una fecha (2026-09-25) o una fecha y hora con su zona (2026-09-25T14:00:00Z), como pide la API.
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2}))?$")


class _Checks(BaseModel):
    """Las validaciones de la entrada, con mensajes en inglés para el modelo. Van aparte para que valgan
    también en el esquema reducido (sin los filtros fijados en el constructor)."""

    @field_validator("query", check_fields=False)
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("query needs at least two letters.")
        return v

    @field_validator("published_after", "published_before", check_fields=False)
    @classmethod
    def _date(cls, v: str | None) -> str | None:
        if v is not None and not _DATE.match(v):
            raise ValueError("must be a date (2026-09-25) or a date-time with its offset (2026-09-25T14:00:00Z).")
        return v


class TypesearchNewsSearchInput(_Checks):
    """What the model can ask for: the parameters, limits and descriptions of ``search_news`` in the
    typesearch MCP server."""

    query: str = Field(
        min_length=2,
        max_length=200,
        description='What to look for, in any language: a topic, event, person, company or place, such as "inflation in Argentina" or "OpenAI funding".',
    )
    days: int | None = Field(
        default=None,
        ge=1,
        le=365,
        description="Only the last N days, 1 to 365. Defaults to 7 unless published_after or published_before are given.",
    )
    published_after: str | None = Field(
        default=None, description="Published on or after this date: 2026-09-25, or a date-time with offset."
    )
    published_before: str | None = Field(default=None, description="Published on or before this date; a bare date includes that whole day.")
    include_domains: list[str] | None = Field(
        default=None, max_length=20, description="Only these domains or paths, such as example.com or example.com/sports."
    )
    exclude_domains: list[str] | None = Field(default=None, max_length=20, description="Never these domains or paths.")
    countries: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=50,
        description='Only sources from these countries: ISO 3166-1 alpha-2 codes, such as ["AR"] or ["US", "GB"].',
    )
    languages: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
        description='Only sources that publish in these languages: ISO 639-1 codes, such as ["es"] or ["en", "pt"].',
    )


_FIXABLE = ("include_domains", "exclude_domains", "countries", "languages")

DESCRIPTION = (
    "Search recent news on any topic across a curated index of news outlets worldwide, judged by a relevance model. "
    "Returns the matching articles: title, link, source, date, country and language, standfirst, and short excerpts in the modes that read, "
    "each with a relevance score from 0 to 1. "
    "Use it for current events and for what outlets reported about a company, person, place or topic. "
    "It covers the last 7 days unless you set days or a date range. Cite the link of every fact."
)


class TypesearchNewsSearch(TypesearchSettings, BaseTool):
    """News search as a LangChain tool, backed by typesearch (https://typesearch.ai).

    The model reads a compact text list of the results. Invoked with a tool call, the ``ToolMessage`` also
    carries the structured results as its ``artifact``.

    Setup:
        .. code-block:: bash

            pip install -U langchain-typesearch
            export TYPESEARCH_API_KEY="ts_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_typesearch import TypesearchNewsSearch

            search = TypesearchNewsSearch(max_results=5)

    Invoke directly with args:
        .. code-block:: python

            search.invoke({"query": "EU AI Act enforcement", "days": 7})

    Invoke with a tool call:
        .. code-block:: python

            msg = search.invoke({"args": {"query": "EU AI Act enforcement"}, "id": "1", "name": search.name, "type": "tool_call"})
            msg.artifact["results"]

    Filters set here (``countries``, ``languages``, ``include_domains``, ``exclude_domains``) are always
    applied and are no longer offered to the model.
    """

    name: str = "typesearch_news_search"
    description: str = DESCRIPTION
    args_schema: type[BaseModel] = TypesearchNewsSearchInput
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    handle_tool_error: bool = True
    max_results: int = Field(default=10, ge=1, le=50)
    """Results per search, 1 to 50. Defaults to 10."""

    @model_validator(mode="after")
    def _hide_fixed_filters(self) -> Self:
        # Lo que fija quien arma la herramienta no se le ofrece al modelo: no puede saltearlo ni gastar
        # tokens en él. La descripción se lo cuenta.
        fixed = [k for k in _FIXABLE if getattr(self, k)]
        if fixed and self.args_schema is TypesearchNewsSearchInput:
            fields: dict[str, Any] = {k: (f.annotation, f) for k, f in TypesearchNewsSearchInput.model_fields.items() if k not in fixed}
            self.args_schema = create_model(
                "TypesearchNewsSearchInput", __base__=_Checks, __doc__=TypesearchNewsSearchInput.__doc__, **fields
            )
            limits = [
                f"sources from {', '.join(self.countries)}" if self.countries else None,
                f"sources in {', '.join(self.languages)}" if self.languages else None,
                f"only {', '.join(self.include_domains)}" if self.include_domains else None,
                f"never {', '.join(self.exclude_domains)}" if self.exclude_domains else None,
            ]
            if self.description == DESCRIPTION:
                self.description = f"{DESCRIPTION} Searches are limited to {'; '.join(x for x in limits if x)}."
        return self

    def _result(self, output: dict[str, Any]) -> tuple[str, dict[str, Any]] | str:
        text = news_search_text(output)
        return (text, output) if self.response_format == "content_and_artifact" else text

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
        **model: Any,
    ) -> tuple[str, dict[str, Any]] | str:
        """Searches the news. ``model`` has the other arguments the model asked for."""
        try:
            res = self._sync_client().search(query, **self._options(self.max_results, **model))
        except Exception as e:
            raise readable_error(e) from e
        return self._result(news_search_output(res, query))

    async def _arun(
        self,
        query: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **model: Any,
    ) -> tuple[str, dict[str, Any]] | str:
        try:
            res = await self._async_client().search(query, **self._options(self.max_results, **model))
        except Exception as e:
            raise readable_error(e) from e
        return self._result(news_search_output(res, query))
