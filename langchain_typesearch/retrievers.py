"""TypesearchRetriever: notas de prensa como documentos de LangChain."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from ._client import TypesearchSettings, readable_error
from ._format import news_search_output


class TypesearchRetriever(TypesearchSettings, BaseRetriever):
    """News articles as LangChain documents, backed by typesearch search (https://typesearch.ai).

    Each document's ``page_content`` is the headline, the standfirst and, in the modes that read, short
    verbatim excerpts — never the full article. ``metadata`` has the link (``url``), the outlet
    (``source``), the date, the country, the language and the relevance ``score``.

    Setup:
        .. code-block:: bash

            pip install -U langchain-typesearch
            export TYPESEARCH_API_KEY="ts_live_..."

    Instantiate:
        .. code-block:: python

            from langchain_typesearch import TypesearchRetriever

            retriever = TypesearchRetriever(k=5, days=30)

    Usage:
        .. code-block:: python

            docs = retriever.invoke("lithium royalties in Chile")
            docs = retriever.invoke("lithium royalties in Chile", k=3)
    """

    k: int = Field(default=10, ge=1, le=50)
    """Documents per query, 1 to 50. Defaults to 10. Also accepted per call: ``invoke(query, k=3)``."""

    def _k(self, kwargs: dict[str, Any]) -> int:
        k = int(kwargs.get("k", self.k))
        if not 1 <= k <= 50:
            raise ValueError("k must be a whole number from 1 to 50.")
        return k

    @staticmethod
    def _documents(output: dict[str, Any]) -> list[Document]:
        docs = []
        for r in output["results"]:
            metadata = {k: v for k, v in r.items() if k not in ("snippet", "highlights")}
            content = "\n\n".join([r["title"], *([r["snippet"]] if r.get("snippet") else []), *r.get("highlights", [])])
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any) -> list[Document]:
        try:
            res = self._sync_client().search(query, **self._options(self._k(kwargs)))
        except Exception as e:
            raise readable_error(e) from e
        return self._documents(news_search_output(res, query))

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        try:
            res = await self._async_client().search(query, **self._options(self._k(kwargs)))
        except Exception as e:
            raise readable_error(e) from e
        return self._documents(news_search_output(res, query))
