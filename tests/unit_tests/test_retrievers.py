"""TypesearchRetriever against the fake API."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from langchain_typesearch import TypesearchRetriever

from ..fake_api import KEY, FakeApi, Scripted, search_response


def retriever(api: FakeApi, **kwargs: object) -> TypesearchRetriever:
    return TypesearchRetriever(typesearch_api_key=KEY, typesearch_base_url=api.url, max_retries=0, **kwargs)  # type: ignore[arg-type]


def test_documents_content_and_metadata(api: FakeApi) -> None:
    docs = retriever(api, k=2).invoke("el dólar")
    assert api.last.body == {"query": "el dólar", "mode": "fast", "max_results": 2}
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == "El dólar cerró estable por 1ª rueda\n\nLa divisa se mantuvo sin cambios frente al cierre anterior."
    assert docs[0].metadata == {
        "title": "El dólar cerró estable por 1ª rueda",
        "url": "https://diarioejemplo.example/economia/nota-1",
        "source": "Diario Ejemplo",
        "published_at": "2026-09-21T18:05Z",
        "country": "AR",
        "language": "es",
        "score": 0.95,
    }
    assert docs[1].page_content == "Peso holds steady for day 2\n\nThe peso ended the session unchanged"
    assert docs[1].metadata["found_in"] == "discovery"


def test_k_per_call_and_the_filters(api: FakeApi) -> None:
    r = retriever(
        api,
        k=5,
        mode="deep",
        days=30,
        countries=["AR"],
        languages=["es"],
        include_domains=["diarioejemplo.example"],
        exclude_domains=["examplewire.example"],
        highlights=True,
        timezone="Europe/Madrid",
    )
    assert len(r.invoke("inflación", k=1)) == 1
    assert api.last.body == {
        "query": "inflación",
        "mode": "deep",
        "max_results": 1,
        "days": 30,
        "include_domains": ["diarioejemplo.example"],
        "exclude_domains": ["examplewire.example"],
        "countries": ["AR"],
        "languages": ["es"],
        "highlights": True,
        "timezone": "Europe/Madrid",
    }
    with pytest.raises(ValueError):
        r.invoke("inflación", k=0)
    with pytest.raises(ValueError):
        TypesearchRetriever(k=51)


def test_nothing_found(api: FakeApi) -> None:
    api.next(Scripted(200, search_response(0, near_misses=[search_response(1)["results"][0]])))
    assert retriever(api).invoke("el dólar") == []


def test_errors_are_raised_readable(api: FakeApi) -> None:
    with pytest.raises(Exception, match=r"typesearch error \(invalid_api_key\): The API key is not valid\."):
        TypesearchRetriever(typesearch_api_key="ts_live_wrong", typesearch_base_url=api.url).invoke("el dólar")
    with pytest.raises(Exception, match="Missing API key"):
        TypesearchRetriever(typesearch_base_url=api.url).invoke("el dólar")


async def test_async(api: FakeApi) -> None:
    docs = await retriever(api, k=3).ainvoke("el dólar")
    assert len(docs) == 3
