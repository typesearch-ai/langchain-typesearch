"""typesearch news search for LangChain: a tool and a retriever.

>>> from langchain_typesearch import TypesearchNewsSearch, TypesearchRetriever
>>> search = TypesearchNewsSearch(max_results=5)  # reads TYPESEARCH_API_KEY
>>> search.invoke({"query": "EU AI Act enforcement", "days": 7})
"""

from ._client import __version__
from .retrievers import TypesearchRetriever
from .tools import TypesearchNewsSearch, TypesearchNewsSearchInput

__all__ = ["TypesearchNewsSearch", "TypesearchNewsSearchInput", "TypesearchRetriever", "__version__"]
