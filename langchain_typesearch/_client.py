"""Lo que comparten la herramienta y el retriever: la conexión con la API, los filtros y los errores."""

from __future__ import annotations

from importlib import metadata
from typing import Any, Literal

import typesearch
from langchain_core.tools import ToolException
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr
from typesearch import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncTypesearch,
    RateLimitError,
    Typesearch,
    TypesearchError,
)

try:
    __version__ = metadata.version("langchain-typesearch")
except metadata.PackageNotFoundError:  # pragma: no cover - sin instalar
    __version__ = "0.0.0"

USER_AGENT = f"langchain-typesearch/{__version__} typesearch-python/{typesearch.__version__}"

Mode = Literal["ultra", "fast", "normal", "deep"]


class TypesearchSettings(BaseModel):
    """The connection and the search filters, shared by :class:`TypesearchNewsSearch` and
    :class:`TypesearchRetriever`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    typesearch_api_key: SecretStr | None = Field(default_factory=secret_from_env("TYPESEARCH_API_KEY", default=None))
    """Your API key. Defaults to the ``TYPESEARCH_API_KEY`` environment variable."""
    typesearch_base_url: str | None = Field(default_factory=from_env("TYPESEARCH_BASE_URL", default=None))
    """Defaults to ``TYPESEARCH_BASE_URL``, or ``https://api.typesearch.ai``."""
    mode: Mode = "fast"
    """How much is read before ranking: ``fast`` (default, the cheapest and quickest: headlines and
    standfirsts), ``ultra`` (headlines only, same price), ``normal`` (also reads the best matches) or ``deep``
    (reads more and finds the topic in other words too). See https://typesearch.ai/docs/modes."""
    days: int | None = Field(default=None, ge=1, le=365)
    """The last N days, when the model (or the query) sets no window. The API's default is 7."""
    include_domains: list[str] | None = None
    """Only these domains or paths."""
    exclude_domains: list[str] | None = None
    """Never these domains or paths."""
    countries: list[str] | None = None
    """Only sources from these countries (ISO 3166-1 alpha-2)."""
    languages: list[str] | None = None
    """Only sources in these languages (ISO 639-1)."""
    highlights: bool | None = None
    """Verbatim excerpts from the articles that were read (``normal`` and ``deep``). On by default in ``deep``."""
    timezone: str | None = None
    """IANA time zone that decides what day "today" is in a query such as "news from today"."""
    timeout: float | None = None
    """Seconds before a request is aborted. Defaults to 70 (a ``deep`` search can take about a minute)."""
    max_retries: int | None = None
    """Retries on connection errors, ``429 rate_limited`` and ``5xx``. Defaults to 2."""
    client: Typesearch | None = Field(default=None, exclude=True)
    """A ``typesearch`` client you already have, used instead of creating one."""
    async_client: AsyncTypesearch | None = Field(default=None, exclude=True)
    """An ``AsyncTypesearch`` client you already have, for the async calls."""

    _sync: Typesearch | None = PrivateAttr(default=None)
    _async: AsyncTypesearch | None = PrivateAttr(default=None)

    # El cliente se crea en la primera llamada: armar la herramienta sin clave no falla, y el error de la clave
    # faltante le llega al agente como error de la herramienta.
    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"default_headers": {"User-Agent": USER_AGENT}}
        if self.typesearch_api_key is not None:
            kwargs["api_key"] = self.typesearch_api_key.get_secret_value()
        if self.typesearch_base_url:
            kwargs["base_url"] = self.typesearch_base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.max_retries is not None:
            kwargs["max_retries"] = self.max_retries
        return kwargs

    def _sync_client(self) -> Typesearch:
        if self.client is not None:
            return self.client
        if self._sync is None:
            self._sync = Typesearch(**self._client_kwargs())
        return self._sync

    def _async_client(self) -> AsyncTypesearch:
        if self.async_client is not None:
            return self.async_client
        if self._async is None:
            self._async = AsyncTypesearch(**self._client_kwargs())
        return self._async

    def _options(self, max_results: int, **model: Any) -> dict[str, Any]:
        """Las opciones del pedido: lo fijado en el constructor manda sobre lo que pide el modelo."""
        dated = any(model.get(k) is not None for k in ("days", "published_after", "published_before"))
        options: dict[str, Any] = {"mode": self.mode, "max_results": max_results}
        days = model.get("days") if model.get("days") is not None else (None if dated else self.days)
        if days is not None:
            options["days"] = days
        for key in ("published_after", "published_before"):
            if model.get(key):
                options[key] = model[key]
        for key in ("include_domains", "exclude_domains", "countries", "languages"):
            value = getattr(self, key) or model.get(key)
            if value:
                options[key] = list(value)
        if self.highlights is not None:
            options["highlights"] = self.highlights
        if self.timezone:
            options["timezone"] = self.timezone
        return options


def readable_error(e: Exception) -> ToolException:
    """Un error que el agente entiende y sobre el que puede actuar, en inglés y sin la clave. El error original
    del SDK queda en ``__cause__``."""
    if isinstance(e, APIError):
        retry = f" Retry after {e.retry_after:g} s." if isinstance(e, RateLimitError) and e.retry_after else ""
        request = f" [request {e.request_id}]" if e.request_id else ""
        message = f"typesearch error ({e.code}): {e.message}{retry}{request}"
    elif isinstance(e, APITimeoutError):
        message = "typesearch error (timeout): the API took too long to answer. Try again, or use a lighter mode."
    elif isinstance(e, APIConnectionError):
        message = "typesearch error (connection): could not reach the typesearch API. Try again."
    elif isinstance(e, TypesearchError):
        message = f"typesearch error: {e}"
    else:
        raise e
    error = ToolException(message)
    error.__cause__ = e
    return error
