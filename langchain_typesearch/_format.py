"""La salida compacta de una búsqueda: los datos sin campos vacíos y un texto breve y legible, que es lo que
lee el modelo. Es el mismo formato que el MCP de typesearch (``salidaDeBusqueda`` en
typesearch-api/lib/api/mcp-contrato.ts): un agente ve lo mismo por el MCP y por este paquete."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from typesearch.types import Result, SearchResponse

_WHERE = {
    "homepage": "from the homepage",
    "section": "from a section page",
    "site_search": "from the site's search",
    "discovery": "found beyond the index",
}


def compact(d: dict[str, Any]) -> dict[str, Any]:
    """Sin None, sin cadenas vacías y sin listas vacías: lo que no dice nada no gasta tokens."""
    return {k: v for k, v in d.items() if v is not None and v != "" and not (isinstance(v, list) and len(v) == 0)}


def shorten(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    cut = clean.rfind(" ", 0, limit)
    return f"{clean[: cut if cut > limit * 0.6 else limit - 1]}…"


def to_minute(iso: str | None) -> str | None:
    """ISO al minuto, en UTC: ``2026-09-25T14:05Z``."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def round2(x: float) -> float:
    """Como ``Math.round(x * 100) / 100`` en JavaScript (el .5 sube), para que dé lo mismo que el MCP."""
    return math.floor(x * 100 + 0.5) / 100


def compact_result(r: Result) -> dict[str, Any]:
    found_in = getattr(r, "found_in", None)
    return compact(
        {
            "title": r.title,
            "url": r.url,
            "source": r.source,
            "published_at": to_minute(r.published_at),
            "country": r.country,
            "language": r.language,
            "snippet": shorten(r.snippet, 300) if r.snippet else None,
            "highlights": list(r.highlights or []),
            "score": round2(r.score),
            "found_in": None if found_in in (None, "index") else found_in,
        }
    )


def news_search_output(res: SearchResponse, query: str) -> dict[str, Any]:
    """Los datos compactos de una búsqueda. ``results`` va siempre, aunque esté vacía."""
    results = [compact_result(r) for r in res.results or []]
    rest = compact(
        {
            "near_misses": [compact_result(r) for r in (res.near_misses or [])[:5]] if not results else [],
            "incomplete": True if res.incomplete else None,
            "cached": True if res.cached_at else None,
            "cost_usd": (res.usage.cost_usd if res.usage and res.usage.cost_usd is not None else 0),
            "warnings": [{"code": w.code, "message": w.message} for w in res.warnings or []],
            "request_id": res.id,
        }
    )
    return {"query": query, "mode": res.mode, "results": results, **rest}


def _result_lines(r: dict[str, Any], i: int) -> list[str]:
    place = "/".join(x for x in (r.get("country"), r.get("language")) if x)
    when = r["published_at"].replace("T", " ").replace("Z", " UTC") if r.get("published_at") else None
    meta = " · ".join(x for x in (r.get("source"), when, place or None, _WHERE.get(r.get("found_in") or "")) if x)
    lines = [f"{i + 1}. {r['title']}"]
    if meta:
        lines.append(meta)
    lines.append(r["url"])
    if r.get("snippet"):
        lines.append(r["snippet"])
    lines += [f"> {h}" for h in r.get("highlights", [])]
    return lines


def news_search_text(s: dict[str, Any]) -> str:
    """El texto que lee el modelo."""
    n = len(s["results"])
    head = f"{n} result{'' if n == 1 else 's'}" if n else "No results"
    cost = "cached, free" if s.get("cached") else f"US${s['cost_usd']:.4f}"
    parts = [f'{head} for "{s["query"]}" · {s["mode"]} · {cost}']
    if s["results"]:
        parts += ["\n".join(_result_lines(r, i)) for i, r in enumerate(s["results"])]
    elif s.get("near_misses"):
        parts.append("Closest articles, which may not be about it:")
        parts += ["\n".join(_result_lines(r, i)) for i, r in enumerate(s["near_misses"])]
    if s.get("incomplete"):
        parts.append("Incomplete: the time or token budget ran out; repeating the search in a minute may bring more.")
    parts += [f"Note ({w['code']}): {w['message']}" for w in s.get("warnings", [])]
    return "\n\n".join(parts)
