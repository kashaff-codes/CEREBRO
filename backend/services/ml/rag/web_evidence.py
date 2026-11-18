"""
Live evidence from the open web, fetched per request.

The curated seed corpus holds 17 documents. Checked against it alone, any claim
outside vaccines, 5G, the Moon landings and a few other myths had nothing to
retrieve, and the score fell to the 50 "unverified" midpoint — the behaviour a
reviewer flagged as "always 50%". This module widens what can be retrieved
without inventing anything: every document it returns is a real page with a
real URL, and goes through the same relevance gate, stance judge and weighting
as the seed corpus.

Sources, both free, keyless, and permitted for this use:

  * Wikipedia (MediaWiki search + intro extracts): encyclopedic background,
    good for settled facts ("Pakistan won the 1992 World Cup").
  * GDELT DOC 2.0: an open index of world news coverage, good for current
    events. It asks for at most one request every five seconds, so calls are
    throttled process-wide and skipped (not queued) when too soon.

Google News RSS was considered and rejected: its terms limit the feed to
personal feed readers.

Failures are silent by design: a source that times out or rate-limits simply
contributes nothing, and the response records which sources answered.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from services.ml.rag.retrieval import Document, tokenize

log = logging.getLogger(__name__)

USER_AGENT = "CEREBRO-factcheck/2.0 (educational fact-checking project)"
TIMEOUT = 6.0

# Source reliability, 0..1. Multiplied into every stance weight, so a passage
# from a wire service counts for more than one from an unknown blog. Kept
# conservative and short: unknown domains get DEFAULT, not zero, because GDELT
# indexes thousands of legitimate local outlets.
RELIABILITY: dict[str, float] = {
    "reuters.com": 0.95, "apnews.com": 0.95, "bbc.co.uk": 0.92, "bbc.com": 0.92,
    "npr.org": 0.9, "theguardian.com": 0.88, "nytimes.com": 0.88,
    "washingtonpost.com": 0.87, "wsj.com": 0.87, "ft.com": 0.88,
    "economist.com": 0.88, "aljazeera.com": 0.82, "dawn.com": 0.82,
    "cnn.com": 0.8, "nbcnews.com": 0.82, "cbsnews.com": 0.82, "abcnews.go.com": 0.82,
    "pbs.org": 0.88, "bloomberg.com": 0.88, "nature.com": 0.95, "science.org": 0.95,
    "who.int": 0.97, "cdc.gov": 0.96, "nasa.gov": 0.97, "nih.gov": 0.95,
    "en.wikipedia.org": 0.8,
    "snopes.com": 0.88, "politifact.com": 0.88, "factcheck.org": 0.9,
    "fullfact.org": 0.9, "afp.com": 0.92,
    # Known low-reliability outlets: counted, but for little.
    "infowars.com": 0.1, "naturalnews.com": 0.1, "breitbart.com": 0.35,
    "dailymail.co.uk": 0.45, "thesun.co.uk": 0.4, "rt.com": 0.2, "sputniknews.com": 0.2,
}
DEFAULT = 0.55


def reliability(domain: str | None) -> float:
    if not domain:
        return DEFAULT
    d = domain.lower().removeprefix("www.")
    while d:
        if d in RELIABILITY:
            return RELIABILITY[d]
        if "." not in d:
            break
        d = d.split(".", 1)[1]
    return DEFAULT


def search_terms(text: str, limit: int = 8) -> str:
    """The claim's content words, in order, de-duplicated: what search engines want."""
    seen: list[str] = []
    for t in tokenize(text):
        if t not in seen and len(t) > 1:
            seen.append(t)
    return " ".join(seen[:limit])


# --- cache --------------------------------------------------------------------

_CACHE: dict[str, tuple[float, list[Document]]] = {}
_CACHE_TTL = 3600.0
_CACHE_MAX = 512


def _cached(key: str) -> list[Document] | None:
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < _CACHE_TTL:
        return hit[1]
    return None


def _store(key: str, docs: list[Document]) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = (time.monotonic(), docs)


# --- Wikipedia ----------------------------------------------------------------

async def wikipedia(client: Any, query: str, n: int = 3) -> list[Document]:
    key = f"wiki:{query}"
    if (c := _cached(key)) is not None:
        return c
    r = await client.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query", "generator": "search", "gsrsearch": query,
            "gsrlimit": n, "prop": "extracts", "exintro": 1, "explaintext": 1,
            "exsentences": 6, "format": "json", "formatversion": 2,
        },
    )
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages") or []
    docs = [
        Document(
            id=f"wiki:{p['pageid']}",
            title=f"Wikipedia: {p.get('title', '')}",
            text=(p.get("extract") or "")[:1800],
            url=f"https://en.wikipedia.org/?curid={p['pageid']}",
            domain="en.wikipedia.org",
            credibility=reliability("en.wikipedia.org"),
            meta={"source": "wikipedia"},
        )
        for p in sorted(pages, key=lambda p: p.get("index", 99))
        if p.get("extract")
    ]
    _store(key, docs)
    return docs


# --- GDELT --------------------------------------------------------------------

_GDELT_LOCK = asyncio.Lock()
_GDELT_LAST = 0.0
_GDELT_GAP = 5.0


async def gdelt(client: Any, query: str, n: int = 10) -> list[Document]:
    global _GDELT_LAST
    key = f"gdelt:{query}"
    if (c := _cached(key)) is not None:
        return c
    async with _GDELT_LOCK:
        if time.monotonic() - _GDELT_LAST < _GDELT_GAP:
            return []  # GDELT's limit; skip rather than queue behind it
        _GDELT_LAST = time.monotonic()
    r = await client.get(
        "https://api.gdeltproject.org/api/v2/doc/doc",
        params={"query": query, "mode": "artlist", "format": "json",
                "maxrecords": n, "sourcelang": "english", "sort": "hybridrel",
                "timespan": "3months"},
    )
    if r.status_code != 200 or not r.text.lstrip().startswith("{"):
        return []  # rate-limit notices come back as plain text
    docs = []
    for i, a in enumerate(r.json().get("articles") or []):
        title = (a.get("title") or "").strip()
        if not title:
            continue
        domain = (a.get("domain") or "").lower()
        docs.append(Document(
            id=f"gdelt:{i}:{a.get('url', '')[:80]}",
            title=title,
            text=f"{title}. Reported by {domain or 'a news outlet'}"
                 + (f" on {a['seendate'][:8]}" if a.get("seendate") else "") + ".",
            url=a.get("url"),
            domain=domain or None,
            credibility=reliability(domain),
            meta={"source": "gdelt", "headline_only": True},
        ))
    _store(key, docs)
    return docs


# --- entry point --------------------------------------------------------------

async def gather(claims: list[str], article: str) -> tuple[list[Document], dict[str, str]]:
    """
    Evidence for a set of claims: Wikipedia per claim, GDELT once for the
    article. Returns the documents and a per-source status for the response.
    """
    import os

    import httpx

    if os.getenv("WEB_EVIDENCE", "1").lower() in ("0", "false", "no"):
        return [], {"web": "disabled"}
    status: dict[str, str] = {}
    docs: list[Document] = []
    seen: set[str] = set()

    async with httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT},
                                 follow_redirects=True) as client:
        wiki_q = [search_terms(c) for c in claims if search_terms(c)]
        tasks = [wikipedia(client, q) for q in dict.fromkeys(wiki_q)]
        news_q = search_terms(article if len(claims) != 1 else claims[0], limit=6)
        tasks.append(gdelt(client, news_q))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    wiki_ok = wiki_fail = 0
    for res in results[:-1]:
        if isinstance(res, BaseException):
            wiki_fail += 1
            log.warning("wikipedia lookup failed: %s", res)
            continue
        wiki_ok += 1
        for d in res:
            if d.id not in seen:
                seen.add(d.id)
                docs.append(d)
    status["wikipedia"] = ("ok" if wiki_ok else "unavailable") if (wiki_ok or wiki_fail) else "not queried"

    news = results[-1]
    if isinstance(news, BaseException):
        log.warning("gdelt lookup failed: %s", news)
        status["gdelt"] = "unavailable"
    else:
        status["gdelt"] = "ok" if news else "no results or rate-limited"
        for d in news:
            if d.id not in seen:
                seen.add(d.id)
                docs.append(d)
    return docs, status
