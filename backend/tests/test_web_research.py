"""
The research step: messy input in, clean claims with quoted sources out.

The fixture follows the documented Messages API shapes for the server-side web
search tool: `server_tool_use`, `web_search_tool_result` (a list of results on
success, a single error object on failure), and `text` blocks whose
`citations` are `web_search_result_location` entries with `cited_text`.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["WEB_EVIDENCE"] = "0"
os.environ["WEB_RESEARCH"] = "0"

from services.ml.rag import web_research  # noqa: E402
from services.ml.rag.web_research import parse  # noqa: E402

EIFFEL_URL = "https://en.wikipedia.org/wiki/Eiffel_Tower"
NVDA_URL = "https://www.reuters.com/technology/nvidia-example"


def _cite(url, title, quote):
    return {"type": "web_search_result_location", "url": url, "title": title,
            "encrypted_index": "x", "cited_text": quote}


CONTENT = [
    {"type": "text", "text": "I'll research each of these."},
    {"type": "server_tool_use", "id": "s1", "name": "web_search",
     "input": {"query": "Eiffel Tower location"}},
    {"type": "web_search_tool_result", "tool_use_id": "s1", "content": [
        {"type": "web_search_result", "url": EIFFEL_URL, "title": "Eiffel Tower - Wikipedia",
         "encrypted_content": "e", "page_age": "2026-10-01"},
        {"type": "web_search_result", "url": "https://www.toureiffel.paris/en",
         "title": "Official site", "encrypted_content": "e"},
    ]},
    {"type": "server_tool_use", "id": "s2", "name": "web_search",
     "input": {"query": "Jensen Huang Nvidia lawsuit 2026"}},
    {"type": "web_search_tool_result", "tool_use_id": "s2",
     "content": {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
    {"type": "text", "text": "CLAIM: The Eiffel Tower is in London.\nFINDING: "},
    {"type": "text", "text": "The Eiffel Tower stands on the Champ de Mars in Paris, France.",
     "citations": [_cite(EIFFEL_URL, "Eiffel Tower - Wikipedia",
                         "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.")]},
    {"type": "text", "text": " It is not in London.\nEXPLORE: Champ de Mars | Gustave Eiffel | Trocadéro\n\n"
                             "CLAIM: Jensen Huang is being sued over Nvidia's disclosures.\nFINDING: "},
    {"type": "text", "text": "A shareholder suit names Nvidia and its chief executive.",
     "citations": [_cite(NVDA_URL, "Reuters", "shareholders sued Nvidia and CEO Jensen Huang")]},
    {"type": "text", "text": " The case is ongoing.\nEXPLORE: Nvidia earnings | SEC filings"},
]


def test_claims_are_split_and_restated():
    r = parse(CONTENT, model="m")
    assert [c.claim for c in r.claims] == [
        "The Eiffel Tower is in London.",
        "Jensen Huang is being sued over Nvidia's disclosures.",
    ]
    assert "Paris, France" in r.claims[0].finding and "not in London" in r.claims[0].finding


def test_each_citation_goes_to_the_claim_it_was_cited_in():
    r = parse(CONTENT)
    assert [c.url for c in r.claims[0].citations] == [EIFFEL_URL]
    assert [c.url for c in r.claims[1].citations] == [NVDA_URL]
    assert "Champ de Mars in Paris" in r.claims[0].citations[0].quote


def test_explore_topics_are_parsed():
    r = parse(CONTENT)
    assert r.claims[0].explore == ["Champ de Mars", "Gustave Eiffel", "Trocadéro"]


def test_search_errors_do_not_break_parsing_and_uncited_results_become_related():
    r = parse(CONTENT)
    assert r.searches == 2
    assert r.queries == ["Eiffel Tower location", "Jensen Huang Nvidia lawsuit 2026"]
    assert [x["url"] for x in r.related] == ["https://www.toureiffel.paris/en"]


def test_preamble_text_is_not_a_claim():
    r = parse([{"type": "text", "text": "Let me look that up."}])
    assert r.claims == []


def test_research_is_skipped_without_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert asyncio.run(web_research.research("anything")) is None


def test_the_route_uses_researched_claims_and_quotes(monkeypatch):
    """End to end through the API, with the research call stubbed."""
    from fastapi.testclient import TestClient

    from services.api.core.config import settings
    from services.api.core.db import db
    from services.api.main import app

    async def fake_research(text, **kw):
        return parse(CONTENT, model="stub")

    monkeypatch.setattr(web_research, "research", fake_research)
    saved = db._dsn, db._pool
    db._dsn, db._pool = "postgresql://nobody:x@127.0.0.1:1/missing", None
    try:
        with TestClient(app) as c:
            tok = c.post("/v1/auth/login", json={"email": settings.demo_email,
                                                 "password": settings.demo_password}).json()["access_token"]
            r = c.post("/v1/analyze/news", headers={"Authorization": f"Bearer {tok}"},
                       json={"text": "eiffel tower is in london? and is jenson hwang getting sued",
                             "depth": "deep"})
    finally:
        db._dsn, db._pool = saved
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["depth"] == "deep"
    assert [c["text"] for c in body["claims"]][0] == "The Eiffel Tower is in London."
    first = body["claims"][0]
    assert "Paris" in first["finding"]
    assert any(e["url"] == EIFFEL_URL for e in first["evidence"])
    assert EIFFEL_URL in body["sources"]
    assert body["related"][0]["url"] == "https://www.toureiffel.paris/en"
    assert body["model_versions"].get("research") == "web_search:stub"
