"""
Web research with Claude's server-side web search: the step that makes the
fact-checker work on what people actually paste in.

Real input is not a clean claim. It is three questions at once, a paragraph
copied from Reddit, a name spelled wrong ("jenson hwang nvidia case"), no
question mark, about something that happened this week. Two jobs follow, and
this module does both in one call:

  1. Normalise. Rewrite the input as atomic, checkable statements: fix the
     spelling, resolve who and what is meant, turn "did X happen" into "X
     happened". Opinions and rhetoric are not claims and are dropped.
  2. Research. Search the live web for each claim and write a short finding
     that says what is actually true, citing sources. When a claim is wrong,
     the finding gives the correction ("the Eiffel Tower is in Paris").

What it does NOT do is decide the verdict. Every sentence of a finding that
rests on a source carries a citation with the quoted passage (`cited_text`),
the page URL and title — those passages are handed to the stance judge as
evidence, weighted by source reliability, exactly like corpus documents. The
prose is shown to the reader; the score is computed from the passages.

Raw HTTP like the other providers in services/ml/providers/llm.py: no SDK is
added to the free-tier image. Reads ANTHROPIC_API_KEY, and RESEARCH_MODEL
falling back to CLAUDE_MODEL, the model the deployment already uses.

Cost, for the record (it is a product decision, not an accident): web search
is billed per search on top of tokens. `standard` caps searches at 4 per
article and `deep` at 10.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

ENDPOINT = "https://api.anthropic.com/v1/messages"
VERSION = "2023-06-01"
MAX_CONTINUATIONS = 3
SEARCHES = {"standard": 4, "deep": 10}

SYSTEM = """You are the research desk of a fact-checking service. Readers paste
in whatever they saw: a headline, a paragraph from Reddit, several questions at
once, with spelling mistakes and no punctuation.

Step 1. Work out the distinct factual claims the reader wants checked. Fix
spelling, resolve names (people, companies, places, events) to their correct
form, and restate questions as the statement being asked about ("did nvidia
get fined" -> "Nvidia was fined"). Drop opinions and rhetoric. At most {max_claims}
claims; merge near-duplicates.

Step 2. Search the web for each claim. Prefer primary sources, wire services and
established outlets; for ongoing stories, prefer the most recent reporting and
say what is still unresolved.

Step 3. Write your answer in exactly this format, one block per claim, and
nothing before the first block:

CLAIM: <the claim, restated cleanly as one sentence>
FINDING: <3-5 sentences: what the sources establish. If the claim is wrong, say
what is actually true. If it is disputed or still developing, say so and say
what is known so far. Cite sources for every factual sentence.>
EXPLORE: <3-5 short follow-up topics a curious reader could look into next,
separated by " | ">

Judge nothing you cannot cite. If the search finds nothing reliable for a
claim, say that plainly in its FINDING."""


@dataclass
class Citation:
    url: str
    title: str
    quote: str


@dataclass
class ClaimResearch:
    claim: str
    finding: str = ""
    explore: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Research:
    claims: list[ClaimResearch]
    related: list[dict[str, str]]          # search results that were not cited
    searches: int = 0
    model: str = ""
    queries: list[str] = field(default_factory=list)


def available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY")) and os.getenv(
        "WEB_RESEARCH", "1").lower() not in ("0", "false", "no")


async def research(text: str, *, depth: str = "standard", max_claims: int = 5,
                   timeout: float = 120.0) -> Research | None:
    """Normalise and research `text`. None when unavailable or on any failure."""
    if not available():
        return None
    import httpx

    model = os.getenv("RESEARCH_MODEL") or os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
    tools = [{"type": "web_search_20260209", "name": "web_search",
              "max_uses": SEARCHES.get(depth, SEARCHES["standard"])}]
    messages: list[dict[str, Any]] = [{"role": "user", "content": text[:12000]}]
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 8000 if depth == "deep" else 5000,
        "system": SYSTEM.format(max_claims=max_claims),
        "tools": tools,
        "messages": messages,
    }
    from services.ml.providers.llm import anthropic_headers
    headers = anthropic_headers(os.environ["ANTHROPIC_API_KEY"], VERSION)

    content: list[dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for _ in range(MAX_CONTINUATIONS + 1):
                r = await client.post(ENDPOINT, json=body, headers=headers)
                r.raise_for_status()
                data = r.json()
                content = data.get("content", [])
                if data.get("stop_reason") != "pause_turn":
                    break
                # The server-side loop paused. Resend the turn as it stands and
                # the API resumes it; no "continue" message is added.
                body["messages"] = messages + [{"role": "assistant", "content": content}]
            if data.get("stop_reason") == "refusal":
                log.warning("web research declined: %s", data.get("stop_details"))
                return None
    except Exception as exc:  # noqa: BLE001 - research is an enhancement, never fatal
        log.warning("web research failed: %s", exc)
        return None

    return parse(content, model=data.get("model", model))


_HEAD = re.compile(r"^\s*(CLAIM|FINDING|EXPLORE)\s*:\s*", re.MULTILINE)


def parse(content: list[dict[str, Any]], *, model: str = "") -> Research:
    """
    Split the answer into claims, and attach each citation to the claim whose
    text it was cited in. Text blocks arrive in order, each with its own
    citations, so a block belongs to the claim whose span contains its start.
    """
    text = ""
    spans: list[tuple[int, list[dict[str, Any]]]] = []
    results: list[dict[str, str]] = []
    queries: list[str] = []
    searches = 0
    after_tool = False
    for b in content:
        t = b.get("type")
        if t == "text":
            # Text resuming after a search is a new paragraph. Blocks are joined
            # without separators, so without this a preamble ("I'll research
            # these.") runs into "CLAIM:" and the first claim is lost.
            if after_tool and text and not text.endswith("\n"):
                text += "\n"
            after_tool = False
            spans.append((len(text), b.get("citations") or []))
            text += b.get("text", "")
            continue
        after_tool = True
        if t == "server_tool_use" and b.get("name") == "web_search":
            searches += 1
            q = (b.get("input") or {}).get("query")
            if q:
                queries.append(q)
        elif t == "web_search_tool_result":
            res = b.get("content")
            if isinstance(res, list):  # an error comes back as an object, not a list
                for item in res:
                    if item.get("type") == "web_search_result" and item.get("url"):
                        results.append({"url": item["url"], "title": item.get("title", ""),
                                        "page_age": item.get("page_age") or ""})

    starts = [m.start() for m in re.finditer(r"^\s*CLAIM\s*:", text, re.MULTILINE)]
    claims: list[ClaimResearch] = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(text)
        claims.append(_fields(text[s:e]))
        for off, cits in spans:
            if s <= off < e:
                for c in cits:
                    if c.get("type") == "web_search_result_location" and c.get("url"):
                        claims[-1].citations.append(Citation(
                            url=c["url"], title=c.get("title", ""),
                            quote=(c.get("cited_text") or "").strip()))

    cited = {c.url for cl in claims for c in cl.citations}
    seen: set[str] = set()
    related = []
    for r in results:
        if r["url"] not in cited and r["url"] not in seen:
            seen.add(r["url"])
            related.append(r)
    return Research(claims=[c for c in claims if c.claim], related=related[:12],
                    searches=searches, model=model, queries=queries)


def _fields(block: str) -> ClaimResearch:
    parts: dict[str, str] = {}
    marks = list(_HEAD.finditer(block))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(block)
        parts[m.group(1)] = block[m.end():end].strip()
    explore = [x.strip(" -•\t") for x in parts.get("EXPLORE", "").split("|")]
    return ClaimResearch(
        claim=" ".join(parts.get("CLAIM", "").split()),
        finding=" ".join(parts.get("FINDING", "").split()),
        explore=[x for x in explore if x][:5],
    )
