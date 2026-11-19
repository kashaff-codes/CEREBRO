"""
Misinformation analysis route.

This wires the RAG verification pipeline (services/ml/rag) to the frontend's
News scanner. The contract is the one the whole architecture is built around:

    a credibility number is DERIVED FROM retrieved evidence, or it is not
    produced at all.

v1's /api/analyze-news asked Gemini for a `credibilityScore` and a `sources[]`
array in a single call with no retrieval — the "sources" were invented, and the
server schema literally called them "simulated references". Here:

  1. the article is segmented into checkable claims (the one legitimate LLM job,
     with a sentence-split fallback when no model is available);
  2. evidence is gathered from live web research (Claude's web search, which
     also normalises messy input into clean claims and quotes its sources),
     Wikipedia, the tenant's corpus and published fact-checks;
  3. each passage's stance toward each claim is judged (an LLM reading the
     passage, the NLI model, or the lexical fallback);
  4. the credibility score is a weighted mean of named components, returned
     with the score so the UI can show exactly where the number came from;
  5. every source returned is a URL of a document that was actually retrieved.

If the corpus has nothing relevant, the honest answer is a low-confidence
"insufficient evidence" — not a confident fabrication. The `sources` array can
therefore be empty, and that is a feature.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator

from services.api.core.config import settings
from services.api.core.db import db, emit_detection
from services.api.core.deps import Principal, current_principal
from services.ml.rag.retrieval import Document
from services.ml.rag import llm_stance, web_evidence, web_research
from services.ml.rag.retrieval import ScoredDocument
from services.ml.rag.verify import ClaimVerdict, Label, build_default_verifier, relevance

log = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["news"])

MAX_TEXT_BYTES = 100 * 1024


class NewsAnalyzeRequest(BaseModel):
    text: str = Field(..., description="Article text, headline, questions, or statement to verify.")
    depth: Literal["standard", "deep"] = Field(
        "standard", description="'deep' searches more widely and returns more coverage.")

    @field_validator("text")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        if len(v.encode("utf-8")) > MAX_TEXT_BYTES:
            raise ValueError(f"text exceeds {MAX_TEXT_BYTES} bytes")
        return v


# Response mirrors the frontend NewsAnalysisResult shape exactly.
class NewsAnalyzeResponse(BaseModel):
    credibilityScore: int = Field(..., ge=0, le=100)
    verdict: str
    summary: str
    reasoning: str
    sources: list[str]
    # Extra fields the v1 UI ignored but that make this auditable. Additive, so
    # the existing client keeps working while new clients can show provenance.
    claims_checked: int
    model_versions: dict[str, str]
    score_source: str = "evidence"
    # How the score was built, so the UI can show its working.
    confidence: float = 0.0
    confidence_label: str = "Style only"
    breakdown: list[dict] = []
    claims: list[dict] = []
    evidence_sources: dict[str, str] = {}
    related: list[dict[str, str]] = []
    depth: str = "standard"
    research_usage: dict[str, int] = {}


async def _load_corpus(tenant_id: str) -> list[Document]:
    """
    The evidence corpus for this tenant: documents of kind 'evidence' or
    'article', joined to their source credibility weight where we know it.

    With no database, the curated seed corpus is served from memory — the
    documents a fresh deploy would have retrieved anyway — rather than failing
    the whole fact-check with a 500.
    """
    if not db.is_connected:
        from services.api.core.seed_corpus import offline_corpus
        log.warning("database unavailable: retrieving from the in-memory seed corpus")
        return offline_corpus()
    rows = await db.fetch(
        tenant_id,
        """
        SELECT d.id, d.title, d.body, d.source_url,
               COALESCE(es.credibility_weight, 0.5) AS credibility,
               es.domain AS source_domain
        FROM cerebro.documents d
        LEFT JOIN cerebro.evidence_sources es
               ON es.domain = split_part(regexp_replace(d.source_url, '^https?://(www\\.)?', ''), '/', 1)
        WHERE d.kind IN ('evidence', 'article')
        ORDER BY d.ingested_at DESC
        LIMIT 5000
        """,
    )
    corpus: list[Document] = []
    for r in rows:
        corpus.append(
            Document(
                id=str(r["id"]),
                title=r["title"] or "",
                text=r["body"] or "",
                url=r["source_url"],
                domain=r["source_domain"],
                credibility=float(r["credibility"]),
            )
        )
    return corpus


def _compose(
    verdicts: list[ClaimVerdict],
    worth: list[float],
    fact_check: tuple[int, str, str, list[str]] | None,
    text: str,
) -> dict:
    """
    Fold everything that was measured into one 0-100 score, and keep the parts.

    The score is a weighted mean of components, and every component is returned
    with its own score, weight and one-line reason, so the number on screen can
    be taken apart. The old version averaged per-claim labels, and every label
    that missed a confidence threshold counted as exactly 0.5 — which is why
    nearly every article scored 50.

      evidence     how strongly the retrieved passages support or refute each
                   claim: P(true) = (support + 0.5) / (support + refute + 1),
                   averaged over claims by check-worthiness x certainty.
                   Weight grows with how much evidence there is.
      fact_check   ratings from published fact-checks (Google Fact Check
                   Tools), when any exist. The strongest single signal.
      language     sensationalism, conspiracy markers, attribution. Style, not
                   truth, so it only ever carries a small weight.
      prior        50, the honest answer with nothing to go on. Its weight
                   shrinks to zero as real evidence arrives, so thin evidence
                   reads as "uncertain" rather than as a confident verdict.
    """
    comps: list[dict] = []

    # Evidence -----------------------------------------------------------------
    w_e = 0.0
    if verdicts:
        cw = [max(0.1, min(1.0, x)) for x in worth]
        mass = [c * v.certainty for c, v in zip(cw, verdicts)]
        coverage = sum(1 for v in verdicts if v.evidence)
        sup = sum(1 for v in verdicts for e in v.evidence if e.stance.value == "entail")
        ref = sum(1 for v in verdicts for e in v.evidence if e.stance.value == "contradict")
        domains = {e.domain for v in verdicts for e in v.evidence if e.domain}
        if sum(mass) > 0:
            ev = 100 * sum(m * v.probability for m, v in zip(mass, verdicts)) / sum(mass)
            certainty = sum(c * v.certainty for c, v in zip(cw, verdicts)) / sum(cw)
            w_e = 1.5 * certainty
            comps.append({
                "key": "evidence", "label": "Evidence agreement",
                "score": round(ev, 1), "weight": round(w_e, 3),
                "detail": (f"{coverage} of {len(verdicts)} claim(s) had relevant evidence: "
                           f"{sup} passage(s) support, {ref} refute, from "
                           f"{len(domains)} source(s)."),
            })
        else:
            comps.append({
                "key": "evidence", "label": "Evidence agreement",
                "score": None, "weight": 0.0,
                "detail": (f"Searched the evidence corpus and the live web for "
                           f"{len(verdicts)} claim(s); {coverage} had related passages, "
                           "none of which took a side."),
            })

    # Published fact-checks ----------------------------------------------------
    w_f = 0.0
    if fact_check:
        fc_score, _fc_verdict, fc_reason, fc_sources = fact_check
        w_f = 1.5 * min(1.0, max(1, len(fc_sources)) / 2)
        comps.append({
            "key": "fact_check", "label": "Published fact-checks",
            "score": float(fc_score), "weight": round(w_f, 3),
            "detail": fc_reason.split(" — ", 1)[-1][:240],
        })

    # Language -----------------------------------------------------------------
    lang_score, _lv, lang_reason = _heuristic_credibility(text)
    w_l = 0.2
    signals = lang_reason.split("Signals — ", 1)[-1] if "Signals — " in lang_reason \
        else "No strong linguistic signals either way."
    comps.append({
        "key": "language", "label": "Language signals",
        "score": float(lang_score), "weight": w_l, "detail": signals,
    })

    # Uncertainty prior --------------------------------------------------------
    w_p = 0.35 * max(0.0, 1.0 - min(1.0, w_e + w_f))
    if w_p > 0:
        comps.append({
            "key": "prior", "label": "Uncertainty prior",
            "score": 50.0, "weight": round(w_p, 3),
            "detail": "Pulls the score toward 50 in proportion to how little evidence was found.",
        })

    total = sum(c["weight"] for c in comps if c["score"] is not None)
    score = sum(c["score"] * c["weight"] for c in comps if c["score"] is not None) / total
    # Decide the verdict on the number the reader sees. Thresholds on the raw
    # value let 64.6 display as "65" yet fall under the 65 line for Credible.
    score = float(max(0, min(100, int(round(score)))))
    for c in comps:
        c["share"] = round(c["weight"] / total, 3) if c["score"] is not None else 0.0

    confidence = (w_e + w_f) / (w_e + w_f + 0.6)
    # Stance from word overlap alone (no NLI model, no LLM answered) can lean,
    # but it cannot verify: "eiffel tower is in london right" overlaps a page
    # about a London tower. Such a run never says Credible or Fake News, only
    # how it leans, unless a published fact-check is behind it.
    weak = (not fact_check and any(
        str(v.model_versions.get("stance", "")).startswith("lexical") for v in verdicts if v.evidence))
    if weak:
        confidence = min(confidence, 0.3)
        for c in comps:
            if c["key"] == "evidence" and c["score"] is not None:
                c["detail"] += " Stance by word overlap only (fallback mode): indicative, not verified."
    conf_label = ("High" if confidence >= 0.6 else "Medium" if confidence >= 0.35
                  else "Low" if confidence >= 0.05 else "Style only")

    both_sides = any(v.support_weight > 0 and v.refute_weight > 0 for v in verdicts)
    if confidence < 0.2 or weak:
        verdict = "Unverified"
    elif score >= 65:
        verdict = "Credible"
    elif score <= 30:
        verdict = "Fake News"
    elif score < 45:
        verdict = "Misleading"
    else:
        verdict = "Disputed" if both_sides else "Unverified"

    supported = sum(1 for v in verdicts if v.label is Label.SUPPORTED)
    refuted = sum(1 for v in verdicts if v.label is Label.REFUTED)
    reasoning = (
        f"Checked {len(verdicts)} claim(s): {supported} supported and {refuted} refuted by "
        f"the evidence. Score {round(score)} is the weighted mean of "
        + ", ".join(f"{c['label'].lower()} {c['score']:.0f} (weight {c['share']:.0%})"
                    for c in comps if c["score"] is not None)
        + f". Confidence: {conf_label.lower()}."
    )
    return {
        "score": max(0, min(100, int(round(score)))),
        "verdict": verdict,
        "reasoning": reasoning,
        "confidence": round(confidence, 3),
        "confidence_label": conf_label,
        "breakdown": comps,
    }


# --- linguistic-credibility fallback (used only when the corpus has no evidence) ---

_SENSATIONAL = (
    "shocking", "unbelievable", "you won't believe", "miracle", "secret",
    "exposed", "they don't want you to know", "wake up", "banned", "cover-up",
    "cover up", "guaranteed", "must see", "the truth about", "mainstream media",
    "big pharma", "what they aren't telling", "no one is talking about",
)
_CONSPIRACY = (
    "microchip", "5g causes", "flat earth", "chemtrail", "illuminati",
    "new world order", "depopulation", "plandemic", "vaccine causes autism",
    "faked the", "staged", "crisis actor",
)
_DEATH_RUMOR = ("is dead", "has died", "found dead", "assassinated", "passed away")
_CREDIBLE_SIGNALS = (
    "according to", "reported", "study", "research", "university", "official",
    "announced", "confirmed", "spokesperson", "data shows", "published in",
    "peer-reviewed", "statement", "court", "report found",
)


def _heuristic_credibility(text: str) -> tuple[int, str, str]:
    """
    A transparent linguistic-credibility estimate for when the evidence corpus
    has nothing relevant. NOT evidence-based verification — it reads writing-style
    signals (sensationalism, conspiracy markers, sourcing/attribution) the way the
    email module reads header signals, and is labeled score_source='heuristic'.
    """
    t = text.lower()
    score = 55
    notes: list[str] = []

    sens = sum(1 for w in _SENSATIONAL if w in t)
    if sens:
        score -= min(30, sens * 10)
        notes.append(f"{sens} sensational/clickbait phrase(s)")
    consp = sum(1 for w in _CONSPIRACY if w in t)
    if consp:
        score -= min(38, consp * 19)
        notes.append(f"{consp} known-conspiracy marker(s)")
    if any(w in t for w in _DEATH_RUMOR):
        score -= 16
        notes.append("unverified death claim (a common misinformation pattern)")
    cred = sum(1 for w in _CREDIBLE_SIGNALS if w in t)
    if cred:
        score += min(24, cred * 8)
        notes.append(f"{cred} attribution/sourcing signal(s)")

    exclam = text.count("!")
    if exclam >= 2:
        score -= min(15, exclam * 4)
        notes.append("excessive exclamation")
    letters = [c for c in text if c.isalpha()]
    if letters and len(letters) > 20:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > 0.3:
            score -= 12
            notes.append("heavy capitalization")
    if re.search(r"\b(19|20)\d{2}\b", text) or re.search(r"\b\d+(\.\d+)?%\b", text):
        score += 6
        notes.append("contains specific figures/dates")

    score = max(5, min(90, score))
    if score < 40:
        verdict = "Fake News" if score < 25 else "Misleading"
    elif score >= 65:
        verdict = "Credible"
    else:
        verdict = "Unverified"

    reasoning = (
        "No matching documents in the evidence corpus, so this is a transparent "
        "linguistic-credibility estimate (heuristic), not evidence-based verification. "
        + ("Signals — " + "; ".join(notes) + "." if notes else
           "No strong linguistic signals either way.")
    )
    return score, verdict, reasoning


async def _fact_check(text: str) -> tuple[int, str, str, list[str]] | None:
    """
    Query the Google Fact Check Tools API (free key) for real fact-checks of the
    claim. Returns (score, verdict, reasoning, source_urls) — where the score is
    derived from what actual fact-checkers rated it, and the sources are their
    article URLs so a reader can verify for themselves. None if no key or no hit.
    """
    key = settings.fact_check_api_key
    if not key:
        return None
    import httpx

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={"query": text[:300], "key": key, "languageCode": "en"},
            )
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        log.warning("fact-check lookup failed: %s", exc)
        return None

    reviews: list[dict[str, str]] = []
    for c in data.get("claims", [])[:6]:
        for rev in (c.get("claimReview") or [])[:2]:
            reviews.append({
                "publisher": (rev.get("publisher") or {}).get("name", "fact-checker"),
                "rating": rev.get("textualRating", ""),
                "url": rev.get("url", ""),
            })
    if not reviews:
        return None

    vals: list[int] = []
    for rev in reviews:
        rt = rev["rating"].lower()
        if "pants on fire" in rt or "fabricated" in rt or (
            "false" in rt and "mostly true" not in rt and "not false" not in rt):
            vals.append(12)
        elif any(k in rt for k in ("mostly false", "misleading", "mixture",
                                    "half true", "partly", "distorts", "exaggerat")):
            vals.append(40)
        elif any(k in rt for k in ("mostly true", "accurate", "correct")) or rt.strip() == "true":
            vals.append(85)
        elif any(k in rt for k in ("unproven", "unverified", "no evidence", "research")):
            vals.append(50)
        else:
            vals.append(45)
    score = int(round(sum(vals) / len(vals)))
    verdict = ("Fake News" if score < 25 else "Misleading" if score < 45
               else "Credible" if score >= 65 else "Unverified")
    reasoning = (
        "Verified against published fact-checks — "
        + "; ".join(f"{rev['publisher']} rated it '{rev['rating']}'"
                    for rev in reviews[:4])
        + ". Open the linked sources to read the full checks and decide for yourself."
    )
    sources = [rev["url"] for rev in reviews if rev["url"]][:8]
    return score, verdict, reasoning, sources


_NARRATE_SYSTEM = (
    "You are a misinformation analyst explaining a verdict that has ALREADY been "
    "decided by retrieval and an entailment model. Your job is to explain WHY, in "
    "plain language, so a non-expert understands the reasoning. Rules: never change "
    "the verdict or the score; never invent evidence, sources, or statistics beyond "
    "what you are given; if the evidence is thin, say plainly that it is thin and "
    "that the verdict is low-confidence. Where a claim is false or unsupported, "
    "explain what specifically fails — the mechanism, the missing evidence, or the "
    "contradiction with the cited sources. Write 3-5 sentences, no bullet points, "
    "no preamble."
)


async def _narrate(
    request: Request,
    reasoning: str,
    verdict: str,
    score: int,
    score_source: str,
    verdicts: list[ClaimVerdict],
) -> str:
    """
    Ask the LLM to expand the deterministic `reasoning` line into an analyst
    explanation. Returns the original reasoning unchanged on any failure, or when
    only the template provider answered — a templated restatement adds nothing.
    """
    if os.getenv("NEWS_LLM_REASONING", "1").lower() in ("0", "false", "no"):
        return reasoning

    lines: list[str] = []
    for v in verdicts[:6]:
        lines.append(f"- claim: {v.claim[:300]}")
        lines.append(f"  model verdict: {v.label.value} (confidence {v.confidence:.2f})")
        for e in v.evidence[:3]:
            lines.append(
                f"    evidence [{e.stance.value}] {e.title[:120]} "
                f"({e.domain or 'unknown source'}): {e.snippet[:240]}"
            )
    evidence_block = "\n".join(lines) or "- no retrieved evidence"

    origin = {
        "fact_check": "a published fact-check from a professional fact-checking organisation",
        "rag_evidence": "retrieval over the evidence corpus plus an entailment (NLI) model",
        "heuristic": "a linguistic heuristic only — NO external evidence was found, so "
                     "confidence is low and you must say so",
    }.get(score_source, score_source)

    prompt = (
        f"Overall verdict: {verdict} (credibility score {score}/100).\n"
        f"How that verdict was reached: {origin}.\n"
        f"System's one-line reasoning: {reasoning}\n\n"
        f"Per-claim findings:\n{evidence_block}\n\n"
        "Explain to the reader why this verdict was reached."
    )

    try:
        resp = await request.app.state.llm.complete(
            prompt, system=_NARRATE_SYSTEM, max_tokens=500, temperature=0.3
        )
        if resp.provider == "template" or not resp.text.strip():
            return reasoning
        return resp.text.strip()
    except Exception as exc:  # noqa: BLE001 - prose is optional, the verdict is not
        log.warning("news reasoning narration skipped: %s", exc)
        return reasoning


MAX_CLAIMS = 5


async def _claims(request: Request, text: str) -> list[tuple[str, float]]:
    """The article's most check-worthy claims, with their check-worthiness."""
    try:
        from services.ml.providers.llm import extract_claims
        extracted = await extract_claims(request.app.state.llm, text)
        rows = [(c["text"].strip(), float(c.get("check_worthy", 0.5)))
                for c in extracted if c.get("text", "").strip()]
    except Exception as exc:  # noqa: BLE001
        log.warning("claim extraction fell back to sentence split: %s", exc)
        from services.ml.providers.llm import split_sentences
        rows = [(s, 0.5) for s in split_sentences(text)]
    rows.sort(key=lambda r: -r[1])
    return rows[:MAX_CLAIMS] or [(text.strip()[:500], 1.0)]


def _host(url: str | None) -> str | None:
    from urllib.parse import urlparse
    try:
        h = urlparse(url or "").hostname
    except ValueError:
        return None
    return h.removeprefix("www.") if h else None


def _cited_hits(rc: web_research.ClaimResearch) -> list[ScoredDocument]:
    """The passages the research step quoted for this claim, as evidence."""
    hits: list[ScoredDocument] = []
    per_url: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for i, c in enumerate(rc.citations):
        key = (c.url, c.quote[:80])
        if not c.quote or key in seen or per_url.get(c.url, 0) >= 2:
            continue
        seen.add(key)
        per_url[c.url] = per_url.get(c.url, 0) + 1
        host = _host(c.url)
        hits.append(ScoredDocument(
            document=Document(
                id=f"cite:{i}:{c.url[:120]}", title=c.title or (host or "source"),
                text=c.quote, url=c.url, domain=host,
                credibility=web_evidence.reliability(host),
                meta={"source": "web_search"},
            ),
            score=1.0,
        ))
    return hits[:6]


def _topics(claim: str, docs: list[Document]) -> list[dict]:
    """Encyclopedia cards for a claim: what the reader can explore, with a picture."""
    out = []
    for d in docs:
        if d.meta.get("source") != "wikipedia":
            continue
        topic = d.meta.get("topic") or d.title
        if relevance(claim, f"{topic} {d.text[:600]}") < 0.2:
            continue
        out.append({"title": topic, "url": d.url, "image": d.meta.get("image"),
                    "summary": d.text[:260]})
        if len(out) == 3:
            break
    return out


@router.post("/news", response_model=NewsAnalyzeResponse, summary="Research and verify the claims in a text")
async def analyze_news(
    req: NewsAnalyzeRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
) -> NewsAnalyzeResponse:
    llm = request.app.state.llm

    # 1. Research (normalise the input into claims, search the live web, cite),
    #    alongside the tenant corpus and published fact-checks.
    rs, corpus, fc = await asyncio.gather(
        web_research.research(req.text, depth=req.depth, max_claims=MAX_CLAIMS),
        _load_corpus(principal.tenant_id),
        _fact_check(req.text),
        return_exceptions=True,
    )
    if isinstance(corpus, BaseException):
        raise corpus
    if isinstance(rs, BaseException):
        rs = None
    if isinstance(fc, BaseException):
        fc = None

    researched = list(rs.claims) if rs and rs.claims else []
    if researched:
        claims = [(c.claim, 1.0) for c in researched]
    else:
        claims = await _claims(request, req.text)
    texts = [c for c, _ in claims]

    # 2. Encyclopedia background for the cleaned-up claims (and news headlines
    #    when the research step was not available to find coverage itself).
    try:
        web_docs, source_status = await web_evidence.gather(
            texts, req.text, include_news=not researched)
    except Exception as exc:  # noqa: BLE001
        log.warning("web evidence failed: %s", exc)
        web_docs, source_status = [], {"web": "unavailable"}
    source_status["web_search"] = (
        f"{rs.searches} searches" if rs else
        ("unavailable" if web_research.available() else "not configured"))
    source_status["corpus"] = f"{len(corpus)} documents"
    source_status["fact_check"] = "ok" if fc else (
        "no published checks" if settings.fact_check_api_key else "not configured")

    verifier = build_default_verifier(list(corpus) + list(web_docs), prefer_transformers=True)
    by_claim = {c.claim: c for c in researched}
    hits: list[list[ScoredDocument]] = []
    for t in texts:
        cited = _cited_hits(by_claim[t]) if t in by_claim else []
        urls = {h.document.url for h in cited}
        extra = [h for h in verifier.candidates(t) if h.document.url not in urls]
        hits.append((cited + extra)[:8])

    # 3. Stance: an LLM reading each passage when one answers; otherwise the NLI
    #    model when installed, or the lexical fallback. Same arithmetic after.
    judged = await asyncio.gather(*(llm_stance.judge(llm, t, h) for t, h in zip(texts, hits)))
    verdicts: list[ClaimVerdict] = []
    for t, h, j in zip(texts, hits, judged):
        if j is None:
            stances = [verifier.stance_model.predict(t, x.document.searchable) for x in h]
            verdicts.append(verifier.assess(t, h, stances))
        else:
            verdicts.append(verifier.assess(t, h, j[0], stance_name=j[1]))

    result = _compose(verdicts, [w for _, w in claims], fc, req.text)
    score, verdict, reasoning = result["score"], result["verdict"], result["reasoning"]

    # Sources: published fact-checks first, then passages that took a side, then
    # the rest of what was judged. Never a document that was not retrieved.
    sources: list[str] = list(fc[3]) if fc else []
    ranked = sorted(
        (e for v in verdicts for e in v.evidence),
        key=lambda e: (e.stance.value == "neutral", -e.credibility),
    )
    for e in ranked:
        if e.url and e.url not in sources:
            sources.append(e.url)

    weights = {c["key"]: c["weight"] for c in result["breakdown"] if c["score"] is not None}
    score_source = max(weights, key=lambda k: weights[k]) if weights else "language"

    summary = req.text.strip().replace("\n", " ")
    summary = (summary[:280] + "…") if len(summary) > 280 else summary

    # Analyst-facing "why". The verdict and score above are already final — this
    # only narrates them in plain language. Best-effort.
    reasoning = await _narrate(request, reasoning, verdict, score, score_source, verdicts)

    model_versions: dict[str, str] = {}
    for v in verdicts:
        model_versions.update(v.model_versions)
    if rs:
        model_versions["research"] = f"web_search:{rs.model}"

    claim_rows = []
    for (text, worth), v in zip(claims, verdicts):
        rc = by_claim.get(text)
        claim_rows.append({
            "text": text,
            "check_worthy": round(worth, 2),
            "label": v.label.value,
            "probability": round(v.probability, 3),
            "certainty": round(v.certainty, 3),
            "support": round(v.support_weight, 3),
            "refute": round(v.refute_weight, 3),
            "finding": rc.finding if rc else "",
            "explore": rc.explore if rc else [],
            "topics": _topics(text, list(web_docs)),
            "evidence": [
                {"title": e.title, "url": e.url, "domain": e.domain,
                 "stance": e.stance.value, "confidence": round(e.nli_score, 3),
                 "reliability": round(e.credibility, 2), "snippet": e.snippet[:280]}
                for e in v.evidence
            ],
        })

    # Persist so metrics populate and the data is collectable. Risk is the
    # inverse of credibility (a low-credibility claim is a high-risk detection).
    try:
        await emit_detection(
            principal.tenant_id, "news", verdict,
            round((100 - score) / 100, 4), summary=summary[:180],
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("news persistence skipped: %s", exc)

    return NewsAnalyzeResponse(
        credibilityScore=score,
        verdict=verdict,
        summary=summary,
        reasoning=reasoning,
        sources=sources[:12],
        claims_checked=len(verdicts),
        model_versions=model_versions,
        score_source=score_source,
        confidence=result["confidence"],
        confidence_label=result["confidence_label"],
        breakdown=result["breakdown"],
        claims=claim_rows,
        evidence_sources=source_status,
        related=(rs.related if rs else []),
        depth=req.depth,
        research_usage=(rs.usage if rs else {}),
    )
