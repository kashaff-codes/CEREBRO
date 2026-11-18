"""
Stance judged by an LLM reading each retrieved passage.

This is the natural-language-inference task (does the passage entail or
contradict the claim) posed to a general model instead of RoBERTa-MNLI, which
does not fit in the free-tier host's memory. The constraint that keeps it an
evidence-based verifier rather than "ask the model if it's fake":

  * the model sees ONLY the claim and numbered passages that were retrieved;
  * it is told to judge from the passage text, not from what it knows;
  * it returns one judgement per passage, and the verdict is still computed by
    ClaimVerifier.assess() from those judgements, source reliability and rank.

So the score stays auditable: every point of it traces to a passage with a URL.
If no real model answers (only the template provider is left), this returns
None and the caller falls back to the NLI or lexical model.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from services.ml.rag.retrieval import ScoredDocument
from services.ml.rag.verify import Stance

log = logging.getLogger(__name__)

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "judgements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "passage": {"type": "integer"},
                    "stance": {"type": "string", "enum": ["supports", "refutes", "unrelated"]},
                    "confidence": {"type": "number"},
                },
                "required": ["passage", "stance", "confidence"],
            },
        }
    },
    "required": ["judgements"],
}

SYSTEM = (
    "You are a natural-language-inference judge for a fact-checking system. "
    "For each numbered passage, decide whether the passage, taken as true, "
    "SUPPORTS the claim, REFUTES it, or is UNRELATED (does not settle it either "
    "way). Judge only from what the passage says, never from your own knowledge: "
    "a passage that merely mentions the same topic is unrelated. A passage that "
    "debunks a different claim on the same topic is unrelated to this one. "
    "A headline counts only for what it actually states. Confidence is 0 to 1."
)

_MAP = {"supports": Stance.ENTAIL, "refutes": Stance.CONTRADICT, "unrelated": Stance.NEUTRAL}


async def judge(
    provider: Any, claim: str, hits: Sequence[ScoredDocument],
) -> tuple[list[tuple[Stance, float]], str] | None:
    """One call per claim. Returns (stances aligned with hits, model name) or None."""
    if not hits:
        return [], "none"
    passages = "\n\n".join(
        f"[{i}] {h.document.title}\n{h.document.text[:900]}"
        for i, h in enumerate(hits, start=1)
    )
    prompt = f"Claim: {claim}\n\nPassages:\n\n{passages}"
    try:
        resp = await provider.complete(
            prompt, system=SYSTEM, max_tokens=700, temperature=0.0, json_schema=SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "use the fallback"
        log.warning("LLM stance judge failed: %s", exc)
        return None
    if getattr(resp, "provider", "template") == "template":
        return None
    try:
        rows = json.loads(resp.text).get("judgements", [])
    except (json.JSONDecodeError, AttributeError):
        log.warning("LLM stance judge returned non-JSON")
        return None

    out: list[tuple[Stance, float]] = [(Stance.NEUTRAL, 0.5)] * len(hits)
    for row in rows:
        try:
            i = int(row["passage"]) - 1
            if 0 <= i < len(hits):
                conf = max(0.0, min(1.0, float(row["confidence"])))
                out[i] = (_MAP.get(str(row["stance"]).lower(), Stance.NEUTRAL), conf)
        except (KeyError, TypeError, ValueError):
            continue
    return out, f"llm-nli:{resp.provider}:{resp.model}"
