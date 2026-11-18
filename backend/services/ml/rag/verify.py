"""
Claim verification: retrieve evidence, judge stance, aggregate to a verdict.

The contract this module enforces, and the reason it exists:

    A verdict may only cite documents that were actually retrieved.

v1 produced a `credibilityScore` and a `sources[]` array from a single Gemini
call with no retrieval step. Here the verdict is *derived from* the evidence —
if retrieval returns nothing, the answer is `insufficient_evidence`, not a
confident guess. That failure mode is the point: a fact-checker that never says
"I don't know" is not a fact-checker.

Stance detection uses an NLI model (RoBERTa fine-tuned on MNLI, optionally
further fine-tuned on FEVER). For each (claim, evidence) pair it answers a
question the model is actually trained for — does this passage entail or
contradict this statement — rather than the open-ended "is this fake news",
which no model is calibrated for.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

from services.ml.rag.retrieval import Document, HybridRetriever, ScoredDocument

log = logging.getLogger(__name__)


class Stance(str, Enum):
    ENTAIL = "entail"
    CONTRADICT = "contradict"
    NEUTRAL = "neutral"


class Label(str, Enum):
    SUPPORTED = "supported"
    REFUTED = "refuted"
    DISPUTED = "disputed"
    INSUFFICIENT = "insufficient_evidence"


@dataclass
class EvidenceItem:
    document_id: str
    url: str | None
    title: str
    snippet: str
    domain: str | None
    credibility: float
    stance: Stance
    nli_score: float
    retrieval_score: float
    rerank_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "domain": self.domain,
            "credibility": round(self.credibility, 3),
            "stance": self.stance.value,
            "nli_score": round(self.nli_score, 4),
            "retrieval_score": round(self.retrieval_score, 6),
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score else None,
        }


@dataclass
class ClaimVerdict:
    claim: str
    label: Label
    confidence: float
    evidence: list[EvidenceItem] = field(default_factory=list)
    support_weight: float = 0.0
    refute_weight: float = 0.0
    explanation: str | None = None
    model_versions: dict[str, str] = field(default_factory=dict)

    @property
    def probability(self) -> float:
        """
        P(claim is true | evidence), as a smoothed share of the evidence weight.

        (support + 0.5) / (support + refute + 1): with no evidence it is exactly
        0.5, and each supporting or refuting passage moves it in proportion to
        its weight (stance confidence x source reliability x rank). Unlike the
        label it never snaps back to 0.5 because a threshold was missed, which
        is why every unfamiliar claim used to score exactly 50.
        """
        return (self.support_weight + 0.5) / (self.support_weight + self.refute_weight + 1.0)

    @property
    def certainty(self) -> float:
        """How much evidence stands behind `probability`, 0..1."""
        total = self.support_weight + self.refute_weight
        return total / (total + 1.0)

    @property
    def citations(self) -> list[str]:
        """Only URLs of documents that were actually retrieved and scored."""
        return [e.url for e in self.evidence if e.url]

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "label": self.label.value,
            "confidence": round(self.confidence, 4),
            "support_weight": round(self.support_weight, 4),
            "refute_weight": round(self.refute_weight, 4),
            "probability": round(self.probability, 4),
            "certainty": round(self.certainty, 4),
            "evidence_count": len(self.evidence),
            "evidence": [e.as_dict() for e in self.evidence],
            "citations": self.citations,
            "explanation": self.explanation,
            "model_versions": self.model_versions,
        }


# ---------------------------------------------------------------------------
# Stance detection
# ---------------------------------------------------------------------------

class StanceModel(Protocol):
    name: str
    def predict(self, claim: str, evidence: str) -> tuple[Stance, float]: ...


class NLIStanceModel:
    """
    Production stance detector: `roberta-large-mnli` (or a FEVER-tuned variant).

    Direction matters and is easy to get backwards: the evidence is the
    *premise* and the claim is the *hypothesis*. Asking "does this passage imply
    this claim" is the correct framing; reversing it changes what entailment
    means and quietly degrades accuracy.
    """

    def __init__(self, model_name: str = "roberta-large-mnli") -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self.name = model_name
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

        # Label order differs between checkpoints; read it from config rather
        # than hardcoding indices.
        self._id2label = {
            int(k): v.lower() for k, v in self._model.config.id2label.items()
        }

    def predict(self, claim: str, evidence: str) -> tuple[Stance, float]:
        inputs = self._tokenizer(
            evidence, claim, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        )
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        probs = self._torch.softmax(logits, dim=-1)[0]

        best_idx = int(probs.argmax())
        label = self._id2label.get(best_idx, "neutral")
        score = float(probs[best_idx])

        if "entail" in label:
            return Stance.ENTAIL, score
        if "contradict" in label:
            return Stance.CONTRADICT, score
        return Stance.NEUTRAL, score


_WORD = re.compile(r"[A-Za-z0-9]+")


class LexicalStanceModel:
    """
    Dependency-free fallback for offline testing and cold start.

    Overlap plus negation heuristics. Genuinely weak — it is here so the
    pipeline is testable and degrades to *worse stance detection* rather than
    crashing when no stronger judge is available. Never report metrics from
    this model as if they were the system's accuracy.

    Two kinds of negation, because treating them alike was a real bug:

    * Debunk markers ("false", "hoax", "debunked", ...) state a verdict about a
      claim. When the passage names the claim it debunks ("the claim that the
      landings were faked is false"), the marker counts only if OUR claim is
      that one; otherwise it counts anywhere.
    * Plain negators ("not", "no", "never", ...) negate the word right after
      them, so they count only when that word is shared with the claim.

    Counting both anywhere made "NASA landed astronauts on the Moon in 1969"
    read as contradicted by "the claim that the Moon landings were faked is
    false", and a true claim scored 17/100.
    """

    name = "lexical-fallback-v2"

    _NEGATORS = frozenset({
        "not", "no", "never", "none", "neither", "nor", "cannot", "cant",
        "didnt", "doesnt", "isnt", "wasnt", "werent", "wont", "denied", "denies",
    })
    _DEBUNK = frozenset({
        "false", "hoax", "debunked", "refuted", "incorrect", "untrue",
        "misleading", "fabricated", "myth", "baseless", "unfounded",
    })
    _NEGATIONS = _NEGATORS | _DEBUNK

    @staticmethod
    def _stem(t: str) -> str:
        return t[:5] if len(t) > 5 else t

    @staticmethod
    def _specifics(text: str) -> set[str]:
        """Names and numbers: capitalised words after the first, and any digits."""
        words = _WORD.findall(text)
        return {w.lower() for i, w in enumerate(words)
                if any(ch.isdigit() for ch in w) or (i > 0 and w[:1].isupper())}

    _DEBUNKED_SPAN = re.compile(
        r"\b(?:claims?|idea|notion|rumou?rs?|theory|theories|allegations?)\s+that\s+"
        r"(.{5,240}?)\s+(?:is|are|was|were|has been|have been)\s+"
        r"(?:false|untrue|baseless|unfounded|a myth|a hoax|debunked|incorrect|"
        r"fabricated|not supported)",
        re.IGNORECASE,
    )

    def _debunks_claim(self, text: str, tokens: list[str], claim_stems: set[str]) -> bool:
        if not any(t in self._DEBUNK for t in tokens):
            return False
        from services.ml.rag.retrieval import tokenize

        spans = self._DEBUNKED_SPAN.findall(text)
        if not spans:
            return True  # the passage calls something false; assume it is this
        for span in spans:
            span_stems = {self._stem(t) for t in tokenize(span)}
            if len(claim_stems & span_stems) >= 0.5 * len(claim_stems):
                return True
        return False

    def _negated(self, text: str, tokens: list[str], shared: set[str],
                 claim_stems: set[str]) -> bool:
        if self._debunks_claim(text, tokens, claim_stems):
            return True
        stems = [self._stem(t) for t in tokens]
        return any(
            t in self._NEGATORS and i + 1 < len(stems) and stems[i + 1] in shared
            for i, t in enumerate(tokens)
        )

    def predict(self, claim: str, evidence: str) -> tuple[Stance, float]:
        from services.ml.rag.retrieval import tokenize

        claim_tokens = tokenize(claim)
        evidence_tokens = tokenize(evidence)
        claim_stems = {self._stem(t) for t in claim_tokens if t not in self._NEGATIONS}
        evidence_stems = {self._stem(t) for t in evidence_tokens}
        if not claim_stems or not evidence_stems:
            return Stance.NEUTRAL, 0.5

        shared = claim_stems & evidence_stems
        overlap = len(shared) / len(claim_stems)
        if overlap < 0.25:
            return Stance.NEUTRAL, 0.5 + 0.3 * (1 - overlap)

        claim_negated = any(t in self._NEGATIONS for t in claim_tokens)
        evidence_negated = self._negated(evidence, evidence_tokens, shared, claim_stems)
        confidence = min(0.5 + overlap * 0.45, 0.95)

        if claim_negated != evidence_negated:
            return Stance.CONTRADICT, confidence
        # Support needs every name and number in the claim to be in the passage:
        # otherwise "The Eiffel Tower is in London" is "supported" by a passage
        # about the Eiffel Tower in Paris, since most of the words match.
        missing = self._specifics(claim) - {t.lower() for t in _WORD.findall(evidence)}
        if missing:
            return Stance.NEUTRAL, 0.5
        return Stance.ENTAIL, confidence


def relevance(claim: str, passage: str) -> float:
    """
    Share of the claim's content words (5-letter stems) that the passage contains.

    Retrieval always returns *something*: the hashing embedder gives every
    document a non-zero cosine. Without this gate, "The Eiffel Tower is in
    Paris" was "checked" against six WHO pages on vaccines and cited them.
    Evidence below the gate is not judged, weighed or cited.
    """
    from services.ml.rag.retrieval import tokenize

    stem = LexicalStanceModel._stem
    c = {stem(t) for t in tokenize(claim) if t not in LexicalStanceModel._NEGATIONS}
    if not c:
        return 0.0
    p = {stem(t) for t in tokenize(passage)}
    return len(c & p) / len(c)


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """
    `ms-marco-MiniLM-L-6-v2` cross-encoder.

    Retrieval optimizes recall over thousands of documents; a cross-encoder
    reads the query and document *together* and optimizes precision over the
    top ~50. Running it on everything would be far too slow — running it on the
    candidate pool is where most of the quality gain in a RAG system lives.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder
        self.name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: Sequence[ScoredDocument],
               top_k: int = 8) -> list[ScoredDocument]:
        if not hits:
            return []
        pairs = [(query, h.document.searchable[:2000]) for h in hits]
        scores = self._model.predict(pairs)
        for hit, score in zip(hits, scores):
            hit.rerank_score = float(score)
        return sorted(hits, key=lambda h: -(h.rerank_score or 0))[:top_k]


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """
    Retrieve → (rerank) → stance → aggregate.

    Aggregation weights each piece of evidence by both its retrieval rank and
    its source credibility, so a contradicting Reuters article outweighs three
    supporting blog posts. Those weights are stored on the verdict, so the
    arithmetic is auditable rather than a black box.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        stance_model: StanceModel | None = None,
        reranker: Any | None = None,
        *,
        candidate_k: int = 20,
        evidence_k: int = 6,
        min_confidence: float = 0.55,
        min_evidence: int = 1,
        min_relevance: float = 0.34,
    ) -> None:
        self.retriever = retriever
        self.stance_model = stance_model or LexicalStanceModel()
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.evidence_k = evidence_k
        self.min_confidence = min_confidence
        self.min_evidence = min_evidence
        self.min_relevance = min_relevance

    def candidates(self, claim: str) -> list[ScoredDocument]:
        """Retrieve, rerank, and keep only passages relevant enough to judge."""
        if not claim or not claim.strip():
            return []
        hits = self.retriever.search(claim, top_k=self.candidate_k)
        if self.reranker is not None:
            hits = self.reranker.rerank(claim, hits, top_k=self.candidate_k)
        hits = [h for h in hits
                if relevance(claim, h.document.searchable) >= self.min_relevance]
        return hits[:self.evidence_k]

    def verify(self, claim: str) -> ClaimVerdict:
        hits = self.candidates(claim)
        stances = [self.stance_model.predict(claim, h.document.searchable) for h in hits]
        return self.assess(claim, hits, stances)

    def assess(
        self,
        claim: str,
        hits: Sequence[ScoredDocument],
        stances: Sequence[tuple[Stance, float]],
        *,
        stance_name: str | None = None,
    ) -> ClaimVerdict:
        """
        Turn judged evidence into a verdict. Stance can come from any judge (the
        NLI model, an LLM reading each passage, or the lexical fallback); the
        arithmetic is the same for all of them and is stored on the verdict.
        """
        versions = self._versions()
        if stance_name:
            versions["stance"] = stance_name
        if not claim or not claim.strip():
            return ClaimVerdict(claim=claim, label=Label.INSUFFICIENT,
                                confidence=0.0, model_versions=versions)

        # No evidence means no verdict. This branch is the whole point of the
        # module — v1 had no equivalent and always produced a confident answer.
        if not hits:
            return ClaimVerdict(
                claim=claim, label=Label.INSUFFICIENT, confidence=0.0,
                explanation="No relevant documents were found for this claim.",
                model_versions=versions,
            )

        evidence: list[EvidenceItem] = []
        support = refute = 0.0

        for rank, (hit, (stance, nli_score)) in enumerate(zip(hits, stances), start=1):
            # Rank decay: the 6th result should not count as much as the 1st.
            rank_weight = 1.0 / math.log2(rank + 1.5)
            weight = nli_score * hit.document.credibility * rank_weight

            if stance is Stance.ENTAIL:
                support += weight
            elif stance is Stance.CONTRADICT:
                refute += weight

            evidence.append(EvidenceItem(
                document_id=hit.document.id,
                url=hit.document.url,
                title=hit.document.title,
                snippet=hit.document.text[:300],
                domain=hit.document.domain,
                credibility=hit.document.credibility,
                stance=stance,
                nli_score=nli_score,
                retrieval_score=hit.score,
                rerank_score=hit.rerank_score,
            ))

        label, confidence = self._aggregate(support, refute, evidence)

        return ClaimVerdict(
            claim=claim, label=label, confidence=confidence,
            evidence=evidence, support_weight=support, refute_weight=refute,
            model_versions=versions,
        )

    def _aggregate(
        self, support: float, refute: float, evidence: Sequence[EvidenceItem],
    ) -> tuple[Label, float]:
        decisive = [e for e in evidence if e.stance is not Stance.NEUTRAL]
        if len(decisive) < self.min_evidence:
            return Label.INSUFFICIENT, 0.0

        total = support + refute
        if total <= 0:
            return Label.INSUFFICIENT, 0.0

        margin = abs(support - refute) / total

        # Genuine disagreement between credible sources is its own answer, and
        # a more honest one than picking the larger pile.
        if margin < 0.2 and len(decisive) >= 2:
            return Label.DISPUTED, round(0.5 + margin, 4)

        confidence = round(min(0.99, margin * (1 - math.exp(-total))), 4)
        if confidence < self.min_confidence:
            return Label.INSUFFICIENT, confidence

        return (Label.SUPPORTED if support > refute else Label.REFUTED), confidence

    def _versions(self) -> dict[str, str]:
        v = {
            "stance": getattr(self.stance_model, "name", "unknown"),
            "embedder": type(self.retriever.embedder).__name__,
        }
        if self.reranker is not None:
            v["reranker"] = getattr(self.reranker, "name", "unknown")
        return v


def build_default_verifier(
    documents: Sequence[Document], *, prefer_transformers: bool = True,
) -> ClaimVerifier:
    """
    Construct the best pipeline the environment supports.

    Tries the real models first and falls back to offline-capable versions,
    logging loudly which one it got — so nobody mistakes fallback quality for
    production quality.
    """
    from services.ml.rag.retrieval import HashingEmbedder, SentenceTransformerEmbedder

    embedder: Any
    stance: Any
    reranker: Any = None

    if prefer_transformers:
        try:
            embedder = SentenceTransformerEmbedder()
            stance = NLIStanceModel()
            try:
                reranker = CrossEncoderReranker()
            except Exception as exc:  # noqa: BLE001
                log.warning("reranker unavailable: %s", exc)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "TRANSFORMERS UNAVAILABLE (%s) — falling back to hashing embedder "
                "and lexical stance. Retrieval quality will be materially worse; "
                "do not report metrics from this configuration.", exc,
            )
            embedder, stance = HashingEmbedder(), LexicalStanceModel()
    else:
        embedder, stance = HashingEmbedder(), LexicalStanceModel()

    retriever = HybridRetriever(embedder=embedder).index(documents)
    return ClaimVerifier(retriever=retriever, stance_model=stance, reranker=reranker)
