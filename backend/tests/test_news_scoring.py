"""
The news credibility score: what it is built from, and the failures it replaced.

A reviewer observed that the score was "always 50%" (or, now and then, 0 or
100) and could not be explained. Reproduced, there were three causes, each with
a test here:

  1. Anything outside the 17-document seed corpus scored exactly 50, while
     citing six unrelated pages: retrieval always returns something, and every
     claim that missed a confidence threshold counted as 0.5.
  2. True claims read as false. "NASA landed astronauts on the Moon in 1969"
     scored 17 / Fake News, because a NASA page saying the landings "were not
     faked" contains a negation.
  3. The number could not be taken apart. It is now a weighted mean of named
     components, each returned with its score and weight.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["WEB_EVIDENCE"] = "0"  # never reach the network from the test suite
os.environ["WEB_RESEARCH"] = "0"

from services.api.routers.news import _compose  # noqa: E402
from services.ml.rag import llm_stance  # noqa: E402
from services.ml.rag.retrieval import Document, HashingEmbedder, HybridRetriever  # noqa: E402
from services.ml.rag.verify import (  # noqa: E402
    ClaimVerdict, ClaimVerifier, EvidenceItem, Label, LexicalStanceModel, Stance, relevance,
)

MOON = Document(
    id="nasa", url="https://nasa.gov/apollo", domain="nasa.gov", credibility=0.97,
    title="NASA: The Apollo Moon landings were not faked",
    text=("The claim that the Apollo Moon landings were faked or filmed in a studio "
          "is false. NASA landed twelve astronauts on the Moon between 1969 and 1972, "
          "beginning with Apollo 11 in July 1969."),
)
AUTISM = Document(
    id="cdc", url="https://cdc.gov/autism", domain="cdc.gov", credibility=0.96,
    title="CDC: Vaccines do not cause autism",
    text="Studies have shown that vaccines do not cause autism in children.",
)
EIFFEL = Document(
    id="wiki", url="https://en.wikipedia.org/?curid=9232", domain="en.wikipedia.org",
    credibility=0.8, title="Wikipedia: Eiffel Tower",
    text="The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
)
CORPUS = [MOON, AUTISM, EIFFEL]


def _verifier() -> ClaimVerifier:
    return ClaimVerifier(retriever=HybridRetriever(embedder=HashingEmbedder()).index(CORPUS),
                         stance_model=LexicalStanceModel())


# --- 1. relevance: no more citing unrelated pages -------------------------------

def test_an_unrelated_claim_cites_nothing():
    v = _verifier().verify("Pakistan won the 1992 Cricket World Cup")
    assert v.evidence == [] and v.citations == []
    assert v.label is Label.INSUFFICIENT


def test_relevance_is_the_share_of_claim_words_present():
    assert relevance("Vaccines cause autism", AUTISM.searchable) == 1.0
    assert relevance("Pakistan won the World Cup", AUTISM.searchable) == 0.0


# --- 2. stance: negation is scoped, names must match -------------------------------

def test_a_debunk_of_a_different_claim_does_not_refute_this_one():
    stance, _ = LexicalStanceModel().predict(
        "NASA landed astronauts on the Moon in 1969", MOON.searchable)
    assert stance is not Stance.CONTRADICT


def test_a_debunk_of_this_claim_refutes_it():
    stance, _ = LexicalStanceModel().predict(
        "Vaccines cause autism in children", AUTISM.searchable)
    assert stance is Stance.CONTRADICT


def test_a_swapped_name_is_not_supported():
    stance, _ = LexicalStanceModel().predict("The Eiffel Tower is in London", EIFFEL.searchable)
    assert stance is not Stance.ENTAIL


def test_the_moon_landing_is_no_longer_called_fake():
    v = _verifier().verify("NASA landed astronauts on the Moon in 1969")
    assert v.label is not Label.REFUTED
    assert v.probability > 0.5


# --- 3. the score is continuous, explained, and honest about thin evidence -------

def _verdict(support: float, refute: float) -> ClaimVerdict:
    return ClaimVerdict(claim="c", label=Label.INSUFFICIENT, confidence=0.0,
                        support_weight=support, refute_weight=refute)


def test_probability_moves_with_evidence_instead_of_snapping_to_half():
    assert _verdict(0, 0).probability == 0.5
    weak, strong = _verdict(0.3, 0), _verdict(2.0, 0)
    assert 0.5 < weak.probability < strong.probability < 1
    assert _verdict(0, 2.0).probability < 0.2


def test_no_evidence_lands_near_50_and_says_so():
    out = _compose([_verdict(0, 0)], [1.0], None, "The council met on Tuesday.")
    assert 40 <= out["score"] <= 65
    assert out["verdict"] == "Unverified"
    assert out["confidence_label"] == "Style only"


def test_strong_refutation_is_fake_news_with_high_confidence():
    out = _compose([_verdict(0, 3.0)], [1.0], None, "Vaccines cause autism.")
    assert out["score"] <= 30 and out["verdict"] == "Fake News"
    assert out["confidence_label"] in ("High", "Medium")


def test_strong_support_is_credible():
    out = _compose([_verdict(3.0, 0)], [1.0], None, "Pakistan won the 1992 World Cup.")
    assert out["score"] >= 65 and out["verdict"] == "Credible"


def test_scores_spread_across_the_range():
    scores = {_compose([_verdict(s, r)], [1.0], None, "A claim.")["score"]
              for s, r in [(0, 3), (0, 1), (0.4, 0.4), (1, 0), (3, 0)]}
    assert len(scores) == 5 and min(scores) < 35 and max(scores) > 65


def test_the_breakdown_adds_up_to_the_score():
    out = _compose([_verdict(1.2, 0.3)], [0.8], (20, "Fake News", "x — y", ["u1", "u2"]),
                   "Officials confirmed the report.")
    parts = [c for c in out["breakdown"] if c["score"] is not None]
    total = sum(c["weight"] for c in parts)
    rebuilt = sum(c["score"] * c["weight"] for c in parts) / total
    assert abs(rebuilt - out["score"]) <= 0.5
    assert abs(sum(c["share"] for c in parts) - 1) < 0.01
    assert {"evidence", "fact_check", "language"} <= {c["key"] for c in parts}


def test_a_published_fact_check_dominates_style():
    shouty = "SHOCKING!!! The truth they don't want you to know!!!"
    out = _compose([_verdict(0, 0)], [1.0], (85, "Credible", "x — rated True", ["u"]), shouty)
    assert out["score"] > 60


# --- the LLM stance judge ----------------------------------------------------------

class _Resp:
    def __init__(self, text, provider="anthropic", model="test-model"):
        self.text, self.provider, self.model = text, provider, model


class _FakeLLM:
    def __init__(self, payload, provider="anthropic"):
        self.payload, self.provider, self.prompts = payload, provider, []

    async def complete(self, prompt, **kw):
        self.prompts.append(prompt)
        return _Resp(json.dumps(self.payload), provider=self.provider)


def test_llm_judge_maps_each_passage_and_names_itself():
    hits = _verifier().candidates("The Eiffel Tower is in London")
    assert hits, "the Eiffel passage should pass the relevance gate"
    fake = _FakeLLM({"judgements": [{"passage": 1, "stance": "refutes", "confidence": 0.93}]})
    stances, name = asyncio.run(llm_stance.judge(fake, "The Eiffel Tower is in London", hits))
    assert stances[0] == (Stance.CONTRADICT, 0.93)
    assert name.startswith("llm-nli:anthropic")
    # The model sees the retrieved passage, not just the claim.
    assert "lattice tower in Paris" in fake.prompts[0]

    v = _verifier().assess("The Eiffel Tower is in London", hits, stances, stance_name=name)
    assert v.refute_weight > 0 and v.probability < 0.5
    assert v.model_versions["stance"] == name


def test_llm_judge_steps_aside_when_only_the_template_answers():
    hits = _verifier().candidates("Vaccines cause autism")
    fake = _FakeLLM({"judgements": []}, provider="template")
    assert asyncio.run(llm_stance.judge(fake, "Vaccines cause autism", hits)) is None


def test_llm_judge_ignores_malformed_rows():
    hits = _verifier().candidates("Vaccines cause autism")
    fake = _FakeLLM({"judgements": [{"passage": 99, "stance": "refutes", "confidence": 1},
                                    {"passage": "x"}]})
    stances, _ = asyncio.run(llm_stance.judge(fake, "Vaccines cause autism", hits))
    assert all(s is Stance.NEUTRAL for s, _ in stances)


def test_web_evidence_reliability_table():
    from services.ml.rag.web_evidence import DEFAULT, reliability
    assert reliability("www.reuters.com") == 0.95
    assert reliability("edition.cnn.com") == 0.8          # parent domain
    assert reliability("unknown-local-paper.example") == DEFAULT
    assert reliability("infowars.com") < 0.2


def test_word_overlap_alone_never_claims_credible():
    """The fallback stance can lean, not verify (it called a London tower page support)."""
    v = _verdict(3.0, 0)
    v.model_versions = {"stance": "lexical-fallback-v2"}
    v.evidence = [EvidenceItem(document_id="w", url="u", title="t", snippet="s",
                               domain="en.wikipedia.org", credibility=0.8,
                               stance=Stance.ENTAIL, nli_score=0.9, retrieval_score=1.0)]
    out = _compose([v], [1.0], None, "eiffel tower is in london right")
    assert out["score"] >= 65            # the lean is still shown
    assert out["verdict"] == "Unverified"
    assert out["confidence_label"] == "Low"


def test_fallback_claim_split_keeps_short_questions_and_drops_lead_ins():
    from services.ml.providers.llm import split_sentences
    assert split_sentences(
        "eiffel tower is in london right? and NASA landed astronauts on the Moon in 1969"
    ) == ["eiffel tower is in london right?", "NASA landed astronauts on the Moon in 1969"]
    assert split_sentences("I saw on reddit that the WHO declared a pandemic. Also that "
                           "Pakistan won the 1992 cricket world cup.") == [
        "the WHO declared a pandemic.", "Pakistan won the 1992 cricket world cup."]
