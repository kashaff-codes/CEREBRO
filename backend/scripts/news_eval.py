"""
Measure the news verifier against labelled claims.

    cd backend && python scripts/news_eval.py               # eval/claims.jsonl
    cd backend && python scripts/news_eval.py --limit 5     # a quick check

Each claim carries a label: TRUE, FALSE, or UNVERIFIABLE (invented people,
places and studies — the right answer is to say there is no evidence, not to
guess). A verdict counts as correct when:

    TRUE          -> Credible
    FALSE         -> Fake News or Misleading
    UNVERIFIABLE  -> Unverified or Disputed

Reports accuracy per label, abstention, how well the score separates true from
false (ROC AUC over the 0-100 score), mean score per label, research cost, and
writes every row to eval/results.json so a number can be traced to its claim.
Runs the real app in-process; reads ANTHROPIC_API_KEY etc. from backend/.env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
env = ROOT / ".env"
if env.exists():
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("DATABASE_URL", "postgresql://nobody:x@127.0.0.1:1/missing")
os.environ.setdefault("SECRET_KEY", "eval-secret-key-0123456789abcdefghijklmn")
os.environ.setdefault("NEWS_LLM_REASONING", "0")   # narration does not affect the verdict

CORRECT = {"TRUE": {"Credible"}, "FALSE": {"Fake News", "Misleading"},
           "UNVERIFIABLE": {"Unverified", "Disputed"}}


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "eval" / "claims.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results.json"))
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from services.api.core.config import settings
    from services.api.main import app

    claims = [json.loads(x) for x in Path(args.file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        claims = claims[: args.limit]
    rows = []
    with TestClient(app) as c:
        tok = c.post("/v1/auth/login", json={"email": settings.demo_email,
                                             "password": settings.demo_password}).json()["access_token"]
        for i, cl in enumerate(claims, 1):
            t0 = time.time()
            r = c.post("/v1/analyze/news", headers={"Authorization": f"Bearer {tok}"},
                       json={"text": cl["text"]})
            b = r.json() if r.status_code == 200 else {}
            u = b.get("research_usage") or {}
            cost = (u.get("input_tokens", 0) * 4e-6 + u.get("output_tokens", 0) * 20e-6
                    + u.get("web_search_requests", 0) * 0.01)
            row = {**cl, "status": r.status_code, "score": b.get("credibilityScore"),
                   "verdict": b.get("verdict"), "confidence": b.get("confidence_label"),
                   "correct": b.get("verdict") in CORRECT[cl["label"]],
                   "claims": [x["text"] for x in b.get("claims", [])],
                   "web_search": (b.get("evidence_sources") or {}).get("web_search"),
                   "passages": sum(len(x["evidence"]) for x in b.get("claims", [])),
                   "seconds": round(time.time() - t0, 1), "cost_usd": round(cost, 4)}
            rows.append(row)
            mark = "✓" if row["correct"] else "✗"
            print(f"{i:>2}/{len(claims)} {mark} {cl['label']:<12} {str(row['verdict']):<10} "
                  f"{row['score']!s:>3}  {row['seconds']:>5}s ${cost:.3f}  {cl['text'][:60]}", flush=True)

    by = {}
    for lab in CORRECT:
        sel = [r for r in rows if r["label"] == lab]
        if sel:
            by[lab] = {"n": len(sel), "accuracy": round(sum(r["correct"] for r in sel) / len(sel), 3),
                       "mean_score": round(sum(r["score"] or 0 for r in sel) / len(sel), 1)}
    summary = {
        "n": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3),
        "by_label": by,
        "auc_true_vs_false": auc([r["score"] for r in rows if r["label"] == "TRUE" and r["score"] is not None],
                                 [r["score"] for r in rows if r["label"] == "FALSE" and r["score"] is not None]),
        "said_unverified_on_true_or_false": sum(
            r["verdict"] in ("Unverified", "Disputed") for r in rows if r["label"] != "UNVERIFIABLE"),
        "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2],
        "cost_usd": round(sum(r["cost_usd"] for r in rows), 2),
        "model": os.getenv("RESEARCH_MODEL") or "claude-opus-5-5",
        "when": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
