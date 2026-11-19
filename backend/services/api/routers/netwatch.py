"""
Network watch: real detectors on real traffic.

    GET  /v1/network/model    what the detectors are, what they were trained on,
                              and how they scored on held-out CIC-IDS2017 traffic
    GET  /v1/network/intel    live threat-feed status (abuse.ch, Spamhaus, Tor)
    GET  /v1/network/replay   real labelled CIC-IDS2017 test flows, scored now,
                              with the true label beside each verdict
    GET  /v1/network/sample   that labelled sample as a CSV, to try the upload with
    POST /v1/network/scan     your own traffic: a Zeek conn.log (TSV or JSON), a
                              CICFlowMeter CSV, or a plain CSV, scored by both
                              detectors and checked against the live feeds

This replaces a list of hand-written "sample" flows stamped with the current
time and labelled as a Zeek sensor, scored by fixed rules under a heading that
named two models neither of which was loaded.
"""
from __future__ import annotations

import csv
import json
import logging
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from services.api.core.db import emit_detection
from services.api.core.deps import Principal, current_principal
from services.ml.network import threat_intel
from services.ml.network.ensemble import ForestClassifierArrays, IsolationForestArrays
from services.ml.network.features import FlowRecord, matrix, parse_any

log = logging.getLogger(__name__)
router = APIRouter(prefix="/network", tags=["network"])
MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "network"
MAX_UPLOAD = 25 * 1024 * 1024
MAX_FLOWS = 50_000


@lru_cache(maxsize=1)
def _models() -> tuple[dict, IsolationForestArrays, ForestClassifierArrays]:
    card_path = MODEL_DIR / "model_card.json"
    if not card_path.exists():
        raise HTTPException(503, "The network detectors are not trained on this deployment "
                                 "(models/network/model_card.json is missing).")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    with np.load(MODEL_DIR / "iforest.npz") as z:
        iforest = IsolationForestArrays({k: z[k] for k in z.files})
    with np.load(MODEL_DIR / "forest.npz") as z:
        forest = ForestClassifierArrays({k: z[k] for k in z.files}, card["random_forest"]["classes"])
    return card, iforest, forest


def _score(records: list[FlowRecord]) -> list[dict[str, Any]]:
    card, iforest, forest = _models()
    X = matrix(records)
    anomaly = iforest.score(X)
    proba = forest.proba(X)
    classes = forest.classes
    benign = classes.index("BENIGN")
    thr = card["isolation_forest"]["threshold"]
    out = []
    for r, a, p in zip(records, anomaly, proba):
        top = int(np.argmax(p))
        out.append({
            "duration_s": round(r.duration_s, 6), "fwd_pkts": r.fwd_pkts, "bwd_pkts": r.bwd_pkts,
            "fwd_bytes": r.fwd_bytes, "bwd_bytes": r.bwd_bytes, "dst_port": r.dst_port,
            "src_ip": r.src_ip, "dst_ip": r.dst_ip, "proto": r.proto, "ts": r.ts, "label": r.label,
            "anomaly_score": round(float(a), 4), "anomalous": bool(a >= thr),
            "predicted_family": classes[top], "family_confidence": round(float(p[top]), 3),
            "attack_probability": round(float(1.0 - p[benign]), 3),
        })
    return out


def _level(rows: list[dict], intel_hits: dict) -> tuple[str, str]:
    n = max(len(rows), 1)
    flagged = sum(r["anomalous"] or r["predicted_family"] != "BENIGN" for r in rows)
    critical = any(h["severity"] == "critical" for hs in intel_hits.values() for h in hs)
    families = {r["predicted_family"] for r in rows if r["predicted_family"] != "BENIGN"}
    if critical or (families & {"DoS/DDoS", "Bot", "Infiltration", "Heartbleed"} and flagged / n > 0.02):
        return "Critical", "Isolate the flagged hosts, block the listed sources at the edge, and open an incident."
    if flagged / n > 0.01 or any(h["severity"] == "high" for hs in intel_hits.values() for h in hs):
        return "High", "Review the flagged flows today; confirm or dismiss each source."
    if flagged:
        return "Medium", "A few unusual flows: worth a look, not an emergency."
    return "Safe", "Nothing stood out against the learned baseline or the threat feeds."


@router.get("/model", summary="The network detectors and their measured accuracy")
async def model_card(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    return _models()[0]


@router.get("/intel", summary="Live threat-feed status")
async def intel(principal: Principal = Depends(current_principal)) -> list[dict[str, Any]]:
    return threat_intel.status()


@lru_cache(maxsize=1)
def _replay_rows() -> list[dict[str, str]]:
    with open(MODEL_DIR / "replay_sample.csv", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@router.get("/replay", summary="Real labelled CIC-IDS2017 test flows, scored now")
async def replay(n: int = 60, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    rows = _replay_rows()
    pick = random.Random(int(time.time() // 30)).sample(rows, k=min(max(n, 10), len(rows), 200))
    records = [FlowRecord(float(r["duration_s"]), float(r["fwd_pkts"]), float(r["bwd_pkts"]),
                          float(r["fwd_bytes"]), float(r["bwd_bytes"]), int(r["dst_port"]),
                          label=r["label"], extra={"family": r["family"], "day": r["day"]})
               for r in pick]
    scored = _score(records)
    for s, r in zip(scored, records):
        s["true_family"] = r.extra["family"]
        s["day"] = r.extra["day"]
    y = np.array([s["true_family"] != "BENIGN" for s in scored])
    flag_if = np.array([s["anomalous"] for s in scored])
    flag_rf = np.array([s["predicted_family"] != "BENIGN" for s in scored])
    flag_any = flag_if | flag_rf

    def conf(flag):
        return {"tp": int((flag & y).sum()), "fp": int((flag & ~y).sum()),
                "fn": int((~flag & y).sum()), "tn": int((~flag & ~y).sum())}
    return {"source": "CIC-IDS2017 held-out test flows (real traffic, real labels); "
                      "this is a replay, not your network",
            "held_out": _models()[0].get("replay_sample", ""),
            "flows": scored, "isolation_forest": conf(flag_if), "random_forest": conf(flag_rf),
            "either": conf(flag_any)}


@router.get("/sample", summary="Download the labelled CIC-IDS2017 test sample (CSV)")
async def sample(principal: Principal = Depends(current_principal)) -> FileResponse:
    return FileResponse(MODEL_DIR / "replay_sample.csv", media_type="text/csv",
                        filename="cerebro-cic-ids2017-test-sample.csv")


def _label_check(rows: list[dict]) -> dict[str, Any] | None:
    """When the upload carries ground truth (a Label column), say how the detectors did on it."""
    labelled = [r for r in rows if r["label"]]
    if not labelled:
        return None
    y = np.array([r["label"].strip().upper() != "BENIGN" for r in labelled])
    flag = np.array([r["anomalous"] or r["predicted_family"] != "BENIGN" for r in labelled])
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    fn, tn = int((~flag & y).sum()), int((~flag & ~y).sum())
    return {"flows": len(labelled), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else None}


@router.post("/scan", summary="Score your own traffic log")
async def scan(file: UploadFile = File(...), principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, f"Upload is over {MAX_UPLOAD // (1024 * 1024)} MB; send a smaller window.")
    try:
        records, fmt = parse_any(raw.decode("utf-8", errors="replace"), limit=MAX_FLOWS)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not records:
        raise HTTPException(422, "No flows found in that file.")
    t0 = time.time()
    rows = _score(records)
    external = sorted({ip for r in records for ip in (r.src_ip, r.dst_ip) if ip})[:2000]
    hits = {ip: h for ip in external if (h := threat_intel.lookup(ip))}
    for row in rows:
        row["intel"] = hits.get(row["src_ip"], []) + hits.get(row["dst_ip"], [])
    level, action = _level(rows, hits)
    ranked = sorted(rows, key=lambda r: (bool(r["intel"]), r["attack_probability"] + r["anomaly_score"]),
                    reverse=True)
    families: dict[str, int] = {}
    for r in rows:
        families[r["predicted_family"]] = families.get(r["predicted_family"], 0) + 1
    summary = {
        "format": fmt, "flows": len(rows), "seconds": round(time.time() - t0, 2),
        "anomalous": sum(r["anomalous"] for r in rows),
        "known_attack_flows": sum(r["predicted_family"] != "BENIGN" for r in rows),
        "by_family": families, "threat_level": level, "recommended_action": action,
        "intel_hits": [{"ip": ip, "matches": h} for ip, h in hits.items()],
        "feeds": threat_intel.status(),
        "has_labels": any(r["label"] for r in rows),
        "label_check": _label_check(rows),
    }
    try:
        risk = {"Critical": 0.95, "High": 0.75, "Medium": 0.5, "Safe": 0.08}[level]
        await emit_detection(principal.tenant_id, "network", f"{level.lower()}_traffic", risk,
                             summary=f"{len(rows)} flows from {file.filename}: {summary['anomalous']} anomalous, "
                                     f"{len(hits)} threat-feed matches")
    except Exception as exc:  # noqa: BLE001
        log.warning("network persistence skipped: %s", exc)
    return {"summary": summary, "top": ranked[:100]}
