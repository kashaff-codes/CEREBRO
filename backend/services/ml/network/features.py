"""
Flow features every common traffic log can supply.

A detector is only useful on traffic people actually have. Security teams run
Zeek (conn.log), export NetFlow/IPFIX, or run CICFlowMeter; each records, per
connection, roughly the same few facts: how long it lasted, how many packets and
payload bytes went each way, and the destination port. The models here are
trained on exactly those facts, so one model scores all three formats:

    duration_s, fwd_pkts, bwd_pkts, fwd_bytes, bwd_bytes, dst_port

plus quantities derived from them (rates, ratios, bytes per packet, the port's
range). CICFlowMeter's 70-odd richer features would score better on its own
CSVs, but a model that needs them cannot read a Zeek log, and a product that
reads only one log format is a demo.

parse_any() accepts Zeek conn.log (TSV with a #fields header, or JSON lines),
CICFlowMeter / CIC-IDS2017 CSV, and a plain CSV with the six column names above
(plus optional src_ip, dst_ip, proto, ts, label).
"""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

BASE = ["duration_s", "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes", "dst_port"]
FEATURES = [
    "log_duration", "log_fwd_pkts", "log_bwd_pkts", "log_fwd_bytes", "log_bwd_bytes",
    "fwd_bytes_per_pkt", "bwd_bytes_per_pkt", "pkt_ratio", "byte_ratio",
    "log_pkts_per_s", "log_bytes_per_s", "port_well_known", "port_registered", "port_dynamic",
    "no_response",
]


@dataclass
class FlowRecord:
    duration_s: float
    fwd_pkts: float
    bwd_pkts: float
    fwd_bytes: float
    bwd_bytes: float
    dst_port: int
    src_ip: str = ""
    dst_ip: str = ""
    proto: str = ""
    ts: str = ""
    label: str = ""          # ground truth when the source has one (CIC-IDS2017), else ""
    extra: dict[str, Any] = field(default_factory=dict)


def vector(r: FlowRecord) -> list[float]:
    d = max(r.duration_s, 0.0)
    fp, bp = max(r.fwd_pkts, 0.0), max(r.bwd_pkts, 0.0)
    fb, bb = max(r.fwd_bytes, 0.0), max(r.bwd_bytes, 0.0)
    span = d + 1e-3
    p = int(r.dst_port) if r.dst_port is not None else 0
    return [
        math.log1p(d), math.log1p(fp), math.log1p(bp), math.log1p(fb), math.log1p(bb),
        fb / (fp + 1.0), bb / (bp + 1.0), (bp + 1.0) / (fp + 1.0), (bb + 1.0) / (fb + 1.0),
        math.log1p((fp + bp) / span), math.log1p((fb + bb) / span),
        float(p < 1024), float(1024 <= p < 49152), float(p >= 49152),
        float(bp == 0),
    ]


def matrix(records: Iterable[FlowRecord]) -> np.ndarray:
    return np.asarray([vector(r) for r in records], dtype=np.float64)


# ── parsers ──────────────────────────────────────────────────────────────────

def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def from_cic_row(row: dict[str, Any]) -> FlowRecord:
    """A CICFlowMeter / CIC-IDS2017 row (headers stripped of whitespace)."""
    g = {k.strip(): v for k, v in row.items() if k}
    return FlowRecord(
        duration_s=_f(g.get("Flow Duration")) / 1e6,          # microseconds
        fwd_pkts=_f(g.get("Total Fwd Packets")),
        bwd_pkts=_f(g.get("Total Backward Packets")),
        fwd_bytes=_f(g.get("Total Length of Fwd Packets")),
        bwd_bytes=_f(g.get("Total Length of Bwd Packets")),
        dst_port=int(_f(g.get("Destination Port") or g.get("Dst Port"))),
        src_ip=str(g.get("Source IP") or g.get("Src IP") or ""),
        dst_ip=str(g.get("Destination IP") or g.get("Dst IP") or ""),
        proto=str(g.get("Protocol") or ""),
        ts=str(g.get("Timestamp") or ""),
        label=str(g.get("Label") or "").strip(),
    )


def from_zeek(rec: dict[str, Any]) -> FlowRecord:
    """A Zeek conn.log record (TSV row as dict, or a JSON line)."""
    return FlowRecord(
        duration_s=_f(rec.get("duration")),
        fwd_pkts=_f(rec.get("orig_pkts")),
        bwd_pkts=_f(rec.get("resp_pkts")),
        fwd_bytes=_f(rec.get("orig_bytes")),
        bwd_bytes=_f(rec.get("resp_bytes")),
        dst_port=int(_f(rec.get("id.resp_p"))),
        src_ip=str(rec.get("id.orig_h") or ""),
        dst_ip=str(rec.get("id.resp_h") or ""),
        proto=str(rec.get("proto") or ""),
        ts=str(rec.get("ts") or ""),
        extra={k: rec.get(k) for k in ("service", "conn_state", "uid") if rec.get(k) not in (None, "-")},
    )


def from_plain(row: dict[str, Any]) -> FlowRecord:
    g = {k.strip().lower(): v for k, v in row.items() if k}
    return FlowRecord(
        duration_s=_f(g.get("duration_s") or g.get("duration")),
        fwd_pkts=_f(g.get("fwd_pkts")), bwd_pkts=_f(g.get("bwd_pkts")),
        fwd_bytes=_f(g.get("fwd_bytes")), bwd_bytes=_f(g.get("bwd_bytes")),
        dst_port=int(_f(g.get("dst_port"))),
        src_ip=str(g.get("src_ip") or ""), dst_ip=str(g.get("dst_ip") or ""),
        proto=str(g.get("proto") or ""), ts=str(g.get("ts") or ""), label=str(g.get("label") or ""),
    )


def parse_any(text: str, limit: int = 50_000) -> tuple[list[FlowRecord], str]:
    """Detect the format and parse up to `limit` flows. Returns (records, format)."""
    head = text.lstrip()[:4000]
    out: list[FlowRecord] = []
    if head.startswith("#separator") or "#fields" in head:                     # Zeek TSV
        fields: list[str] = []
        for line in text.splitlines():
            if line.startswith("#fields"):
                fields = line.split("\t")[1:]
                continue
            if not line or line.startswith("#") or not fields:
                continue
            vals = line.split("\t")
            rec = {k: (None if v in ("-", "(empty)") else v) for k, v in zip(fields, vals)}
            out.append(from_zeek(rec))
            if len(out) >= limit:
                break
        return out, "zeek_tsv"
    if head.startswith("{"):                                                    # Zeek JSON lines
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                out.append(from_zeek(json.loads(line)))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out, "zeek_json"
    reader = csv.DictReader(io.StringIO(text))
    cols = {c.strip() for c in (reader.fieldnames or [])}
    if "Flow Duration" in cols or "Total Fwd Packets" in cols:                  # CICFlowMeter
        kind, conv = "cicflowmeter_csv", from_cic_row
    elif {"fwd_pkts", "bwd_pkts"} <= {c.lower() for c in cols}:
        kind, conv = "plain_csv", from_plain
    else:
        raise ValueError(
            "Unrecognised format. Upload a Zeek conn.log (TSV or JSON), a CICFlowMeter CSV, "
            "or a CSV with columns " + ", ".join(BASE) + ".")
    for row in reader:
        out.append(conv(row))
        if len(out) >= limit:
            break
    return out, kind
