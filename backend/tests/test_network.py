"""
Network watch: real detectors, real traffic formats, live-feed parsing, Gmail state.
All offline: no feed is fetched and no Google endpoint is called.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-0123456789")

from services.ml.network import threat_intel as T  # noqa: E402
from services.ml.network.ensemble import ForestClassifierArrays, IsolationForestArrays  # noqa: E402
from services.ml.network.features import FEATURES, matrix, parse_any  # noqa: E402

MODELS = Path(__file__).resolve().parents[1] / "models" / "network"


# ── the scorers are scikit-learn, exactly ───────────────────────────────────

def test_numpy_scorers_equal_sklearn():
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, len(FEATURES)))
    y = np.where(X[:, 0] + X[:, 3] > 0.4, "ATTACK", "BENIGN")
    f = IsolationForest(n_estimators=40, max_samples=200, max_features=0.6, random_state=0).fit(X)
    assert np.abs(IsolationForestArrays(IsolationForestArrays.export(f)).score(X[:300])
                  - (-f.score_samples(X[:300]))).max() < 1e-9
    r = RandomForestClassifier(n_estimators=15, max_depth=7, random_state=0).fit(X, y)
    assert np.abs(ForestClassifierArrays(ForestClassifierArrays.export(r), list(r.classes_)).proba(X[:300])
                  - r.predict_proba(X[:300])).max() < 1e-9


# ── the shipped model card says what it is ──────────────────────────────────

def test_model_card_is_complete_and_honest():
    card = json.loads((MODELS / "model_card.json").read_text(encoding="utf-8"))
    assert card["dataset"].startswith("CIC-IDS2017") and card["flows"] > 2_000_000
    iso = card["isolation_forest"]
    assert iso["max_abs_diff_vs_sklearn"] < 1e-9
    budgets = [o["false_alarm_budget"] for o in iso["operating_points"]]
    assert budgets == sorted(budgets) and 0.05 in budgets
    recalls = [o["recall"] for o in iso["operating_points"]]
    assert recalls == sorted(recalls), "recall must rise as the false-alarm budget grows"
    rf = card["random_forest"]
    assert {"random_split", "held_out_days"} <= set(rf)       # both published, not just the flattering one
    assert rf["held_out_days"]["attack_vs_normal"]["recall"] < rf["random_split"]["attack_vs_normal"]["recall"]


# ── every log format a team actually has ────────────────────────────────────

ZEEK_TSV = """#separator \\x09
#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tservice\tduration\torig_bytes\tresp_bytes\tconn_state\torig_pkts\tresp_pkts
1700000000.1\tC1\t10.0.0.5\t51000\t93.184.216.34\t443\ttcp\tssl\t1.52\t1200\t48000\tSF\t14\t40
1700000001.2\tC2\t10.0.0.9\t40000\t10.0.0.1\t22\ttcp\t-\t0.0001\t0\t0\tS0\t1\t0
"""


def test_parse_zeek_tsv_json_cic_and_plain():
    recs, kind = parse_any(ZEEK_TSV)
    assert kind == "zeek_tsv" and len(recs) == 2
    assert recs[0].dst_port == 443 and recs[0].bwd_bytes == 48000 and recs[1].bwd_pkts == 0
    js = '{"ts":1,"id.orig_h":"10.0.0.5","id.resp_h":"1.2.3.4","id.resp_p":80,"duration":0.5,' \
         '"orig_bytes":10,"resp_bytes":20,"orig_pkts":2,"resp_pkts":2}\n'
    recs, kind = parse_any(js)
    assert kind == "zeek_json" and recs[0].dst_ip == "1.2.3.4"
    cic = " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets," \
          "Total Length of Fwd Packets, Total Length of Bwd Packets, Label\n80,2000000,3,2,120,300,BENIGN\n"
    recs, kind = parse_any(cic)
    assert kind == "cicflowmeter_csv" and recs[0].duration_s == 2.0 and recs[0].label == "BENIGN"
    plain = "duration_s,fwd_pkts,bwd_pkts,fwd_bytes,bwd_bytes,dst_port,src_ip\n1,2,3,4,5,53,8.8.8.8\n"
    recs, kind = parse_any(plain)
    assert kind == "plain_csv" and recs[0].dst_port == 53
    assert matrix(recs).shape == (1, len(FEATURES))


def test_unknown_format_is_a_clear_error():
    with pytest.raises(ValueError, match="Zeek conn.log"):
        parse_any("a,b,c\n1,2,3\n")


# ── threat feeds ────────────────────────────────────────────────────────────

def test_feed_parsing_and_lookup(monkeypatch):
    T._FEEDS.clear()
    feodo = T._parse("feodo", json.dumps([{"ip_address": "45.142.212.61", "malware": "QakBot", "status": "online"}]))
    drop = T._parse("spamhaus_drop", '{"cidr":"91.200.12.0/24","sblid":"SBL1"}\n{"type":"metadata"}\n')
    tor = T._parse("tor_exit", "185.220.101.5\n")
    now = time.time()
    for f in (feodo, drop, tor):
        f.fetched_at = now
        T._FEEDS[f.name] = f
    monkeypatch.setattr(T, "refresh", lambda force=False: dict(T._FEEDS))
    assert T.lookup("45.142.212.61")[0]["malware"] == "QakBot"
    assert T.lookup("91.200.12.99")[0]["netblock"] == "91.200.12.0/24"
    assert T.lookup("10.1.2.3") == []            # private never matches
    assert T.lookup("not-an-ip") == []


# ── the API ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from services.api.core.db import db
    from services.api.main import app
    saved = db._dsn, db._pool
    db._dsn, db._pool = "postgresql://nobody:x@127.0.0.1:1/missing", None
    try:
        with TestClient(app) as c:
            yield c
    finally:
        db._dsn, db._pool = saved


def _token(client):
    from services.api.core.config import settings
    return client.post("/v1/auth/login", json={"email": settings.demo_email,
                                               "password": settings.demo_password}).json()["access_token"]


def test_replay_scores_real_labelled_flows(client):
    r = client.get("/v1/network/replay?n=80", headers={"Authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert "CIC-IDS2017" in body["source"] and "not your network" in body["source"]
    assert all("true_family" in f and "anomaly_score" in f for f in body["flows"])
    c = body["either"]
    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == len(body["flows"])


def test_scan_an_uploaded_zeek_log(client, monkeypatch):
    monkeypatch.setattr(T, "lookup", lambda ip: [{"feed": "feodo", "label": "C2", "severity": "critical"}]
                        if ip == "93.184.216.34" else [])
    monkeypatch.setattr(T, "status", lambda: [])
    r = client.post("/v1/network/scan", headers={"Authorization": f"Bearer {_token(client)}"},
                    files={"file": ("conn.log", ZEEK_TSV, "text/plain")})
    assert r.status_code == 200, r.text[:300]
    s = r.json()["summary"]
    assert s["format"] == "zeek_tsv" and s["flows"] == 2
    assert s["threat_level"] == "Critical"          # a live C2 match outranks everything
    assert s["intel_hits"][0]["ip"] == "93.184.216.34"


def test_scan_rejects_unknown_files(client):
    r = client.post("/v1/network/scan", headers={"Authorization": f"Bearer {_token(client)}"},
                    files={"file": ("x.csv", "a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422


# ── Gmail sign-in state ─────────────────────────────────────────────────────

def test_gmail_state_is_bound_to_the_user_and_expires():
    from fastapi import HTTPException

    from services.api.routers import gmail as G
    now = int(time.time())
    state = G._sign("user-a", now)
    G._check_state(state, "user-a")
    with pytest.raises(HTTPException):
        G._check_state(state, "user-b")               # another user's session cannot finish it
    with pytest.raises(HTTPException):
        G._check_state(G._sign("user-a", now - 3600), "user-a")   # expired
    with pytest.raises(HTTPException):
        G._check_state(f"{now}.forged", "user-a")


def test_gmail_redirect_only_to_allowed_origins():
    from fastapi import HTTPException

    from services.api.routers import gmail as G
    assert G._redirect_uri("http://localhost:5173") == "http://localhost:5173/"
    with pytest.raises(HTTPException):
        G._redirect_uri("https://evil.example")


def test_labelled_upload_reports_its_own_accuracy(client, monkeypatch):
    monkeypatch.setattr(T, "lookup", lambda ip: [])
    monkeypatch.setattr(T, "status", lambda: [])
    h = {"Authorization": f"Bearer {_token(client)}"}
    csv_text = client.get("/v1/network/sample", headers=h).text
    assert csv_text.startswith("day,duration_s")
    r = client.post("/v1/network/scan", headers=h, files={"file": ("sample.csv", csv_text, "text/csv")})
    assert r.status_code == 200, r.text[:300]
    s = r.json()["summary"]
    lc = s["label_check"]
    assert s["format"] == "plain_csv" and lc["flows"] == s["flows"]
    assert lc["tp"] + lc["fp"] + lc["fn"] + lc["tn"] == lc["flows"]
    assert lc["recall"] > 0.5            # the shipped forest has seen these days; it must catch most


# ── Gmail inbox scan, against a fake Gmail API ──────────────────────────────

PHISH = (b'From: "PayPal Security" <alert@paypa1.com>\r\nReply-To: x@secure-paypa1.ru\r\n'
         b'Authentication-Results: mx.google.com; spf=fail smtp.mailfrom=paypa1.com; dmarc=fail\r\n'
         b'Subject: Urgent: your account has been limited\r\n\r\n'
         b'Your account is suspended. Verify now at http://paypa1.com/login-verify within 24 hours.\r\n')
CLEAN = (b'From: GitHub <noreply@github.com>\r\n'
         b'Authentication-Results: mx.google.com; spf=pass smtp.mailfrom=github.com; '
         b'dkim=pass header.d=github.com; dmarc=pass\r\n'
         b'Subject: Your weekly digest\r\n\r\nThree new stars on your repository.\r\n')


class _Resp:
    def __init__(self, code, body):
        self.status_code, self._body = code, body

    def json(self):
        return self._body


class _FakeGmail:
    """Just enough of httpx.AsyncClient for the list + get calls the scan makes."""
    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None, params=None):
        import base64
        assert headers["Authorization"] == "Bearer tok"
        if url.endswith("/messages"):
            assert "-in:sent" in params["q"]
            return _Resp(200, {"messages": [{"id": "a"}, {"id": "b"}, {"id": "gone"}]})
        mid = url.rsplit("/", 1)[1]
        if mid == "gone":
            return _Resp(404, {})
        raw = {"a": PHISH, "b": CLEAN}[mid]
        return _Resp(200, {"id": mid, "threadId": "t", "internalDate": "1795000000000", "labelIds": ["INBOX"],
                           "raw": base64.urlsafe_b64encode(raw).decode().rstrip("=")})


def test_gmail_scan_scores_the_inbox_and_builds_analytics(client, monkeypatch):
    import httpx

    from services.api.routers import gmail as G

    async def token(_p):
        return "tok"
    monkeypatch.setattr(G, "_access_token", token)
    monkeypatch.setattr(httpx, "AsyncClient", _FakeGmail)
    r = client.post("/v1/gmail/scan?max=500", headers={"Authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body["requested"] == 3 and body["scanned"] == 2          # the missing message is skipped, not fatal
    first, second = body["messages"]
    assert first["from_domain"] == "paypa1.com" and first["risk"] > second["risk"]   # riskiest first
    assert first["verdict"] in {"phishing", "suspicious"} and second["verdict"] == "benign"
    assert body["auth_pass_rate"]["spf"] == 0.5 and body["auth_pass_rate"]["dmarc"] == 0.5
    assert dict(body["top_sender_domains"]) == {"paypa1.com": 1, "github.com": 1}
    assert first["date"] == 1795000000


def test_gmail_not_configured_is_a_clear_503(client, monkeypatch):
    from services.api.core.config import settings
    monkeypatch.setattr(settings, "google_client_id", "")
    r = client.get("/v1/gmail/auth-url?origin=http://localhost:5173",
                   headers={"Authorization": f"Bearer {_token(client)}"})
    assert r.status_code == 503 and "GOOGLE_CLIENT_ID" in r.json()["detail"]


def test_gmail_connect_refuses_when_gmail_permission_was_unticked(client, monkeypatch):
    import time as _t

    from services.api.routers import auth as A
    from services.api.routers import gmail as G
    calls = []

    async def exchange(req, principal):
        return {"scopes": ["openid", "email"]}          # user unticked the Gmail box
    monkeypatch.setattr(A, "google_exchange", exchange)
    monkeypatch.setattr(G.settings, "google_client_id", "id")
    monkeypatch.setattr(G.settings, "google_client_secret", "s")
    monkeypatch.setattr(G.settings, "token_encryption_key", "k")

    async def fetch(*a):
        calls.append(a[0])
        return []
    monkeypatch.setattr(G.db, "fetch_unscoped", fetch)
    tok = _token(client)
    from services.api.core.config import settings
    from services.api.core.security import decode_access_token
    uid = decode_access_token(tok, settings.secret_key)["sub"]
    r = client.post("/v1/gmail/connect", headers={"Authorization": f"Bearer {tok}"},
                    json={"code": "c", "state": G._sign(uid, int(_t.time())), "origin": "http://localhost:5173"})
    assert r.status_code == 400 and "View your email" in r.json()["detail"]
    assert any(c.startswith("DELETE") for c in calls)   # nothing half-connected is left behind
