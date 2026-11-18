"""
CEREBRO with no database — the state production is in.

The Render free-tier PostgreSQL behind the deployed API was deleted when its
free period ended. After that every login answered 500 and so did the news
fact-check, although neither the analysers nor the demo evidence corpus need a
database. These tests boot the real app against a database that does not exist.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["WEB_EVIDENCE"] = "0"  # the suite never reaches the network
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-0123456789")
UNREACHABLE = "postgresql://nobody:nothing@127.0.0.1:1/missing"

pytest.importorskip("fastapi")
pytest.importorskip("asyncpg")
from fastapi.testclient import TestClient  # noqa: E402

from services.api.core.config import settings  # noqa: E402
from services.api.core.db import db  # noqa: E402
from services.api.main import app  # noqa: E402

DEMO = {"email": settings.demo_email, "password": settings.demo_password}


@pytest.fixture(scope="module")
def client():
    """
    Point the pool at a database that does not exist, on the object itself.

    CI runs this suite against a real PostgreSQL, and `settings` may already
    have been imported with that URL by an earlier test module — an environment
    variable set here would be too late, the app would connect, and these tests
    would quietly test nothing.
    """
    saved_dsn, saved_pool = db._dsn, db._pool
    db._dsn, db._pool = UNREACHABLE, None
    try:
        with TestClient(app) as c:      # the lifespan's connect fails; the app still starts
            assert not db.is_connected, "the outage was not simulated"
            yield c
    finally:
        db._dsn, db._pool = saved_dsn, saved_pool


def _login(client, body=DEMO):
    return client.post("/v1/auth/login", json=body)


def test_service_starts_and_reports_the_missing_database(client):
    r = client.get("/ready").json()
    assert r["capabilities"]["database"] is False
    assert r["capabilities"]["email_forensics"] is True


def test_demo_account_can_sign_in_without_a_database(client):
    r = _login(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["display_name"] == "Demo (offline mode)" and body["access_token"]
    me = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200 and me.json()["offline"] is True


def test_any_other_account_is_told_the_truth_not_wrong_password(client):
    r = _login(client, {"email": "someone@example.com", "password": "whatever"})
    assert r.status_code == 503
    assert "offline" in r.json()["detail"]


def test_demo_email_with_wrong_password_is_refused(client):
    assert _login(client, {"email": DEMO["email"], "password": DEMO["password"] + "x"}).status_code == 503


def test_offline_login_can_be_switched_off(client, monkeypatch):
    monkeypatch.setattr(settings, "offline_demo_login", False)
    assert _login(client).status_code == 503


def test_database_routes_answer_503_not_an_opaque_500(client):
    token = _login(client).json()["access_token"]
    r = client.get("/v1/metrics/summary", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 503
    assert r.json()["error"] == "database_unavailable"


def test_news_fact_check_works_from_the_in_memory_corpus(client):
    token = _login(client).json()["access_token"]
    r = client.post("/v1/analyze/news", headers={"Authorization": f"Bearer {token}"},
                    json={"text": "The Earth is flat and space agencies fake every photo of it."})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert body.get("claims") or body.get("verdicts") or "credibility" in str(body).lower()


def test_offline_corpus_mirrors_the_seed():
    from services.api.core.seed_corpus import DOCUMENTS, offline_corpus
    docs = offline_corpus()
    assert len(docs) == len(DOCUMENTS)
    assert all(d.meta["curated"] and d.url.startswith("https://") for d in docs)
    assert all(0 <= d.credibility <= 1 for d in docs)


def test_prometheus_metrics(client):
    _login(client)
    _login(client)
    text = client.get("/metrics").text
    assert re.search(r"^cerebro_database_up 0$", text, re.M)
    assert re.search(r'^cerebro_api_requests_total\{method="POST",route="/v1/auth/login",status="200"\} \d+$',
                     text, re.M)
    line = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[0-9.e+-]+$')
    for row in text.strip().splitlines():
        if not row.startswith("#"):
            assert line.match(row), row
