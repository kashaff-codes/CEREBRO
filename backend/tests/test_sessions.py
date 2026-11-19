"""
Sessions: an expired access token renews itself; a stolen refresh token does not.

The database is replaced by a small in-memory stand-in for cerebro.users and
cerebro.sessions that answers the exact statements the auth router issues, so
these run anywhere. The last test repeats the round trip against the real
schema when a database is reachable (CI's Postgres).
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-0123456789")

from services.api.core.security import hash_password  # noqa: E402


class FakeAccounts:
    """cerebro.users + cerebro.sessions, as far as routers/auth.py uses them."""

    def __init__(self, grace: float):
        self.grace = grace
        self.users: dict[str, dict] = {}
        self.sessions: dict[str, dict] = {}       # id -> row

    def add_user(self, email: str, password: str) -> str:
        uid = str(uuid.uuid4())
        self.users[uid] = {"id": uid, "tenant_id": str(uuid.uuid4()), "email": email, "role": "owner",
                           "display_name": "Ana", "password_hash": hash_password(password),
                           "is_active": True, "last_seen_at": None}
        return uid

    def live(self, uid: str) -> list[dict]:
        return [s for s in self.sessions.values() if s["user_id"] == uid and s["revoked_at"] is None]

    async def fetch_unscoped(self, sql: str, *a):
        q = " ".join(sql.split())
        now = time.time()
        if q.startswith("SELECT id, tenant_id, role, display_name, password_hash, is_active FROM cerebro.users"):
            return [u for u in self.users.values() if u["email"] == a[0]][:1]
        if q.startswith("SELECT id, tenant_id, email, display_name, role, last_seen_at FROM cerebro.users"):
            return [u for u in self.users.values() if u["id"] == str(a[0])]
        if q.startswith("UPDATE cerebro.users SET last_seen_at"):
            return []
        if q.startswith("INSERT INTO cerebro.sessions"):
            sid = str(uuid.uuid4())
            self.sessions[sid] = {"id": sid, "user_id": str(a[0]), "token_hash": a[1],
                                  "expires_at": a[4].timestamp(), "revoked_at": None}
            return [{"id": sid}]
        if q.startswith("SELECT s.id, s.user_id, s.revoked_at"):
            out = []
            for srow in self.sessions.values():
                if srow["token_hash"] == a[0]:
                    u = self.users[srow["user_id"]]
                    rv = srow["revoked_at"]
                    out.append({"id": srow["id"], "user_id": srow["user_id"], "revoked_at": rv,
                                "recently_revoked": rv is not None and rv > now - float(a[1]),
                                "expired": srow["expires_at"] <= now, "tenant_id": u["tenant_id"],
                                "role": u["role"], "display_name": u["display_name"],
                                "is_active": u["is_active"]})
            return out
        if q.startswith("UPDATE cerebro.sessions SET revoked_at = now() WHERE id = $1"):
            srow = self.sessions.get(a[0])
            if srow and srow["revoked_at"] is None:
                srow["revoked_at"] = now
                return [{"id": srow["id"]}]
            return []
        if q.startswith("UPDATE cerebro.sessions SET revoked_at = now() WHERE user_id = $1"):
            for srow in self.live(str(a[0])):
                srow["revoked_at"] = now
            return []
        if q.startswith("UPDATE cerebro.sessions SET revoked_at = now() WHERE token_hash = $1"):
            for srow in self.sessions.values():
                if srow["token_hash"] == a[0] and srow["revoked_at"] is None:
                    srow["revoked_at"] = now
            return []
        raise AssertionError(f"unexpected SQL: {q[:120]}")


@pytest.fixture()
def env(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from services.api.core.db import db
    from services.api.main import app
    from services.api.routers import auth as A
    saved = db._dsn, db._pool
    db._dsn, db._pool = "postgresql://nobody:x@127.0.0.1:1/missing", None
    try:
        with TestClient(app) as c:
            fake = FakeAccounts(A.ROTATION_GRACE_SECONDS)
            monkeypatch.setattr(db, "_pool", object())          # "connected"
            monkeypatch.setattr(db, "fetch_unscoped", fake.fetch_unscoped)
            fake.uid = fake.add_user("ana@example.com", "correct-horse-9")
            yield c, fake
    finally:
        db._dsn, db._pool = saved


def _login(c):
    r = c.post("/v1/auth/login", json={"email": "ana@example.com", "password": "correct-horse-9"})
    assert r.status_code == 200, r.text
    return r.json()


def _expired_access_token(uid, tid):
    from services.api.core.config import settings
    from services.api.core.security import create_access_token
    return create_access_token(user_id=uid, tenant_id=tid, role="owner", secret=settings.secret_key,
                               ttl_minutes=-1)


def test_login_returns_a_refresh_token_and_lifetime(env):
    c, fake = env
    body = _login(c)
    assert body["refresh_token"] and body["expires_in"] == 3600
    assert len(fake.live(fake.uid)) == 1
    stored = next(iter(fake.sessions.values()))["token_hash"]
    assert stored != body["refresh_token"]               # only the hash is kept


def test_expired_access_token_is_refused_then_renewed(env):
    c, fake = env
    body = _login(c)
    old = _expired_access_token(body["user_id"], body["tenant_id"])
    r = c.get("/v1/auth/me", headers={"Authorization": f"Bearer {old}"})
    assert r.status_code == 401 and "expired" in r.json()["detail"]   # the bug the user hit
    r = c.post("/v1/auth/refresh", json={"refresh_token": body["refresh_token"]})
    assert r.status_code == 200, r.text
    new = r.json()
    assert new["refresh_token"] != body["refresh_token"]   # rotated
    me = c.get("/v1/auth/me", headers={"Authorization": f"Bearer {new['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == "ana@example.com"
    assert len(fake.live(fake.uid)) == 1                    # old revoked, new open


def test_two_tabs_renewing_at_once_do_not_sign_the_user_out(env):
    c, fake = env
    body = _login(c)
    first = c.post("/v1/auth/refresh", json={"refresh_token": body["refresh_token"]})
    second = c.post("/v1/auth/refresh", json={"refresh_token": body["refresh_token"]})
    assert first.status_code == 200
    assert second.status_code == 401 and second.json()["detail"] == "refresh token already rotated"
    assert len(fake.live(fake.uid)) == 1                    # the winner's session survives


def test_reusing_a_rotated_token_later_revokes_every_session(env):
    c, fake = env
    stolen = _login(c)["refresh_token"]
    other_device = _login(c)["refresh_token"]
    assert c.post("/v1/auth/refresh", json={"refresh_token": stolen}).status_code == 200
    for srow in fake.sessions.values():                     # move the rotation out of the grace window
        if srow["revoked_at"] is not None:
            srow["revoked_at"] -= 3600
    r = c.post("/v1/auth/refresh", json={"refresh_token": stolen})
    assert r.status_code == 401 and r.json()["detail"].startswith("session ended")
    assert fake.live(fake.uid) == []                        # everyone is out, thief included
    assert c.post("/v1/auth/refresh", json={"refresh_token": other_device}).status_code == 401


def test_logout_revokes_the_refresh_token(env):
    c, fake = env
    body = _login(c)
    assert c.post("/v1/auth/logout", json={"refresh_token": body["refresh_token"]}).status_code == 200
    assert fake.live(fake.uid) == []
    assert c.post("/v1/auth/refresh", json={"refresh_token": body["refresh_token"]}).status_code == 401


def test_expired_or_unknown_or_missing_refresh_tokens_are_refused(env):
    c, fake = env
    body = _login(c)
    for srow in fake.sessions.values():
        srow["expires_at"] = time.time() - 1
    assert c.post("/v1/auth/refresh", json={"refresh_token": body["refresh_token"]}).status_code == 401
    assert c.post("/v1/auth/refresh", json={"refresh_token": "made-up"}).status_code == 401
    assert c.post("/v1/auth/refresh", json={}).status_code == 401


def test_a_disabled_account_cannot_renew(env):
    c, fake = env
    body = _login(c)
    fake.users[fake.uid]["is_active"] = False
    assert c.post("/v1/auth/refresh", json={"refresh_token": body["refresh_token"]}).status_code == 401


# ── the same round trip on the real schema, where a database is available ──

def test_refresh_round_trip_on_the_real_database():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from services.api.core.db import db
    from services.api.main import app
    with TestClient(app) as c:
        if not db.is_connected:
            pytest.skip("no database reachable (runs in CI against Postgres)")
        email = f"refresh-{uuid.uuid4().hex[:8]}@example.com"
        r = c.post("/v1/auth/register", json={"name": "Refresh Test", "email": email,
                                              "password": "correct-horse-9"})
        assert r.status_code == 200, r.text
        rt = r.json()["refresh_token"]
        r2 = c.post("/v1/auth/refresh", json={"refresh_token": rt})
        assert r2.status_code == 200, r2.text
        assert c.get("/v1/auth/me", headers={"Authorization": f"Bearer {r2.json()['access_token']}"}).json()[
            "email"] == email
        assert c.post("/v1/auth/refresh", json={"refresh_token": rt}).json()["detail"] == \
            "refresh token already rotated"
        assert c.post("/v1/auth/logout", json={"refresh_token": r2.json()["refresh_token"]}).status_code == 200
        assert c.post("/v1/auth/refresh", json={"refresh_token": r2.json()["refresh_token"]}).status_code == 401


def test_a_client_address_that_is_not_an_ip_is_stored_as_null(env):
    # sessions.ip is inet: "testclient", a socket path or a proxy name must not break sign-in
    c, fake = env
    _login(c)
    from services.api.routers.auth import _client_ip

    class R:
        class client:
            host = "testclient"
    assert _client_ip(R) is None
    R.client.host = "203.0.113.9"
    assert _client_ip(R) == "203.0.113.9"
