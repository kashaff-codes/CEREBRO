"""
Connect a Gmail inbox and scan it with the email forensics engine.

    GET    /v1/gmail/auth-url?origin=…   where to send the browser (Google consent)
    POST   /v1/gmail/connect             {code, state, origin}: exchange, encrypt, store
    GET    /v1/gmail/status              connected? which address? last scan?
    POST   /v1/gmail/scan?max=25         read recent messages, score each, return analytics
    DELETE /v1/gmail                     revoke at Google and delete the stored tokens

Scope: gmail.readonly, plus openid/email to know which inbox it is. CEREBRO
never sends, deletes or labels mail, and asks for nothing that would let it.

Tokens: exchanged server-side (the client secret never reaches the browser),
stored AES-GCM-encrypted and bound to the user (auth.py, TokenVault), refreshed
here when expired, never returned to the browser.

`state` is an HMAC over the user id and a timestamp, checked on the way back:
without it, an attacker could complete the flow with their own Google code in a
victim's session ("login CSRF") and have the victim's analyses land in the
attacker's inbox link. It expires after ten minutes.

Google treats gmail.readonly as a restricted scope. Until the OAuth app passes
Google's verification, only the test users listed on its consent screen can
connect; anyone else sees Google's "app not verified" block.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import time
from collections import Counter
from typing import Any
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from services.api.core.config import settings
from services.api.core.db import db
from services.api.core.deps import Principal, current_principal

log = logging.getLogger(__name__)
router = APIRouter(prefix="/gmail", tags=["gmail"])

SCOPES = ["openid", "email", "https://www.googleapis.com/auth/gmail.readonly"]
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GMAIL = "https://gmail.googleapis.com/gmail/v1/users/me"
STATE_TTL = 600


def _origins() -> list[str]:
    raw = getattr(settings, "cors_origins", "") or ""
    items = raw if isinstance(raw, list) else [o.strip() for o in str(raw).split(",")]
    return [o.rstrip("/") for o in items if o.strip()] + ["http://localhost:5173"]


def _redirect_uri(origin: str) -> str:
    origin = origin.rstrip("/")
    if origin not in _origins():
        raise HTTPException(400, "That origin is not allowed to receive Gmail sign-in.")
    return f"{origin}/"


def _require_config() -> None:
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(503, "Gmail connection is not configured on this deployment "
                                 "(GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET).")
    if not settings.token_encryption_key:
        raise HTTPException(503, "TOKEN_ENCRYPTION_KEY is not set; refusing to store mailbox tokens.")


def _sign(user_id: str, ts: int) -> str:
    mac = hmac.new(settings.secret_key.encode(), f"gmail|{user_id}|{ts}".encode(), hashlib.sha256).digest()
    return f"{ts}.{base64.urlsafe_b64encode(mac[:18]).decode().rstrip('=')}"


def _check_state(state: str, user_id: str) -> None:
    try:
        ts_s, _ = state.split(".", 1)
        ts = int(ts_s)
    except ValueError as exc:
        raise HTTPException(400, "Invalid sign-in state.") from exc
    if time.time() - ts > STATE_TTL or not hmac.compare_digest(_sign(user_id, ts), state):
        raise HTTPException(400, "The Gmail sign-in link expired or was not started here. Try again.")


@router.get("/auth-url", summary="Google consent URL for read-only Gmail access")
async def auth_url(origin: str, principal: Principal = Depends(current_principal)) -> dict[str, str]:
    _require_config()
    return {"url": AUTH_URL + "?" + urlencode({
        "client_id": settings.google_client_id, "redirect_uri": _redirect_uri(origin),
        "response_type": "code", "scope": " ".join(SCOPES), "access_type": "offline",
        "prompt": "consent", "include_granted_scopes": "true",
        "state": _sign(principal.user_id, int(time.time())),
    })}


class ConnectIn(BaseModel):
    code: str
    state: str
    origin: str


@router.post("/connect", summary="Finish connecting: exchange the code, store encrypted tokens")
async def connect(body: ConnectIn, principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    _require_config()
    _check_state(body.state, principal.user_id)
    from services.api.routers.auth import GoogleExchangeRequest, google_exchange
    stored = await google_exchange(GoogleExchangeRequest(code=body.code, redirect_uri=_redirect_uri(body.origin)),
                                   principal)
    if SCOPES[-1] not in (stored.get("scopes") or []):
        # Google's consent screen lets people untick individual permissions.
        await db.fetch_unscoped("DELETE FROM cerebro.oauth_credentials WHERE user_id = $1 AND provider = 'google'",
                                principal.user_id)
        raise HTTPException(400, "Google connected, but without permission to read Gmail. Connect again and "
                                 "leave “View your email messages and settings” ticked.")
    token = await _access_token(principal)
    import httpx
    async with httpx.AsyncClient(timeout=15) as c:
        prof = await c.get(f"{GMAIL}/profile", headers={"Authorization": f"Bearer {token}"})
    if prof.status_code != 200:
        raise HTTPException(502, f"Connected, but Gmail refused the first read ({prof.status_code}). "
                                 "Check that the Gmail API is enabled for this Google Cloud project.")
    profile = prof.json()
    return {"connected": True, "email": profile.get("emailAddress"),
            "messages_total": profile.get("messagesTotal"), "scopes": stored.get("scopes", [])}


async def _creds(principal: Principal):
    rows = await db.fetch_unscoped(
        "SELECT access_token, refresh_token, expires_at, scopes, updated_at FROM cerebro.oauth_credentials "
        "WHERE user_id = $1 AND provider = 'google'", principal.user_id)
    return rows[0] if rows else None


async def _access_token(principal: Principal) -> str:
    """A valid access token for this user, refreshing (and re-storing) it when expired."""
    from services.api.core.security import TokenVault
    row = await _creds(principal)
    if not row:
        raise HTTPException(404, "No Gmail account is connected.")
    vault = TokenVault(settings.token_encryption_key)
    from datetime import datetime, timezone
    if row["expires_at"] and row["expires_at"] > datetime.now(timezone.utc).replace(microsecond=0):
        return vault.decrypt(bytes(row["access_token"]), aad=principal.user_id)
    if not row["refresh_token"]:
        raise HTTPException(401, "Gmail access expired. Connect the account again.")
    import httpx
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.post(settings.google_token_url, data={
            "client_id": settings.google_client_id, "client_secret": settings.google_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": vault.decrypt(bytes(row["refresh_token"]), aad=principal.user_id)})
    if r.status_code != 200:
        raise HTTPException(401, "Google refused to refresh access (it may have been revoked). Connect again.")
    tok = r.json()
    await db.fetch_unscoped(
        "UPDATE cerebro.oauth_credentials SET access_token = $2, expires_at = now() + ($3 || ' seconds')::interval, "
        "updated_at = now() WHERE user_id = $1 AND provider = 'google'",
        principal.user_id, vault.encrypt(tok["access_token"], aad=principal.user_id),
        str(int(tok.get("expires_in", 3600))))
    return tok["access_token"]


@router.get("/status", summary="Is a Gmail inbox connected?")
async def gmail_status(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    configured = bool(settings.google_client_id and settings.google_client_secret)
    row = await _creds(principal) if configured else None
    if not row:
        return {"configured": configured, "connected": False}
    try:
        token = await _access_token(principal)
        import httpx
        async with httpx.AsyncClient(timeout=15) as c:
            prof = await c.get(f"{GMAIL}/profile", headers={"Authorization": f"Bearer {token}"})
        p = prof.json() if prof.status_code == 200 else {}
        return {"configured": True, "connected": prof.status_code == 200, "email": p.get("emailAddress"),
                "messages_total": p.get("messagesTotal"), "scopes": list(row["scopes"] or [])}
    except HTTPException as exc:
        return {"configured": True, "connected": False, "problem": exc.detail}


def _header(msg_headers: list[dict], name: str) -> str:
    return next((h.get("value", "") for h in msg_headers if h.get("name", "").lower() == name), "")


@router.post("/scan", summary="Scan recent messages with the forensics engine")
async def scan(request: Request, limit: int = Query(25, alias="max"),
               query: str = "newer_than:30d -in:sent -in:drafts",
               principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    import httpx

    from services.api.routers.email import _persist_email, _verdict_from_score
    from services.ml.email.forensics import analyze_email

    limit = min(max(limit, 1), 100)
    token = await _access_token(principal)
    headers = {"Authorization": f"Bearer {token}"}
    t0 = time.time()
    async with httpx.AsyncClient(timeout=30) as c:
        lst = await c.get(f"{GMAIL}/messages", headers=headers, params={"maxResults": limit, "q": query})
        if lst.status_code != 200:
            raise HTTPException(502, f"Gmail list failed ({lst.status_code}).")
        ids = [m["id"] for m in lst.json().get("messages", [])]
        sem = asyncio.Semaphore(6)

        async def fetch(mid: str):
            async with sem:
                r = await c.get(f"{GMAIL}/messages/{mid}", headers=headers, params={"format": "raw"})
                return mid, (r.json() if r.status_code == 200 else None)
        raws = await asyncio.gather(*(fetch(i) for i in ids))

    classifier = getattr(request.app.state, "models", {}).get("email_classifier")
    rows = []
    for mid, msg in raws:
        if not msg or "raw" not in msg:
            continue
        raw = base64.urlsafe_b64decode(msg["raw"] + "=" * (-len(msg["raw"]) % 4))
        try:
            res = analyze_email(raw)
        except Exception:  # noqa: BLE001 - one unparseable message must not stop the scan
            continue
        score = float(classifier.predict_proba(res.features)) if classifier else res.heuristic_risk
        verdict = _verdict_from_score(score)
        await _persist_email(principal, res, score, "model" if classifier else "heuristic", verdict)
        rows.append({
            "id": mid, "thread": msg.get("threadId"), "subject": res.subject, "from": res.from_addr,
            "from_display": res.from_display, "from_domain": res.from_domain,
            "date": int(msg.get("internalDate", 0)) // 1000, "verdict": verdict, "risk": round(score, 4),
            "spf": res.spf, "dkim": res.dkim, "dmarc": res.dmarc, "dkim_aligned": res.dkim_aligned,
            "indicators": [i.as_dict() for i in sorted(res.indicators, key=lambda i: -i.weight)[:4]],
            "labels": msg.get("labelIds", []),
        })

    def rate(key: str, ok: set[str]) -> float | None:
        seen = [r[key] for r in rows if r[key]]
        return round(sum(v in ok for v in seen) / len(seen), 3) if seen else None
    domains = Counter(r["from_domain"] for r in rows if r["from_domain"])
    return {
        "scanned": len(rows), "requested": len(ids), "query": query, "seconds": round(time.time() - t0, 1),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "auth_pass_rate": {"spf": rate("spf", {"pass"}), "dkim": rate("dkim", {"pass"}),
                           "dmarc": rate("dmarc", {"pass"})},
        "top_indicators": Counter(i["code"] for r in rows for i in r["indicators"]).most_common(8),
        "top_sender_domains": domains.most_common(8),
        "messages": sorted(rows, key=lambda r: -r["risk"]),
        "score_source": "model" if classifier else "heuristic",
    }


@router.delete("", summary="Disconnect Gmail: revoke at Google and delete the tokens")
async def disconnect(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    row = await _creds(principal)
    if row:
        try:
            from services.api.core.security import TokenVault
            import httpx
            vault = TokenVault(settings.token_encryption_key)
            tok = row["refresh_token"] or row["access_token"]
            async with httpx.AsyncClient(timeout=15) as c:
                await c.post("https://oauth2.googleapis.com/revoke",
                             params={"token": vault.decrypt(bytes(tok), aad=principal.user_id)})
        except Exception as exc:  # noqa: BLE001 - deleting locally must still happen
            log.warning("google revoke failed: %s", exc)
        await db.fetch_unscoped("DELETE FROM cerebro.oauth_credentials WHERE user_id = $1 AND provider = 'google'",
                                principal.user_id)
    return {"connected": False}
