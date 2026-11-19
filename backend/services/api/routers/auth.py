"""
Authentication routes.

This is the server-side replacement for two v1 liabilities:

  - `authService.ts`, which compared passwords with `u.password === password`
    and stored them in plaintext in localStorage.
  - a Gmail token with `gmail.send` scope living in browser localStorage,
    readable by any XSS on the origin.

Here credentials are verified server-side against a scrypt hash, the access
token is a short-lived signed JWT, and it is delivered in an httpOnly cookie so
page JavaScript can never read it. Access tokens live an hour; a refresh token
(30 days) renews them. Refresh tokens are opaque random strings stored only as a
SHA-256 hash, so read access to the sessions table cannot mint a login. Each is
single-use: /refresh revokes it and issues a new one. Presenting a token that
was already rotated (outside a short multi-tab grace window) is treated as theft
and signs the user out everywhere.

The Google OAuth *exchange* endpoint is scaffolded with the contract the
frontend calls, and is explicit that the authorization-code exchange itself is
not yet wired — it returns 501 rather than pretending to succeed.
"""

from __future__ import annotations

import hmac
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

from services.api.core.config import settings
from services.api.core.db import db
from services.api.core.deps import Principal, current_principal
from services.api.core.security import (
    create_access_token,
    generate_token,
    hash_password,
    hash_token,
    session_expiry,
    verify_password,
)

log = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

_COOKIE_NAME = "cerebro_session"
_REFRESH_COOKIE = "cerebro_refresh"
# Two tabs that share a refresh token can both try to renew at the same moment;
# the loser should pick up the winner's tokens, not set off theft detection.
ROTATION_GRACE_SECONDS = 30


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    user_id: str
    tenant_id: str
    role: str
    display_name: str | None
    # The access token is also returned in the body for non-browser API clients
    # (scripts, the parent FYP service). Browsers should rely on the cookie and
    # ignore this field.
    access_token: str
    token_type: str = "bearer"
    # Seconds until access_token expires, and the single-use token that renews
    # it (absent in offline demo mode, where there is no session store).
    expires_in: int = 0
    refresh_token: str | None = None


def _set_session_cookie(response: Response, token: str) -> None:
    samesite = settings.session_cookie_samesite
    if samesite not in {"lax", "none", "strict"}:
        samesite = "lax"
    # SameSite=None is only valid on a Secure cookie. Force Secure in that case
    # even outside production, or the browser silently drops the cookie.
    secure = settings.is_production or samesite == "none"
    response.set_cookie(
        _COOKIE_NAME,
        token,
        max_age=settings.access_token_ttl_minutes * 60,
        httponly=True,
        secure=secure,
        samesite=samesite,  # type: ignore[arg-type]
        path="/",
    )


def _set_refresh_cookie(response: Response, token: str) -> None:
    samesite = settings.session_cookie_samesite
    if samesite not in {"lax", "none", "strict"}:
        samesite = "lax"
    secure = settings.is_production or samesite == "none"
    response.set_cookie(_REFRESH_COOKIE, token, max_age=settings.refresh_token_ttl_days * 86400,
                        httponly=True, secure=secure, samesite=samesite,  # type: ignore[arg-type]
                        path="/v1/auth")


async def _open_session(user_id: Any, request: Request) -> str:
    """Store a new session (hash only) and return its refresh token."""
    refresh_token = generate_token()
    await db.fetch_unscoped(
        """
        INSERT INTO cerebro.sessions (user_id, token_hash, user_agent, ip, expires_at)
        VALUES ($1, $2, $3, $4, $5)
        """,
        user_id, hash_token(refresh_token),
        request.headers.get("user-agent"),
        request.client.host if request.client else None,
        session_expiry(settings.refresh_token_ttl_days),
    )
    return refresh_token


def _issue(response: Response, *, user_id: str, tenant_id: str, role: str, display_name: str | None,
           refresh_token: str | None) -> "LoginResponse":
    access_token = create_access_token(
        user_id=user_id, tenant_id=tenant_id, role=role,
        secret=settings.secret_key, ttl_minutes=settings.access_token_ttl_minutes,
    )
    _set_session_cookie(response, access_token)
    if refresh_token:
        _set_refresh_cookie(response, refresh_token)
    return LoginResponse(user_id=user_id, tenant_id=tenant_id, role=role, display_name=display_name,
                         access_token=access_token, expires_in=settings.access_token_ttl_minutes * 60,
                         refresh_token=refresh_token)


# Stable identifiers for the offline demo principal, so a token issued while the
# database is down stays meaningful across restarts.
OFFLINE_TENANT_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "https://cerebro.app/offline-demo/tenant"))
OFFLINE_USER_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "https://cerebro.app/offline-demo/user"))


def _offline_demo_login(req: LoginRequest, response: Response) -> LoginResponse:
    """
    Sign in the demo account when there is no database to look it up in.

    Why: the Render free-tier PostgreSQL behind this service was deleted when its
    free period ended. From then on every login answered 500, so nobody could
    reach the email, news and network analysers — which need no database at
    all. This keeps them demonstrable while the database is gone.

    Limits, deliberately narrow:
      * only when the database is genuinely unreachable (no pool) — with a
        database, login is the normal path and this is never consulted;
      * only the configured demo account (DEMO_EMAIL / DEMO_PASSWORD), whose
        credentials are already published in the README, so this grants
        nothing a visitor could not already do with the database up;
      * constant-time comparison of both fields;
      * no session row is written (there is nowhere to write it), so the token
        simply expires; OFFLINE_DEMO_LOGIN=false turns the whole path off.
    Any other account gets 503, not 401: its credentials were never checked,
    so "wrong password" would be a lie.
    """
    ok_email = hmac.compare_digest(req.email.lower().encode(), settings.demo_email.lower().encode())
    ok_pw = hmac.compare_digest(req.password.encode(), settings.demo_password.encode())
    if not (settings.offline_demo_login and ok_email and ok_pw):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The account database is offline, so only the demo account can sign in right now.",
        )
    log.warning("database unavailable: issuing an offline demo session")
    token = create_access_token(
        user_id=OFFLINE_USER_ID, tenant_id=OFFLINE_TENANT_ID, role="owner",
        secret=settings.secret_key, ttl_minutes=settings.access_token_ttl_minutes,
    )
    _set_session_cookie(response, token)
    return LoginResponse(user_id=OFFLINE_USER_ID, tenant_id=OFFLINE_TENANT_ID, role="owner",
                         display_name="Demo (offline mode)", access_token=token,
                         expires_in=settings.access_token_ttl_minutes * 60)


@router.post("/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request, response: Response) -> LoginResponse:
    if not db.is_connected:
        return _offline_demo_login(req, response)
    # Uniform failure for "no such user" and "wrong password": revealing which
    # one is true hands an attacker an account-enumeration oracle.
    user = await db.fetch_unscoped(
        """
        SELECT id, tenant_id, role, display_name, password_hash, is_active
        FROM cerebro.users
        WHERE email = $1
        ORDER BY created_at
        LIMIT 1
        """,
        req.email,
    )
    row = user[0] if user else None
    if (
        row is None
        or not row["is_active"]
        or not row["password_hash"]
        or not verify_password(req.password, row["password_hash"])
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid email or password",
        )

    user_id, tenant_id, role = str(row["id"]), str(row["tenant_id"]), row["role"]
    refresh_token = await _open_session(row["id"], request)
    await db.fetch_unscoped(
        "UPDATE cerebro.users SET last_seen_at = now() WHERE id = $1", row["id"]
    )
    return _issue(response, user_id=user_id, tenant_id=tenant_id, role=role,
                  display_name=row["display_name"], refresh_token=refresh_token)


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=200)


@router.post("/register", response_model=LoginResponse, summary="Create an account and sign in")
async def register(req: RegisterRequest, request: Request, response: Response) -> LoginResponse:
    # Each signup gets its own tenant (workspace), so a new account starts with a
    # clean, isolated view — which is exactly what row-level security enforces.
    # Global email uniqueness keeps the sign-in lookup unambiguous.
    existing = await db.fetch_unscoped(
        "SELECT 1 FROM cerebro.users WHERE email = $1 LIMIT 1", req.email
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists. Sign in instead.",
        )

    tenant = await db.fetch_unscoped(
        """
        INSERT INTO cerebro.tenants (name, slug)
        VALUES ($1, $2)
        ON CONFLICT (slug) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        f"{req.name}'s workspace", req.email,
    )
    tenant_id = str(tenant[0]["id"])

    user = await db.fetch_unscoped(
        """
        INSERT INTO cerebro.users (tenant_id, email, display_name, role, password_hash)
        VALUES ($1, $2, $3, 'owner', $4)
        RETURNING id
        """,
        tenant_id, req.email, req.name, hash_password(req.password),
    )
    user_id = str(user[0]["id"])
    refresh_token = await _open_session(user_id, request)
    return _issue(response, user_id=user_id, tenant_id=tenant_id, role="owner",
                  display_name=req.name, refresh_token=refresh_token)


class RefreshRequest(BaseModel):
    # Optional in the body: a browser that kept the httpOnly cookie sends nothing here.
    refresh_token: str | None = None


def _ended(detail: str = "session ended: sign in again") -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail,
                         headers={"WWW-Authenticate": "Bearer"})


@router.post("/refresh", response_model=LoginResponse, summary="Renew the session with a refresh token")
async def refresh(request: Request, response: Response, body: RefreshRequest | None = None) -> LoginResponse:
    token = (body.refresh_token if body else None) or request.cookies.get(_REFRESH_COOKIE)
    if not token:
        raise _ended()
    if not db.is_connected:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="The account database is offline; sign in again.")
    rows = await db.fetch_unscoped(
        """
        SELECT s.id, s.user_id, s.revoked_at,
               (s.revoked_at > now() - make_interval(secs => $2)) AS recently_revoked,
               (s.expires_at <= now()) AS expired,
               u.tenant_id, u.role, u.display_name, u.is_active
        FROM cerebro.sessions s JOIN cerebro.users u ON u.id = s.user_id
        WHERE s.token_hash = $1
        """,
        hash_token(token), float(ROTATION_GRACE_SECONDS),
    )
    row = rows[0] if rows else None
    if row is None:
        raise _ended()
    if row["revoked_at"] is not None:
        if row["recently_revoked"]:
            # Another tab renewed with this token a moment ago: not theft.
            raise _ended("refresh token already rotated")
        # A rotated token used again later means a copy exists somewhere else.
        # Revoke every session this user has, so whoever holds it is out too.
        log.warning("refresh token reuse for user %s: revoking all sessions", row["user_id"])
        await db.fetch_unscoped(
            "UPDATE cerebro.sessions SET revoked_at = now() WHERE user_id = $1 AND revoked_at IS NULL",
            row["user_id"])
        raise _ended()
    if row["expired"] or not row["is_active"]:
        raise _ended()
    # Single use: of two concurrent callers, only one can claim the row.
    claimed = await db.fetch_unscoped(
        "UPDATE cerebro.sessions SET revoked_at = now() WHERE id = $1 AND revoked_at IS NULL RETURNING id",
        row["id"])
    if not claimed:
        raise _ended("refresh token already rotated")
    new_refresh = await _open_session(row["user_id"], request)
    await db.fetch_unscoped("UPDATE cerebro.users SET last_seen_at = now() WHERE id = $1", row["user_id"])
    return _issue(response, user_id=str(row["user_id"]), tenant_id=str(row["tenant_id"]), role=row["role"],
                  display_name=row["display_name"], refresh_token=new_refresh)


@router.post("/logout", summary="End this session: revoke its refresh token, clear cookies")
async def logout(request: Request, response: Response, body: RefreshRequest | None = None) -> dict[str, str]:
    token = (body.refresh_token if body else None) or request.cookies.get(_REFRESH_COOKIE)
    if token and db.is_connected:
        try:
            await db.fetch_unscoped(
                "UPDATE cerebro.sessions SET revoked_at = now() WHERE token_hash = $1 AND revoked_at IS NULL",
                hash_token(token))
        except Exception as exc:  # noqa: BLE001 - signing out locally must still succeed
            log.warning("could not revoke session on logout: %s", exc)
    response.delete_cookie(_COOKIE_NAME, path="/")
    response.delete_cookie(_REFRESH_COOKIE, path="/v1/auth")
    return {"status": "logged_out"}


@router.get("/me", summary="The authenticated principal")
async def me(principal: Principal = Depends(current_principal)) -> dict[str, Any]:
    if not db.is_connected and principal.user_id == OFFLINE_USER_ID:
        return {"user_id": OFFLINE_USER_ID, "tenant_id": OFFLINE_TENANT_ID, "email": settings.demo_email,
                "display_name": "Demo (offline mode)", "role": "owner", "last_seen_at": None,
                "offline": True}
    row = await db.fetch_unscoped(
        """
        SELECT id, tenant_id, email, display_name, role, last_seen_at
        FROM cerebro.users WHERE id = $1
        """,
        principal.user_id,
    )
    if not row:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="user not found")
    u = row[0]
    return {
        "user_id": str(u["id"]),
        "tenant_id": str(u["tenant_id"]),
        "email": str(u["email"]),
        "display_name": u["display_name"],
        "role": u["role"],
        "last_seen_at": u["last_seen_at"].isoformat() if u["last_seen_at"] else None,
    }


class GoogleExchangeRequest(BaseModel):
    code: str = Field(..., description="OAuth 2.0 authorization code from Google")
    redirect_uri: str


@router.post("/google/exchange", summary="Exchange a Google auth code for stored, encrypted tokens")
async def google_exchange(
    req: GoogleExchangeRequest,
    principal: Principal = Depends(current_principal),
) -> dict[str, Any]:
    """
    Server-side Gmail OAuth. The browser sends only the short-lived
    authorization code; the code-for-token swap happens here, where the client
    secret lives, and the resulting tokens are encrypted at rest and never
    returned to the browser.

    This is the fix for v1 keeping a `gmail.send`-scoped access token in
    localStorage: with this flow the token exists only server-side, AES-GCM
    encrypted and bound to the user via AAD, so a database dump does not hand an
    attacker live mailbox access and a row copied between users won't decrypt.
    """
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google OAuth is not configured. Set GOOGLE_CLIENT_ID and "
                   "GOOGLE_CLIENT_SECRET in the environment.",
        )
    if not settings.token_encryption_key:
        # Refuse rather than store a token we cannot protect.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TOKEN_ENCRYPTION_KEY is not set; refusing to store OAuth tokens in the clear.",
        )

    import httpx

    from services.api.core.security import TokenVault

    # 1. Swap the authorization code for tokens at Google's token endpoint.
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                settings.google_token_url,
                data={
                    "code": req.code,
                    "client_id": settings.google_client_id,
                    "client_secret": settings.google_client_secret,
                    "redirect_uri": req.redirect_uri,
                    "grant_type": "authorization_code",
                },
                headers={"Accept": "application/json"},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach Google's token endpoint.",
        ) from exc

    if resp.status_code != 200:
        # Google returns {error, error_description} on a bad/expired code. Do not
        # echo the raw body — it can contain the code; log server-side instead.
        log.warning("google token exchange failed: %s", resp.status_code)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code exchange was rejected by Google (expired or already used).",
        )

    tokens = resp.json()
    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Google response did not contain an access token.",
        )
    refresh_token = tokens.get("refresh_token")  # only present on first consent
    expires_in = int(tokens.get("expires_in", 3600))
    scopes = (tokens.get("scope") or "").split()

    # 2. Encrypt both tokens, binding each to this user via AAD.
    vault = TokenVault(settings.token_encryption_key)
    enc_access = vault.encrypt(access_token, aad=principal.user_id)
    enc_refresh = vault.encrypt(refresh_token, aad=principal.user_id) if refresh_token else None

    # 3. Upsert. If Google withheld a refresh token (repeat consent), keep the
    #    one we already stored rather than nulling it.
    await db.fetch_unscoped(
        """
        INSERT INTO cerebro.oauth_credentials
            (user_id, provider, scopes, access_token, refresh_token, expires_at, updated_at)
        VALUES ($1, 'google', $2, $3, $4, now() + ($5 || ' seconds')::interval, now())
        ON CONFLICT (user_id, provider) DO UPDATE SET
            scopes        = EXCLUDED.scopes,
            access_token  = EXCLUDED.access_token,
            refresh_token = COALESCE(EXCLUDED.refresh_token, cerebro.oauth_credentials.refresh_token),
            expires_at    = EXCLUDED.expires_at,
            updated_at    = now()
        """,
        principal.user_id, scopes, enc_access, enc_refresh, str(expires_in),
    )

    # The browser gets confirmation and metadata only — never the token itself.
    return {
        "status": "stored",
        "provider": "google",
        "scopes": scopes,
        "expires_in": expires_in,
        "has_refresh_token": refresh_token is not None,
    }
