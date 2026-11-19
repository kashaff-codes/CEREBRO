import { describe, it, expect, beforeEach } from 'vitest';
import { getToken, setToken, clearToken, authHeaders } from './session';

// Regression coverage for the mobile/iOS auth fix: iOS Safari blocks the
// cross-site session cookie, so auth must ride on a bearer token instead.
describe('session token (mobile-safe auth)', () => {
  beforeEach(() => clearToken());

  it('stores and returns the token', () => {
    setToken('t123');
    expect(getToken()).toBe('t123');
  });

  it('authHeaders carries the bearer token when present', () => {
    setToken('t123');
    expect(authHeaders()).toEqual({ Authorization: 'Bearer t123' });
  });

  it('authHeaders is empty when logged out', () => {
    clearToken();
    expect(authHeaders()).toEqual({});
  });

  it('clearToken removes it', () => {
    setToken('t');
    clearToken();
    expect(getToken()).toBe('');
  });
});

describe('authFetch: expired access tokens renew themselves', () => {
  beforeEach(() => clearToken());

  it('renews once on 401, retries with the new token, and stores the rotated refresh token', async () => {
    const { http, HttpResponse } = await import('msw');
    const { server } = await import('../test/msw-handlers');
    const { authFetch, setSession, getRefreshToken } = await import('./session');
    let refreshCalls = 0;
    server.use(
      http.get('*/v1/metrics/summary', ({ request }) =>
        request.headers.get('authorization') === 'Bearer fresh'
          ? HttpResponse.json({ ok: true })
          : HttpResponse.json({ detail: 'invalid token: token expired' }, { status: 401 }),
      ),
      http.post('*/v1/auth/refresh', async ({ request }) => {
        refreshCalls++;
        const body = (await request.json()) as { refresh_token: string };
        expect(body.refresh_token).toBe('r1');
        return HttpResponse.json({ access_token: 'fresh', refresh_token: 'r2' });
      }),
    );
    setSession('stale', 'r1');
    // two calls at once share one renewal (the refresh token is single-use)
    const [a, b] = await Promise.all([
      authFetch('/v1/metrics/summary'),
      authFetch('/v1/metrics/summary'),
    ]);
    expect(a.status).toBe(200);
    expect(b.status).toBe(200);
    expect(refreshCalls).toBe(1);
    expect(getToken()).toBe('fresh');
    expect(getRefreshToken()).toBe('r2');
  });

  it('ends the session cleanly when renewal is refused', async () => {
    const { http, HttpResponse } = await import('msw');
    const { server } = await import('../test/msw-handlers');
    const { authFetch, setSession, SESSION_ENDED } = await import('./session');
    server.use(
      http.get('*/v1/metrics/summary', () =>
        HttpResponse.json({ detail: 'invalid token: token expired' }, { status: 401 }),
      ),
      http.post('*/v1/auth/refresh', () =>
        HttpResponse.json({ detail: 'session ended: sign in again' }, { status: 401 }),
      ),
    );
    let ended = 0;
    const on = () => ended++;
    window.addEventListener(SESSION_ENDED, on);
    setSession('stale', 'r-old');
    const res = await authFetch('/v1/metrics/summary');
    window.removeEventListener(SESSION_ENDED, on);
    expect(res.status).toBe(401);
    expect(ended).toBe(1);
    expect(getToken()).toBe('');
  });
});
