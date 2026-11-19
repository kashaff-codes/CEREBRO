// Bearer-token session, stored client-side.
//
// The backend also sets httpOnly cookies, but cross-site cookies (frontend on
// *.vercel.app, API on *.onrender.com) are blocked by default on mobile
// Safari/Chrome and in-app browsers (WhatsApp, Instagram). So we ALSO keep the
// tokens here and send `Authorization: Bearer …`, which works everywhere.
//
// The access token lives an hour. When a request comes back 401, `authFetch`
// renews it once with the refresh token (single-use: the server hands back a
// new one each time) and retries. If renewal is impossible the session is
// cleared and a `cerebro:session-ended` event tells the app to show sign-in,
// instead of leaving a signed-in-looking page whose every button fails.
const KEY = 'cerebro_token';
const REFRESH_KEY = 'cerebro_refresh';
const API_BASE = import.meta.env.VITE_API_BASE ?? '';
export const SESSION_ENDED = 'cerebro:session-ended';

const read = (k: string): string => {
  try {
    return localStorage.getItem(k) || '';
  } catch {
    return '';
  }
};

const write = (k: string, v: string): void => {
  try {
    if (v) localStorage.setItem(k, v);
    else localStorage.removeItem(k);
  } catch {
    /* storage unavailable */
  }
};

export const getToken = (): string => read(KEY);
export const getRefreshToken = (): string => read(REFRESH_KEY);

export const setToken = (t: string): void => {
  if (t) write(KEY, t);
};

/** Store what /login, /register or /refresh returned. */
export const setSession = (access: string, refresh?: string | null): void => {
  setToken(access);
  if (refresh) write(REFRESH_KEY, refresh);
};

export const clearToken = (): void => {
  write(KEY, '');
  write(REFRESH_KEY, '');
};

export const authHeaders = (): Record<string, string> => {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
};

const NO_RETRY = ['/v1/auth/login', '/v1/auth/register', '/v1/auth/refresh', '/v1/auth/logout'];

let inflight: Promise<boolean> | null = null;

const pause = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Renew the access token. Concurrent callers share one request; if another tab
 * renewed first (the server says "already rotated"), adopt the tokens it saved.
 */
export function refreshSession(): Promise<boolean> {
  if (inflight) return inflight;
  inflight = (async () => {
    const sent = getRefreshToken();
    try {
      const res = await fetch(`${API_BASE}/v1/auth/refresh`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sent ? { refresh_token: sent } : {}),
      });
      if (res.ok) {
        const data = await res.json();
        setSession(data.access_token, data.refresh_token);
        return true;
      }
      // Another tab may have just rotated the shared token; give its write a moment.
      for (let i = 0; i < 5; i++) {
        if (getRefreshToken() && getRefreshToken() !== sent) return true;
        await pause(100);
      }
      return false;
    } catch {
      return false;
    } finally {
      setTimeout(() => (inflight = null), 0);
    }
  })();
  return inflight;
}

const endSession = () => {
  clearToken();
  window.dispatchEvent(new CustomEvent(SESSION_ENDED));
};

/** fetch() against the API with the session attached, renewed once on 401. */
export async function authFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const go = () =>
    fetch(`${API_BASE}${path}`, {
      ...init,
      credentials: 'include',
      headers: { ...(init.headers ?? {}), ...authHeaders() },
    });
  const res = await go();
  if (res.status !== 401 || NO_RETRY.some((p) => path.startsWith(p))) return res;
  if (!getToken() && !getRefreshToken()) return res; // never signed in: nothing to renew
  if (await refreshSession()) {
    const again = await go();
    if (again.status !== 401) return again;
  }
  endSession();
  return res;
}
