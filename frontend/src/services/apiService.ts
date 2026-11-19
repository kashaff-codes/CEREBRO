// CEREBRO API client.
//
// Renamed from geminiService.ts: it no longer talks to Gemini. Detection is
// performed by the FastAPI backend (deterministic features + fitted models);
// this module just calls it. Requests go to the `/v1` namespace, which the Vite
// dev server proxies to the API (see vite.config.ts) so httpOnly session cookies
// work the same in dev and prod. `VITE_API_BASE` overrides the origin in builds
// that call a remote API directly.
import { NewsAnalysisResult } from '../types';
import { authFetch } from './session';

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  // Bearer covers mobile; the cookie is a desktop bonus. An expired access token
  // is renewed once here, so callers never see "token expired".
  const response = await authFetch(path, init);
  if (response.status === 401) throw new Error('Your session has ended. Please sign in again.');

  if (!response.ok) {
    const errText = await response.text();
    let detail: string | undefined;
    try {
      const parsed = JSON.parse(errText);
      // FastAPI puts human-readable errors in `detail`; our 500 handler uses `message`.
      detail = parsed?.detail || parsed?.message || parsed?.error;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail || `Request failed (${response.status} ${response.statusText})`);
  }

  const type = response.headers.get('content-type') ?? '';
  return (type.includes('json') ? response.json() : response.text()) as Promise<T>;
}

export const getJSON = <T>(path: string): Promise<T> => request<T>(path);

export const postJSON = <T>(path: string, body?: unknown): Promise<T> =>
  request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const deleteJSON = <T>(path: string): Promise<T> => request<T>(path, { method: 'DELETE' });

/** Multipart upload of one file under the field name `file`. */
export const postFile = <T>(path: string, file: Blob, name: string): Promise<T> => {
  const form = new FormData();
  form.append('file', file, name);
  return request<T>(path, { method: 'POST', body: form });
};

export const analyzeNewsText = async (
  text: string,
  depth: 'standard' | 'deep' = 'standard',
): Promise<NewsAnalysisResult> => {
  if (!text) throw new Error('Text is required for analysis');
  return postJSON<NewsAnalysisResult>('/v1/analyze/news', { text, depth });
};
