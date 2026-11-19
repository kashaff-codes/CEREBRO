import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Inbox, LogOut, RefreshCw, ScanLine } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { deleteJSON, getJSON, postJSON } from '../services/apiService';
import { playCyberSFX } from '../utils/audio';

/**
 * Mail forensics.
 *
 * Two ways in. Connect a Gmail inbox (read-only) and the forensics engine
 * scores the recent mail in it: authentication results, lookalike senders,
 * link tricks, each message with the signals that drove its score. Or paste
 * one raw message. Either way the score is ~35 deterministic signals, each
 * listed, not a model's opinion.
 */

// ── shapes ─────────────────────────────────────────────────────────────────

interface Indicator {
  code: string;
  severity: string;
  weight: number;
  detail: string;
}
interface EmailResult {
  from_display: string | null;
  from_addr: string | null;
  from_domain: string | null;
  subject: string | null;
  spf: string | null;
  dkim: string | null;
  dmarc: string | null;
  dkim_aligned: boolean | null;
  verdict: string;
  risk_score: number;
  score_source: string;
  indicators: Indicator[];
}
interface GmailStatus {
  configured: boolean;
  connected: boolean;
  email?: string;
  messages_total?: number;
  problem?: string;
}
interface InboxMessage {
  id: string;
  subject: string | null;
  from: string | null;
  from_display: string | null;
  from_domain: string | null;
  date: number;
  verdict: string;
  risk: number;
  spf: string | null;
  dkim: string | null;
  dmarc: string | null;
  indicators: Indicator[];
}
interface InboxScan {
  scanned: number;
  requested: number;
  seconds: number;
  verdicts: Record<string, number>;
  auth_pass_rate: { spf: number | null; dkim: number | null; dmarc: number | null };
  top_indicators: [string, number][];
  top_sender_domains: [string, number][];
  messages: InboxMessage[];
  score_source: string;
}

// ── vocabulary ─────────────────────────────────────────────────────────────

const VERDICT: Record<string, { word: string; tone: string; glyph: string }> = {
  phishing: { word: 'Phishing', tone: '#D2654E', glyph: '✕' },
  suspicious: { word: 'Suspicious', tone: '#D08A4E', glyph: '▲' },
  low_confidence: { word: 'Worth a look', tone: '#C9A45C', glyph: '◆' },
  benign: { word: 'Clean', tone: '#86AD90', glyph: '●' },
};
const verdictOf = (v: string) => VERDICT[v] ?? VERDICT.low_confidence;

const SEVERITY_TONE: Record<string, string> = {
  critical: '#D2654E',
  high: '#D08A4E',
  medium: '#C9A45C',
  low: '#A29C90',
  info: '#6F6A62',
};

const SAMPLE = `From: "PayPal Security" <alert@paypa1.com>
Reply-To: recover@secure-paypa1.ru
To: you@example.com
Subject: Urgent: your account has been limited

Dear customer,

We detected unusual activity. Your account is suspended. Verify your
identity immediately at http://paypa1.com/login-verify or it will be
permanently closed within 24 hours.

PayPal Security Team`;

const pct = (x: number | null | undefined) => (x == null ? '—' : `${Math.round(x * 100)}%`);
const when = (s: number) =>
  s ? new Date(s * 1000).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) : '';

// ── parts ──────────────────────────────────────────────────────────────────

const AuthMark: React.FC<{ name: string; value: string | null }> = ({ name, value }) => {
  const ok = value === 'pass';
  const bad = value === 'fail' || value === 'softfail';
  return (
    <span
      className="font-mono text-[11px]"
      style={{ color: ok ? '#86AD90' : bad ? '#D2654E' : '#6F6A62' }}
      title={`${name}: ${value ?? 'not present'}`}
    >
      {name} {ok ? '✓' : bad ? '✕' : '–'}
    </span>
  );
};

const IndicatorList: React.FC<{ items: Indicator[] }> = ({ items }) =>
  items.length === 0 ? (
    <p className="text-[13px] text-sage-300">No risk indicators found.</p>
  ) : (
    <ul className="space-y-2.5">
      {items.map((ind, i) => (
        <li
          key={i}
          className="rounded-lg border px-4 py-3"
          style={{ borderColor: `${SEVERITY_TONE[ind.severity] ?? '#38363A'}40` }}
        >
          <div className="mb-1 flex items-center justify-between gap-3">
            <span className="font-mono text-[12px] tracking-wide text-slate-100">{ind.code}</span>
            <span
              className="text-[11px] uppercase tracking-wider"
              style={{ color: SEVERITY_TONE[ind.severity] ?? '#A29C90' }}
            >
              {ind.severity} · +{ind.weight.toFixed(2)}
            </span>
          </div>
          <p className="text-[13px] leading-relaxed text-slate-400">{ind.detail}</p>
        </li>
      ))}
    </ul>
  );

const RateBar: React.FC<{ label: string; value: number | null; i: number }> = ({
  label,
  value,
  i,
}) => (
  <li className="grid grid-cols-[64px_1fr_44px] items-center gap-3 text-[13px]">
    <span className="text-slate-300">{label}</span>
    <span className="h-[6px] overflow-hidden rounded-full bg-[rgba(236,230,217,0.06)]">
      <span
        className="animate-rise block h-full rounded-full"
        style={{
          width: `${Math.max(1.5, (value ?? 0) * 100)}%`,
          background: value != null && value < 0.8 ? '#D08A4E' : '#86AD90',
          animationDelay: `${i * 70}ms`,
        }}
      />
    </span>
    <span className="text-right font-mono text-[12px] text-slate-400">{pct(value)}</span>
  </li>
);

const CountList: React.FC<{ rows: [string, number][]; mono?: boolean }> = ({ rows, mono }) => {
  const top = Math.max(...rows.map((r) => r[1]), 1);
  return (
    <ul className="space-y-2.5">
      {rows.map(([k, n], i) => (
        <li key={k} className="grid grid-cols-[1fr_90px_28px] items-center gap-3 text-[13px]">
          <span className={`truncate text-slate-300 ${mono ? 'font-mono text-[12px]' : ''}`}>
            {k}
          </span>
          <span className="h-[5px] overflow-hidden rounded-full bg-[rgba(236,230,217,0.06)]">
            <span
              className="animate-rise block h-full rounded-full bg-brass-400"
              style={{ width: `${(n / top) * 100}%`, animationDelay: `${i * 60}ms` }}
            />
          </span>
          <span className="text-right font-mono text-[12px] text-slate-400">{n}</span>
        </li>
      ))}
    </ul>
  );
};

// ── Gmail ──────────────────────────────────────────────────────────────────

/** Finish the Google round trip if this page load is the return leg. */
function takeOAuthReturn(): { code: string; state: string } | null {
  const q = new URLSearchParams(window.location.search);
  const code = q.get('code');
  const state = q.get('state');
  if (!code || !state || !/^\d+\./.test(state)) return null;
  window.history.replaceState({}, '', window.location.pathname);
  return { code, state };
}

const GmailPanel: React.FC = () => {
  const { addToHistory } = useAuth();
  const [status, setStatus] = useState<GmailStatus | null>(null);
  const [busy, setBusy] = useState<'connect' | 'scan' | 'disconnect' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [scan, setScan] = useState<InboxScan | null>(null);
  const [count, setCount] = useState(25);
  const [open, setOpen] = useState<string | null>(null);
  const started = useRef(false);

  const refresh = useCallback(async () => {
    try {
      setStatus(await getJSON<GmailStatus>('/v1/gmail/status'));
    } catch (e: any) {
      setStatus({ configured: false, connected: false, problem: e.message });
    }
  }, []);

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    const back = takeOAuthReturn();
    if (!back) {
      refresh();
      return;
    }
    setBusy('connect');
    postJSON<GmailStatus>('/v1/gmail/connect', { ...back, origin: window.location.origin })
      .then((s) => {
        setStatus({ ...s, configured: true });
        playCyberSFX('success');
      })
      .catch((e) => {
        setError(e.message);
        refresh();
      })
      .finally(() => setBusy(null));
  }, [refresh]);

  const connect = async () => {
    setError(null);
    setBusy('connect');
    try {
      const { url } = await getJSON<{ url: string }>(
        `/v1/gmail/auth-url?origin=${encodeURIComponent(window.location.origin)}`,
      );
      window.location.assign(url);
    } catch (e: any) {
      setError(e.message);
      setBusy(null);
    }
  };

  const runScan = async () => {
    setError(null);
    setBusy('scan');
    playCyberSFX('scan');
    try {
      const r = await postJSON<InboxScan>(`/v1/gmail/scan?max=${count}`);
      setScan(r);
      const bad = (r.verdicts.phishing ?? 0) + (r.verdicts.suspicious ?? 0);
      playCyberSFX(bad ? 'alarm' : 'success');
      addToHistory({
        id: Math.random().toString(36).slice(2, 11),
        date: new Date().toISOString(),
        type: 'CYBER',
        summary: `Inbox scan: ${r.scanned} messages, ${r.verdicts.phishing ?? 0} phishing, ${r.verdicts.suspicious ?? 0} suspicious`,
        result: r.verdicts.phishing ? 'Critical' : bad ? 'High' : 'Safe',
      });
    } catch (e: any) {
      playCyberSFX('alarm');
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  const disconnect = async () => {
    setBusy('disconnect');
    try {
      await deleteJSON('/v1/gmail');
      setScan(null);
      await refresh();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  const connected = status?.connected;

  return (
    <div className="space-y-6">
      <section className="panel p-7">
        <div className="grid gap-8 md:grid-cols-[1fr_auto] md:items-center">
          <div>
            <p className="label-caps mb-3">Your inbox</p>
            {status === null || busy === 'connect' ? (
              <p className="text-[15px] text-slate-400">
                {busy === 'connect' ? 'Connecting to Gmail…' : 'Checking the connection…'}
              </p>
            ) : connected ? (
              <>
                <p className="font-display text-[30px] leading-tight text-slate-100">
                  {status.email}
                </p>
                <p className="mt-2 text-[14px] text-slate-400">
                  Connected, read-only
                  {status.messages_total != null &&
                    ` · ${status.messages_total.toLocaleString('en-GB')} messages in the mailbox`}
                </p>
              </>
            ) : (
              <>
                <p className="max-w-[62ch] text-[15px] leading-relaxed text-slate-300">
                  Connect a Gmail account and the recent mail in it is scored: who really sent each
                  message, whether it passed SPF, DKIM and DMARC, and what in it looks wrong.
                </p>
                <p className="mt-3 max-w-[62ch] text-[12.5px] leading-relaxed text-slate-500">
                  Read-only access. CEREBRO cannot send, delete or move mail. Google’s tokens are
                  stored encrypted on the server and never reach this browser; disconnecting revokes
                  them at Google.
                  {!status.configured &&
                    ' Gmail is not configured on this deployment yet (an administrator needs to add the Google client).'}
                </p>
              </>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            {connected ? (
              <>
                <label className="sr-only" htmlFor="scan-count">
                  Messages to scan
                </label>
                <select
                  id="scan-count"
                  value={count}
                  onChange={(e) => setCount(Number(e.target.value))}
                  className="rounded-lg border border-slate-800 bg-[#101013] px-3 py-3 text-[13px] text-slate-200 outline-none focus:border-brass-600"
                >
                  {[25, 50, 100].map((n) => (
                    <option key={n} value={n}>
                      Last {n} messages
                    </option>
                  ))}
                </select>
                <button
                  onClick={runScan}
                  disabled={busy !== null}
                  className="flex items-center gap-2 rounded-lg bg-brass-400 px-6 py-3 text-[14px] font-semibold text-[#16120A] transition-colors hover:bg-brass-300 disabled:opacity-40"
                >
                  {busy === 'scan' ? (
                    <RefreshCw className="h-4 w-4 animate-spin" strokeWidth={2} />
                  ) : (
                    <ScanLine className="h-4 w-4" strokeWidth={2} />
                  )}
                  {busy === 'scan' ? 'Scanning…' : 'Scan inbox'}
                </button>
                <button
                  onClick={disconnect}
                  disabled={busy !== null}
                  className="flex items-center gap-2 rounded-lg border border-slate-700 px-4 py-3 text-[13px] text-slate-400 transition-colors hover:border-slate-500 hover:text-slate-200 disabled:opacity-40"
                >
                  <LogOut className="h-4 w-4" strokeWidth={1.6} /> Disconnect
                </button>
              </>
            ) : (
              <button
                onClick={connect}
                disabled={busy !== null || !status?.configured}
                className="flex items-center gap-2 rounded-lg bg-brass-400 px-6 py-3 text-[14px] font-semibold text-[#16120A] transition-colors hover:bg-brass-300 disabled:opacity-40"
              >
                <Inbox className="h-4 w-4" strokeWidth={2} /> Connect Gmail
              </button>
            )}
          </div>
        </div>
        {status?.problem && connected === false && status.configured && (
          <p className="mt-4 text-[13px] text-copper-300">{status.problem}</p>
        )}
      </section>

      {error && (
        <p
          role="alert"
          className="rounded-xl border border-signal-800 bg-signal-950/40 px-5 py-4 text-[14px] text-signal-200"
        >
          {error}
        </p>
      )}

      {scan && (
        <>
          <section className="panel p-7">
            <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
              {(['phishing', 'suspicious', 'low_confidence', 'benign'] as const).map((k) => {
                const v = VERDICT[k];
                const n = scan.verdicts[k] ?? 0;
                return (
                  <div key={k}>
                    <p className="label-caps mb-2">
                      <span style={{ color: v.tone }} aria-hidden="true">
                        {v.glyph}
                      </span>{' '}
                      {v.word}
                    </p>
                    <p
                      className="font-display text-[38px] leading-none"
                      style={{ color: n && k !== 'benign' ? v.tone : '#ECE6D9' }}
                    >
                      {n}
                    </p>
                  </div>
                );
              })}
            </div>
            <p className="mt-5 text-[12.5px] text-slate-500">
              {scan.scanned} of {scan.requested} recent messages scored in {scan.seconds}s (last 30
              days, excluding sent and drafts) · scores from the{' '}
              {scan.score_source === 'model' ? 'trained classifier' : 'rule engine'}
            </p>
          </section>

          <section className="grid gap-6 lg:grid-cols-3">
            <div className="panel p-7">
              <p className="label-caps mb-5">Authentication passed</p>
              <ul className="space-y-3">
                <RateBar label="SPF" value={scan.auth_pass_rate.spf} i={0} />
                <RateBar label="DKIM" value={scan.auth_pass_rate.dkim} i={1} />
                <RateBar label="DMARC" value={scan.auth_pass_rate.dmarc} i={2} />
              </ul>
              <p className="mt-5 text-[12px] leading-relaxed text-slate-500">
                Share of messages whose sender proved it was allowed to send for its domain.
              </p>
            </div>
            <div className="panel p-7">
              <p className="label-caps mb-5">Most common warnings</p>
              {scan.top_indicators.length ? (
                <CountList rows={scan.top_indicators} mono />
              ) : (
                <p className="text-[13px] text-slate-400">None.</p>
              )}
            </div>
            <div className="panel p-7">
              <p className="label-caps mb-5">Who writes to you most</p>
              <CountList rows={scan.top_sender_domains} />
            </div>
          </section>

          <section className="panel p-7">
            <p className="label-caps mb-5">Messages, riskiest first</p>
            <ul className="divide-y divide-[var(--line)]">
              {scan.messages.map((m) => {
                const v = verdictOf(m.verdict);
                const isOpen = open === m.id;
                return (
                  <li key={m.id}>
                    <button
                      onClick={() => setOpen(isOpen ? null : m.id)}
                      aria-expanded={isOpen}
                      className="grid w-full grid-cols-[120px_1fr_auto] items-center gap-5 py-3.5 text-left transition-colors hover:bg-[rgba(236,230,217,0.02)]"
                    >
                      <span
                        className="flex items-center gap-2 text-[13px]"
                        style={{ color: v.tone }}
                      >
                        <span aria-hidden="true">{v.glyph}</span>
                        {v.word}
                        <span className="font-mono text-[11px] text-slate-500">{pct(m.risk)}</span>
                      </span>
                      <span className="min-w-0">
                        <span className="block truncate text-[14px] text-slate-100">
                          {m.subject || '(no subject)'}
                        </span>
                        <span className="block truncate text-[12.5px] text-slate-500">
                          {m.from_display ? `${m.from_display} · ` : ''}
                          {m.from}
                        </span>
                      </span>
                      <span className="flex items-center gap-3">
                        <AuthMark name="SPF" value={m.spf} />
                        <AuthMark name="DKIM" value={m.dkim} />
                        <AuthMark name="DMARC" value={m.dmarc} />
                        <span className="w-14 text-right text-[12px] text-slate-500">
                          {when(m.date)}
                        </span>
                      </span>
                    </button>
                    {isOpen && (
                      <div className="animate-rise pb-5 pl-[140px]">
                        <IndicatorList items={m.indicators} />
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </section>
        </>
      )}
    </div>
  );
};

// ── page ───────────────────────────────────────────────────────────────────

export const EmailScanner: React.FC = () => {
  const [raw, setRaw] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<EmailResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { addToHistory } = useAuth();

  const analyze = async () => {
    if (!raw.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    playCyberSFX('scan');
    try {
      const data = await postJSON<EmailResult>('/v1/analyze/email', {
        raw_message: raw,
        generate_explanation: false,
      });
      setResult(data);
      playCyberSFX('success');
      addToHistory({
        id: Math.random().toString(36).slice(2, 11),
        date: new Date().toISOString(),
        type: 'CYBER',
        summary: `${data.verdict.toUpperCase()} — ${data.from_addr || 'unknown sender'}`,
        result:
          data.verdict === 'phishing'
            ? 'Critical'
            : data.verdict === 'suspicious'
              ? 'High'
              : 'Safe',
      });
    } catch (e: any) {
      playCyberSFX('alarm');
      setError(e.message || 'Could not analyze the message.');
    } finally {
      setLoading(false);
    }
  };

  const v = result ? verdictOf(result.verdict) : null;

  return (
    <div className="space-y-8">
      <GmailPanel />

      <div className="flex items-center gap-4">
        <div className="hairline flex-1" />
        <span className="label-caps">or examine one message</span>
        <div className="hairline flex-1" />
      </div>

      <section className="panel p-7">
        <label htmlFor="raw-email" className="label-caps mb-3 block">
          Mail forensics · raw message
        </label>
        <textarea
          id="raw-email"
          rows={9}
          value={raw}
          onChange={(e) => setRaw(e.target.value)}
          placeholder="Paste the full source: From:, Subject:, Received: headers and the body. In Gmail: ⋮ → Show original → Copy to clipboard."
          className="custom-scrollbar w-full resize-y rounded-xl border border-slate-800 bg-[#101013] px-5 py-4 font-mono text-[13px] leading-relaxed text-slate-200 outline-none transition-colors placeholder:font-sans placeholder:text-slate-600 focus:border-brass-600"
        />
        <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
          <button
            onClick={() => {
              setRaw(SAMPLE);
              playCyberSFX('hover');
            }}
            className="text-[13px] text-slate-500 underline-offset-4 transition-colors hover:text-slate-200 hover:underline"
          >
            Load sample phishing email
          </button>
          <button
            onClick={analyze}
            disabled={loading || !raw.trim()}
            className="flex items-center gap-2 rounded-lg border border-brass-700 px-5 py-2.5 text-[13px] text-brass-200 transition-colors hover:bg-brass-950 disabled:opacity-40"
          >
            {loading && <RefreshCw className="h-4 w-4 animate-spin" strokeWidth={1.6} />}
            {loading ? 'Analysing…' : 'Analyse message'}
          </button>
        </div>
      </section>

      {error && (
        <p
          role="alert"
          className="rounded-xl border border-signal-800 bg-signal-950/40 px-5 py-4 text-[14px] text-signal-200"
        >
          {error}
        </p>
      )}

      {result && v && (
        <section className="panel animate-rise p-8">
          <div className="grid gap-8 md:grid-cols-[1fr_auto] md:items-start">
            <div>
              <p className="label-caps mb-3">Verdict</p>
              <p className="font-display text-[42px] leading-none" style={{ color: v.tone }}>
                <span className="mr-3 align-middle text-[28px]" aria-hidden="true">
                  {v.glyph}
                </span>
                {v.word}
              </p>
              <p className="mt-3 text-[14px] text-slate-400">
                Risk {pct(result.risk_score)} · scored by the{' '}
                {result.score_source === 'model' ? 'trained classifier' : 'rule engine'}
              </p>
            </div>
            <div className="flex gap-4 md:pt-8">
              <AuthMark name="SPF" value={result.spf} />
              <AuthMark name="DKIM" value={result.dkim} />
              <AuthMark name="DMARC" value={result.dmarc} />
            </div>
          </div>
          <div className="hairline my-7" />
          <dl className="grid gap-x-8 gap-y-3 text-[14px] md:grid-cols-[120px_1fr]">
            <dt className="label-caps pt-0.5">From</dt>
            <dd className="text-slate-200">
              {result.from_display}{' '}
              <span className="text-slate-500">&lt;{result.from_addr}&gt;</span>
            </dd>
            <dt className="label-caps pt-0.5">Domain</dt>
            <dd className="font-mono text-[13px] text-slate-200">{result.from_domain}</dd>
            <dt className="label-caps pt-0.5">Subject</dt>
            <dd className="text-slate-200">{result.subject}</dd>
            <dt className="label-caps pt-0.5">DKIM aligned</dt>
            <dd style={{ color: result.dkim_aligned === false ? '#D2654E' : '#ECE6D9' }}>
              {result.dkim_aligned == null ? 'not signed' : result.dkim_aligned ? 'yes' : 'no'}
            </dd>
          </dl>
          <p className="label-caps mb-4 mt-8">Indicators ({result.indicators.length})</p>
          <IndicatorList items={result.indicators} />
        </section>
      )}
    </div>
  );
};
