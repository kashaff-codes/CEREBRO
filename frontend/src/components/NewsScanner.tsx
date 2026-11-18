import React, { useEffect, useRef, useState } from 'react';
import { analyzeNewsText } from '../services/apiService';
import type { CheckedClaim, NewsAnalysisResult, ScoreComponent } from '../types';
import { useAuth } from '../context/AuthContext';
import { jsPDF } from 'jspdf';
import { playCyberSFX } from '../utils/audio';
import { ArrowUpRight, Download, Search, Telescope, Share2 } from 'lucide-react';

/**
 * Verify a claim.
 *
 * The page is built around one promise: every number can be taken apart. The
 * score is shown with the components it was computed from, each claim with the
 * passages that support or refute it, and each passage with its source and how
 * much that source is trusted. Verdict colours always travel with a word and a
 * glyph, never alone.
 */

const VERDICT: Record<string, { tone: string; glyph: string; line: string }> = {
  Credible: { tone: '#86AD90', glyph: '●', line: 'The evidence supports it.' },
  Disputed: { tone: '#C9A45C', glyph: '◆', line: 'Credible sources disagree.' },
  Unverified: { tone: '#A29C90', glyph: '○', line: 'Not enough evidence either way.' },
  Misleading: { tone: '#D08A4E', glyph: '▲', line: 'Partly wrong, or missing context.' },
  'Fake News': { tone: '#D2654E', glyph: '✕', line: 'The evidence contradicts it.' },
};
const verdictOf = (v: string) => VERDICT[v] ?? VERDICT.Unverified;

const COMPONENT_TONE: Record<string, string> = {
  evidence: '#C9A45C',
  fact_check: '#86AD90',
  language: '#7C776E',
  prior: '#38363A',
};

const STANCE = {
  entail: { glyph: '✓', word: 'Supports', tone: '#86AD90' },
  contradict: { glyph: '✕', word: 'Refutes', tone: '#D2654E' },
  neutral: { glyph: '–', word: 'Context', tone: '#7C776E' },
} as const;

const EXAMPLES = [
  'eiffel tower is in london right',
  'Is it true the Great Wall is visible from space, goldfish have a 3 second memory, and Einstein failed math?',
  'whats going on with the nvidia case, did jensen huang get sued',
];

const STEPS = [
  'Reading what you pasted',
  'Separating the claims',
  'Searching the web and the archive',
  'Weighing each source',
  'Scoring',
];

/** Animated count-up for the headline score. */
function useCountUp(target: number | null, ms = 1100) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (target == null) return;
    const start = performance.now();
    let raf = 0;
    const tick = (t: number) => {
      const p = Math.min(1, (t - start) / ms);
      setV(Math.round(target * (1 - Math.pow(1 - p, 3))));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v;
}

const ScoreRing: React.FC<{ score: number; tone: string }> = ({ score, tone }) => {
  const shown = useCountUp(score);
  const r = 70;
  const c = 2 * Math.PI * r;
  return (
    <svg
      width="176"
      height="176"
      viewBox="0 0 176 176"
      role="img"
      aria-label={`Credibility ${score} out of 100`}
    >
      <circle cx="88" cy="88" r={r} fill="none" stroke="rgba(236,230,217,0.08)" strokeWidth="2" />
      {Array.from({ length: 40 }).map((_, i) => {
        const a = (i / 40) * Math.PI * 2 - Math.PI / 2;
        return (
          <line
            key={i}
            x1={88 + Math.cos(a) * 80}
            y1={88 + Math.sin(a) * 80}
            x2={88 + Math.cos(a) * (i % 10 === 0 ? 74 : 77)}
            y2={88 + Math.sin(a) * (i % 10 === 0 ? 74 : 77)}
            stroke="rgba(236,230,217,0.18)"
          />
        );
      })}
      <circle
        cx="88"
        cy="88"
        r={r}
        fill="none"
        stroke={tone}
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeDasharray={`${(score / 100) * c} ${c}`}
        transform="rotate(-90 88 88)"
        className="animate-draw"
        style={{ ['--len' as string]: c }}
      />
      <text
        x="88"
        y="96"
        textAnchor="middle"
        fill="#ECE6D9"
        fontSize="54"
        fontFamily="'Bodoni Moda', serif"
      >
        {shown}
      </text>
      <text
        x="88"
        y="120"
        textAnchor="middle"
        fill="#7C776E"
        fontSize="10"
        letterSpacing="3"
        fontFamily="'Instrument Sans'"
      >
        OUT OF 100
      </text>
    </svg>
  );
};

/** How the score was built: one bar split by each component's share, and its table twin. */
const Breakdown: React.FC<{ parts: ScoreComponent[] }> = ({ parts }) => {
  const used = parts.filter((p) => p.score !== null && p.share > 0);
  return (
    <section className="panel p-7">
      <div className="mb-5 flex items-baseline justify-between">
        <h3 className="font-display text-[22px] text-slate-100">How this score was built</h3>
        <p className="file-no">weighted mean of the parts below</p>
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-900" aria-hidden="true">
        {used.map((p, i) => (
          <div
            key={p.key}
            className="h-full animate-rise"
            style={{
              width: `${p.share * 100}%`,
              background: COMPONENT_TONE[p.key] ?? '#5B5852',
              marginLeft: i ? 2 : 0,
              animationDelay: `${i * 90}ms`,
            }}
          />
        ))}
      </div>
      <table className="mt-6 w-full text-left text-[14px]">
        <thead>
          <tr className="label-caps">
            <th className="pb-3 font-semibold">Component</th>
            <th className="pb-3 text-right font-semibold">Score</th>
            <th className="pb-3 text-right font-semibold">Weight</th>
            <th className="pb-3 pl-6 font-semibold">Why</th>
          </tr>
        </thead>
        <tbody>
          {parts.map((p) => (
            <tr key={p.key} className="border-t border-slate-900 align-top">
              <td className="py-3 pr-4 text-slate-200">
                <span
                  className="mr-2.5 inline-block h-2.5 w-2.5 rounded-sm align-middle"
                  style={{ background: COMPONENT_TONE[p.key] ?? '#5B5852' }}
                />
                {p.label}
              </td>
              <td className="py-3 text-right font-mono text-slate-200">
                {p.score === null ? '—' : Math.round(p.score)}
              </td>
              <td className="py-3 text-right font-mono text-slate-400">
                {Math.round(p.share * 100)}%
              </td>
              <td className="py-3 pl-6 text-[13px] leading-relaxed text-slate-400">{p.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
};

/** Support against refutation for one claim, as a diverging bar around "no evidence". */
const Balance: React.FC<{ claim: CheckedClaim }> = ({ claim }) => {
  const total = claim.support + claim.refute;
  const s = total ? claim.support / total : 0;
  const r = total ? claim.refute / total : 0;
  return (
    <div
      className="flex items-center gap-3"
      aria-label={`Support ${claim.support.toFixed(2)}, refute ${claim.refute.toFixed(2)}`}
    >
      <span className="file-no w-14 text-right" style={{ color: STANCE.contradict.tone }}>
        ✕ {claim.refute.toFixed(2)}
      </span>
      <div className="relative flex h-1.5 flex-1 overflow-hidden rounded-full bg-slate-900">
        <div className="flex w-1/2 justify-end">
          <div
            className="h-full rounded-l-full"
            style={{ width: `${r * 100}%`, background: STANCE.contradict.tone }}
          />
        </div>
        <div className="absolute left-1/2 top-[-3px] h-[12px] w-px bg-slate-600" />
        <div className="w-1/2">
          <div
            className="h-full rounded-r-full"
            style={{ width: `${s * 100}%`, background: STANCE.entail.tone }}
          />
        </div>
      </div>
      <span className="file-no w-14" style={{ color: STANCE.entail.tone }}>
        {claim.support.toFixed(2)} ✓
      </span>
    </div>
  );
};

const likelihood = (c: CheckedClaim) => {
  if (c.certainty < 0.08) return { word: 'No evidence found', tone: '#7C776E' };
  const p = c.probability;
  if (p >= 0.7) return { word: `Likely true · ${Math.round(p * 100)}%`, tone: STANCE.entail.tone };
  if (p <= 0.3)
    return { word: `Likely false · ${Math.round(p * 100)}%`, tone: STANCE.contradict.tone };
  return { word: `Unsettled · ${Math.round(p * 100)}%`, tone: '#C9A45C' };
};

const host = (u: string | null) => {
  try {
    return u ? new URL(u).hostname.replace(/^www\./, '') : '';
  } catch {
    return '';
  }
};

const ClaimCard: React.FC<{
  claim: CheckedClaim;
  index: number;
  onExplore: (t: string) => void;
}> = ({ claim, index, onExplore }) => {
  const l = likelihood(claim);
  const passages = [...claim.evidence].sort(
    (a, b) => Number(a.stance === 'neutral') - Number(b.stance === 'neutral'),
  );
  return (
    <article className="panel animate-rise p-7" style={{ animationDelay: `${index * 120}ms` }}>
      <div className="flex items-start justify-between gap-6">
        <div>
          <p className="file-no mb-2">CLAIM {String(index + 1).padStart(2, '0')}</p>
          <h4 className="font-display text-[24px] leading-snug text-slate-100">{claim.text}</h4>
        </div>
        <span
          className="shrink-0 rounded-full border px-3 py-1 text-[12px] font-medium"
          style={{ borderColor: `${l.tone}55`, color: l.tone }}
        >
          {l.word}
        </span>
      </div>

      <div className="mt-5">
        <Balance claim={claim} />
      </div>

      {claim.finding && (
        <div className="mt-6 border-l border-brass-700 pl-5">
          <p className="label-caps mb-2">What the sources say</p>
          <p className="text-[15px] leading-relaxed text-slate-200">{claim.finding}</p>
        </div>
      )}

      {claim.topics && claim.topics.length > 0 && (
        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          {claim.topics.map((t) => (
            <a
              key={t.title}
              href={t.url ?? '#'}
              target="_blank"
              rel="noreferrer"
              className="group overflow-hidden rounded-xl border border-slate-800 bg-[#101013] transition-colors hover:border-brass-700"
            >
              {t.image ? (
                <img
                  src={t.image}
                  alt=""
                  loading="lazy"
                  className="h-28 w-full object-cover opacity-85 transition-opacity group-hover:opacity-100"
                />
              ) : (
                <div className="h-28 w-full bg-slate-900" />
              )}
              <div className="p-3">
                <p className="flex items-center justify-between text-[13px] font-medium text-slate-200">
                  {t.title}{' '}
                  <ArrowUpRight className="h-3.5 w-3.5 text-slate-600 group-hover:text-brass-300" />
                </p>
                <p className="mt-1 line-clamp-2 text-[12px] leading-snug text-slate-500">
                  {t.summary}
                </p>
              </div>
            </a>
          ))}
        </div>
      )}

      {passages.length > 0 && (
        <div className="mt-6">
          <p className="label-caps mb-3">Evidence · {passages.length} passages</p>
          <ul className="divide-y divide-slate-900">
            {passages.map((e, i) => {
              const st = STANCE[e.stance] ?? STANCE.neutral;
              return (
                <li key={i} className="grid grid-cols-[88px_1fr_auto] gap-4 py-3">
                  <span className="text-[12px] font-medium" style={{ color: st.tone }}>
                    {st.glyph} {st.word}
                  </span>
                  <div className="min-w-0">
                    <a
                      href={e.url ?? '#'}
                      target="_blank"
                      rel="noreferrer"
                      className="text-[13px] text-slate-300 underline decoration-slate-700 underline-offset-4 hover:text-brass-200 hover:decoration-brass-700"
                    >
                      {e.domain || host(e.url)}
                    </a>
                    <p className="mt-1 line-clamp-2 text-[13px] leading-relaxed text-slate-500">
                      “{e.snippet}”
                    </p>
                  </div>
                  <span
                    className="file-no whitespace-nowrap"
                    title="How much this source is trusted"
                  >
                    trust {Math.round(e.reliability * 100)}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {claim.explore && claim.explore.length > 0 && (
        <div className="mt-6 flex flex-wrap items-center gap-2">
          <span className="label-caps mr-1">Look into</span>
          {claim.explore.map((x) => (
            <button
              key={x}
              onClick={() => onExplore(x)}
              className="rounded-full border border-slate-800 px-3 py-1 text-[12px] text-slate-300 transition-colors hover:border-brass-600 hover:text-brass-200"
            >
              {x}
            </button>
          ))}
        </div>
      )}
    </article>
  );
};

export const NewsScanner: React.FC = () => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [step, setStep] = useState(0);
  const [result, setResult] = useState<NewsAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { addToHistory } = useAuth();
  const resultRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!loading) return;
    setStep(0);
    const t = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 2600);
    return () => clearInterval(t);
  }, [loading]);

  const run = async (input: string, depth: 'standard' | 'deep' = 'standard') => {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    if (depth === 'standard') setResult(null);
    playCyberSFX('scan');
    try {
      const data = await analyzeNewsText(input, depth);
      setResult(data);
      playCyberSFX('success');
      addToHistory({
        id: Math.random().toString(36).slice(2, 11),
        date: new Date().toISOString(),
        type: 'NEWS',
        summary: data.summary.substring(0, 80) + '…',
        result: `${data.credibilityScore}% · ${data.verdict}`,
      });
      setTimeout(
        () => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        80,
      );
    } catch (e: any) {
      playCyberSFX('alarm');
      setError(e?.message || 'The analysis did not complete. Try again in a moment.');
    } finally {
      setLoading(false);
    }
  };

  const explore = (topic: string) => {
    setText(topic);
    run(topic, 'deep');
  };

  const exportPDF = () => {
    if (!result) return;
    const doc = new jsPDF();
    const m = 20;
    const w = doc.internal.pageSize.getWidth() - m * 2;
    let y = 22;
    const line = (s: string, size = 10, bold = false, gap = 5) => {
      doc.setFont('times', bold ? 'bold' : 'normal');
      doc.setFontSize(size);
      const lines = doc.splitTextToSize(s, w);
      lines.forEach((l: string) => {
        if (y > 280) {
          doc.addPage();
          y = 20;
        }
        doc.text(l, m, y);
        y += gap;
      });
    };
    line('CEREBRO — Verification report', 18, true, 9);
    line(new Date().toLocaleString(), 9, false, 8);
    line(
      `Verdict: ${result.verdict} · ${result.credibilityScore}/100 · confidence ${result.confidence_label ?? '—'}`,
      12,
      true,
      8,
    );
    line(`Text examined: ${result.summary}`);
    y += 3;
    (result.breakdown ?? []).forEach((p) =>
      line(
        `${p.label}: ${p.score === null ? '—' : Math.round(p.score)} (weight ${Math.round(p.share * 100)}%) — ${p.detail}`,
      ),
    );
    y += 3;
    (result.claims ?? []).forEach((c, i) => {
      line(`Claim ${i + 1}: ${c.text}`, 11, true, 6);
      if (c.finding) line(c.finding);
      c.evidence.forEach((e) =>
        line(
          `  [${STANCE[e.stance]?.word ?? e.stance}] ${e.domain ?? ''}: “${e.snippet}” ${e.url ?? ''}`,
          9,
          false,
          4.5,
        ),
      );
      y += 3;
    });
    line('Reasoning', 11, true, 6);
    line(result.reasoning);
    doc.save(`cerebro-verification-${Date.now()}.pdf`);
  };

  const share = () => {
    if (!result) return;
    const shareText = `CEREBRO: ${result.verdict} (${result.credibilityScore}/100) — ${result.summary}`;
    if (navigator.share) navigator.share({ text: shareText }).catch(() => undefined);
    else navigator.clipboard?.writeText(shareText);
  };

  const v = result ? verdictOf(result.verdict) : null;

  return (
    <div className="space-y-8">
      {/* Input */}
      <section className="panel p-7">
        <label htmlFor="claim-input" className="label-caps mb-3 block">
          What did you read?
        </label>
        <textarea
          id="claim-input"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run(text);
          }}
          rows={5}
          placeholder="A headline, a paragraph from Reddit, or several questions at once. Spelling does not need to be perfect."
          className="w-full resize-y rounded-xl border border-slate-800 bg-[#101013] px-5 py-4 text-[16px] leading-relaxed text-slate-100 outline-none transition-colors placeholder:text-slate-600 focus:border-brass-600"
        />
        <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-2">
            <span className="label-caps mr-1">Try</span>
            {EXAMPLES.map((x) => (
              <button
                key={x}
                onClick={() => setText(x)}
                className="max-w-[260px] truncate rounded-full border border-slate-800 px-3 py-1 text-[12px] text-slate-400 transition-colors hover:border-slate-600 hover:text-slate-200"
              >
                {x}
              </button>
            ))}
          </div>
          <button
            onClick={() => run(text)}
            disabled={loading || !text.trim()}
            className="flex items-center gap-2 rounded-lg bg-brass-400 px-6 py-3 text-[14px] font-semibold text-[#16120A] transition-colors hover:bg-brass-300 disabled:opacity-40"
          >
            <Search className="h-4 w-4" strokeWidth={2} /> Verify
          </button>
        </div>
      </section>

      {/* Progress: honest stage names, no fake log lines */}
      {loading && (
        <section className="panel flex items-center gap-8 p-7" aria-live="polite">
          <svg width="56" height="56" viewBox="0 0 56 56" aria-hidden="true">
            <circle cx="28" cy="28" r="26" fill="none" stroke="rgba(236,230,217,0.1)" />
            <circle
              cx="28"
              cy="28"
              r="26"
              fill="none"
              stroke="#C9A45C"
              strokeWidth="1.5"
              strokeDasharray="30 134"
              className="animate-sweep"
              style={{ transformOrigin: '28px 28px' }}
            />
          </svg>
          <ol className="flex flex-1 flex-wrap gap-x-6 gap-y-2">
            {STEPS.map((s, i) => (
              <li
                key={s}
                className={`text-[13px] transition-colors ${
                  i < step ? 'text-slate-500' : i === step ? 'text-slate-100' : 'text-slate-700'
                }`}
              >
                <span className="file-no mr-2">{String(i + 1).padStart(2, '0')}</span>
                {s}
                {i === step && <span className="animate-breathe">…</span>}
              </li>
            ))}
          </ol>
        </section>
      )}

      {error && (
        <p
          role="alert"
          className="rounded-xl border border-signal-800 bg-signal-950/40 px-5 py-4 text-[14px] text-signal-200"
        >
          {error}
        </p>
      )}

      {result && v && (
        <div ref={resultRef} className="space-y-8">
          {/* Verdict */}
          <section className="panel grid items-center gap-8 p-8 md:grid-cols-[176px_1fr]">
            <ScoreRing score={result.credibilityScore} tone={v.tone} />
            <div>
              <p className="label-caps mb-3">Verdict</p>
              <p className="font-display text-[46px] leading-none" style={{ color: v.tone }}>
                <span className="mr-3 text-[30px] align-middle">{v.glyph}</span>
                {result.verdict}
              </p>
              <p className="mt-3 text-[16px] text-slate-300">{v.line}</p>
              <div className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px] text-slate-400">
                <span>
                  Confidence{' '}
                  <span className="text-slate-100">{result.confidence_label ?? '—'}</span>
                </span>
                <span>
                  Claims checked{' '}
                  <span className="text-slate-100">{result.claims?.length ?? '—'}</span>
                </span>
                <span>
                  Sources <span className="text-slate-100">{result.sources.length}</span>
                </span>
              </div>
              <div className="mt-6 flex flex-wrap gap-3">
                <button
                  onClick={() => run(text || result.summary, 'deep')}
                  disabled={loading}
                  className="flex items-center gap-2 rounded-lg border border-brass-700 px-4 py-2 text-[13px] text-brass-200 transition-colors hover:bg-brass-950 disabled:opacity-40"
                >
                  <Telescope className="h-4 w-4" strokeWidth={1.6} />
                  {result.depth === 'deep' ? 'Search again, deeper' : 'Dig deeper'}
                </button>
                <button
                  onClick={exportPDF}
                  className="flex items-center gap-2 rounded-lg border border-slate-800 px-4 py-2 text-[13px] text-slate-300 transition-colors hover:border-slate-600"
                >
                  <Download className="h-4 w-4" strokeWidth={1.6} /> Report (PDF)
                </button>
                <button
                  onClick={share}
                  className="flex items-center gap-2 rounded-lg border border-slate-800 px-4 py-2 text-[13px] text-slate-300 transition-colors hover:border-slate-600"
                >
                  <Share2 className="h-4 w-4" strokeWidth={1.6} /> Share
                </button>
              </div>
            </div>
          </section>

          {result.breakdown && result.breakdown.length > 0 && (
            <Breakdown parts={result.breakdown} />
          )}

          {(result.claims ?? []).map((c, i) => (
            <ClaimCard key={`${c.text}-${i}`} claim={c} index={i} onExplore={explore} />
          ))}

          <section className="panel p-7">
            <h3 className="mb-3 font-display text-[22px] text-slate-100">The reasoning</h3>
            <p className="text-[15px] leading-relaxed text-slate-300">{result.reasoning}</p>
            {result.evidence_sources && (
              <p className="mt-5 file-no">
                {Object.entries(result.evidence_sources)
                  .map(([k, val]) => `${k.replace('_', ' ')}: ${val}`)
                  .join('   ·   ')}
              </p>
            )}
          </section>

          {result.related && result.related.length > 0 && (
            <section className="panel p-7">
              <h3 className="mb-4 font-display text-[22px] text-slate-100">More coverage</h3>
              <ul className="grid gap-x-8 sm:grid-cols-2">
                {result.related.map((r) => (
                  <li key={r.url} className="border-t border-slate-900 py-3">
                    <a
                      href={r.url}
                      target="_blank"
                      rel="noreferrer"
                      className="group flex items-start justify-between gap-3 text-[14px] text-slate-300 hover:text-brass-200"
                    >
                      <span>
                        {r.title || host(r.url)}
                        <span className="mt-0.5 block file-no">
                          {host(r.url)}
                          {r.page_age ? ` · ${r.page_age}` : ''}
                        </span>
                      </span>
                      <ArrowUpRight className="mt-1 h-3.5 w-3.5 shrink-0 text-slate-600 group-hover:text-brass-300" />
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </div>
      )}
    </div>
  );
};
