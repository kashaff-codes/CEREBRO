import React, { useCallback, useEffect, useRef, useState } from 'react';
import { jsPDF } from 'jspdf';
import { Download, FileUp, Play, RefreshCw } from 'lucide-react';
import { getJSON, postFile } from '../services/apiService';
import { useAuth } from '../context/AuthContext';
import { playCyberSFX } from '../utils/audio';

/**
 * Network watch.
 *
 * Two trained detectors, both measured on CIC-IDS2017 (2.8 million labelled
 * flows of real traffic): an Isolation Forest that learned only what normal
 * looks like, and a Random Forest that knows eight attack families. Three ways
 * in: upload your own Zeek or flow log, replay real labelled test traffic with
 * the true answer beside each verdict, or read the measured accuracy, including
 * the numbers that are not flattering.
 */

// ── API shapes ─────────────────────────────────────────────────────────────

interface Feed {
  feed: string;
  label: string;
  entries: number;
  fetched_at: number;
  ok: boolean;
  error: string;
}
interface IntelHit {
  feed: string;
  label: string;
  severity: string;
  malware?: string;
  netblock?: string;
}
interface Flow {
  duration_s: number;
  fwd_pkts: number;
  bwd_pkts: number;
  fwd_bytes: number;
  bwd_bytes: number;
  dst_port: number;
  src_ip: string;
  dst_ip: string;
  proto: string;
  ts: string;
  label: string;
  anomaly_score: number;
  anomalous: boolean;
  predicted_family: string;
  family_confidence: number;
  attack_probability: number;
  intel?: IntelHit[];
  true_family?: string;
  day?: string;
}
interface Confusion {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}
interface LabelCheck extends Confusion {
  flows: number;
  recall: number | null;
  precision: number | null;
  false_positive_rate: number | null;
}
interface ScanResult {
  summary: {
    format: string;
    flows: number;
    seconds: number;
    anomalous: number;
    known_attack_flows: number;
    by_family: Record<string, number>;
    threat_level: Level;
    recommended_action: string;
    intel_hits: { ip: string; matches: IntelHit[] }[];
    feeds: Feed[];
    has_labels: boolean;
    label_check: LabelCheck | null;
  };
  top: Flow[];
}
interface Replay {
  source: string;
  held_out?: string;
  flows: Flow[];
  isolation_forest: Confusion;
  random_forest: Confusion;
  either: Confusion;
}
interface Metrics {
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  roc_auc: number;
  pr_auc: number;
}
interface Card {
  dataset: string;
  flows: number;
  families: Record<string, number>;
  trained_at: string;
  isolation_forest: {
    trained_on: string;
    threshold_rule: string;
    test: string;
    metrics: Metrics;
    recall_by_family: Record<string, number>;
    operating_points: {
      false_alarm_budget: number;
      recall: number;
      precision: number;
      test_false_positive_rate: number;
    }[];
  };
  random_forest: {
    classes: string[];
    random_split: { attack_vs_normal: Metrics };
    held_out_days: { attack_vs_normal: Metrics };
    shipped_model: string;
  };
}

type Level = 'Critical' | 'High' | 'Medium' | 'Safe';

// ── vocabulary ─────────────────────────────────────────────────────────────

const LEVEL: Record<Level, { tone: string; glyph: string }> = {
  Critical: { tone: '#D2654E', glyph: '✕' },
  High: { tone: '#D08A4E', glyph: '▲' },
  Medium: { tone: '#C9A45C', glyph: '◆' },
  Safe: { tone: '#86AD90', glyph: '●' },
};

const SEVERITY_TONE: Record<string, string> = {
  critical: '#D2654E',
  high: '#D08A4E',
  info: '#7C776E',
};

const FEED_NAME: Record<string, string> = {
  feodo: 'abuse.ch Feodo Tracker',
  spamhaus_drop: 'Spamhaus DROP',
  tor_exit: 'Tor exit relays',
};

const pct = (x: number | null | undefined, digits = 1) =>
  x == null ? '—' : `${(x * 100).toFixed(digits)}%`;
const int = (n: number) => n.toLocaleString('en-GB');
const bytes = (n: number) =>
  n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${(n / 1e3).toFixed(1)} kB` : `${n} B`;
const ago = (t: number) => {
  if (!t) return 'never';
  const m = Math.round((Date.now() / 1000 - t) / 60);
  return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : `${Math.round(m / 60)} h ago`;
};

// ── small parts ────────────────────────────────────────────────────────────

const Stat: React.FC<{ label: string; value: string; note?: string; tone?: string }> = ({
  label,
  value,
  note,
  tone,
}) => (
  <div className="min-w-0">
    <p className="label-caps mb-2">{label}</p>
    <p className="font-display text-[34px] leading-none" style={{ color: tone ?? '#ECE6D9' }}>
      {value}
    </p>
    {note && <p className="mt-2 text-[12.5px] leading-snug text-slate-500">{note}</p>}
  </div>
);

const Verdict: React.FC<{ flow: Flow }> = ({ flow }) => {
  const known = flow.predicted_family !== 'BENIGN';
  const flagged = known || flow.anomalous;
  return (
    <span className="inline-flex items-center gap-2 text-[13px]">
      <span style={{ color: flagged ? '#D2654E' : '#86AD90' }} aria-hidden="true">
        {flagged ? '✕' : '●'}
      </span>
      <span className={flagged ? 'text-slate-100' : 'text-slate-400'}>
        {known ? flow.predicted_family : flow.anomalous ? 'Unusual' : 'Normal'}
      </span>
    </span>
  );
};

/** A 2×2 confusion grid: what was really an attack, against what was flagged. */
const ConfusionGrid: React.FC<{ c: Confusion; caption: string }> = ({ c, caption }) => {
  const cell = (n: number, label: string, good: boolean) => (
    <div
      className="rounded-lg border px-4 py-3"
      style={{
        borderColor: good ? 'rgba(134,173,144,0.25)' : 'rgba(210,101,78,0.25)',
        background: good ? 'rgba(134,173,144,0.05)' : 'rgba(210,101,78,0.05)',
      }}
    >
      <p className="font-display text-[26px] leading-none text-slate-100">{n}</p>
      <p className="mt-1.5 text-[12px] text-slate-400">{label}</p>
    </div>
  );
  return (
    <figure>
      <div className="grid grid-cols-2 gap-2">
        {cell(c.tp, 'attacks caught', true)}
        {cell(c.fn, 'attacks missed', false)}
        {cell(c.fp, 'false alarms', false)}
        {cell(c.tn, 'normal, left alone', true)}
      </div>
      <figcaption className="mt-2 text-[12px] text-slate-500">{caption}</figcaption>
    </figure>
  );
};

/** Recall as the false-alarm budget grows: the trade-off a SOC actually chooses. */
const OperatingCurve: React.FC<{
  points: Card['isolation_forest']['operating_points'];
  shipped: number;
}> = ({ points, shipped }) => {
  const W = 520;
  const H = 220;
  const pad = { l: 44, r: 16, t: 14, b: 34 };
  const maxX = Math.max(...points.map((p) => p.false_alarm_budget));
  const x = (v: number) => pad.l + (v / maxX) * (W - pad.l - pad.r);
  const y = (v: number) => H - pad.b - v * (H - pad.t - pad.b);
  const d = points
    .map((p, i) => `${i ? 'L' : 'M'}${x(p.false_alarm_budget)},${y(p.recall)}`)
    .join(' ');
  const len = points.reduce((acc, p, i) => {
    if (!i) return 0;
    const q = points[i - 1];
    return (
      acc + Math.hypot(x(p.false_alarm_budget) - x(q.false_alarm_budget), y(p.recall) - y(q.recall))
    );
  }, 0);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      role="img"
      aria-label="Share of attacks caught at each false-alarm budget"
    >
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke="rgba(236,230,217,0.07)" />
          <text x={pad.l - 8} y={y(t) + 4} textAnchor="end" fontSize="10.5" fill="#6F6A62">
            {t * 100}%
          </text>
        </g>
      ))}
      {points.map((p) => (
        <text
          key={p.false_alarm_budget}
          x={x(p.false_alarm_budget)}
          y={H - pad.b + 18}
          textAnchor="middle"
          fontSize="10.5"
          fill="#6F6A62"
        >
          {Math.round(p.false_alarm_budget * 100)}%
        </text>
      ))}
      <path
        d={d}
        fill="none"
        stroke="#C9A45C"
        strokeWidth="2"
        strokeLinejoin="round"
        className="animate-draw"
        style={{ ['--len' as string]: len }}
      />
      {points.map((p) => {
        const on = p.false_alarm_budget === shipped;
        return (
          <g key={p.false_alarm_budget}>
            <circle
              cx={x(p.false_alarm_budget)}
              cy={y(p.recall)}
              r={on ? 6 : 4}
              fill={on ? '#C9A45C' : '#121215'}
              stroke="#C9A45C"
              strokeWidth="2"
            >
              <title>
                {`${Math.round(p.false_alarm_budget * 100)}% false-alarm budget: catches ${pct(p.recall)} of attacks, ${pct(p.precision)} of alerts are real`}
              </title>
            </circle>
            {on && (
              <text
                x={x(p.false_alarm_budget) + 10}
                y={y(p.recall) - 10}
                fontSize="11"
                fill="#E2C487"
              >
                shipped
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
};

const Bars: React.FC<{
  rows: [string, number][];
  max?: number;
  format?: (v: number) => string;
}> = ({ rows, max, format = (v) => pct(v, 0) }) => {
  const top = max ?? Math.max(...rows.map((r) => r[1]), 1e-9);
  return (
    <ul className="space-y-2.5">
      {rows.map(([name, v], i) => (
        <li key={name} className="grid grid-cols-[120px_1fr_56px] items-center gap-3 text-[13px]">
          <span className="truncate text-slate-300">{name}</span>
          <span className="h-[6px] overflow-hidden rounded-full bg-[rgba(236,230,217,0.06)]">
            <span
              className="animate-rise block h-full rounded-full bg-brass-400"
              style={{ width: `${Math.max(1.5, (v / top) * 100)}%`, animationDelay: `${i * 60}ms` }}
            />
          </span>
          <span className="text-right font-mono text-[12px] text-slate-400">{format(v)}</span>
        </li>
      ))}
    </ul>
  );
};

const FlowTable: React.FC<{ flows: Flow[]; truth?: boolean }> = ({ flows, truth }) => (
  <div className="custom-scrollbar max-h-[440px] overflow-auto rounded-xl border border-[var(--line)]">
    <table className="w-full text-left text-[13px]">
      <thead className="sticky top-0 bg-[#141417]">
        <tr className="label-caps">
          {truth ? (
            <th className="px-4 py-3 font-semibold">Day</th>
          ) : (
            <th className="px-4 py-3 font-semibold">Source → destination</th>
          )}
          <th className="px-4 py-3 font-semibold">Port</th>
          <th className="px-4 py-3 font-semibold">Packets ⇄</th>
          <th className="px-4 py-3 font-semibold">Bytes ⇄</th>
          <th className="px-4 py-3 font-semibold">Anomaly</th>
          <th className="px-4 py-3 font-semibold">Verdict</th>
          {truth && <th className="px-4 py-3 font-semibold">Really was</th>}
        </tr>
      </thead>
      <tbody className="divide-y divide-[var(--line)]">
        {flows.map((f, i) => {
          const flagged = f.anomalous || f.predicted_family !== 'BENIGN';
          const attack = truth ? f.true_family !== 'BENIGN' : false;
          const right = flagged === attack;
          return (
            <tr key={i} className="transition-colors hover:bg-[rgba(236,230,217,0.025)]">
              <td className="px-4 py-2.5 font-mono text-[12px] text-slate-400">
                {truth ? (
                  f.day
                ) : (
                  <>
                    {f.src_ip || '—'} → {f.dst_ip || '—'}
                    {f.intel && f.intel.length > 0 && (
                      <span
                        className="ml-2 rounded px-1.5 py-0.5 font-sans text-[10.5px]"
                        style={{
                          color: SEVERITY_TONE[f.intel[0].severity] ?? '#A29C90',
                          border: `1px solid ${SEVERITY_TONE[f.intel[0].severity] ?? '#38363A'}55`,
                        }}
                        title={f.intel.map((h) => h.label).join('; ')}
                      >
                        {FEED_NAME[f.intel[0].feed] ?? f.intel[0].feed}
                      </span>
                    )}
                  </>
                )}
              </td>
              <td className="px-4 py-2.5 font-mono text-[12px] text-slate-300">{f.dst_port}</td>
              <td className="px-4 py-2.5 font-mono text-[12px] text-slate-400">
                {f.fwd_pkts} / {f.bwd_pkts}
              </td>
              <td className="px-4 py-2.5 font-mono text-[12px] text-slate-400">
                {bytes(f.fwd_bytes)} / {bytes(f.bwd_bytes)}
              </td>
              <td className="px-4 py-2.5">
                <span className="flex items-center gap-2">
                  <span className="h-[5px] w-14 overflow-hidden rounded-full bg-[rgba(236,230,217,0.07)]">
                    <span
                      className="block h-full rounded-full"
                      style={{
                        width: `${Math.min(100, f.anomaly_score * 130)}%`,
                        background: f.anomalous ? '#D2654E' : '#7C776E',
                      }}
                    />
                  </span>
                  <span className="font-mono text-[11.5px] text-slate-500">
                    {f.anomaly_score.toFixed(2)}
                  </span>
                </span>
              </td>
              <td className="px-4 py-2.5">
                <Verdict flow={f} />
              </td>
              {truth && (
                <td className="px-4 py-2.5 text-[13px]">
                  <span className={attack ? 'text-slate-100' : 'text-slate-400'}>
                    {attack ? f.true_family : 'Normal'}
                  </span>
                  <span
                    className="ml-2 text-[12px]"
                    style={{ color: right ? '#86AD90' : '#D2654E' }}
                    aria-label={right ? 'correct' : 'wrong'}
                  >
                    {right ? '✓' : '✕'}
                  </span>
                </td>
              )}
            </tr>
          );
        })}
      </tbody>
    </table>
  </div>
);

// ── page ───────────────────────────────────────────────────────────────────

export const CyberMonitor: React.FC = () => {
  const { addToHistory } = useAuth();
  const [card, setCard] = useState<Card | null>(null);
  const [feeds, setFeeds] = useState<Feed[] | null>(null);
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [scanName, setScanName] = useState('');
  const [replay, setReplay] = useState<Replay | null>(null);
  const [busy, setBusy] = useState<'scan' | 'replay' | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getJSON<Card>('/v1/network/model')
      .then(setCard)
      .catch((e) => setError(e.message));
    getJSON<Feed[]>('/v1/network/intel')
      .then(setFeeds)
      .catch(() => setFeeds([]));
  }, []);

  const runReplay = useCallback(async () => {
    setBusy('replay');
    setError(null);
    try {
      setReplay(await getJSON<Replay>('/v1/network/replay?n=60'));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }, []);

  useEffect(() => {
    runReplay();
  }, [runReplay]);

  const upload = async (file: Blob, name: string) => {
    setBusy('scan');
    setError(null);
    setScan(null);
    setScanName(name);
    playCyberSFX('scan');
    try {
      const r = await postFile<ScanResult>('/v1/network/scan', file, name);
      setScan(r);
      if (r.summary.feeds?.length) setFeeds(r.summary.feeds);
      playCyberSFX(r.summary.threat_level === 'Safe' ? 'success' : 'alarm');
      addToHistory({
        id: Math.random().toString(36).slice(2, 11),
        date: new Date().toISOString(),
        type: 'CYBER',
        summary: `${name}: ${r.summary.flows} flows, ${r.summary.anomalous} unusual, ${r.summary.intel_hits.length} feed matches`,
        result: r.summary.threat_level,
      });
      setTimeout(
        () => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        80,
      );
    } catch (e: any) {
      playCyberSFX('alarm');
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  const trySample = async () => {
    try {
      const csv = await getJSON<string>('/v1/network/sample');
      await upload(new Blob([csv], { type: 'text/csv' }), 'cic-ids2017-test-sample.csv');
    } catch (e: any) {
      setError(e.message);
    }
  };

  const exportPDF = () => {
    if (!scan) return;
    const s = scan.summary;
    const doc = new jsPDF();
    const m = 20;
    const w = doc.internal.pageSize.getWidth() - m * 2;
    let y = 22;
    const line = (t: string, size = 10, bold = false, gap = 5) => {
      doc.setFont('times', bold ? 'bold' : 'normal');
      doc.setFontSize(size);
      doc.splitTextToSize(t, w).forEach((l: string) => {
        if (y > 280) {
          doc.addPage();
          y = 20;
        }
        doc.text(l, m, y);
        y += gap;
      });
    };
    line('CEREBRO — Network watch report', 18, true, 9);
    line(`${scanName} · ${new Date().toLocaleString()}`, 9, false, 8);
    line(`Threat level: ${s.threat_level}`, 13, true, 7);
    line(s.recommended_action);
    y += 3;
    line(
      `${s.flows} flows (${s.format}) · ${s.anomalous} unusual · ${s.known_attack_flows} matched a known attack family · ${s.intel_hits.length} threat-feed matches`,
    );
    if (s.label_check)
      line(
        `Against the file's own labels: caught ${pct(s.label_check.recall)} of attacks, ${pct(s.label_check.precision)} of flags correct.`,
      );
    y += 3;
    line('Threat-feed matches', 12, true, 6);
    if (!s.intel_hits.length) line('None.');
    s.intel_hits.forEach((h) =>
      line(
        `${h.ip}: ${h.matches.map((x) => x.label + (x.malware ? ` (${x.malware})` : '')).join('; ')}`,
      ),
    );
    y += 3;
    line('Highest-risk flows', 12, true, 6);
    scan.top
      .slice(0, 25)
      .forEach((f) =>
        line(
          `${f.src_ip || '—'} → ${f.dst_ip || '—'}:${f.dst_port} · ${f.predicted_family} (${pct(f.family_confidence, 0)}) · anomaly ${f.anomaly_score.toFixed(2)}${f.anomalous ? ' (unusual)' : ''}`,
          9,
        ),
      );
    y += 4;
    line(
      'Detectors: Isolation Forest (trained on normal traffic only) and Random Forest (8 families), both trained on CIC-IDS2017. Threat feeds: abuse.ch Feodo Tracker, Spamhaus DROP, Tor exit list.',
      8,
    );
    doc.save(`cerebro-network-${Date.now()}.pdf`);
  };

  const iso = card?.isolation_forest;
  const rf = card?.random_forest;
  const level = scan ? LEVEL[scan.summary.threat_level] : null;

  return (
    <div className="space-y-8">
      {/* Scan your own traffic + live feeds */}
      <section className="grid gap-6 lg:grid-cols-[1fr_360px]">
        {/* Drag-and-drop is a shortcut; the "Choose a log file" button is the keyboard path. */}
        {/* eslint-disable-next-line jsx-a11y/no-noninteractive-element-interactions */}
        <div
          role="region"
          aria-label="Drop a traffic log here to score it"
          className={`panel flex flex-col p-7 transition-colors ${dragging ? 'border-brass-600' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const f = e.dataTransfer.files?.[0];
            if (f) upload(f, f.name);
          }}
        >
          <p className="label-caps mb-3">Score your traffic</p>
          <p className="max-w-[60ch] text-[15px] leading-relaxed text-slate-300">
            Drop a Zeek <span className="font-mono text-[13px]">conn.log</span> (TSV or JSON), a
            CICFlowMeter export, or any CSV with durations, packet and byte counts and a destination
            port. Every flow is scored by both detectors, and every public address is checked
            against three live threat feeds.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <input
              ref={fileRef}
              type="file"
              accept=".log,.csv,.json,.tsv,.txt"
              className="hidden"
              data-testid="flow-file"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) upload(f, f.name);
                e.target.value = '';
              }}
            />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={busy !== null}
              className="flex items-center gap-2 rounded-lg bg-brass-400 px-6 py-3 text-[14px] font-semibold text-[#16120A] transition-colors hover:bg-brass-300 disabled:opacity-40"
            >
              <FileUp className="h-4 w-4" strokeWidth={2} />
              {busy === 'scan' ? 'Scoring…' : 'Choose a log file'}
            </button>
            <button
              onClick={trySample}
              disabled={busy !== null}
              className="rounded-lg border border-slate-700 px-4 py-3 text-[13px] text-slate-300 transition-colors hover:border-slate-500 hover:text-slate-100 disabled:opacity-40"
            >
              Try it with a labelled CIC-IDS2017 file
            </button>
          </div>
          <p className="mt-auto pt-6 text-[12px] text-slate-500">
            Up to 25 MB or 50,000 flows per file. Files are scored in memory and not stored; the
            summary is kept in your history.
          </p>
        </div>

        <aside className="panel p-7">
          <div className="mb-4 flex items-center justify-between">
            <p className="label-caps">Live threat feeds</p>
            <span className="file-no">refreshed hourly</span>
          </div>
          {feeds === null ? (
            <p className="text-[13px] text-slate-500">Checking feeds…</p>
          ) : feeds.length === 0 ? (
            <p className="text-[13px] text-slate-500">Feed status unavailable.</p>
          ) : (
            <ul className="space-y-4">
              {feeds.map((f) => (
                <li key={f.feed} className="flex items-start justify-between gap-4">
                  <div className="min-w-0">
                    <p className="text-[14px] text-slate-200">{FEED_NAME[f.feed] ?? f.feed}</p>
                    <p className="text-[12px] text-slate-500">{f.label}</p>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className="font-mono text-[13px] text-slate-200">
                      {f.ok ? int(f.entries) : '—'}
                    </p>
                    <p className="text-[11.5px]" style={{ color: f.ok ? '#86AD90' : '#D2654E' }}>
                      {f.ok ? `● ${ago(f.fetched_at)}` : `✕ unavailable`}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-5 text-[12px] leading-relaxed text-slate-500">
            A feed that fails to load is shown as unavailable, never as “no matches”.
          </p>
        </aside>
      </section>

      {error && (
        <p
          role="alert"
          className="rounded-xl border border-signal-800 bg-signal-950/40 px-5 py-4 text-[14px] text-signal-200"
        >
          {error}
        </p>
      )}

      {/* Scan result */}
      {scan && level && (
        <div ref={resultRef} className="space-y-6">
          <section className="panel p-8">
            <div className="grid gap-8 md:grid-cols-[1fr_auto] md:items-start">
              <div>
                <p className="label-caps mb-3">Threat level · {scanName}</p>
                <p className="font-display text-[46px] leading-none" style={{ color: level.tone }}>
                  <span className="mr-3 align-middle text-[30px]">{level.glyph}</span>
                  {scan.summary.threat_level}
                </p>
                <p className="mt-3 max-w-[62ch] text-[16px] text-slate-300">
                  {scan.summary.recommended_action}
                </p>
              </div>
              <button
                onClick={exportPDF}
                className="flex items-center gap-2 self-start rounded-lg border border-slate-700 px-4 py-2 text-[13px] text-slate-300 transition-colors hover:border-slate-500"
              >
                <Download className="h-4 w-4" strokeWidth={1.6} /> Report (PDF)
              </button>
            </div>
            <div className="hairline my-7" />
            <div className="grid grid-cols-2 gap-6 md:grid-cols-4">
              <Stat
                label="Flows scored"
                value={int(scan.summary.flows)}
                note={`${scan.summary.format.replace('_', ' ')} · ${scan.summary.seconds}s`}
              />
              <Stat
                label="Unusual"
                value={int(scan.summary.anomalous)}
                note="outside the learned normal"
                tone={scan.summary.anomalous ? '#D08A4E' : undefined}
              />
              <Stat
                label="Known attack"
                value={int(scan.summary.known_attack_flows)}
                note="matched a trained family"
                tone={scan.summary.known_attack_flows ? '#D2654E' : undefined}
              />
              <Stat
                label="Feed matches"
                value={int(scan.summary.intel_hits.length)}
                note="addresses on a live list"
                tone={scan.summary.intel_hits.length ? '#D2654E' : undefined}
              />
            </div>
          </section>

          <section className="grid gap-6 lg:grid-cols-2">
            <div className="panel p-7">
              <p className="label-caps mb-5">What the classifier called each flow</p>
              <Bars
                rows={Object.entries(scan.summary.by_family).sort((a, b) => b[1] - a[1])}
                format={(v) => int(v)}
              />
            </div>
            <div className="panel p-7">
              {scan.summary.label_check ? (
                <>
                  <p className="label-caps mb-1">Checked against the file’s own labels</p>
                  <p className="mb-5 text-[13px] text-slate-500">
                    This file says which flows were attacks, so the verdicts can be marked. One file
                    is a small sample: read the measured accuracy below for the full picture.
                  </p>
                  <ConfusionGrid
                    c={scan.summary.label_check}
                    caption={`Caught ${pct(scan.summary.label_check.recall)} of attacks · ${pct(scan.summary.label_check.precision)} of flags were real · ${pct(scan.summary.label_check.false_positive_rate)} of normal flows flagged`}
                  />
                </>
              ) : (
                <>
                  <p className="label-caps mb-5">Threat-feed matches</p>
                  {scan.summary.intel_hits.length === 0 ? (
                    <p className="text-[14px] text-slate-400">
                      No address in this file is on Feodo Tracker, Spamhaus DROP or the Tor exit
                      list right now.
                    </p>
                  ) : (
                    <ul className="space-y-3">
                      {scan.summary.intel_hits.map((h) => (
                        <li key={h.ip} className="text-[13.5px]">
                          <span className="font-mono text-slate-100">{h.ip}</span>
                          {h.matches.map((x, i) => (
                            <span
                              key={i}
                              className="ml-3"
                              style={{ color: SEVERITY_TONE[x.severity] ?? '#A29C90' }}
                            >
                              {x.label}
                              {x.malware ? ` · ${x.malware}` : ''}
                              {x.netblock ? ` · ${x.netblock}` : ''}
                            </span>
                          ))}
                        </li>
                      ))}
                    </ul>
                  )}
                </>
              )}
            </div>
          </section>

          <section className="panel p-7">
            <p className="label-caps mb-5">
              Highest-risk flows (top {Math.min(100, scan.top.length)})
            </p>
            <FlowTable flows={scan.top} />
          </section>
        </div>
      )}

      {/* Replay */}
      <section className="panel p-7">
        <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="label-caps mb-2">Replay · real labelled traffic</p>
            <p className="max-w-[70ch] text-[14px] leading-relaxed text-slate-400">
              Sixty flows drawn from the CIC-IDS2017 test days, scored live by the shipped
              detectors, with the true answer beside each verdict. These flows were removed before
              any model was trained, but they come from the same days the classifier learned from,
              so expect its seen-days accuracy here. This is a replay of recorded traffic, not your
              network.
            </p>
          </div>
          <button
            onClick={runReplay}
            disabled={busy !== null}
            className="flex items-center gap-2 rounded-lg border border-brass-700 px-4 py-2 text-[13px] text-brass-200 transition-colors hover:bg-brass-950 disabled:opacity-40"
          >
            {busy === 'replay' ? (
              <RefreshCw className="h-4 w-4 animate-spin" strokeWidth={1.6} />
            ) : (
              <Play className="h-4 w-4" strokeWidth={1.6} />
            )}
            Draw another sixty
          </button>
        </div>
        {replay ? (
          <div className="grid gap-6 xl:grid-cols-[1fr_300px]">
            <FlowTable flows={replay.flows} truth />
            <div className="space-y-6">
              <ConfusionGrid
                c={replay.either}
                caption="Either detector raising a flag, on this draw."
              />
              <div className="space-y-1.5 text-[12.5px] text-slate-400">
                <p>
                  Isolation Forest alone: {replay.isolation_forest.tp} caught,{' '}
                  {replay.isolation_forest.fp} false alarms
                </p>
                <p>
                  Random Forest alone: {replay.random_forest.tp} caught, {replay.random_forest.fp}{' '}
                  false alarms
                </p>
              </div>
            </div>
          </div>
        ) : (
          <p className="text-[13px] text-slate-500">Loading the replay…</p>
        )}
      </section>

      {/* Model card */}
      {card && iso && rf && (
        <section className="panel p-7">
          <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="label-caps mb-2">How good is it · measured, not claimed</p>
              <p className="max-w-[72ch] text-[14px] leading-relaxed text-slate-400">
                Trained and tested on {card.dataset.split(',')[0]}: {int(card.flows)} labelled flows
                over five working days, eight families of attack. Trained {card.trained_at}.
              </p>
            </div>
          </div>

          <div className="grid gap-8 lg:grid-cols-2">
            <div>
              <p className="text-[15px] text-slate-100">Isolation Forest · learns only normal</p>
              <p className="mb-4 text-[12.5px] text-slate-500">
                {iso.trained_on}. Never saw an attack. Threshold: {iso.threshold_rule}.
              </p>
              <OperatingCurve points={iso.operating_points} shipped={0.05} />
              <p className="mt-1 text-center text-[11.5px] text-slate-500">
                False-alarm budget on normal traffic → share of attacks caught
              </p>
              <table className="mt-4 w-full text-[12.5px]">
                <thead>
                  <tr className="label-caps text-left">
                    <th className="py-1.5 font-semibold">Budget</th>
                    <th className="py-1.5 font-semibold">Caught</th>
                    <th className="py-1.5 font-semibold">Alerts real</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-slate-300">
                  {iso.operating_points.map((o) => (
                    <tr key={o.false_alarm_budget} className="border-t border-[var(--line)]">
                      <td className="py-1.5">
                        {Math.round(o.false_alarm_budget * 100)}%
                        {o.false_alarm_budget === 0.05 && (
                          <span className="ml-2 font-sans text-brass-300">shipped</span>
                        )}
                      </td>
                      <td className="py-1.5">{pct(o.recall)}</td>
                      <td className="py-1.5">{pct(o.precision)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div>
              <p className="text-[15px] text-slate-100">Random Forest · knows eight families</p>
              <p className="mb-5 text-[12.5px] text-slate-500">{rf.shipped_model}.</p>
              <div className="grid grid-cols-2 gap-6">
                <Stat
                  label="Seen days"
                  value={pct(rf.random_split.attack_vs_normal.recall)}
                  note={`attacks caught on a random hold-out · ${pct(rf.random_split.attack_vs_normal.precision)} of flags real`}
                  tone="#86AD90"
                />
                <Stat
                  label="Unseen days"
                  value={pct(rf.held_out_days.attack_vs_normal.recall)}
                  note={`trained Mon–Wed, tested Thu–Fri · ${pct(rf.held_out_days.attack_vs_normal.precision)} of flags real`}
                  tone="#D08A4E"
                />
              </div>
              <p className="mt-5 text-[13px] leading-relaxed text-slate-400">
                The gap is the honest part. On traffic like what it learned from, the classifier
                catches nearly everything. Against days and attack types it never saw, it catches
                about a quarter. That is why the Isolation Forest runs beside it, and why the threat
                feeds matter: they flag known-bad addresses no model has to learn.
              </p>
              <p className="label-caps mb-4 mt-7">Isolation Forest · caught by family</p>
              <Bars
                rows={Object.entries(iso.recall_by_family).sort((a, b) => b[1] - a[1])}
                max={1}
              />
            </div>
          </div>
        </section>
      )}
    </div>
  );
};
