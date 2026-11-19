import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

// Mock the CEREBRO API for component/integration tests. Path-prefixed with `*`
// so it matches whatever origin VITE_API_BASE resolves to in the test env.
export const handlers = [
  http.post('*/v1/auth/login', () =>
    HttpResponse.json({
      user_id: 'u1',
      tenant_id: 't1',
      role: 'owner',
      display_name: 'Demo Analyst',
      access_token: 'test.jwt.token',
      token_type: 'bearer',
    }),
  ),
  http.post('*/v1/auth/register', () =>
    HttpResponse.json({
      user_id: 'u2',
      tenant_id: 't2',
      role: 'owner',
      display_name: 'New Analyst',
      access_token: 'test.jwt.token2',
      token_type: 'bearer',
    }),
  ),
  http.post('*/v1/auth/logout', () => HttpResponse.json({ status: 'logged_out' })),
  // Signed-out by default so AuthContext mount resolves to the login screen.
  http.get('*/v1/auth/me', () => new HttpResponse(null, { status: 401 })),

  http.post('*/v1/analyze/email', () =>
    HttpResponse.json({
      from_display: 'PayPal Security',
      from_addr: 'alert@paypa1.com',
      from_domain: 'paypa1.com',
      subject: 'Urgent: verify your account',
      spf: 'fail',
      dkim: 'fail',
      dmarc: 'fail',
      dkim_aligned: false,
      verdict: 'phishing',
      risk_score: 0.91,
      score_source: 'heuristic',
      indicators: [
        {
          code: 'LOOKALIKE_SENDER_DOMAIN',
          severity: 'critical',
          weight: 0.35,
          detail: "Sender domain 'paypa1.com' is 1 character from 'paypal'.",
        },
        { code: 'DMARC_FAIL', severity: 'critical', weight: 0.35, detail: 'DMARC failed.' },
      ],
      features: {},
      urls: [],
      explanation: null,
      model_version: null,
    }),
  ),
  http.post('*/v1/analyze/news', () =>
    HttpResponse.json({
      credibilityScore: 26,
      verdict: 'Misleading',
      summary: 'test claim',
      reasoning: 'heuristic reasoning',
      sources: [],
      claims_checked: 1,
      model_versions: {},
      score_source: 'heuristic',
    }),
  ),
  http.get('*/v1/network/model', () =>
    HttpResponse.json({
      dataset: 'CIC-IDS2017 (UNB), MachineLearningCVE CSVs',
      flows: 2830628,
      families: { BENIGN: 2272982 },
      trained_at: '2026-11-19 02:50 UTC',
      isolation_forest: {
        trained_on: '300,000 Monday flows (all normal)',
        threshold_rule: '95th percentile of Tuesday normal',
        test: '',
        metrics: {
          precision: 0.85,
          recall: 0.24,
          f1: 0.38,
          false_positive_rate: 0.06,
          roc_auc: 0.86,
          pr_auc: 0.82,
        },
        recall_by_family: { 'DoS/DDoS': 0.35, PortScan: 0.002 },
        operating_points: [
          {
            false_alarm_budget: 0.01,
            recall: 0.0025,
            precision: 0.12,
            test_false_positive_rate: 0.03,
          },
          {
            false_alarm_budget: 0.05,
            recall: 0.24,
            precision: 0.85,
            test_false_positive_rate: 0.06,
          },
          { false_alarm_budget: 0.2, recall: 0.89, precision: 0.8, test_false_positive_rate: 0.2 },
        ],
      },
      random_forest: {
        classes: ['BENIGN', 'DoS/DDoS'],
        random_split: {
          attack_vs_normal: {
            precision: 0.9,
            recall: 0.997,
            f1: 0.95,
            false_positive_rate: 0.04,
            roc_auc: 0.99,
            pr_auc: 0.99,
          },
        },
        held_out_days: {
          attack_vs_normal: {
            precision: 0.92,
            recall: 0.253,
            f1: 0.4,
            false_positive_rate: 0.01,
            roc_auc: 0.8,
            pr_auc: 0.72,
          },
        },
        shipped_model: 'trained on 538,028 flows from all five days',
      },
    }),
  ),
  http.get('*/v1/network/intel', () =>
    HttpResponse.json([
      {
        feed: 'feodo',
        label: 'Botnet C2 server',
        entries: 12,
        fetched_at: Date.now() / 1000,
        ok: true,
        error: '',
      },
    ]),
  ),
  http.get('*/v1/network/replay', () =>
    HttpResponse.json({
      source: 'CIC-IDS2017 held-out test flows; this is a replay, not your network',
      flows: [
        {
          ...{
            duration_s: 0.01,
            fwd_pkts: 2,
            bwd_pkts: 0,
            fwd_bytes: 0,
            bwd_bytes: 0,
            dst_port: 80,
            src_ip: '10.0.0.5',
            dst_ip: '45.142.212.61',
            proto: 'tcp',
            ts: '',
            label: '',
            anomaly_score: 0.61,
            anomalous: true,
            predicted_family: 'DoS/DDoS',
            family_confidence: 0.9,
            attack_probability: 0.9,
          },
          true_family: 'DoS/DDoS',
          day: 'Wednesday',
        },
      ],
      isolation_forest: { tp: 1, fp: 0, fn: 0, tn: 0 },
      random_forest: { tp: 1, fp: 0, fn: 0, tn: 0 },
      either: { tp: 1, fp: 0, fn: 0, tn: 0 },
    }),
  ),
  http.post('*/v1/network/scan', () =>
    HttpResponse.json({
      summary: {
        format: 'zeek_tsv',
        flows: 1,
        seconds: 0.01,
        anomalous: 1,
        known_attack_flows: 1,
        by_family: { 'DoS/DDoS': 1 },
        threat_level: 'Critical',
        recommended_action: 'Isolate the flagged hosts.',
        intel_hits: [
          {
            ip: '45.142.212.61',
            matches: [
              { feed: 'feodo', label: 'Botnet C2 server', severity: 'critical', malware: 'QakBot' },
            ],
          },
        ],
        feeds: [],
        has_labels: false,
        label_check: null,
      },
      top: [
        {
          duration_s: 0.01,
          fwd_pkts: 2,
          bwd_pkts: 0,
          fwd_bytes: 0,
          bwd_bytes: 0,
          dst_port: 80,
          src_ip: '10.0.0.5',
          dst_ip: '45.142.212.61',
          proto: 'tcp',
          ts: '',
          label: '',
          anomaly_score: 0.61,
          anomalous: true,
          predicted_family: 'DoS/DDoS',
          family_confidence: 0.9,
          attack_probability: 0.9,
          intel: [{ feed: 'feodo', label: 'Botnet C2 server', severity: 'critical' }],
        },
      ],
    }),
  ),
  http.get('*/v1/metrics/summary', () =>
    HttpResponse.json({
      total_detections: 0,
      email_detections: 0,
      news_detections: 0,
      network_detections: 0,
      critical_count: 0,
      suspicious_count: 0,
      avg_risk: 0,
      last_detection_at: null,
      window_hours: 168,
      pct_change: null,
    }),
  ),
  http.get('*/v1/metrics/threat-volume', () => HttpResponse.json({ days: 7, buckets: [] })),
  http.get('*/ready', () =>
    HttpResponse.json({
      status: 'ready',
      uptime_seconds: 1,
      capabilities: {
        email_forensics: true,
        database: true,
        llm_explanations: true,
        models_loaded: [],
      },
    }),
  ),
];

export const server = setupServer(...handlers);
