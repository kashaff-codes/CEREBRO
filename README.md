# CEREBRO

**Threat & misinformation intelligence** — an evidence-producing security platform.
Give it an email, a news claim, or network traffic; it tells you whether it's
dangerous or false, and shows you *why*: the exact features it measured, the
named model or heuristic that scored them, and — for misinformation — real
fact-check sources you can open and read.

It's built to be the **data & feature layer** for a larger predictive system
(the *Apollo* TFT + RoBERTa project) — the eyes and ears that feed the brain.

---

## 🚀 Live demo

| | |
|---|---|
| **App** | **https://cerebro-sandy-beta.vercel.app** |
| **API docs** | https://cerebro-api-nmah.onrender.com/docs |
| **Demo login** | `demo@cerebro.app` / `cerebro-demo-2026` (or use **SIGN_UP**) |
| **Monitoring** | https://apollo-m-monitoring.onrender.com — Grafana, read-only, no login (shared with APOLLO-M) |

> The backend runs on a free tier that sleeps after ~15 min idle, so the first
> request after a quiet spell takes ~50 s to wake up, then it's fast. Works on
> desktop and mobile (iOS/Android).

> **Database status (November 2026).** The Render free-tier PostgreSQL was
> deleted when its free period ended. CEREBRO keeps working without it: the
> email, news and network analysers run, news retrieves from the curated seed
> corpus in memory, and the demo account signs in ("Demo (offline mode)").
> History, dashboard metrics and new accounts return 503 with an explanation,
> and the dashboard says so. To bring them back:
> [`docs/RESTORE_DATABASE.md`](docs/RESTORE_DATABASE.md) — three environment
> variables on Render.

**Stack:** React 19 + Vite (Vercel) · FastAPI (Render) · Postgres 16 + pgvector
(Render). Frontend and API are separate services talking over HTTPS + WebSocket.

---

## The architectural rule

> **Deterministic code and trained models decide. The language model only explains.**

The previous version asked a chatbot for a "credibility score" — invented
sources, no measurable accuracy, not reproducible, and it shipped inboxes to a
third party. CEREBRO inverts that: every verdict is measurable, repeatable,
auditable, and runs locally. The LLM (optional) only writes prose about results
that already exist.

---

## What works today

| Module | What it does | How it decides |
|---|---|---|
| **Email forensics** | Phishing / spoofing, one message or a whole inbox | ~35 deterministic signals from the raw RFC 5322 message — SPF/DKIM/DMARC + DKIM alignment, lookalike domains, Unicode homographs, URL tricks. **Connect Gmail** (read-only, `gmail.readonly`) and the last 25–100 messages are scored: verdict counts, SPF/DKIM/DMARC pass rates, most common warnings, top senders, each message expandable to its signals. Tokens are exchanged server-side, AES-GCM-encrypted and bound to the user, refreshed when they expire and revoked at Google on disconnect. A trained classifier (F1 0.97 on SpamAssassin) can replace the heuristic prior. |
| **Misinformation** | True / false / unverifiable, for anything pasted in: a headline, a Reddit paragraph, several questions at once, typos and all | **Research**: Claude with web search restates the input as clean claims (fixing spelling, resolving names, turning questions into the statement asked about), searches the live web, and writes a cited finding per claim, including the correction when a claim is wrong. **Evidence**: the quoted passages it cited, plus Wikipedia (with images) and the curated corpus, each judged supports / refutes / unrelated from its own text. **Score**: a weighted mean of named components (evidence agreement, published fact-checks, language signals, an uncertainty prior) returned with the result, so every point can be traced to a passage with a URL. *Dig deeper* searches more widely. |
| **Network watch** | Intrusions / attacks in your own logs | Upload a Zeek `conn.log` (TSV or JSON), a CICFlowMeter CSV or a plain flow CSV. Two detectors trained on **CIC-IDS2017** (2.83 M labelled flows): an **Isolation Forest** that learned only Monday's normal traffic, and a **Random Forest** over eight attack families. Every public address is checked against live feeds (abuse.ch Feodo Tracker, Spamhaus DROP, Tor exits). A replay of 351 held-out labelled flows shows the true answer beside each verdict. Measured accuracy below. |
| **Dashboard** | Live metrics | Real counts over the `detections` table, event-driven via Postgres `NOTIFY` → WebSocket. |
| **Auth** | Multi-tenant | scrypt + stdlib JWT (rejects `alg=none`), AES-GCM-encrypted OAuth tokens, row-level security in the database, Bearer-token sessions that work on iOS Safari. With no database, only the published demo account can sign in (constant-time check; `OFFLINE_DEMO_LOGIN=false` disables it). |
| **Monitoring** | Prometheus | `GET /metrics`: database up, models loaded, LLM chain, request counts and latency per route — scraped with APOLLO-M's by [apollo-m/monitoring/hosted](https://github.com/kashaff-codes/apollo-m/tree/main/monitoring/hosted). |

Every verdict carries a `score_source` and a `breakdown` of what the score was
built from, so it's never mistaken for more certainty than it has.

### How accurate are the network detectors? (measured, not claimed)

Trained by `backend/scripts/train_network_detector.py` on the eight
MachineLearningCVE CSVs (2,830,628 flows, Monday–Friday). The models ship as
numpy arrays in `backend/models/network/` (2.2 MB) and score identically to
scikit-learn (max difference < 1e-15, checked at training time and in CI).

| Detector | Test | Attacks caught | Alerts that were real | Normal flows flagged |
|---|---|---|---|---|
| Isolation Forest, 5% false-alarm budget (shipped) | Tue–Fri, 957 k flows | 24.1% | 84.7% | 6.1% |
| Isolation Forest, 20% budget | same | 89.1% | 85.0% | 22.0% |
| Random Forest | random 30% hold-out | 99.7% | 90.4% | 3.6% |
| Random Forest | **train Mon–Wed, test Thu–Fri** | **30.9%** | 94.9% | 1.0% |

The last row is the one that matters for a buyer: against days and attack
types it never saw, the classifier catches under a third. That is why the
unsupervised detector runs beside it and why known-bad addresses come from
live feeds rather than a model. Per-family recall and the full operating curve
are served at `GET /v1/network/model` and drawn on the page.

### How accurate is the news verifier? (measured, not claimed)

[`backend/eval/claims.jsonl`](backend/eval/claims.jsonl) holds 36 labelled claims:
true facts, popular myths, swapped names and dates, invented towns, institutes
and companies (where the only right answer is "unverified"), and typo'd,
unpunctuated versions. [`scripts/news_eval.py`](backend/scripts/news_eval.py)
runs the real app against them; [`backend/eval/results.json`](backend/eval/results.json)
has every row.

| | Result |
|---|---|
| Overall | **35 / 36 correct (97.2%)** |
| True claims → Credible | 14 / 15 (mean score 73) |
| False claims → Fake News or Misleading | **18 / 18** (mean score 28) |
| Invented entities → Unverified / Disputed | **3 / 3**: no invented answers |
| Does the score separate true from false? | ROC AUC **1.0**: every true claim scored above every false one |
| Cost and speed | ≈ $0.10 and 11 s per check (Claude Opus 5.5 research + web search) |

The one miss exposed a real bug, since fixed: the verdict was decided on the
unrounded score (64.6) while the page showed 65. Before this pipeline, the score
sat at 50 for almost anything outside a 17-document corpus, and "NASA landed
astronauts on the Moon in 1969" scored 17 / Fake News; tests now pin both.

Run it yourself: put `ANTHROPIC_API_KEY` (and `ANTHROPIC_WORKSPACE_ID` for a
user-scoped key) in the git-ignored `backend/.env`, then
`python scripts/news_eval.py`, or `python scripts/news_battery.py` for messy,
multi-question, current-events inputs with the full output printed.

---

## The data it collects (and how it feeds Apollo)

Every analysis is **persisted to Postgres**, which is the whole point for the
parent project:

- `detections` — the unified stream: `(ts, tenant, module, threat_type,
  risk_score, summary)`. This is the hourly time-series the **TFT forecaster**
  consumes.
- `email_analyses` — the full forensic record (SPF/DKIM/DMARC, from-domain,
  features, indicators, verdict) → labeled **classifier training data**.
- `verdicts`, `anomalies`, `analyst_feedback` — verdicts with evidence, network
  anomalies, and the correction→retrain flywheel.

**Apollo connects to CEREBRO via any of:**
1. **Shared database** — query the tables above with plain SQL (the intended
   integration; the schema is designed for it).
2. **Realtime event stream** — `pg_notify('cerebro_events', …)` fires on every
   detection; a consumer gets it with no polling.
3. **REST API** — `/v1/analyze/*`, `/v1/metrics/*`, `/v1/flows/*`, `/v1/stream`.
4. **`model_registry`** — one shared table so both projects version models the
   same way.

Grab the data for Apollo:
```bash
psql "<cerebro-db connection string>" \
  -c "SELECT ts, module, threat_type, risk_score, summary FROM cerebro.detections ORDER BY ts DESC;"
```

CEREBRO turns raw text and telemetry into the engineered features Apollo's
predictive core consumes — a clean, defensible module boundary.

---

## Run it locally

Everything runs in Docker on a laptop — no accounts, no keys required.

```bash
git clone https://github.com/kashaff-codes/CEREBRO.git
cd CEREBRO
docker compose up -d          # Postgres + pgvector, Redis, API (seeds a demo login)
cd frontend && npm install && npm run dev
```

- App: http://localhost:5173  ·  API docs: http://localhost:8000/docs
- Login: `demo@cerebro.app` / `cerebro-demo-2026`

> On a restricted network that can't pull the `pgvector` image, build it from
> the cached base first: `docker build -f deploy/local/Dockerfile.pgvector -t
> cerebro-pgvector:pg16 .` (a compose override wires it in automatically).

### Optional: better results
```bash
cd backend
python scripts/fetch_datasets.py                                  # SpamAssassin corpus
python scripts/train_email_classifier.py --phishing data/email/spam --benign data/email/ham \
    --out models/email_classifier.joblib                          # F1 ~0.97 → set EMAIL_CLASSIFIER_PATH
```
Set `ANTHROPIC_API_KEY` for live web research (the news verifier's main path;
`RESEARCH_MODEL` defaults to `claude-opus-5-5`, `ANTHROPIC_WORKSPACE_ID` is needed
for a user-scoped `sk-ant-usr-` key) and `FACT_CHECK_API_KEY` (free from Google)
for published fact-checks. Without either, the verifier still runs on Wikipedia,
GDELT and the curated corpus, and never says Credible or Fake News on word
overlap alone.

**Design.** The interface follows one system ("Dossier"): tuxedo black and ivory
with a single brass accent; Bodoni Moda for what matters, Instrument Sans for
everything else, IBM Plex Mono only for file metadata. The map behind sign-in and
on the Overview is Natural Earth land as a dot matrix, with the agencies whose
reporting the verifier weighs most at their real coordinates: no invented
telemetry.

---

## Deploy your own

See [`infra/DEPLOY.md`](infra/DEPLOY.md). The live instance uses **Render**
(backend + Postgres, one-click via [`render.yaml`](render.yaml)) + **Vercel**
(frontend). Set `VITE_API_BASE` to the API URL, and `CORS_ORIGINS` /
`SESSION_COOKIE_SAMESITE=none` for the cross-site setup.

---

## Project layout

```
frontend/     React 19 + Vite SPA — dashboard, scanners, auth, live threat-map UI
backend/
  services/api/     FastAPI: routers (auth, email, gmail, news, netwatch, metrics,
                    stream, forecast), core (db/asyncpg, queries, security, deps)
  services/ml/      email forensics · RAG verify · network (features, numpy forests,
                    threat feeds) · TFT · LLM providers
  models/network/   shipped detectors (npz), model card, held-out replay sample
  db/               schema_portable.sql · queries.sql
  scripts/          dataset fetch, model training, backtest
docs/         ARCHITECTURE · STATUS · OVERVIEW · FRONTEND_MIGRATION
infra/        render.yaml · netlify.toml · DEPLOY.md
deploy/       huggingface/ · local/ (pgvector build)
```

---

## Honest gaps / roadmap

Stating these is what makes the rest credible.

- **Automated ingestion** (RSS/IMAP/Zeek workers) isn't built — data currently
  enters through the API. This is the next step to a self-feeding pipeline.
- **Network watch** scores uploaded logs; it does not sit on a live tap. A
  Zeek sensor shipping `conn.log` on a schedule is the next step. Its classifier
  generalises poorly to unseen attack days (30.9% recall, table above).
- **Gmail** uses a restricted scope. Until Google verifies the app (which for
  `gmail.readonly` includes a third-party security assessment), only test users
  listed on the consent screen can connect, and Google expires their refresh
  tokens after 7 days, so they reconnect weekly.
- **The email classifier** ships as a script; the live instance scores mail with
  the deterministic rule engine.
- **TFT forecasting** is scaffolded and validated on real data (see
  [`docs/STATUS.md`](docs/STATUS.md)) but needs ~2 weeks of accumulated
  `detections` history before it produces production forecasts.
- The accuracy set is 36 claims; a production claim would need several hundred
  (e.g. AVeriTeC) and a calibration check, not one run.
- Live research costs about $0.10 per check; at volume it needs caching and a
  per-tenant budget.

---

## Tests

```bash
cd backend
python tests/test_forensics.py   # 11 — SPF/DKIM/DMARC, homographs
python tests/test_security.py    # 23 — alg=none, tampering, cross-tenant, AEAD
python tests/test_rag.py         # 17 — retrieval, citations, refuses to guess
python tests/test_queries.py     #  7 — named-query loader
python tests/test_anomaly.py     #  9 — fits on benign only (needs scikit-learn)
pytest tests/test_offline.py     #  9 — the real app with its database gone
pytest tests/test_network.py     # 14 — numpy = sklearn, Zeek/CIC parsing, feeds, Gmail scan + OAuth state
```

**117 backend tests**, including 23 adversarial security tests, run against
Postgres 16 + pgvector in CI. `test_offline` boots the app against a database
that does not exist: demo-only sign-in, honest 503s, news from the in-memory
corpus, valid Prometheus output.

### The frontend gate

Eight layered checks, each answering a different question, all in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Check | Command | Catches |
|---|---|---|
| Type safety | `npm run typecheck` | contract breaks at the boundaries |
| Format | `npm run format:check` | Prettier drift |
| Lint + accessibility | `npm run lint` | real bugs, and `jsx-a11y` violations |
| Unit + component | `npm test` | Vitest + Testing Library |
| Contract | (same run) | MSW mocks the API, so a changed shape fails here |
| Build | `npm run build:client` | anything the type-checker cannot see |
| Bundle-size budget | `npm run size` | silent weight regressions — currently 473 KB gzip of a 750 KB budget |
| End-to-end + visual | `npm run test:e2e` | Playwright on desktop Chrome and an emulated iPhone, screenshots compared against committed baselines |

Visual baselines are platform-specific, so they are generated on the CI runner
by [`update-snapshots.yml`](.github/workflows/update-snapshots.yml) rather than
on a laptop — a baseline captured on Windows reports Linux font rasterisation as
a regression on every run. That workflow is manual on purpose: refreshing
baselines automatically would mean the check could never fail.

### Bugs that became checks

The point of the gate is that nothing gets fixed twice:

- **iOS/WhatsApp in-app browser sign-in** failed on a session-storage
  assumption. Now covered by the Playwright run on an emulated iPhone.
- **An API error shape change** broke a scanner silently. Now covered by an MSW
  contract test.
- **A named SQL query was renamed** without updating its caller. Now covered by
  a build-time loader test.
- **Layout could break with everything else green.** Now covered by the visual
  regression baselines.
- **The hosted database was deleted** and every login and news check answered
  500. Now covered by `test_offline`, which runs the app with no database.
- **Clickable `div`s on the sign-in and dashboard screens** could not be
  reached by keyboard. They are buttons now, and `jsx-a11y` reports nothing.

---

*Full design rationale in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · current
status in [`docs/STATUS.md`](docs/STATUS.md) · plain-English overview in
[`docs/OVERVIEW.md`](docs/OVERVIEW.md).*
