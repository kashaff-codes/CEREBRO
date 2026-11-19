"""
Train and evaluate CEREBRO's network detectors on CIC-IDS2017.

    python scripts/train_network_detector.py            # data in backend/data/cicids2017/

CIC-IDS2017 (Canadian Institute for Cybersecurity, University of New Brunswick):
five working days of real traffic on an instrumented network, with attacks run
on known days and every flow labelled. Monday is entirely normal.

Two detectors, because they answer different questions:

  Isolation Forest — "is this flow unlike normal traffic?" Unsupervised: it
    sees only Monday's normal flows. Its alert threshold is fixed at the 99th
    percentile of scores on Tuesday's NORMAL flows (a 1% false-alarm budget),
    without looking at a single attack. Then it is tested on every attack from
    Tuesday to Friday. This is the honest measure of catching attacks it was
    never shown.

  Random Forest — "which known attack family is this?" Supervised, reported two
    ways: a random 70/30 split across all days (what most CIC-IDS2017 papers
    report; optimistic, because flows from one attack session land on both
    sides), and a held-out-days split (train Mon-Wed, test Thu-Fri: attack
    families it never saw count against it). Both numbers are published.

Both use only the portable features in services/ml/network/features.py, so the
same models score Zeek conn.log and NetFlow, not just CICFlowMeter CSVs.

Outputs (committed; small): backend/models/network/
  iforest.npz, forest.npz      the trees, as arrays (ensemble.py scores them)
  model_card.json              every metric below, the thresholds, the data
  replay_sample.csv            labelled test flows for the console's replay
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.ml.network.ensemble import ForestClassifierArrays, IsolationForestArrays  # noqa: E402
from services.ml.network.features import FEATURES, FlowRecord, vector  # noqa: E402

DATA = ROOT / "data" / "cicids2017"
OUT = ROOT / "models" / "network"
SEED = 2026
COLS = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Destination Port", "Label"]
DAYS = {
    "Monday": ["Monday-WorkingHours.pcap_ISCX.csv"],
    "Tuesday": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
    "Wednesday": ["Wednesday-workingHours.pcap_ISCX.csv"],
    "Thursday": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                 "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
    "Friday": ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
               "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
               "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
}


def family(label: str) -> str:
    lab = label.lower()
    if lab == "benign":
        return "BENIGN"
    if "ddos" in lab or lab.startswith("dos"):
        return "DoS/DDoS"
    if "portscan" in lab:
        return "PortScan"
    if "patator" in lab:
        return "BruteForce"
    if "web attack" in lab or "xss" in lab or "sql" in lab:
        return "WebAttack"
    if "bot" in lab:
        return "Bot"
    if "infiltration" in lab:
        return "Infiltration"
    if "heartbleed" in lab:
        return "Heartbleed"
    return "Other"


def load() -> pd.DataFrame:
    frames = []
    for day, files in DAYS.items():
        for f in files:
            df = pd.read_csv(DATA / f, skipinitialspace=True, encoding="latin-1", low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            df = df[COLS].copy()
            df["day"] = day
            frames.append(df)
            print(f"  {f}: {len(df):,} flows", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["Label"] = df["Label"].astype(str).str.strip().str.replace("�", "-", regex=False)
    num = COLS[:-1]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=num)
    df = df[(df["Flow Duration"] >= 0)]
    df["family"] = df["Label"].map(family)
    return df


def features(df: pd.DataFrame) -> np.ndarray:
    recs = (FlowRecord(r[0] / 1e6, r[1], r[2], r[3], r[4], int(r[5]))
            for r in df[COLS[:-1]].itertuples(index=False, name=None))
    return np.asarray([vector(r) for r in recs], dtype=np.float64)


def binary_metrics(y: np.ndarray, flag: np.ndarray, score: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    tp = int(((flag == 1) & (y == 1)).sum())
    fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum())
    tn = int(((flag == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
            "roc_auc": round(float(roc_auc_score(y, score)), 4),
            "pr_auc": round(float(average_precision_score(y, score)), 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    import sklearn
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import classification_report

    t0 = time.time()
    print("loading CIC-IDS2017 ...", flush=True)
    df = load()
    print(f"{len(df):,} flows after cleaning; {df['family'].value_counts().to_dict()}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    card: dict = {"dataset": "CIC-IDS2017 (Canadian Institute for Cybersecurity, UNB), MachineLearningCVE CSVs",
                  "flows": int(len(df)), "families": df["family"].value_counts().to_dict(),
                  "features": FEATURES, "sklearn": sklearn.__version__, "seed": SEED,
                  "trained_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}

    # ── Replay sample, drawn FIRST and kept out of every model's training ───
    # The console replays these flows and the upload page offers them as a
    # labelled file. If the shipped classifier had trained on them, a perfect
    # score there would be memory, not detection.
    later = df[df["day"] != "Monday"]
    parts = [later[later["family"] == "BENIGN"].sample(n=240, random_state=SEED)]
    for fam, g in later[later["family"] != "BENIGN"].groupby("family"):
        # never more than a quarter of a family: rare ones (Heartbleed: 11) must stay trainable
        parts.append(g.sample(n=max(1, min(20, len(g) // 4)), random_state=SEED))
    rep = pd.concat(parts).sample(frac=1.0, random_state=SEED)
    pd.DataFrame({
        "day": rep["day"], "duration_s": (rep["Flow Duration"] / 1e6).round(6),
        "fwd_pkts": rep["Total Fwd Packets"], "bwd_pkts": rep["Total Backward Packets"],
        "fwd_bytes": rep["Total Length of Fwd Packets"], "bwd_bytes": rep["Total Length of Bwd Packets"],
        "dst_port": rep["Destination Port"].astype(int), "label": rep["Label"], "family": rep["family"],
    }).to_csv(OUT / "replay_sample.csv", index=False)
    card["replay_sample"] = (f"{len(rep)} flows from Tuesday-Friday, removed before any model was trained "
                             "or evaluated; same days as the classifier's training data, so expect its "
                             "seen-days numbers on them, not its unseen-days numbers")
    df = df.drop(rep.index)

    # ── Isolation Forest: unknown-attack detector ───────────────────────────
    mon = df[df["day"] == "Monday"]
    train = mon.sample(n=min(300_000, len(mon)), random_state=SEED)
    tue_benign = df[(df["day"] == "Tuesday") & (df["family"] == "BENIGN")]
    calib = tue_benign.sample(n=min(100_000, len(tue_benign)), random_state=SEED)
    test = df[(df["day"] != "Monday") & ~df.index.isin(calib.index)]
    benign_test = test[test["family"] == "BENIGN"]
    attacks = test[test["family"] != "BENIGN"]
    test = pd.concat([benign_test.sample(n=min(400_000, len(benign_test)), random_state=SEED), attacks])

    print(f"isolation forest: train {len(train):,} Monday flows", flush=True)
    iforest = IsolationForest(n_estimators=200, max_samples=512, random_state=SEED, n_jobs=-1)
    iforest.fit(features(train))
    arrays = IsolationForestArrays.export(iforest)
    portable = IsolationForestArrays(arrays)

    Xc = features(calib)
    sc = portable.score(Xc)
    Xt = features(test)
    st = portable.score(Xt)
    yt = (test["family"] != "BENIGN").to_numpy().astype(int)
    # One threshold hides the trade-off, so publish the curve: recall at several
    # false-alarm budgets, each fixed on Tuesday's normal flows alone.
    operating = []
    for budget in (0.01, 0.02, 0.05, 0.10, 0.20):
        t = float(np.quantile(sc, 1 - budget))
        f = st >= t
        operating.append({"false_alarm_budget": budget, "threshold": round(t, 6),
                          "recall": round(float(f[yt == 1].mean()), 4),
                          "test_false_positive_rate": round(float(f[yt == 0].mean()), 4),
                          "precision": round(float(yt[f].mean()), 4) if f.any() else 0.0})
    SHIP_BUDGET = 0.05
    threshold = next(o["threshold"] for o in operating if o["false_alarm_budget"] == SHIP_BUDGET)
    m = binary_metrics(yt, (st >= threshold).astype(int), st)
    per = {fam: round(float((st[test["family"].to_numpy() == fam] >= threshold).mean()), 4)
           for fam in sorted(attacks["family"].unique())}
    # Identity check: the numpy scorer must equal scikit-learn's own.
    probe = Xt[:2000]
    diff = float(np.abs(portable.score(probe) - (-iforest.score_samples(probe))).max())
    print(f"  threshold {threshold:.4f}; {m}; recall by family {per}; max |numpy-sklearn| {diff:.2e}",
          flush=True)
    card["isolation_forest"] = {
        "trained_on": f"{len(train):,} Monday flows (all normal)",
        "threshold": round(threshold, 6),
        "threshold_rule": "95th percentile of scores on Tuesday's normal flows (5% false-alarm budget, no attacks seen)",
        "operating_points": operating,
        "test": f"{len(test):,} flows, Tuesday-Friday: {len(attacks):,} attacks + normal",
        "metrics": m, "recall_by_family": per, "max_abs_diff_vs_sklearn": diff,
    }
    np.savez_compressed(OUT / "iforest.npz", **arrays)

    # ── Random Forest: known-attack-family classifier ───────────────────────
    def fit_forest(tr: pd.DataFrame):
        rf = RandomForestClassifier(n_estimators=60, max_depth=16, min_samples_leaf=3,
                                    class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
        rf.fit(features(tr), tr["family"].to_numpy())
        return rf

    def eval_forest(rf, te: pd.DataFrame) -> dict:
        X = features(te)
        pred = rf.predict(X)
        y = te["family"].to_numpy()
        rep = classification_report(y, pred, output_dict=True, zero_division=0)
        attack_score = 1.0 - rf.predict_proba(X)[:, list(rf.classes_).index("BENIGN")]
        b = binary_metrics((y != "BENIGN").astype(int), (pred != "BENIGN").astype(int), attack_score)
        return {"attack_vs_normal": b, "macro_f1": round(rep["macro avg"]["f1-score"], 4),
                "per_family": {k: {"precision": round(v["precision"], 4), "recall": round(v["recall"], 4),
                                   "support": int(v["support"])}
                               for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")}}

    def balanced(frame: pd.DataFrame, cap_benign: int, cap_attack: int) -> pd.DataFrame:
        parts = []
        for fam, g in frame.groupby("family"):
            cap = cap_benign if fam == "BENIGN" else cap_attack
            parts.append(g.sample(n=min(cap, len(g)), random_state=SEED))
        return pd.concat(parts)

    pool = balanced(df, 400_000, 60_000)
    shuffled = pool.sample(frac=1.0, random_state=SEED)
    cut = int(len(shuffled) * 0.7)
    rf_random = fit_forest(shuffled.iloc[:cut])
    random_eval = eval_forest(rf_random, shuffled.iloc[cut:])
    print(f"random forest, random split: {random_eval['attack_vs_normal']}, macro F1 {random_eval['macro_f1']}",
          flush=True)

    early = balanced(df[df["day"].isin(["Monday", "Tuesday", "Wednesday"])], 300_000, 60_000)
    late = balanced(df[df["day"].isin(["Thursday", "Friday"])], 200_000, 60_000)
    rf_days = fit_forest(early)
    days_eval = eval_forest(rf_days, late)
    print(f"random forest, held-out days: {days_eval['attack_vs_normal']}, macro F1 {days_eval['macro_f1']}",
          flush=True)

    # The shipped classifier is trained on everything, like a deployed model would be.
    rf_final = fit_forest(pool)
    farr = ForestClassifierArrays.export(rf_final)
    fport = ForestClassifierArrays(farr, list(rf_final.classes_))
    probe = features(pool.sample(n=2000, random_state=SEED))
    fdiff = float(np.abs(fport.proba(probe) - rf_final.predict_proba(probe)).max())
    np.savez_compressed(OUT / "forest.npz", **farr)
    card["random_forest"] = {
        "classes": list(rf_final.classes_),
        "random_split": random_eval, "held_out_days": days_eval,
        "shipped_model": f"trained on {len(pool):,} flows from all five days (balanced by family)",
        "max_abs_diff_vs_sklearn": fdiff,
    }

    card["seconds"] = round(time.time() - t0, 1)
    (OUT / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"done in {card['seconds']} s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
