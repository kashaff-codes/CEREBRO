"""
Run the news analyser against hard, realistic input and print what comes back.

    cd backend && python scripts/news_battery.py            # all cases
    cd backend && python scripts/news_battery.py --deep     # the deeper search
    cd backend && python scripts/news_battery.py "your own text here"

Runs the real app in-process (no database needed) with whatever providers the
environment has. Reads ANTHROPIC_API_KEY (and anything else) from backend/.env
if present; that file is git-ignored. Each case says what a good answer looks
like, so the output can be judged, not just read.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
env = ROOT / ".env"
if env.exists():
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
os.environ.setdefault("DATABASE_URL", "postgresql://nobody:x@127.0.0.1:1/missing")
os.environ.setdefault("SECRET_KEY", "battery-secret-key-0123456789abcdefghij")

CASES: list[tuple[str, str]] = [
    ("The Eiffel Tower is in London.",
     "Refuted. Finding says Paris, France; a Wikipedia card with a picture."),
    ("eiffel tower is in london right",
     "Same as above despite lower case and no question mark."),
    ("did jenson hwang get sued over nvidia? whats going on with the nvidia case",
     "Name corrected to Jensen Huang; current reporting cited; says what is unresolved."),
    ("Is it true that the Great Wall of China is visible from space, that goldfish have a "
     "3 second memory, and that Einstein failed math?",
     "Three separate claims, each refuted with a source."),
    ("I saw on reddit that the WHO declared a new pandemic last week and that airports in "
     "Europe are closing again. Also that Pakistan won the 1992 cricket world cup.",
     "Two unsupported current claims (unless true this week) and one supported historical one."),
    ("NASA landed astronauts on the Moon in 1969.", "Supported, high confidence."),
    ("Vaccines cause autism in children.", "Refuted by health agencies; low score."),
    ("SHOCKING!!! Doctors don't want you to know this one miracle cure the mainstream media hides!!",
     "No checkable claim; language signals low; confidence 'Style only'."),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    from fastapi.testclient import TestClient

    from services.api.core.config import settings
    from services.api.main import app

    deep = "--deep" in sys.argv
    custom = [a for a in sys.argv[1:] if not a.startswith("--")]
    cases = [(c, "") for c in custom] or CASES
    print(f"research: {'on' if os.getenv('ANTHROPIC_API_KEY') else 'OFF (no ANTHROPIC_API_KEY)'}"
          f"  depth: {'deep' if deep else 'standard'}\n")

    with TestClient(app) as c:
        tok = c.post("/v1/auth/login", json={"email": settings.demo_email,
                                             "password": settings.demo_password}).json()["access_token"]
        for text, expect in cases:
            t0 = time.time()
            r = c.post("/v1/analyze/news", headers={"Authorization": f"Bearer {tok}"},
                       json={"text": text, "depth": "deep" if deep else "standard"})
            took = time.time() - t0
            print("=" * 100)
            print(f"INPUT  {text}")
            if expect:
                print(f"EXPECT {expect}")
            if r.status_code != 200:
                print(f"ERROR  {r.status_code} {r.text[:300]}")
                continue
            b = r.json()
            print(f"RESULT {b['credibilityScore']}/100  {b['verdict']}  confidence {b['confidence_label']}"
                  f"  ({took:.1f}s)  sources: {json.dumps(b['evidence_sources'])}")
            for comp in b["breakdown"]:
                if comp["score"] is not None:
                    print(f"       {comp['label']:<22} {comp['score']:5.1f}  x{comp['share']:.0%}  {comp['detail'][:90]}")
            for cl in b["claims"]:
                stances = [e["stance"][0].upper() for e in cl["evidence"]]
                print(f"  - CLAIM {cl['text']}")
                print(f"      p(true)={cl['probability']:.2f} certainty={cl['certainty']:.2f} "
                      f"label={cl['label']} evidence={''.join(stances) or '-'}")
                if cl.get("finding"):
                    print(f"      finding: {cl['finding'][:300]}")
                for e in cl["evidence"][:3]:
                    print(f"      [{e['stance'][:4]}] {e['domain']}: {e['snippet'][:110]}")
                if cl.get("topics"):
                    print(f"      explore: {[t['title'] for t in cl['topics']]} "
                          f"images={sum(1 for t in cl['topics'] if t.get('image'))}")
            if b.get("related"):
                print(f"  related coverage: {len(b['related'])} more articles")


if __name__ == "__main__":
    main()
