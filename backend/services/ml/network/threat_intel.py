"""
Live threat intelligence from public feeds, refreshed hourly.

Each external IP in a scanned log is checked against lists maintained by
others, which change by the hour:

  abuse.ch Feodo Tracker   botnet command-and-control servers (Dridex, Emotet,
                           QakBot, ...). CC0.            feodotracker.abuse.ch
  Spamhaus DROP            netblocks hijacked or run by criminal operations;
                           "do not route or peer".        spamhaus.org/drop
  Tor Project exit list    current Tor exit relays (not malicious in itself:
                           context, reported as such).    check.torproject.org

A feed that fails to load is reported as unavailable, never treated as "no
matches": an empty list and a failed download mean different things.
"""
from __future__ import annotations

import ipaddress
import json
import logging
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
TTL_S = 3600
TIMEOUT = 15

FEEDS = {
    "feodo": ("https://feodotracker.abuse.ch/downloads/ipblocklist.json",
              "Botnet C2 server (abuse.ch Feodo Tracker)", "critical"),
    "spamhaus_drop": ("https://www.spamhaus.org/drop/drop_v4.json",
                      "Spamhaus DROP: hijacked or criminal netblock", "high"),
    "tor_exit": ("https://check.torproject.org/torbulkexitlist",
                 "Tor exit relay", "info"),
}


@dataclass
class Feed:
    name: str
    ips: set = field(default_factory=set)
    nets: list = field(default_factory=list)
    detail: dict = field(default_factory=dict)       # ip -> extra (malware family, ...)
    fetched_at: float = 0.0
    error: str = ""


_FEEDS: dict[str, Feed] = {}
_LOCK = threading.Lock()


def _parse(name: str, text: str) -> Feed:
    f = Feed(name)
    if name == "feodo":
        for row in json.loads(text):
            ip = row.get("ip_address")
            if ip:
                f.ips.add(ip)
                f.detail[ip] = {k: row.get(k) for k in ("malware", "status", "last_online", "port")}
    elif name == "spamhaus_drop":
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cidr = row.get("cidr")
            if cidr:
                try:
                    f.nets.append((ipaddress.ip_network(cidr, strict=False), row.get("sblid")))
                except ValueError:
                    continue
    elif name == "tor_exit":
        f.ips = {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")}
    return f


def refresh(force: bool = False) -> dict[str, Feed]:
    import requests
    now = time.time()
    for name, (url, _, _) in FEEDS.items():
        cur = _FEEDS.get(name)
        if cur and not force and now - cur.fetched_at < TTL_S and not cur.error:
            continue
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "CEREBRO-threat-intel/1.0"})
            r.raise_for_status()
            feed = _parse(name, r.text)
            feed.fetched_at = now
        except Exception as exc:  # noqa: BLE001 - keep the previous copy, say it failed
            log.warning("threat feed %s failed: %s", name, exc)
            feed = cur or Feed(name)
            feed.error = type(exc).__name__
        with _LOCK:
            _FEEDS[name] = feed
    return dict(_FEEDS)


def status() -> list[dict]:
    feeds = refresh()
    return [{"feed": n, "label": FEEDS[n][1], "entries": len(f.ips) + len(f.nets),
             "fetched_at": f.fetched_at, "ok": not f.error, "error": f.error}
            for n, f in feeds.items()]


def lookup(ip: str) -> list[dict]:
    """Every feed this IP appears in. Private and invalid addresses never match."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return []
    if addr.is_private or addr.is_loopback or addr.is_reserved:
        return []
    hits = []
    for name, f in refresh().items():
        _, label, severity = FEEDS[name]
        if ip in f.ips:
            hits.append({"feed": name, "label": label, "severity": severity, **f.detail.get(ip, {})})
        elif f.nets:
            for net, sbl in f.nets:
                if addr in net:
                    hits.append({"feed": name, "label": label, "severity": severity,
                                 "netblock": str(net), "sbl": sbl})
                    break
    return hits
