"""Online-visitor count with per-country totals (WS-04) and per-country visit history (WS-05).

The service runs as one uvicorn process, so a process-local map from client id to last heartbeat
and country is the whole online state: no IPs, and a restart refills within one heartbeat interval.
The country is Cloudflare's CF-IPCountry header (requests reach the service through a Cloudflare
Tunnel, so the peer address is always local). A heartbeat from a browser that is not counted online
starts a visit; the per-country visit totals persist in one small JSON file (no ids, no IPs), so they
survive a restart, which counts browsers that are still open as new visits. Running several workers
or replicas would need this state in a shared store.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_COUNTRY = re.compile(r"^[A-Z]{2}$")


@dataclass(frozen=True)
class PresenceSnapshot:
    online: int
    countries: list[tuple[str | None, int]]  # largest first; None = unknown or local


@dataclass(frozen=True)
class VisitHistory:
    since: str | None  # ISO time of the first recorded visit
    countries: list[tuple[str | None, int]]  # visits per country, largest first


def normalize_country(header: str | None) -> str | None:
    """Keep real alpha-2 codes; Cloudflare's XX (unknown) and T1 (Tor) become unknown."""
    code = (header or "").strip().upper()
    return code if _COUNTRY.match(code) and code != "XX" else None


def _ranked(counts: Counter[str | None]) -> list[tuple[str | None, int]]:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0] is None, kv[0] or ""))


class PresenceService:
    def __init__(self, history_file: Path, window_s: int = 90, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s = window_s
        self._clock = clock
        self._seen: dict[str, tuple[float, str | None]] = {}
        self._lock = threading.Lock()
        self._history_file = history_file
        self._since: str | None = None
        self._visits: Counter[str | None] = Counter()
        if history_file.is_file():
            saved = json.loads(history_file.read_text())
            self._since = saved["since"]
            self._visits.update({code or None: n for code, n in saved["visits"].items()})  # "" = unknown

    def beat(self, client_id: str, country: str | None = None) -> PresenceSnapshot:
        """Record a heartbeat, drop browsers silent for longer than the window, return the totals.

        A browser that is not counted online (its first heartbeat, or one after a longer silence) starts a visit.
        """
        with self._lock:
            now = self._clock()
            cutoff = now - self.window_s
            self._seen = {cid: v for cid, v in self._seen.items() if v[0] >= cutoff}
            if client_id not in self._seen:
                self._record_visit(country)
            self._seen[client_id] = (now, country)
            return PresenceSnapshot(len(self._seen), _ranked(Counter(c for _, c in self._seen.values())))

    def history(self) -> VisitHistory:
        with self._lock:
            return VisitHistory(self._since, _ranked(self._visits))

    def _record_visit(self, country: str | None) -> None:
        self._visits[country] += 1
        self._since = self._since or datetime.now(timezone.utc).isoformat(timespec="seconds")
        saved = {"since": self._since, "visits": {code or "": n for code, n in self._visits.items()}}
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._history_file.with_name(self._history_file.name + ".tmp")
        tmp.write_text(json.dumps(saved, indent=2, sort_keys=True))
        os.replace(tmp, self._history_file)  # a crash mid-write keeps the previous file, never a torn one
