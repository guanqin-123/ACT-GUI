"""Presence count: browsers are deduplicated by client id, expire after the window, and group by country;
each new visit adds to per-country totals that survive a restart."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from visual_bench.workspace.service import PresenceService, normalize_country


def test_count_dedupes_browsers_and_expires_silent_ones(tmp_path: Path):
    now = [1000.0]
    svc = PresenceService(tmp_path / "presence.json", window_s=90, clock=lambda: now[0])
    assert svc.beat("browser-a1").online == 1
    assert svc.beat("browser-a1").online == 1
    now[0] += 60
    assert svc.beat("browser-b2").online == 2
    now[0] += 31  # a1 last seen 91 s ago
    assert svc.beat("browser-b2").online == 1


def test_countries_are_ranked_and_unknown_codes_collapse(tmp_path: Path):
    svc = PresenceService(tmp_path / "presence.json")
    svc.beat("browser-a1", "AU")
    svc.beat("browser-b2", "CN")
    svc.beat("browser-c3", "AU")
    svc.beat("browser-d4", normalize_country("XX"))
    snap = svc.beat("browser-e5", normalize_country("T1"))
    assert snap.countries == [("AU", 2), (None, 2), ("CN", 1)]
    assert normalize_country(" au ") == "AU" and normalize_country(None) is None


def test_visits_count_browser_sessions_and_survive_a_restart(tmp_path: Path):
    now = [1000.0]
    svc = PresenceService(tmp_path / "presence.json", window_s=90, clock=lambda: now[0])
    svc.beat("browser-a1", "AU")
    svc.beat("browser-a1", "AU")  # same visit
    svc.beat("browser-b2", None)
    now[0] += 91
    svc.beat("browser-a1", "AU")  # back after the window: a second visit
    restarted = PresenceService(tmp_path / "presence.json", window_s=90, clock=lambda: now[0])
    assert restarted.history() == svc.history()
    assert restarted.history().countries == [("AU", 2), (None, 1)] and restarted.history().since


def test_heartbeat_route_reads_cloudflare_country_and_rejects_bad_ids(client: TestClient):
    first = client.post("/api/presence", json={"client_id": "tab-0000001"}, headers={"CF-IPCountry": "AU"})
    assert first.status_code == 200
    assert first.json() == {"online": 1, "window_s": 90, "countries": [{"country": "AU", "count": 1}]}
    local = client.post("/api/presence", json={"client_id": "tab-0000002"}).json()
    assert local["online"] == 2 and {"country": None, "count": 1} in local["countries"]
    bad = client.post("/api/presence", json={"client_id": "<script>"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_request"


def test_history_route_lists_visits_per_country(client: TestClient):
    assert client.get("/api/presence/history").json() == {"since": None, "countries": []}
    for client_id in ("tab-0000001", "tab-0000002"):
        client.post("/api/presence", json={"client_id": client_id}, headers={"CF-IPCountry": "AU"})
    history = client.get("/api/presence/history").json()
    assert history["countries"] == [{"country": "AU", "count": 2}] and history["since"]
