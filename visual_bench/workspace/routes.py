"""HTTP routes of the workspace module: POST /api/presence (heartbeat in, online count out) and
GET /api/presence/history (visits per country)."""
from __future__ import annotations

from fastapi import APIRouter, Header

from visual_bench.workspace.schemas import CountryCount, PresenceHeartbeat, PresenceHistoryOut, PresenceOut
from visual_bench.workspace.service import PresenceService, normalize_country


def _counts(countries: list[tuple[str | None, int]]) -> list[CountryCount]:
    return [CountryCount(country=c, count=n) for c, n in countries]


def router(presence: PresenceService) -> APIRouter:
    r = APIRouter(prefix="/api", tags=["workspace"])

    @r.post("/presence", response_model=PresenceOut)
    def heartbeat(body: PresenceHeartbeat, cf_ipcountry: str | None = Header(default=None)) -> PresenceOut:
        snap = presence.beat(body.client_id, normalize_country(cf_ipcountry))
        return PresenceOut(online=snap.online, window_s=presence.window_s, countries=_counts(snap.countries))

    @r.get("/presence/history", response_model=PresenceHistoryOut)
    def history() -> PresenceHistoryOut:
        visits = presence.history()
        return PresenceHistoryOut(since=visits.since, countries=_counts(visits.countries))

    return r
