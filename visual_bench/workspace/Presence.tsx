import { lazy, Suspense, useState } from "react";
import type { CountryCount } from "../generated/wire";
import { usePresence, useVisitHistory } from "./hooks";

const WorldMap = lazy(() => import("./WorldMap"));
const regionNames = new Intl.DisplayNames(["en"], { type: "region" });
const dayFormat = new Intl.DateTimeFormat("en", { day: "numeric", month: "short", year: "numeric" });

const flag = (code: string) => String.fromCodePoint(...[...code].map((ch) => 0x1f1a5 + ch.charCodeAt(0)));
const countryLabel = (code: string | null | undefined) => (code ? `${flag(code)} ${regionNames.of(code) ?? code}` : "Local / unknown");

// Footer bar under Run history: how many browsers have the workbench open; hover maps them with past visits.
export function PresenceBar() {
  const presence = usePresence();
  const [open, setOpen] = useState(false);
  const n = presence.data?.online;
  return (
    <div className="presence" tabIndex={0} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)} onBlur={() => setOpen(false)}>
      <span className={presence.isError ? "presencedot off" : "presencedot"} />
      {presence.isError ? "Online count unavailable"
        : n === undefined ? "Counting visitors…"
        : <><b key={n}>{n}</b> {n === 1 ? "person" : "people"} online</>}
      {open && presence.data && <CountryCard online={presence.data.countries} windowS={presence.data.window_s} />}
    </div>
  );
}

type CountryRow = { country: string | null; online: number; visits: number };

// One row per country online now or in the visit history: most online first, then most visits.
function countryRows(online: CountryCount[], visits: CountryCount[]): CountryRow[] {
  const rows = new Map<string | null, CountryRow>();
  const row = (country: string | null) => {
    let r = rows.get(country);
    if (!r) rows.set(country, (r = { country, online: 0, visits: 0 }));
    return r;
  };
  for (const c of online) row(c.country).online = c.count;
  for (const c of visits) row(c.country).visits = c.count;
  return [...rows.values()].sort((a, b) => b.online - a.online || b.visits - a.visits);
}

function CountryCard({ online, windowS }: { online: CountryCount[]; windowS: number }) {
  const history = useVisitHistory();
  const visits = history.data?.countries ?? [];
  const since = history.data?.since;
  const visitsNote = history.isError ? "unavailable" : history.isPending ? "loading…"
    : since ? `browser sessions since ${dayFormat.format(new Date(since))}` : "none yet";
  return (
    <div className="presencepop" role="tooltip">
      <Suspense fallback={<div className="worldmap loading">Loading map…</div>}>
        <WorldMap online={online} visits={visits} />
      </Suspense>
      <div className="presencehead">
        <span>Country</span><span className="presencekey online">Online</span><span className="presencekey visits">Visits</span>
      </div>
      <ul className="presencelist">
        {countryRows(online, visits).map((r) => (
          <li key={r.country ?? "unknown"}>
            <span>{countryLabel(r.country)}</span><b>{r.online || "–"}</b><b>{r.visits || (history.isPending ? "…" : "–")}</b>
          </li>
        ))}
      </ul>
      <div className="note">Online: browsers active in the last {windowS} s. Visits: {visitsNote}. Countries from Cloudflare.</div>
    </div>
  );
}
