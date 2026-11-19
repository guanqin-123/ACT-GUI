// Small choropleth of past visits per country with the countries online now in green; loaded on first hover so the
// atlas stays out of the main bundle.
import { geoNaturalEarth1, geoPath } from "d3-geo";
import { feature } from "topojson-client";
import type { GeometryCollection, Topology } from "topojson-specification";
import atlas from "world-atlas/countries-110m.json";
import type { CountryCount } from "../generated/wire";
import { ATLAS_ID } from "./countryCodes";

const W = 320, H = 160;
const topo = atlas as unknown as Topology<{ countries: GeometryCollection<{ name: string }> }>;
const land = feature(topo, topo.objects.countries).features.filter((f) => f.id !== "010"); // drop Antarctica
const path = geoPath(geoNaturalEarth1().fitSize([W, H], { type: "FeatureCollection", features: land }));
const SHAPES = land.map((f, i) => ({ key: String(f.id ?? i), id: f.id === undefined ? null : String(f.id), name: f.properties.name, d: path(f) ?? "" }));

// Atlas id to count; countries too small for the 110m atlas have no shape and stay list-only.
function byAtlasId(countries: CountryCount[]): Map<string, number> {
  const byId = new Map<string, number>();
  for (const c of countries) {
    const id = c.country ? ATLAS_ID[c.country] : undefined;
    if (id) byId.set(id, c.count);
  }
  return byId;
}

export default function WorldMap({ online, visits }: { online: CountryCount[]; visits: CountryCount[] }) {
  const onlineById = byAtlasId(online);
  const visitsById = byAtlasId(visits);
  const maxOnline = Math.max(1, ...onlineById.values());
  const maxVisits = Math.log1p(Math.max(1, ...visitsById.values()));
  return (
    <svg className="worldmap" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Online browsers and past visits by country">
      {SHAPES.map((s) => {
        const n = s.id ? onlineById.get(s.id) : undefined;
        const v = s.id ? visitsById.get(s.id) : undefined;
        // Visits shade on a log scale so one busy country does not wash out the rest; online now overrides in green.
        const fill = n ? `color-mix(in srgb, var(--good) ${Math.round(55 + 45 * n / maxOnline)}%, var(--panel2))`
          : v ? `color-mix(in srgb, var(--accent) ${Math.round(30 + 45 * Math.log1p(v) / maxVisits)}%, var(--panel2))`
          : undefined;
        const counts = [n && `${n} online`, v && `${v} ${v === 1 ? "visit" : "visits"}`].filter(Boolean).join(", ");
        return (
          <path key={s.key} d={s.d} className={n ? "on" : undefined} style={fill ? { fill } : undefined}>
            <title>{counts ? `${s.name}: ${counts}` : s.name}</title>
          </path>
        );
      })}
    </svg>
  );
}
