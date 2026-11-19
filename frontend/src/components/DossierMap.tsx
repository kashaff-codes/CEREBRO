import React, { useMemo } from 'react';
import { LAND_DOTS } from './landDots';

/**
 * The dossier map: where CEREBRO's evidence comes from.
 *
 * A dot-matrix world (Natural Earth land, not invented shapes), a graticule, a
 * slow brass sweep, and the newsrooms and agencies whose reporting the verifier
 * weighs most, each pinged as the sweep passes over it. Thin arcs run from each
 * source to the desk. Nothing on it is fake telemetry: the markers are real
 * places with their real coordinates, and the map says what they are.
 *
 * Equirectangular, cropped to 58S–78N. One <path> for all land dots, so it is
 * cheap to render; motion is pure CSS and stops under prefers-reduced-motion.
 */

const W = 1000;
const TOP = 78;
const BOTTOM = -58;
const H = Math.round(((TOP - BOTTOM) / 360) * W * 1.38); // a touch taller than true equirect
const x = (lon: number) => ((lon + 180) / 360) * W;
const y = (lat: number) => ((TOP - lat) / (TOP - BOTTOM)) * H;

const SWEEP_S = 16;

/**
 * Label placement is set per source, by hand: London, Paris, Berlin and Geneva
 * sit within a few pixels of each other, as do New York and Washington, and an
 * automatic "label to the right" stacks them on top of each other.
 * `lx`/`ly` offset the label from the marker; `end` right-aligns it.
 */
export const SOURCES: {
  name: string;
  city: string;
  lat: number;
  lon: number;
  lx: number;
  ly: number;
  end?: boolean;
}[] = [
  { name: 'Reuters · BBC', city: 'London', lat: 51.51, lon: -0.13, lx: -10, ly: -16, end: true },
  { name: 'AFP', city: 'Paris', lat: 48.86, lon: 2.35, lx: -10, ly: 14, end: true },
  { name: 'Nature', city: 'Berlin', lat: 52.52, lon: 13.4, lx: 10, ly: -16 },
  { name: 'WHO', city: 'Geneva', lat: 46.2, lon: 6.14, lx: 10, ly: 16 },
  { name: 'Associated Press', city: 'New York', lat: 40.71, lon: -74.01, lx: 10, ly: -14 },
  { name: 'NASA · CDC', city: 'Washington', lat: 38.9, lon: -77.04, lx: -10, ly: 16, end: true },
  {
    name: 'Wikipedia',
    city: 'San Francisco',
    lat: 37.78,
    lon: -122.42,
    lx: -10,
    ly: -6,
    end: true,
  },
  { name: 'Al Jazeera', city: 'Doha', lat: 25.29, lon: 51.53, lx: -10, ly: 18, end: true },
  { name: 'Dawn', city: 'Karachi', lat: 24.86, lon: 67.0, lx: 22, ly: 20 },
  { name: 'NHK', city: 'Tokyo', lat: 35.68, lon: 139.69, lx: -10, ly: -14, end: true },
  { name: 'ABC', city: 'Sydney', lat: -33.87, lon: 151.21, lx: -10, ly: 18, end: true },
];
const DESK = { lat: 24.86, lon: 67.0 }; // the verification desk (Karachi)

const fmt = (v: number, pos: string, neg: string) =>
  `${Math.abs(v).toFixed(2).padStart(5, '0')}${v >= 0 ? pos : neg}`;

export const DossierMap: React.FC<{
  className?: string;
  labels?: boolean;
  arcs?: boolean;
  opacity?: number;
  /** 'slice' fills a background (crops); 'meet' shows the whole map (panels). */
  fit?: 'slice' | 'meet';
  /** Source markers and the desk reticle; off for a pure background. */
  markers?: boolean;
}> = ({
  className = '',
  labels = true,
  arcs = true,
  opacity = 1,
  fit = 'slice',
  markers = true,
}) => {
  const land = useMemo(() => {
    let d = '';
    for (const p of LAND_DOTS.split(';')) {
      const [lon, lat] = p.split(',').map(Number);
      d += `M${x(lon).toFixed(1)} ${y(lat).toFixed(1)}h0`;
    }
    return d;
  }, []);

  const desk = { x: x(DESK.lon), y: y(DESK.lat) };

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={className}
      style={{ opacity }}
      role="img"
      aria-label="World map of the news agencies and institutions whose reporting CEREBRO weighs most"
      preserveAspectRatio={`xMidYMid ${fit}`}
    >
      <defs>
        <linearGradient id="sweep" x1="0" x2="1">
          <stop offset="0" stopColor="#C9A45C" stopOpacity="0" />
          <stop offset="0.85" stopColor="#C9A45C" stopOpacity="0.035" />
          <stop offset="1" stopColor="#C9A45C" stopOpacity="0.16" />
        </linearGradient>
        <radialGradient id="vignette" cx="0.5" cy="0.5" r="0.75">
          <stop offset="0.55" stopColor="#0B0B0D" stopOpacity="0" />
          <stop offset="1" stopColor="#0B0B0D" stopOpacity="0.9" />
        </radialGradient>
        <style>{`
          .dm-sweep { animation: dm-sweep ${SWEEP_S}s linear infinite; }
          @keyframes dm-sweep { from { transform: translateX(-140px); } to { transform: translateX(${W}px); } }
          .dm-ping { transform-box: fill-box; transform-origin: center; animation: dm-ping ${SWEEP_S}s linear infinite; opacity: 0; }
          @keyframes dm-ping { 0% { opacity: .9; transform: scale(.4); } 7% { opacity: 0; transform: scale(3.2); } 100% { opacity: 0; } }
          .dm-arc { stroke-dasharray: 3 5; animation: dm-flow 9s linear infinite; }
          @keyframes dm-flow { to { stroke-dashoffset: -80; } }
          @media (prefers-reduced-motion: reduce) { .dm-sweep, .dm-ping, .dm-arc { animation: none; } }
        `}</style>
      </defs>

      {/* graticule every 30 degrees; the equator a little firmer */}
      <g stroke="rgba(236,230,217,0.06)" strokeWidth="0.6">
        {[-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150].map((lon) => (
          <line key={`m${lon}`} x1={x(lon)} x2={x(lon)} y1={0} y2={H} />
        ))}
        {[60, 30, -30].map((lat) => (
          <line key={`p${lat}`} x1={0} x2={W} y1={y(lat)} y2={y(lat)} />
        ))}
      </g>
      <line
        x1={0}
        x2={W}
        y1={y(0)}
        y2={y(0)}
        stroke="rgba(201,164,92,0.16)"
        strokeDasharray="2 6"
      />

      {/* land */}
      <path
        d={land}
        stroke="rgba(236,230,217,0.20)"
        strokeWidth="2.6"
        strokeLinecap="round"
        fill="none"
      />

      {/* evidence flowing to the desk */}
      {arcs &&
        SOURCES.filter((s) => s.city !== 'Karachi').map((s) => {
          const sx = x(s.lon);
          const sy = y(s.lat);
          const mx = (sx + desk.x) / 2;
          const my = Math.min(sy, desk.y) - Math.abs(sx - desk.x) * 0.18;
          return (
            <path
              key={`a-${s.city}`}
              d={`M${sx} ${sy} Q${mx} ${my} ${desk.x} ${desk.y}`}
              fill="none"
              stroke="rgba(201,164,92,0.22)"
              strokeWidth="0.8"
              className="dm-arc"
            />
          );
        })}

      {markers && (
        <>
          {/* sources */}
          {SOURCES.map((s) => {
            const sx = x(s.lon);
            const sy = y(s.lat);
            const delay = `${((sx + 140) / (W + 140)) * SWEEP_S}s`;
            return (
              <g key={s.city}>
                <circle
                  cx={sx}
                  cy={sy}
                  r="5"
                  fill="none"
                  stroke="#C9A45C"
                  className="dm-ping"
                  style={{ animationDelay: delay }}
                />
                <circle cx={sx} cy={sy} r="2.2" fill="#C9A45C" />
                <circle
                  cx={sx}
                  cy={sy}
                  r="5.5"
                  fill="none"
                  stroke="rgba(201,164,92,0.45)"
                  strokeWidth="0.7"
                />
                {labels && (
                  <text
                    x={sx + s.lx}
                    y={sy + s.ly}
                    textAnchor={s.end ? 'end' : 'start'}
                    fill="rgba(236,230,217,0.6)"
                    fontSize="7.5"
                    fontFamily="'IBM Plex Mono', monospace"
                    letterSpacing="0.6"
                  >
                    {s.city.toUpperCase()}
                    <tspan x={sx + s.lx} dy="9" fill="rgba(201,164,92,0.75)">
                      {s.name}
                    </tspan>
                    <tspan x={sx + s.lx} dy="9" fill="rgba(236,230,217,0.35)">
                      {fmt(s.lat, 'N', 'S')} {fmt(s.lon, 'E', 'W')}
                    </tspan>
                  </text>
                )}
              </g>
            );
          })}

          {/* the desk: a reticle */}
          <g stroke="#ECE6D9" strokeWidth="0.8" fill="none" opacity="0.75">
            <circle cx={desk.x} cy={desk.y} r="11" />
            <line x1={desk.x - 18} x2={desk.x - 6} y1={desk.y} y2={desk.y} />
            <line x1={desk.x + 6} x2={desk.x + 18} y1={desk.y} y2={desk.y} />
            <line x1={desk.x} x2={desk.x} y1={desk.y - 18} y2={desk.y - 6} />
            <line x1={desk.x} x2={desk.x} y1={desk.y + 6} y2={desk.y + 18} />
          </g>
        </>
      )}

      {/* the sweep */}
      <g className="dm-sweep">
        <rect x={0} y={0} width={140} height={H} fill="url(#sweep)" />
        <line x1={140} x2={140} y1={0} y2={H} stroke="rgba(201,164,92,0.38)" strokeWidth="0.7" />
      </g>

      <rect x={0} y={0} width={W} height={H} fill="url(#vignette)" pointerEvents="none" />

      {labels && (
        <g
          fontFamily="'IBM Plex Mono', monospace"
          fontSize="8"
          fill="rgba(236,230,217,0.35)"
          letterSpacing="1.2"
        >
          <text x={14} y={18}>
            PROJECTION EQUIRECTANGULAR · GRATICULE 30°
          </text>
          <text x={W - 14} y={H - 12} textAnchor="end">
            {SOURCES.length} SOURCES · DESK {fmt(DESK.lat, 'N', 'S')} {fmt(DESK.lon, 'E', 'W')}
          </text>
        </g>
      )}
    </svg>
  );
};
