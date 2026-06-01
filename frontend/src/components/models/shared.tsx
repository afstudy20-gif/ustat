// Shared presentational helpers for the model result views.

export function adjustP(p: number, beta: number, nullHyp: string): number {
  if (nullHyp === "leq") return beta > 0 ? Math.min(p / 2, 1) : Math.min(1 - p / 2, 1);
  if (nullHyp === "geq") return beta < 0 ? Math.min(p / 2, 1) : Math.min(1 - p / 2, 1);
  return p; // "eq" = two-tailed default
}

// ── Mini bell-curve (sampling distribution of the estimator) ─────────────────
export function MiniNormalSVG({ beta, se, p }: { beta: number; se: number; p: number }) {
  if (!isFinite(beta) || !isFinite(se) || se <= 0)
    return <span className="text-amber-400 text-[11px]">⚠</span>;
  const W = 64, H = 24, span = 3.8 * se;
  const lo = beta - span, hi = beta + span;
  const N  = 60;
  const toSX = (x: number) => ((x - lo) / (hi - lo)) * W;
  const toSY = (y: number) => H - 2 - y * (H - 4);
  const pts = Array.from({ length: N + 1 }, (_, i) => {
    const x = lo + (hi - lo) * i / N;
    return [x, Math.exp(-0.5 * ((x - beta) / se) ** 2)] as [number, number];
  });
  const curve = pts.map(([x, y]) => `${toSX(x).toFixed(1)},${toSY(y).toFixed(1)}`).join(" ");
  const fill  = [`0,${H}`, ...pts.map(([x, y]) => `${toSX(x).toFixed(1)},${toSY(y).toFixed(1)}`), `${W},${H}`].join(" ");
  const zx    = toSX(0);
  const color = p < 0.001 ? "#3730a3" : p < 0.01 ? "#4338ca" : p < 0.05 ? "#6366f1" : "#9ca3af";
  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      <polygon points={fill}  fill={`${color}${p < 0.05 ? "22" : "0e"}`} />
      <polyline points={curve} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
      {zx >= 0 && zx <= W && (
        <line x1={zx.toFixed(1)} y1="1" x2={zx.toFixed(1)} y2={H}
          stroke="#9ca3af" strokeWidth="0.8" strokeDasharray="2,2" />
      )}
    </svg>
  );
}

// ── Significance bar ──────────────────────────────────────────────────────────
export function SigBar({ p }: { p: number }) {
  const pct   = p < 0.001 ? 100 : p < 0.01 ? 80 : p < 0.05 ? 55 : p < 0.1 ? 22 : 7;
  const color = p < 0.001 ? "#3730a3" : p < 0.01 ? "#4338ca" : p < 0.05 ? "#6366f1" : "#d1d5db";
  return (
    <div style={{ width: 56, height: 10, backgroundColor: "#f3f4f6", borderRadius: 3, overflow: "hidden" }}>
      <div style={{ width: `${pct}%`, height: "100%", backgroundColor: color }} />
    </div>
  );
}
