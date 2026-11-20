"use client";
// STRATEGIES screen, PREDICTIONS and SCOREBOARD panels. Read-only: every number comes from GET /api/predictions, which reads the
// prediction layer's files (regime, forecast, scores, returns) and the allocator's newest weights line. A missing file shows
// "not computed yet", never a zero. The panels sit under the allocator panel and above the strategy cards.
import { useStore } from "@/lib/store";
import { age, fx, isNum, NA, sg, ud } from "@/lib/format";

type Feature = { key: string; meaning: string; unit: string; value: number | null };
type P7 = { state: string; p: number | null; help: string };
type Row = {
  id: string; label: string; core: number | null; shadow_tau: number | null; shadow_w: number | null; tau: number | null; sr: number | null;
  mu_ann_pct: Record<string, number | null>; sigma_ann_pct: number | null; p_decider: number | null; live_weight: number | null;
  reason: string | null; clipped: string[];
};
type Score = {
  id: string; loss: Record<string, number | null>; weight: Record<string, number | null>; brier: number | null; brier_base: number | null;
  n_brier: number | null; n_scored: number | null; decider_no_skill: boolean;
};
type Pnl = {
  id: string; label: string; live_pnl_pct: number | null; live_pnl_usdt: number | null; shadow_pct: number | null; shadow_weighted_pct: number | null;
  weight: number | null; days: number; last_day: string | null;
};
export type PredDoc = {
  generated_at: number; go_live: string; error?: string; missing: string[];
  regime: { date: string | null; state: string; state_help: string; features: Feature[]; p7: P7[] } | null;
  forecast: { date: string | null; made_ms: number | null; age_h: number | null; stale: boolean; kappa: number | null } | null;
  allocator: { t_ms: number | null; weights_mode: string | null; kappa: number | null; t_eff: number | null; model: string | null } | null;
  strategies: Row[];
  scoreboard: {
    date: string | null; per_strategy: Score[];
    tilt: { n_days: number; days_needed: number; kappa: number | null; skill_90_ann_pct: number | null; skill_180_ann_pct: number | null; live: boolean; line: string };
  } | null;
  pnl: Pnl[];
  history: { since: string; days: string[]; core_pct: number[]; tilted_pct: number[] } | null;
};

const KS = ["base", "regime", "ridge", "structural", "decider"] as const;
const NOT_YET = "not computed yet";
const pct = (x: number | null | undefined, nd = 1) => (isNum(x) ? sg(x, nd, "%") : NA);
const num = (x: number | null | undefined, nd = 2) => (isNum(x) ? fx(x, nd) : NA);
const SIGNED = new Set(["btc_trend", "vol_ratio", "funding_level"]); // the others cannot be negative
const featValue = (f: Feature) => {
  if (!isNum(f.value)) return NA;
  const v = Number(f.value) * (f.unit === "pct" ? 100 : 1), suf = f.unit === "pct" ? "%" : "";
  return SIGNED.has(f.key) ? sg(v, f.unit === "pct" ? 1 : 2, suf) : fx(v, f.unit === "pct" ? 1 : 2) + suf;
};

function NotYet({ what }: { what: string }) {
  return <div className="empty">{what}: {NOT_YET}</div>;
}

function RegimeBlock({ doc }: { doc: PredDoc }) {
  const r = doc.regime;
  if (!r) return <NotYet what="Regime (regime/latest.json)" />;
  return (
    <div className="pred-regime">
      <div className="pred-col">
        <div className="muted">REGIME NOW{r.date ? ` (${r.date})` : ""}</div>
        <div className="pred-state"><b className="amb">{r.state.toUpperCase()}</b></div>
        <div className="muted">{r.state_help}</div>
        <div className="muted pred-gap">NEXT 7 DAYS, probability of each state</div>
        <div className="pbars" role="list" aria-label="7-day regime probabilities">
          {r.p7.map((p) => (
            <div key={p.state} className={`pbar ${p.state === r.state ? "now" : ""}`} role="listitem" data-tip={p.help}>
              <span className="pbar-l">{p.state}</span>
              <span className="pbar-t"><span className="pbar-f" style={{ width: `${isNum(p.p) ? Math.round(Number(p.p) * 100) : 0}%` }} /></span>
              <span className="pbar-v">{isNum(p.p) ? `${(Number(p.p) * 100).toFixed(0)}%` : NA}</span>
            </div>
          ))}
        </div>
      </div>
      <table className="t pred-feat">
        <thead><tr><th>Feature</th><th className="r">Value</th><th>Meaning</th></tr></thead>
        <tbody>{r.features.map((f) => (
          <tr key={f.key}><td className="sym">{f.key}</td><td className="r">{featValue(f)}</td><td className="wrap muted">{f.meaning}</td></tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function ForecastTable({ doc }: { doc: PredDoc }) {
  if (!doc.strategies.length) return <NotYet what="Forecasts (forecast/latest.json and the allocator's weights line)" />;
  return (
    <div className="pred-scroll">
      <table className="t">
        <thead>
          <tr><th rowSpan={2}>Strategy</th><th className="r" rowSpan={2}>Core w</th><th className="r" colSpan={2}>Shadow tilt</th>
            <th className="r" colSpan={6}>Expected net return, %/yr, by forecaster</th>
            <th className="r" rowSpan={2}>Risk %/yr</th><th className="r" rowSpan={2}>Decider p</th><th className="r" rowSpan={2}>Live w</th></tr>
          <tr><th className="r">tau</th><th className="r">fcst SR</th>
            {KS.map((k) => <th key={k} className="r">{k}</th>)}<th className="r">combined</th></tr>
        </thead>
        <tbody>
          {doc.strategies.flatMap((s) => [
            <tr key={s.id}>
              <td className="sym" data-tip={s.label}>{s.id}</td>
              <td className="r">{num(s.core)}</td>
              <td className={`r ${ud(s.shadow_tau)}`}>{isNum(s.shadow_tau) ? sg(Number(s.shadow_tau) * 100, 0, "%") : NA}</td>
              <td className={`r ${ud(s.sr)}`}>{isNum(s.sr) ? sg(s.sr, 2) : NA}</td>
              {KS.map((k) => <td key={k} className={`r ${ud(s.mu_ann_pct[k])}`}>{k in s.mu_ann_pct ? pct(s.mu_ann_pct[k]) : NA}</td>)}
              <td className={`r ${ud(s.mu_ann_pct.combined)}`}><b>{pct(s.mu_ann_pct.combined)}</b></td>
              <td className="r">{isNum(s.sigma_ann_pct) ? `${fx(s.sigma_ann_pct, 1)}%` : NA}</td>
              <td className="r">{num(s.p_decider, 2)}</td>
              <td className="r"><b className="w">{num(s.live_weight)}</b></td>
            </tr>,
            <tr key={`${s.id}-why`} className="pred-why"><td colSpan={12} className="wrap muted">{s.reason || "no allocator line yet for this strategy"}
              {s.clipped.length ? ` | Sharpe-clipped: ${s.clipped.join(", ")}` : ""}</td></tr>,
          ])}
        </tbody>
      </table>
    </div>
  );
}

export function PredictionsPanel() {
  const doc = useStore((s) => s.pred) as PredDoc | null;
  if (!doc) return <section className="pnl strat-alloc"><div className="ph"><span className="code">PRED</span>Predictions</div><div className="pb"><div className="empty">Loading</div></div></section>;
  const f = doc.forecast;
  return (
    <section className="pnl strat-alloc">
      <div className="ph"><span className="code">PRED</span>Predictions
        {f?.stale && <span className="spill latched">FORECAST STALE</span>}
        <span className="ph-r">{f?.date ? `forecast of ${f.date}${isNum(f.age_h) ? `, made ${age(Number(f.age_h) * 3600)} ago` : ""}` : "no forecast yet"}
          {doc.allocator?.t_ms ? ` · weights line ${doc.allocator.weights_mode || ""}` : ""}</span></div>
      <div className="pb">
        {doc.error && <div className="alert crit"><b>PREDICTIONS</b> {doc.error}</div>}
        {f?.stale && <div className="alert crit"><b>STALE</b> the forecast is older than 36 h, so the allocator holds the weights.</div>}
        <div className="muted strat-help">
          The prediction layer runs daily. The core weight is the static inverse-volatility mix and is what trades. The shadow tilt is what the forecasts
          would add (tau, as a share of the core weight, at full strength); it only trades once it has earned it, see the scoreboard.
          Returns are net, annualised, of the strategy&apos;s book at weight 1. The decider is the LLM&apos;s vote.
        </div>
        <RegimeBlock doc={doc} />
        <div className="ph sub"><span className="code">FCST</span>Forecasts per strategy<span className="ph-r">%/yr · core and live weight in 0 to 1</span></div>
        <ForecastTable doc={doc} />
      </div>
    </section>
  );
}

function Chart({ h }: { h: NonNullable<PredDoc["history"]> }) {
  const W = 640, H = 120, P = 6;
  const all = [0, ...h.core_pct, ...h.tilted_pct];
  const lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  const n = h.days.length;
  const x = (i: number) => (n === 1 ? W / 2 : P + (i / (n - 1)) * (W - 2 * P));
  const y = (v: number) => H - P - ((v - lo) / span) * (H - 2 * P);
  const line = (vs: number[]) => vs.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const last = (vs: number[]) => vs[vs.length - 1];
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="pred-chart" role="img"
        aria-label={`Cumulative return since ${h.since}: core ${pct(last(h.core_pct), 2)}, shadow tilted ${pct(last(h.tilted_pct), 2)}`}>
        <line x1={0} x2={W} y1={y(0)} y2={y(0)} stroke="var(--line)" strokeDasharray="3 3" />
        <polyline points={line(h.core_pct)} fill="none" stroke="var(--s1)" strokeWidth={1.5} />
        <polyline points={line(h.tilted_pct)} fill="none" stroke="var(--amber)" strokeWidth={1.5} />
        {n === 1 && <><circle cx={x(0)} cy={y(h.core_pct[0])} r={3} fill="var(--s1)" /><circle cx={x(0)} cy={y(h.tilted_pct[0])} r={3} fill="var(--amber)" /></>}
      </svg>
      <div className="muted pred-legend">
        <span><i className="sw" style={{ background: "var(--s1)" }} /> core {pct(last(h.core_pct), 2)}</span>
        <span><i className="sw" style={{ background: "var(--amber)" }} /> shadow tilted {pct(last(h.tilted_pct), 2)}</span>
        <span>{h.days[0]} to {h.days[n - 1]} · {n} day{n === 1 ? "" : "s"} · scale {pct(lo, 2)} to {pct(hi, 2)}</span>
      </div>
    </div>
  );
}

export function ScoreboardPanel() {
  const doc = useStore((s) => s.pred) as PredDoc | null;
  if (!doc) return null;
  const sb = doc.scoreboard;
  const t = sb?.tilt;
  return (
    <section className="pnl strat-alloc">
      <div className="ph"><span className="code">SCORE</span>Scoreboard<span className="ph-r">{sb?.date ? `scored through ${sb.date}` : "no scores yet"}</span></div>
      <div className="pb">
        {t ? (
          <div className={`pred-tilt ${t.live ? "on" : "off"}`}>
            <b>{t.line}</b>
            <div className="pred-prog" aria-hidden="true"><span style={{ width: `${Math.min(100, Math.round((t.n_days / t.days_needed) * 100))}%` }} /></div>
            <span className="muted">
              kappa {num(t.kappa)} · tilt skill (shadow tilted minus core): 90-day {isNum(t.skill_90_ann_pct) ? `${pct(t.skill_90_ann_pct, 2)}/yr` : NA}, 180-day {isNum(t.skill_180_ann_pct) ? `${pct(t.skill_180_ann_pct, 2)}/yr` : NA}
              {!isNum(t.skill_90_ann_pct) ? " (a window has no value until it has all its days)" : ""}
            </span>
          </div>
        ) : <NotYet what="Tilt skill and scores (forecast/scores.json)" />}
        {sb && (
          <div className="pred-scroll">
            <table className="t">
              <thead>
                <tr><th rowSpan={2}>Strategy</th><th className="r" colSpan={5}>Forecast error L (lower is better) and weight W</th>
                  <th className="r" colSpan={3}>Decider Brier (lower is better)</th><th className="r" rowSpan={2}>Days scored</th></tr>
                <tr>{KS.map((k) => <th key={k} className="r">{k}</th>)}<th className="r">score</th><th className="r">base rate</th><th className="r">n</th></tr>
              </thead>
              <tbody>{sb.per_strategy.map((s) => (
                <tr key={s.id}>
                  <td className="sym">{s.id}</td>
                  {KS.map((k) => <td key={k} className="r">{k in s.loss || k in s.weight ? <>{isNum(s.loss[k]) ? fx(s.loss[k], 3) : NA} <span className="muted">/ {isNum(s.weight[k]) ? fx(s.weight[k], 2) : NA}</span></> : <span className="muted">{NA}</span>}</td>)}
                  <td className={`r ${isNum(s.brier) && isNum(s.brier_base) ? (s.brier < s.brier_base ? "up" : "dn") : ""}`}>{isNum(s.brier) ? fx(s.brier, 3) : NA}</td>
                  <td className="r">{isNum(s.brier_base) ? fx(s.brier_base, 3) : NA}</td>
                  <td className="r">{s.n_brier ?? NA}{s.decider_no_skill ? <span className="tag err" data-tip="No better than the base rate after 30 scored days: the decider's weight is held at the floor"> NO SKILL</span> : null}</td>
                  <td className="r">{s.n_scored ?? NA}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
        <div className="ph sub"><span className="code">PNL</span>Demo P&amp;L against the modelled shadow since go-live ({doc.go_live})
          <span className="ph-r">shadow = returns/&lt;id&gt;.csv, weight 1, to yesterday; live = status.json since the book started</span></div>
        {doc.pnl.length ? (
          <div className="pred-scroll">
            <table className="t">
              <thead><tr><th>Strategy</th><th className="r">Live P&amp;L</th><th className="r">Live USDT</th><th className="r">Shadow, weight 1</th>
                <th className="r">Shadow at live w</th><th className="r">Live w</th><th className="r">Shadow days</th></tr></thead>
              <tbody>{doc.pnl.map((p) => (
                <tr key={p.id}>
                  <td className="sym" data-tip={p.label}>{p.id}</td>
                  <td className={`r ${ud(p.live_pnl_pct)}`}>{pct(p.live_pnl_pct, 2)}</td>
                  <td className={`r ${ud(p.live_pnl_usdt)}`}>{sg(p.live_pnl_usdt, 2)}</td>
                  <td className={`r ${ud(p.shadow_pct)}`}>{p.days ? pct(p.shadow_pct, 2) : NOT_YET}</td>
                  <td className={`r ${ud(p.shadow_weighted_pct)}`}>{p.days ? pct(p.shadow_weighted_pct, 2) : NOT_YET}</td>
                  <td className="r">{num(p.weight)}</td>
                  <td className="r">{p.days}{p.last_day ? <span className="muted"> to {p.last_day}</span> : null}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        ) : <NotYet what="Returns (returns/<id>.csv)" />}
        <div className="ph sub"><span className="code">HIST</span>Core against shadow tilted, cumulative since go-live
          <span className="ph-r">full tilt, no rules, no costs, no decider: the series the tilt skill is judged on</span></div>
        {doc.history ? <Chart h={doc.history} /> : <div className="empty">History: {NOT_YET}. It starts after the first forecast made on or after go-live has a day of returns.</div>}
      </div>
    </section>
  );
}
