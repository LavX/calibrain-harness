"use client";
// STRATEGIES: the live strategies that trade on the Bybit DEMO account. A separate runner (a systemd user unit)
// does the trading; this screen reads the files it writes and writes only control.json files, plus start/stop of
// the runner unit. The data lives in the store (polled every 10 s by the terminal) so the header shows the same mode.
import { useState } from "react";
import { post } from "@/lib/api";
import { loadOverview, loadPredictions, loadStrat } from "@/lib/data";
import { PredictionsPanel, ScoreboardPanel } from "@/components/Predictions";
import { useStore } from "@/lib/store";
import { age, dt, fx, isNum, px, sg, ud } from "@/lib/format";

type Pos = { symbol: string; side: string; market: string; qty: number | null; entry_price: number | null; mark_price: number | null; upnl_usdt: number | null };
type Status = {
  enabled: boolean; state: string; updated_ms: number | null; last_rebalance_ms: number | null; next_rebalance_ms: number | null; last_error: string | null;
  equity_usdt: number | null; start_equity_usdt: number | null; pnl_usdt: number | null; pnl_pct: number | null; fees_usdt: number | null; funding_usdt: number | null;
  positions: Pos[]; targets: { symbol: string; market: string; target_qty: number | null }[]; trades_today: number | null; trades_total: number | null;
};
type Trade = { t_ms: number; symbol: string; market: string; side: string; qty: number; price: number; fee_usdt: number };
type Control = { enabled: boolean; weight: number | null; set_by: "llm" | "owner" | null; owner_lock: boolean; reason: string | null; flatten_pending: boolean; reset_pending: boolean };
type Strat = {
  id: string; label: string; unit: string | null; account: string; status: Status | null; pill: string; control: Control;
  trades: Trade[]; equity_history: [number, number][]; runner_active: boolean;
};
type Decision = { t_ms: number; model: string; trigger?: string; risk_floor?: string[]; error?: string | null; weights: Record<string, number>; enabled: Record<string, boolean>; reasoning: string; applied: Record<string, boolean> | null; skipped_owner_lock: string[] | null };
type Allocator = {
  available: boolean;
  status: { state: string; model: string | null; last_decision_ms: number | null; next_decision_ms: number | null; last_error: string | null; cost_usd_today: number | null } | null;
  control: { enabled: boolean; interval_min: number | null; decide_now_pending: boolean };
  decisions: Decision[];
};
type Runner = { unit: string; load: string; active: string; since?: string };
export type StratDoc = {
  mode: "llm" | "manual" | "bot"; auto_on: boolean; error: string | null; root: string; venue: string; llm_trading_enabled: boolean;
  allocator: Allocator; runners: Runner[]; strategies: Strat[];
};

export const MODE_LABEL: Record<string, string> = { llm: "LLM DECIDES", manual: "MANUAL", bot: "PER-SYMBOL BOT" };
const MODE_HELP: Record<string, string> = {
  llm: "The LLM allocator decides which strategies trade and how much, once the AUTO-TRADE switch is on. Strategies you switch by hand stay locked to your choice until you unlock them.",
  manual: "You switch strategies per card. The LLM allocator is paused.",
  bot: "The older per-symbol LLM bot trades. The strategy runner and the allocator stay off, so only one thing trades the demo account.",
};
export const PILL: Record<string, [string, string]> = {
  running: ["RUNNING", "run"], paused: ["PAUSED", "stop"], latched: ["LATCHED", "latched"], error: ["ERROR", "latched"],
  stale: ["STALE", "latched"], runner_stopped: ["RUNNER STOPPED", "stop"], no_data: ["NO DATA", "notlive"],
};
const TEXT: Record<string, string> = {
  flatten: "Close every position of this strategy at market, then pause it.",
  reset: "Clear the latch after you have checked the demo account.",
  start: "Start the runner unit. Enabled strategies begin to place demo orders.",
  stop: "Stop the runner unit. Positions stay open on the demo account.",
  decide: "Ask the allocator for a decision right now.",
};
const usdt = (x: number | null | undefined, nd = 2) => (isNum(x) ? fx(x, nd) : "n/a");
const SEL_KEY = "ctb.strat.manual-set";

/** Strategy ids the header switch turns on in MANUAL mode (default all). Kept in this browser only. */
function manualSet(all: string[]): string[] {
  try {
    const v = JSON.parse(localStorage.getItem(SEL_KEY) || "null");
    if (Array.isArray(v)) return all.filter((i) => v.includes(i));
  } catch {}
  return all;
}
function setManualSet(ids: string[]) {
  try { localStorage.setItem(SEL_KEY, JSON.stringify(ids)); } catch {}
}

async function reloadAll() {
  await Promise.all([loadStrat().catch(() => {}), loadOverview().catch(() => {}), loadPredictions().catch(() => {})]);
}

/** Header: change the auto-trade mode (with a confirm that says what gets paused first). */
export async function switchMode(from: string, to: string) {
  if (from === to) return;
  const what = to === "bot"
    ? "The LLM allocator and every strategy are paused and the strategy runner is stopped. Positions stay open on the demo account (flatten them first if you want them closed)."
    : from === "bot"
      ? "The per-symbol bot's AUTO-TRADE is turned off first."
      : to === "manual" ? "The LLM allocator is paused. Strategies stay as they are and you switch them per card."
        : "When AUTO-TRADE is on, the allocator decides which strategies trade and how much. Strategies you switched by hand stay locked until you unlock them.";
  if (!confirm(`Switch the auto-trade mode to ${MODE_LABEL[to]}?\n\n${what}`)) return;
  try {
    await post("/api/mode", { mode: to });
  } catch (e: any) {
    alert(e.message);
  }
  await reloadAll();
}

/** Header AUTO-TRADE switch in LLM DECIDES or MANUAL mode. */
export async function headerAuto(doc: StratDoc, on: boolean) {
  const ids = doc.strategies.map((s) => s.id);
  const set = doc.mode === "manual" ? manualSet(ids) : ids;
  if (on && !confirm(doc.mode === "llm"
    ? "Turn AUTO-TRADE on? The LLM allocator will decide which strategies trade on your Bybit DEMO account and how much. The strategy runner is started if needed."
    : `Turn AUTO-TRADE on for ${set.join(", ") || "no strategy"} on your Bybit DEMO account? The strategy runner is started if needed.`)) return;
  try {
    await post("/api/strategies/auto", { enabled: on, ids: set });
  } catch (e: any) {
    alert(e.message);
  }
  await reloadAll();
}

/** A button that needs a second click: CONFIRM and CANCEL replace it and say what will happen. */
function Confirmed({ id, label, cn, disabled, onRun, open, setOpen, busy }: {
  id: string; label: string; cn?: string; disabled?: boolean; onRun: () => void; open: string | null; setOpen: (v: string | null) => void; busy: boolean;
}) {
  if (open === id)
    return (
      <span className="confirm">
        <button className="kbtn amber" disabled={busy} onClick={onRun}>CONFIRM {label}</button>
        <button className="kbtn" disabled={busy} onClick={() => setOpen(null)}>CANCEL</button>
        <span className="muted">{TEXT[id.split(":")[0]]}</span>
      </span>
    );
  return <button className={`kbtn ${cn || ""}`} disabled={busy || !!open || disabled} onClick={() => setOpen(id)}>{label}</button>;
}

function Spark({ h }: { h: [number, number][] }) {
  if (h.length < 2) return <span className="muted">equity chart starts after a few samples</span>;
  const W = 260, H = 36, v = h.map((p) => p[1]), lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
  const pts = h.map((p, i) => `${(i / (h.length - 1)) * W},${H - 3 - ((p[1] - lo) / span) * (H - 6)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H} className="spark" role="img" aria-label="equity since the dashboard started">
      <polyline points={pts} fill="none" stroke={v[v.length - 1] >= v[0] ? "var(--up)" : "var(--down)"} strokeWidth={1.5} />
    </svg>
  );
}

function Card({ s, mode, inSet, toggleSet }: { s: Strat; mode: string; inSet: boolean; toggleSet: () => void }) {
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [label, cls] = PILL[s.pill] || [s.pill.toUpperCase(), "notlive"];
  const st = s.status, c = s.control;
  const act = async (action: string) => {
    setBusy(true);
    setErr(null);
    try {
      await post(`/api/strategies/${s.id}/${action}`);
    } catch (e: any) {
      setErr(e.message);
    }
    setOpen(null);
    setBusy(false);
    await reloadAll();
  };
  const flip = (on: boolean) => {
    if (on && !confirm(`Switch ${s.label} on? It will place orders on your Bybit DEMO account (the runner is started if needed). Your choice is locked: the LLM allocator will not change it until you unlock it.`)) return;
    if (!on && !confirm(`Switch ${s.label} off? It stops placing new orders and keeps its positions. Your choice is locked against the LLM allocator until you unlock it.`)) return;
    act(on ? "enable" : "pause");
  };
  const btn = (action: string, text: string, cn?: string, disabled?: boolean) => (
    <Confirmed id={action} label={text} cn={cn} disabled={disabled} open={open} setOpen={setOpen} busy={busy} onRun={() => act(action)} />
  );
  const now = Date.now() / 1000;
  return (
    <section className="pnl strat-card">
      <div className="ph"><span className="code">{s.id.toUpperCase()}</span>{s.label}<span className={`spill ${cls}`}>{label}</span>
        <span className="ph-r">{s.unit || "no unit"}{s.account ? ` · account ${s.account}` : ""}</span></div>
      <div className="pb">
        <div className="strat-actions">
          <label className="arm" data-tip="Auto trade: on = the runner places this strategy's orders, off = it stops placing new orders and keeps its positions. Flipping it by hand locks it against the LLM allocator.">
            <input type="checkbox" checked={c.enabled} disabled={busy || mode === "bot"} onChange={(e) => flip(e.target.checked)} />
            <span className="arm-box" /><span>AUTO TRADE</span>
          </label>
          {btn("flatten", "FLATTEN", "red", !st?.positions.length)}
          {btn("reset", "RESET LATCH", "", s.pill !== "latched")}
          {mode === "manual" && (
            <label className="muted" data-tip="Include this strategy when the header AUTO-TRADE switch is turned on in MANUAL mode (kept in this browser)">
              <input type="checkbox" checked={inSet} onChange={toggleSet} /> in header set
            </label>
          )}
          {(c.flatten_pending || c.reset_pending) && (
            <span className="muted">waiting for the runner: {[c.flatten_pending && "flatten", c.reset_pending && "reset latch"].filter(Boolean).join(", ")}</span>
          )}
        </div>
        <div className="strat-weight">
          <span className="muted">WEIGHT</span> <b className="w">{isNum(c.weight) ? fx(c.weight, 2) : "n/a"}</b>
          <span className="muted"> · set by</span> <b className="w">{c.set_by === "llm" ? "LLM" : c.set_by === "owner" ? "owner" : "n/a"}</b>
          {c.owner_lock && (
            <>
              <span className="tag amb">OWNER LOCK</span>
              <button className="kbtn" disabled={busy} onClick={() => act("unlock")} data-tip="Hand this strategy back to the LLM allocator at its next decision">UNLOCK TO LLM</button>
            </>
          )}
          {c.reason && <span className="strat-reason"><span className="muted">reason</span> {c.reason}</span>}
        </div>
        {err && <div className="alert crit"><b>REFUSED</b> {err}</div>}
        {s.pill === "latched" && <div className="alert crit"><b>LATCHED</b> {st?.last_error || "read the runner journal"}. Check the demo account, then reset the latch.</div>}
        {s.pill === "stale" && <div className="alert crit"><b>STALE</b> status.json has not been updated for {age(now - (st?.updated_ms || 0) / 1000)} while the runner unit is active.</div>}
        {s.pill === "no_data" && <div className="empty">No status.json yet for this strategy.</div>}
        {st && (
          <>
            <div className="strip strat-strip">
              <div className="cell"><div className="l">Equity (USDT)</div><div className="n">{usdt(st.equity_usdt)}</div><div className="s">start {usdt(st.start_equity_usdt)}</div></div>
              <div className="cell"><div className="l">P&amp;L</div><div className={`n ${ud(st.pnl_usdt)}`}>{sg(st.pnl_usdt, 2)}</div><div className={`s ${ud(st.pnl_pct)}`}>{sg(st.pnl_pct, 2, "%")}</div></div>
              <div className="cell"><div className="l">Fees (USDT)</div><div className="n">{usdt(st.fees_usdt)}</div><div className="s">paid</div></div>
              <div className="cell"><div className="l">Funding (USDT)</div><div className={`n ${ud(st.funding_usdt)}`}>{sg(st.funding_usdt, 2)}</div><div className="s">received if positive</div></div>
              <div className="cell"><div className="l">Trades</div><div className="n">{st.trades_today ?? "n/a"}</div><div className="s">today · {st.trades_total ?? "n/a"} total</div></div>
              <div className="cell"><div className="l">Last rebalance</div><div className="n">{st.last_rebalance_ms ? dt(st.last_rebalance_ms / 1000) : "n/a"}</div><div className="s">{st.last_rebalance_ms ? `${age(now - st.last_rebalance_ms / 1000)} ago` : "none yet"}</div></div>
              <div className="cell"><div className="l">Next rebalance</div><div className="n">{st.next_rebalance_ms ? dt(st.next_rebalance_ms / 1000) : "n/a"}</div><div className="s">{st.next_rebalance_ms ? `in ${age(Math.max(0, st.next_rebalance_ms / 1000 - now))}` : "not scheduled"}</div></div>
            </div>
            <div className="strat-spark"><span className="muted">EQUITY since the dashboard started</span> <Spark h={s.equity_history} /></div>
            {st.last_error && s.pill !== "latched" && <div className="strat-err"><span className="amb">LAST ERROR</span> {st.last_error}</div>}
            <div className="ph sub"><span className="code">POS</span>Positions<span className="ph-r">{st.positions.length} open</span></div>
            {st.positions.length ? (
              <table className="t">
                <thead><tr><th>Symbol</th><th>Market</th><th>Side</th><th className="r">Qty</th><th className="r">Entry</th><th className="r">Mark</th><th className="r">Unrealised P&amp;L</th></tr></thead>
                <tbody>{st.positions.map((p, i) => (
                  <tr key={`${p.symbol}${p.market}${i}`}><td className="sym">{p.symbol}</td><td>{p.market}</td>
                    <td><span className={`tag ${["long", "buy"].includes(p.side.toLowerCase()) ? "long" : "short"}`}>{p.side.toUpperCase()}</span></td>
                    <td className="r">{isNum(p.qty) ? fx(p.qty, 4) : "n/a"}</td><td className="r">{isNum(p.entry_price) ? px(p.entry_price) : "n/a"}</td>
                    <td className="r">{isNum(p.mark_price) ? px(p.mark_price) : "n/a"}</td><td className={`r ${ud(p.upnl_usdt)}`}>{sg(p.upnl_usdt, 2)}</td></tr>
                ))}</tbody>
              </table>
            ) : <div className="empty">No open positions</div>}
            {st.targets.length > 0 && (
              <>
                <div className="ph sub"><span className="code">TGT</span>Targets at the next rebalance</div>
                <table className="t">
                  <thead><tr><th>Symbol</th><th>Market</th><th className="r">Target qty</th></tr></thead>
                  <tbody>{st.targets.map((t, i) => <tr key={`${t.symbol}${t.market}${i}`}><td className="sym">{t.symbol}</td><td>{t.market}</td><td className="r">{isNum(t.target_qty) ? fx(t.target_qty, 4) : "n/a"}</td></tr>)}</tbody>
                </table>
              </>
            )}
          </>
        )}
        <div className="ph sub"><span className="code">TRD</span>Recent trades<span className="ph-r">last {s.trades.length}</span></div>
        {s.trades.length ? (
          <table className="t">
            <thead><tr><th>Time</th><th>Symbol</th><th>Market</th><th>Side</th><th className="r">Qty</th><th className="r">Price</th><th className="r">Fee</th></tr></thead>
            <tbody>{s.trades.map((t, i) => (
              <tr key={`${t.t_ms}${i}`}><td>{dt(t.t_ms / 1000)}</td><td className="sym">{t.symbol}</td><td>{t.market}</td>
                <td><span className={`tag ${["buy", "long"].includes(String(t.side).toLowerCase()) ? "long" : "short"}`}>{String(t.side).toUpperCase()}</span></td>
                <td className="r">{fx(t.qty, 4)}</td><td className="r">{px(t.price)}</td><td className="r">{fx(t.fee_usdt, 4)}</td></tr>
            ))}</tbody>
          </table>
        ) : <div className="empty">No trades yet</div>}
      </div>
    </section>
  );
}

function AllocatorPanel({ doc }: { doc: StratDoc }) {
  const a = doc.allocator, st = a.status;
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const now = Date.now() / 1000;
  const run = async (path: string, body: object = {}) => {
    setBusy(true);
    setErr(null);
    try {
      await post(`/api/allocator/${path}`, body);
    } catch (e: any) {
      setErr(e.message);
    }
    setOpen(null);
    setBusy(false);
    await reloadAll();
  };
  const setInterval_ = () => {
    const r = prompt("Decision interval in minutes (1 to 1440)", String(a.control.interval_min ?? 60));
    if (r != null) run("interval", { interval_min: Number(r) });
  };
  const live = doc.mode === "llm";
  const pillTxt = !a.available ? "NOT INSTALLED" : !live ? "PAUSED (MANUAL)" : !a.control.enabled ? "OFF" : (st?.state || "no status").toUpperCase();
  const pillCls = !a.available || !live || !a.control.enabled ? "notlive" : st?.state === "error" ? "latched" : "run";
  return (
    <section className="pnl strat-alloc">
      <div className="ph"><span className="code">ALLOC</span>LLM allocator<span className={`spill ${pillCls}`}>{pillTxt}</span>
        <span className="ph-r">{st?.model || "model n/a"}</span></div>
      <div className="pb">
        {!a.available && <div className="muted">ctb.allocator is not part of this build, so the LLM cannot decide yet. Use MANUAL.</div>}
        <div className="strip strat-strip">
          <div className="cell"><div className="l">Last decision</div><div className="n">{st?.last_decision_ms ? dt(st.last_decision_ms / 1000) : "n/a"}</div><div className="s">{st?.last_decision_ms ? `${age(now - st.last_decision_ms / 1000)} ago` : "none yet"}</div></div>
          <div className="cell"><div className="l">Next decision</div><div className="n">{st?.next_decision_ms && live && a.control.enabled ? dt(st.next_decision_ms / 1000) : "n/a"}</div><div className="s">{st?.next_decision_ms && live && a.control.enabled ? `in ${age(Math.max(0, st.next_decision_ms / 1000 - now))}` : "not scheduled"}</div></div>
          <div className="cell"><div className="l">Interval</div><div className="n">{a.control.interval_min ? `${a.control.interval_min} min` : "n/a"}</div><div className="s"><a href="#" onClick={(e) => { e.preventDefault(); if (live && a.available) setInterval_(); }}>change</a></div></div>
          <div className="cell"><div className="l">Cost today</div><div className="n">{isNum(st?.cost_usd_today) ? `$${fx(st?.cost_usd_today, 4)}` : "n/a"}</div><div className="s">model calls, USD</div></div>
        </div>
        <div className="strat-actions">
          <Confirmed id="decide" label="DECIDE NOW" disabled={!live || !a.available || !a.control.enabled || a.control.decide_now_pending}
            open={open} setOpen={setOpen} busy={busy} onRun={() => run("decide_now")} />
          {a.control.decide_now_pending && <span className="muted">waiting for the allocator</span>}
        </div>
        {err && <div className="alert crit"><b>REFUSED</b> {err}</div>}
        {st?.last_error && <div className="strat-err"><span className="amb">LAST ERROR</span> {st.last_error}</div>}
        <div className="ph sub"><span className="code">FEED</span>Decisions<span className="ph-r">last {a.decisions.length}</span></div>
        {a.decisions.length ? (
          <table className="t">
            <thead><tr><th>Time</th><th>Weights</th><th>Reasoning</th></tr></thead>
            <tbody>{a.decisions.map((d) => (
              <tr key={d.t_ms}><td>{dt(d.t_ms / 1000)}</td>
                <td className="wrap">{Object.entries(d.weights || {}).map(([id, w]) => (
                  <div key={id}><span className="amb">{id}</span> {fx(w, 2)}{d.enabled?.[id] === false ? " (off)" : ""}{d.skipped_owner_lock?.includes(id) ? " (owner lock, not applied)" : ""}</div>
                ))}</td>
                <td className="wrap">{d.error && <div className="dn">failed: {d.error}</div>}{d.risk_floor?.length ? <div className="amb">risk floor: {d.risk_floor.join(", ")}</div> : null}{d.reasoning}<span className="muted"> {d.trigger ? `(${d.trigger})` : ""}</span></td></tr>
            ))}</tbody>
          </table>
        ) : <div className="empty">No decisions yet</div>}
      </div>
    </section>
  );
}

function RunnerBar({ doc }: { doc: StratDoc }) {
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const run = async (unit: string, action: string) => {
    setBusy(true);
    setErr(null);
    try {
      await post(`/api/runner/${action}`, { unit });
    } catch (e: any) {
      setErr(e.message);
    }
    setOpen(null);
    setBusy(false);
    await reloadAll();
  };
  if (!doc.runners.length) return <div className="alert"><b>NO RUNNER KNOWN</b> strategies.json does not exist yet under {doc.root}. Set CTB_LIVE_UNITS to name the runner unit so it can be started from here.</div>;
  return (
    <div className="strat-runner">
      {doc.runners.map((r) => {
        const active = ["active", "activating", "reloading", "deactivating"].includes(r.active);
        return (
          <div key={r.unit} className="strat-runner-row">
            <span className="code">RUNNER</span><b>{r.unit}</b>
            <span className={`spill ${active ? "run" : "stop"}`}>{r.load === "not-found" ? "NOT INSTALLED" : active ? "RUNNING" : String(r.active || "unknown").toUpperCase()}</span>
            {r.since && active && <span className="muted">since {r.since}</span>}
            <Confirmed id={`start:${r.unit}`} label="START" disabled={active || r.load === "not-found"} open={open} setOpen={setOpen} busy={busy} onRun={() => run(r.unit, "start")} />
            <Confirmed id={`stop:${r.unit}`} label="STOP" cn="red" disabled={!active} open={open} setOpen={setOpen} busy={busy} onRun={() => run(r.unit, "stop")} />
          </div>
        );
      })}
      {err && <div className="alert crit"><b>REFUSED</b> {err}</div>}
    </div>
  );
}

export function StrategiesView() {
  const doc = useStore((s) => s.strat) as StratDoc | null;
  const [, bump] = useState(0);
  if (!doc) return <main className="view on"><div className="empty">Loading</div></main>;
  const ids = doc.strategies.map((s) => s.id);
  const set = manualSet(ids);
  return (
    <main className="view on">
      <div className="toolbar">
        <span className="code">STRATEGIES</span><b>Live demo strategies</b>
        <span className="muted">mode <b className="amb">{MODE_LABEL[doc.mode]}</b> · AUTO-TRADE <b className={doc.auto_on ? "up" : "w"}>{doc.auto_on ? "ON" : "OFF"}</b> · venue <b className="w">{doc.venue.toUpperCase()}</b> · orders go to Bybit Demo Trading only</span>
        <button className="kbtn" onClick={() => reloadAll()}>REFRESH</button>
      </div>
      <div className="muted strat-help">{MODE_HELP[doc.mode]} Change the mode in the header.</div>
      {doc.error && <div className="alert crit"><b>CONFIG</b> {doc.error}</div>}
      <RunnerBar doc={doc} />
      {(doc.allocator.available || doc.mode === "llm") && <AllocatorPanel doc={doc} />}
      <PredictionsPanel />
      <ScoreboardPanel />
      <div className="strat-grid">
        {doc.strategies.map((s) => (
          <Card key={s.id} s={s} mode={doc.mode} inSet={set.includes(s.id)}
            toggleSet={() => { setManualSet(set.includes(s.id) ? set.filter((i) => i !== s.id) : [...set, s.id]); bump((n) => n + 1); }} />
        ))}
      </div>
    </main>
  );
}
