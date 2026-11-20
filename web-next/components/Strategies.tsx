"use client";
// STRATEGIES (F7): the live strategies of one venue (venue selector on top). Each venue's runner (a systemd user unit)
// does the trading; this screen reads the files it writes and writes only control.json files, plus start/stop of the
// runner unit, through /api/venues/{vid}/... The allocator venue's document is the store's `strat` (polled every 10 s by the
// terminal, the header reads it too); another venue's is polled here while it is selected.
import { useEffect, useState } from "react";
import { post } from "@/lib/api";
import { loadOverview, loadPredictions, loadStrat, loadVenues, loadVenueStrat } from "@/lib/data";
import { PredictionsPanel, ScoreboardPanel } from "@/components/Predictions";
import { setState, useStore } from "@/lib/store";
import { age, ago, dt, fx, isNum, money, NA, pf, sg, smoney, ud } from "@/lib/format";
import { PRESENCE, type VenuesDoc } from "./Venues";

type Pos = { symbol: string; side: string; market: string; qty: number | null; entry_price: number | null; mark_price: number | null; upnl_usdt: number | null };
type Status = {
  enabled: boolean; state: string; weight: number | null; set_by: "llm" | "owner" | null; reason: string | null; owner_lock: boolean; updated_ms: number | null; last_rebalance_ms: number | null; next_rebalance_ms: number | null; last_error: string | null;
  equity_usdt: number | null; start_equity_usdt: number | null; pnl_usdt: number | null; pnl_pct: number | null; fees_usdt: number | null; funding_usdt: number | null;
  positions: Pos[]; targets: { symbol: string; market: string; target_qty: number | null }[]; trades_today: number | null; trades_total: number | null;
};
type Trade = { t_ms: number; symbol: string; market: string; side: string; qty: number; price: number; fee_usdt: number };
type Control = { enabled: boolean; weight: number | null; set_by: "llm" | "owner" | null; owner_lock: boolean; reason: string | null; flatten_pending: boolean; reset_pending: boolean };
type Strat = {
  id: string; name: string; label: string; pnl_today?: number | null; pnl_today_since_ms?: number | null; pnl_today_since?: string | null; unit: string | null; account: string; status: Status | null; pill: string; control: Control;
  trades: Trade[]; equity_history: [number, number][]; runner_active: boolean;
};
type Decision = { t_ms: number; model: string; trigger?: string; risk_floor?: string[]; error?: string | null; weights: Record<string, number>; enabled: Record<string, boolean>; reasoning: string; applied: Record<string, boolean> | null; skipped_owner_lock: string[] | null; skipped_kill_pending?: string[] | null; skipped_paused?: string[] | null };
type Allocator = {
  available: boolean;
  status: { state: string; model: string | null; last_decision_ms: number | null; next_decision_ms: number | null; last_error: string | null; cost_usd_today: number | null } | null;
  control: { enabled: boolean; interval_min: number | null; decide_now_pending: boolean };
  decisions: Decision[];
};
type Runner = { unit: string; load: string; active: string; since?: string; since_s?: number | null };
export type StratDoc = {
  mode: "llm" | "manual" | "bot"; auto_on: boolean; error: string | null; root: string; venue: string; llm_trading_enabled: boolean;
  allocator: Allocator; runners: Runner[]; strategies: Strat[]; auto_set: string[] | null;
  venue_id: string; venue_label: string; ccy: string; controlled_by: "allocator" | "owner"; allocator_venue: boolean;
  dashboard_account: string | null;
};

export const MODE_LABEL: Record<string, string> = { llm: "LLM DECIDES", manual: "MANUAL", bot: "PER-SYMBOL BOT" };
const MODE_HELP: Record<string, string> = {
  llm: "The LLM allocator decides which strategies trade on this venue and how much, once the AUTO-TRADE switch is on. Strategies you switch by hand stay locked to your choice until you unlock them.",
  manual: "You switch strategies per card. The LLM allocator is paused.",
  bot: "The older per-symbol LLM bot trades this account. The strategy runner and the allocator stay off, so only one thing trades it.",
};
/** Strategy state words and classes: colour only when something is wrong (red fill: error, latched, stale; amber: unit down). */
export const PILL: Record<string, [string, string]> = {
  running: ["RUNNING", "plain"], idle: ["IDLE", "notlive"], paused: ["PAUSED", "plain"], latched: ["LATCHED", "latched"], error: ["ERROR", "latched"],
  stale: ["STALE", "latched"], runner_stopped: ["UNIT DOWN", "warn"], no_data: ["NO DATA", "notlive"],
};
const TEXT: Record<string, string> = {
  flatten: "Close every position of this strategy at market, then pause it.",
  reset: "Clear the latch after you have checked the account on the exchange.",
  start: "Start the runner unit. Enabled strategies begin to place orders.",
  stop: "Stop the runner unit. Positions stay open on the exchange.",
  decide: "Ask the allocator for a decision right now.",
};
const usdt = (x: number | null | undefined, nd = 2) => (isNum(x) ? fx(x, nd) : NA);

async function reloadAll(vid?: string) {
  await Promise.all([loadStrat().catch(() => {}), loadVenues().catch(() => {}), loadOverview().catch(() => {}), loadPredictions().catch(() => {}),
    vid ? loadVenueStrat(vid).catch(() => {}) : Promise.resolve()]);
}

/** Header: change the auto-trade mode (with a confirm that says what gets paused first). PER-SYMBOL BOT needs the word typed. */
export async function switchMode(from: string, to: string, doc?: StratDoc | null) {
  if (from === to) return;
  // what the server's set_mode("bot") really does: it pauses and stops only when the bot trades the strategies' own account
  const shares = !!doc && doc.venue === "demo" && doc.strategies.some((s) => !doc.dashboard_account || s.account === doc.dashboard_account);
  const unit = doc?.runners[0]?.unit || "the strategy runner";
  const what = to === "bot"
    ? shares
      ? `The per-symbol bot trades the strategies' Bybit demo account, so the LLM allocator and every BYBIT·DEMO strategy are paused and ${unit} is stopped first (refused while a strategy holds positions: flatten them first). Other venues are not affected.`
      : "The per-symbol bot trades another account, so nothing of the strategies changes: the LLM allocator and the BYBIT·DEMO strategies keep running as they are. Only the header mode changes. Other venues are not affected."
    : from === "bot"
      ? "The per-symbol bot's AUTO-TRADE is turned off first."
      : to === "manual" ? "The LLM allocator is paused. Strategies stay as they are and you switch them per card."
        : "When AUTO-TRADE is on, the allocator decides which BYBIT·DEMO strategies trade and how much. Strategies you switched by hand stay locked until you unlock them.";
  if (to === "bot") {
    const typed = prompt(`Switch the auto-trade mode to PER-SYMBOL BOT?\n\n${what}\n\nType BOT to confirm.`);
    if ((typed || "").trim().toUpperCase() !== "BOT") return;
  } else if (!confirm(`Switch the auto-trade mode to ${MODE_LABEL[to]}?\n\n${what}`)) return;
  try {
    await post("/api/mode", { mode: to });
  } catch (e: any) {
    alert(e.message);
  }
  await reloadAll();
}

/** The MANUAL header set: kept by the harness (every strategy until one is saved). */
function manualSet(doc: StratDoc): string[] {
  const all = doc.strategies.map((s) => s.id);
  return doc.auto_set ? all.filter((i) => doc.auto_set!.includes(i)) : all;
}

/** Header AUTO-TRADE switch in LLM DECIDES or MANUAL mode: the allocator venue only. */
export async function headerAuto(doc: StratDoc, on: boolean, venue = "BYBIT·DEMO") {
  const set = doc.mode === "manual" ? manualSet(doc) : doc.strategies.map((s) => s.id);
  const msg = on
    ? doc.mode === "llm"
      ? `Turn AUTO-TRADE on for ${venue}? The LLM allocator will decide which strategies trade there and how much. The strategy runner is started if needed. Other venues are not affected.`
      : `Turn AUTO-TRADE on for ${set.join(", ") || "no strategy"} on ${venue}? The strategy runner is started if needed. Other venues are not affected.`
    : `Turn AUTO-TRADE off for ${venue}? ${doc.mode === "llm" ? "The allocator pauses and every" : "Every"} ${venue} strategy is paused: it places no new orders. Positions stay open (use KILL to close them).`;
  const pending = on && doc.mode === "manual" ? doc.strategies.filter((s) => set.includes(s.id) && s.control.flatten_pending) : [];
  if (pending.length) {
    if (!confirm(`A kill is pending for ${pending.map((s) => s.name || s.label).join(", ")} on ${venue}. Turning AUTO-TRADE on cancels it for them: if the runner has not started closing yet, their positions will not be closed and they trade from them. Cancel the kill and turn AUTO-TRADE on?`)) return;
  } else if (!confirm(msg)) return;
  try {
    await post("/api/strategies/auto", { enabled: on, ...(pending.length ? { cancel_kill: true } : {}) });
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

function Card({ s, doc, inSet, toggleSet }: { s: Strat; doc: StratDoc; inSet: boolean; toggleSet: () => void }) {
  const mode = doc.allocator_venue ? doc.mode : "owner";
  const venue = doc.venue_label, base = `/api/venues/${encodeURIComponent(doc.venue_id)}`;
  const [open, setOpen] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [label, cls] = PILL[s.pill] || [s.pill.toUpperCase(), "notlive"];
  const st = s.status, c = s.control;
  const act = async (action: string, body?: object) => {
    setBusy(true);
    setErr(null);
    try {
      await post(`${base}/strategies/${s.id}/${action}`, body);
    } catch (e: any) {
      setErr(e.message);
    }
    setOpen(null);
    setBusy(false);
    await reloadAll(doc.venue_id);
  };
  const lockTxt = doc.allocator_venue ? " Your choice is locked: the LLM allocator will not change it until you unlock it." : "";
  const open_n = st?.positions.length ?? 0;
  // A pending kill ends when the runner, flat, clears it, or when the owner cancels it here, never as a side effect.
  const cancelTxt = `A kill is pending for ${s.label} on ${venue}${open_n ? ` and ${open_n} positions are still open` : ""}. Cancel the kill? If the runner has not started closing yet, the positions will not be closed.`;
  const flip = (on: boolean) => {
    if (on && c.flatten_pending) {
      if (confirm(`${cancelTxt} It is then switched on and trades from them.${lockTxt}`)) act("enable", { cancel_kill: true });
      return;
    }
    if (on && !confirm(`Switch ${s.label} on? Orders go to ${venue} (the runner is started if needed).${lockTxt}`)) return;
    if (!on && !confirm(`Switch ${s.label} off on ${venue}? It stops placing new orders and keeps its positions open.${lockTxt}`)) return;
    act(on ? "enable" : "pause");
  };
  const unlock = () => {
    if (c.flatten_pending) {
      if (confirm(`${cancelTxt} It is then handed back to the LLM allocator, which may switch it on.`)) act("unlock", { cancel_kill: true });
      return;
    }
    act("unlock");
  };
  const cancelKill = () => {
    if (confirm(`${cancelTxt} It stays paused and owner-locked.`)) act("cancel_kill");
  };
  const btn = (action: string, text: string, cn?: string, disabled?: boolean) => (
    <Confirmed id={action} label={text} cn={cn} disabled={disabled} open={open} setOpen={setOpen} busy={busy} onRun={() => act(action)} />
  );
  const now = Date.now() / 1000;
  return (
    <section className="pnl strat-card">
      <div className="ph"><span className="code">{s.id.toUpperCase()}</span>{s.name || s.label}<span className={`spill ${cls}`}>{label}</span>
        <span className="ph-r" data-tip={s.label}>{venue} · {s.unit || "no unit"}{s.account ? ` · account ${s.account}` : ""}</span></div>
      <div className="pb">
        <div className="strat-actions">
          <label className="arm" data-tip={`Auto trade: on = the runner places this strategy's orders on ${venue}, off = it stops placing new orders and keeps its positions open.${doc.allocator_venue ? " Flipping it by hand locks it against the LLM allocator." : ""}`}>
            <input type="checkbox" checked={c.enabled} disabled={busy || mode === "bot"} onChange={(e) => flip(e.target.checked)} />
            <span className="arm-box" /><span>AUTO TRADE</span>
          </label>
          {btn("flatten", "FLATTEN", "red", !st?.positions.length)}
          {btn("reset", "RESET LATCH", "", s.pill !== "latched")}
          {mode === "manual" && (
            <label className="muted" data-tip="Include this strategy when the header AUTO-TRADE switch is turned on in MANUAL mode (kept by the harness)">
              <input type="checkbox" checked={inSet} onChange={toggleSet} /> in header set
            </label>
          )}
          {c.flatten_pending && (
            <button className="kbtn" disabled={busy} onClick={cancelKill} data-tip="Clear the pending kill's flatten flag: the positions stay open and the strategy stays paused">CANCEL KILL</button>
          )}
          {((c.flatten_pending && (open_n > 0 || c.enabled)) || c.reset_pending) && (
            <span className="amb">waiting for the runner: {[c.flatten_pending && (open_n > 0 || c.enabled) && `flatten (${open_n} open)`, c.reset_pending && "reset latch"].filter(Boolean).join(", ")}</span>
          )}
        </div>
        <div className="strat-weight">
          <span className="muted" data-tip="The weight the runner applied (status.json); the request in control.json follows when it differs">WEIGHT</span> <b className="w">{isNum(st?.weight) ? fx(st!.weight, 2) : NA}</b>
          {isNum(c.weight) && (!isNum(st?.weight) || Math.abs(Number(c.weight) - Number(st!.weight)) >= 0.005) && <span className="amb" data-tip="control.json asks for this; the runner applies it at its next loop"> (requested {fx(c.weight, 2)})</span>}
          <span className="muted"> · set by</span> <b className="w">{(st?.set_by || c.set_by) === "llm" ? "LLM allocator" : (st?.set_by || c.set_by) === "owner" ? "owner" : NA}</b>
          {doc.allocator_venue && c.owner_lock && (
            <>
              <span className="tag amb">OWNER LOCK</span>
              <button className="kbtn" disabled={busy} onClick={unlock} data-tip="Hand this strategy back to the LLM allocator at its next decision">UNLOCK TO LLM</button>
            </>
          )}
          {(st?.reason || c.reason) && <span className="strat-reason"><span className="muted">reason</span> {st?.reason || c.reason}</span>}
        </div>
        {err && <div className="alert crit"><b>REFUSED</b> {err}</div>}
        {s.pill === "latched" && <div className="alert crit"><b>LATCHED</b> {st?.last_error || "read the runner journal"}. Check the account on {venue}, then reset the latch.</div>}
        {s.pill === "stale" && <div className="alert crit"><b>STALE</b> status.json was last written {ago((st?.updated_ms || 0) / 1000)} while the runner unit is active.</div>}
        {s.pill === "no_data" && <div className="empty">No status.json yet for this strategy.</div>}
        {st && (
          <>
            <div className="strip strat-strip">
              <div className="cell"><div className="l">Equity (book, {doc.ccy})</div><div className="n">{usdt(st.equity_usdt)}</div><div className="s">start {usdt(st.start_equity_usdt)}</div></div>
              <div className="cell"><div className="l">P&amp;L today</div><div className={`n ${ud(s.pnl_today)}`}>{sg(s.pnl_today, 2)}</div><div className="s">{s.pnl_today_since || "no sample yet today"}</div></div>
              <div className="cell"><div className="l">P&amp;L since start</div><div className={`n ${ud(st.pnl_usdt)}`}>{sg(st.pnl_usdt, 2)}</div><div className={`s ${ud(st.pnl_pct)}`}>{sg(st.pnl_pct, 2, "% since start")}</div></div>
              <div className="cell"><div className="l">Fees (USDT)</div><div className="n">{usdt(st.fees_usdt)}</div><div className="s">paid</div></div>
              <div className="cell"><div className="l">Funding (USDT)</div><div className={`n ${ud(st.funding_usdt)}`}>{sg(st.funding_usdt, 2)}</div><div className="s">received if positive</div></div>
              <div className="cell"><div className="l">Trades</div><div className="n">{st.trades_today ?? NA}</div><div className="s">today (UTC) · {st.trades_total ?? NA} total</div></div>
              <div className="cell"><div className="l">Last rebalance</div><div className="n">{st.last_rebalance_ms ? dt(st.last_rebalance_ms / 1000) : NA}</div><div className="s">{st.last_rebalance_ms ? ago(st.last_rebalance_ms / 1000) : "none yet"}</div></div>
              <div className="cell"><div className="l">Next rebalance</div><div className="n">{st.next_rebalance_ms ? dt(st.next_rebalance_ms / 1000) : NA}</div><div className="s">{st.next_rebalance_ms ? `in ${age(Math.max(0, st.next_rebalance_ms / 1000 - now))}` : "not scheduled"}</div></div>
            </div>
            <div className="strat-spark"><span className="muted">EQUITY, last 12 h of the harness&apos;s samples</span> <Spark h={s.equity_history} /></div>
            {st.last_error && s.pill !== "latched" && <div className="strat-err"><span className="amb">LAST ERROR</span> {st.last_error}</div>}
            <div className="ph sub"><span className="code">POS</span>Positions<span className="ph-r">{st.positions.length} open</span></div>
            {st.positions.length ? (
              <table className="t">
                <thead><tr><th>Symbol</th><th>Market</th><th>Side</th><th className="r">Qty</th><th className="r">Entry</th><th className="r">Mark</th><th className="r">Unrealised P&amp;L</th></tr></thead>
                <tbody>{st.positions.map((p, i) => (
                  <tr key={`${p.symbol}${p.market}${i}`}><td className="sym">{p.symbol}</td><td>{p.market}</td>
                    <td><span className={`tag ${["long", "buy"].includes(p.side.toLowerCase()) ? "long" : "short"}`}>{p.side.toUpperCase()}</span></td>
                    <td className="r">{isNum(p.qty) ? fx(p.qty, 4) : NA}</td><td className="r">{pf(p.entry_price)}</td>
                    <td className="r">{pf(p.mark_price)}</td><td className={`r ${ud(p.upnl_usdt)}`}>{sg(p.upnl_usdt, 2)}</td></tr>
                ))}</tbody>
              </table>
            ) : <div className="empty">No open positions</div>}
            {st.targets.length > 0 && (
              <>
                <div className="ph sub"><span className="code">TGT</span>Targets at the next rebalance</div>
                <table className="t">
                  <thead><tr><th>Symbol</th><th>Market</th><th className="r">Target qty</th></tr></thead>
                  <tbody>{st.targets.map((t, i) => <tr key={`${t.symbol}${t.market}${i}`}><td className="sym">{t.symbol}</td><td>{t.market}</td><td className="r">{isNum(t.target_qty) ? fx(t.target_qty, 4) : NA}</td></tr>)}</tbody>
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
                <td className="r">{fx(t.qty, 4)}</td><td className="r">{pf(t.price)}</td><td className="r">{fx(t.fee_usdt, 4)}</td></tr>
            ))}</tbody>
          </table>
        ) : <div className="empty">No trades yet</div>}
      </div>
    </section>
  );
}

function AllocatorPanel({ doc }: { doc: StratDoc }) {
  const a = doc.allocator, st = a.status;
  const names = new Map(doc.strategies.map((s) => [s.id, s.name || s.id]));
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
        <span className="ph-r">{st?.model || NA}</span></div>
      <div className="pb">
        {!a.available && <div className="muted">ctb.allocator is not part of this build, so the LLM cannot decide yet. Use MANUAL.</div>}
        <div className="strip strat-strip">
          <div className="cell"><div className="l">Last decision</div><div className="n">{st?.last_decision_ms ? dt(st.last_decision_ms / 1000) : NA}</div><div className="s">{st?.last_decision_ms ? ago(st.last_decision_ms / 1000) : "none yet"}</div></div>
          <div className="cell"><div className="l">Next decision</div><div className="n">{st?.next_decision_ms && live && a.control.enabled ? dt(st.next_decision_ms / 1000) : NA}</div><div className="s">{st?.next_decision_ms && live && a.control.enabled ? `in ${age(Math.max(0, st.next_decision_ms / 1000 - now))}` : "not scheduled"}</div></div>
          <div className="cell"><div className="l">Interval</div><div className="n">{a.control.interval_min ? `${a.control.interval_min} min` : NA}</div><div className="s"><a href="#" onClick={(e) => { e.preventDefault(); if (live && a.available) setInterval_(); }}>change</a></div></div>
          <div className="cell"><div className="l">Cost today</div><div className="n">{isNum(st?.cost_usd_today) ? `$${fx(st?.cost_usd_today, 4)}` : NA}</div><div className="s">model calls, USD</div></div>
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
                  <div key={id}><span className="amb" data-tip={id}>{names.get(id) || id}</span> {fx(w, 2)}{d.enabled?.[id] === false ? " (off)" : ""}{d.skipped_owner_lock?.includes(id) ? " (owner lock, not applied)" : ""}{d.skipped_kill_pending?.includes(id) ? " (kill pending, not switched on)" : ""}{d.skipped_paused?.includes(id) ? " (allocator paused during the decision, not applied)" : ""}</div>
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
      await post(`/api/venues/${encodeURIComponent(doc.venue_id)}/runner/${action}`, { unit });
    } catch (e: any) {
      setErr(e.message);
    }
    setOpen(null);
    setBusy(false);
    await reloadAll(doc.venue_id);
  };
  if (!doc.runners.length) return <div className="alert"><b>NO RUNNER KNOWN</b> strategies.json does not exist yet under {doc.root}.{doc.allocator_venue ? " Set CTB_LIVE_UNITS to name the runner unit so it can be started from here." : ""}</div>;
  return (
    <div className="strat-runner">
      {doc.runners.map((r) => {
        const active = ["active", "activating", "reloading", "deactivating"].includes(r.active);
        return (
          <div key={r.unit} className="strat-runner-row">
            <span className="code">RUNNER</span><b>{r.unit}</b>
            <span className={`spill ${active ? "plain" : "warn"}`}>{r.load === "not-found" ? "NOT INSTALLED" : active ? "UP" : String(r.active || "unknown").toUpperCase()}</span>
            {r.since && active && <span className="muted">since {r.since_s ? dt(r.since_s) : `${r.since} (host time)`}</span>}
            <Confirmed id={`start:${r.unit}`} label="START" disabled={active || r.load === "not-found"} open={open} setOpen={setOpen} busy={busy} onRun={() => run(r.unit, "start")} />
            <Confirmed id={`stop:${r.unit}`} label="STOP" cn="red" disabled={!active} open={open} setOpen={setOpen} busy={busy} onRun={() => run(r.unit, "stop")} />
          </div>
        );
      })}
      {err && <div className="alert crit"><b>REFUSED</b> {err}</div>}
    </div>
  );
}

/** Which venue F7 shows. */
function VenueTabs({ venues, sel }: { venues: VenuesDoc | null; sel: string }) {
  if (!venues || venues.venues.length < 2) return null;
  return (
    <span className="seg venue-tabs" data-tip="Pick the venue: cards, runner and controls below act on that venue only">
      {venues.venues.map((v) => (
        <button key={v.id} className={v.id === sel ? "on" : ""} onClick={() => setState({ stratVenue: v.id })}
          data-tip={v.presence_text || undefined}>{v.label}{v.presence !== "live" ? ` · ${(PRESENCE[v.presence]?.[0] || v.presence).toLowerCase()}` : ""}</button>
      ))}
    </span>
  );
}

export function StrategiesView() {
  const venues = useStore((s) => s.venues) as VenuesDoc | null;
  const primary = venues?.primary || "bybit-demo";
  const sel = useStore((s) => s.stratVenue) || primary;
  const own = useStore((s) => s.strat) as StratDoc | null;
  const other = useStore((s) => s.venueStrat[sel]) as StratDoc | undefined;
  const [err, setErr] = useState<string | null>(null);
  const isPrimary = sel === primary;
  useEffect(() => {
    if (isPrimary) return;
    const load = () => loadVenueStrat(sel).then(() => setErr(null), (e) => setErr(e.message));
    load();
    const id = setInterval(() => { if (!document.hidden) load(); }, 10_000);
    return () => clearInterval(id);
  }, [sel, isPrimary]);
  const doc = isPrimary ? own : other;
  const v = venues?.venues.find((x) => x.id === sel);
  if (!doc) return <main className="view on"><div className="toolbar"><span className="code">STRATEGIES</span><VenueTabs venues={venues} sel={sel} /></div><div className="empty">{err || "Loading"}</div></main>;
  const set = manualSet(doc);
  const toggleSet = async (id: string) => {
    const next = set.includes(id) ? set.filter((i) => i !== id) : [...set, id];
    try { await post("/api/strategies/auto-set", { ids: next }); } catch (e: any) { alert(e.message); }
    await loadStrat().catch(() => {});
  };
  return (
    <main className="view on">
      <div className="toolbar">
        <span className="code">STRATEGIES</span><VenueTabs venues={venues} sel={sel} /><b>{doc.venue_label}</b>
        {doc.allocator_venue
          ? <span className="muted">mode <b className="w">{MODE_LABEL[doc.mode]}</b> · AUTO-TRADE <b className="w">{doc.auto_on ? "ON" : "OFF"}</b> · orders go to {doc.venue_label} only</span>
          : <span className="muted">controlled by the owner (no allocator) · orders go to {doc.venue_label} only · settles in {doc.ccy}</span>}
        {v && v.presence !== "live" && <span className={`vbadge ${PRESENCE[v.presence]?.[1] || "bad"}`} data-tip={v.presence_text || undefined}>{PRESENCE[v.presence]?.[0] || v.presence}</span>}
        <button className="kbtn" onClick={() => reloadAll(isPrimary ? undefined : sel)}>REFRESH</button>
      </div>
      {doc.allocator_venue && <div className="muted strat-help">{MODE_HELP[doc.mode]} Change the mode in the header.</div>}
      {v?.presence === "absent" && <div className="alert"><b>NOT STARTED</b> {v.presence_text}. The runner ({v.unit || "its unit"}) writes strategies.json when it first starts.</div>}
      {doc.error && <div className="alert crit"><b>CONFIG</b> {doc.error}</div>}
      <RunnerBar doc={doc} />
      {doc.allocator_venue && (doc.allocator.available || doc.mode === "llm") && <AllocatorPanel doc={doc} />}
      {doc.allocator_venue && <PredictionsPanel />}
      {doc.allocator_venue && <ScoreboardPanel />}
      {v && v.account && <div className="muted strat-help">{doc.venue_label} account: {v.account.ccy} equity {money(v.account.equity, v.account.ccy)} · {v.account.source === "exchange" ? "derivatives unrealised" : "unrealised"} {smoney(v.account.unrealised, v.account.ccy)}</div>}
      <div className="strat-grid">
        {doc.strategies.map((s) => (
          <Card key={`${doc.venue_id}:${s.id}`} s={s} doc={doc} inSet={set.includes(s.id)} toggleSet={() => toggleSet(s.id)} />
        ))}
      </div>
    </main>
  );
}
