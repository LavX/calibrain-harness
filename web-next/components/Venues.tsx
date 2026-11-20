"use client";
// F1: the strategies world. Every venue (one live runner per exchange account) on one screen: the venue strip, the
// strategy x venue matrix, alerts, positions per venue and one honest line about the LLM allocator. Everything comes from
// GET /api/venues (polled every 10 s); KILL and KILL ALL go to /api/venues/{vid}/kill and /api/kill-all, which refuse
// when a runner is down or stale instead of answering ok.
import { useState } from "react";
import { post } from "@/lib/api";
import { loadOverview, loadStrat, loadVenues } from "@/lib/data";
import { setState, useStore } from "@/lib/store";
import { age, ago, fx, hhmm, isNum, money, NA, pf, sg, smoney, ud } from "@/lib/format";
import { MODE_LABEL } from "./Strategies";
import type { PredDoc } from "./Predictions";
import { useNow } from "./TopBar";

export type VStrat = {
  id: string; name: string; label: string; pill: string; enabled: boolean; weight: number | null; weight_requested: number | null;
  set_by: string | null; owner_lock: boolean; reason: string | null; flatten_pending: boolean; positions_n: number | null; targets_n: number | null;
  equity: number | null; start_equity: number | null; pnl: number | null; pnl_today: number | null; pnl_today_since_ms: number | null;
  /** "since 00:00 UTC" or the real time of this strategy's baseline sample ("since 09:12 UTC") */
  pnl_today_since: string | null;
  unrealised: number | null; last_error: string | null; updated_ms: number | null; last_rebalance_ms: number | null; last_trade_ms: number | null;
  trades_today: number | null;
};
export type VPos = { symbol: string; market: string; side: string; size: number | null; entry: number | null; mark: number | null; upnl: number | null; held_by: string[]; legs: number | null; source?: "exchange" | "books"; account?: string };
export type VAccount = {
  source: "exchange" | "account.json"; ccy: string; equity: number | null; collateral_usd: number | null; available: number | null; unrealised: number | null;
  /** the books' perp and future legs' uPnL, the like-for-like figure the exchange's unrealised is compared with */
  books_unrealised: number | null;
  /** only from a runner's account.json (its own day start); never computed from the exchange's USDT coin equity */
  pnl_today: number | null; pnl_today_since: string | null; unattributed: number | null; positions_n: number | null; last_sync: number | null;
  /** the dashboard's own exchange read failed or is older than 60 s: the figures are the last known, positions come from the books */
  stale: boolean; sync_error: string | null; sync_age_s: number | null;
};
export type Venue = {
  id: string; label: string; ccy: string; controlled_by: "allocator" | "owner"; root: string; unit: string | null;
  presence: "live" | "stale" | "stopped" | "absent" | "error"; presence_text: string | null; updated_ms: number | null;
  counts: { running: number; idle: number; paused: number; abnormal: number; total: number };
  books: { equity: number | null; start: number | null; pnl: number | null; unrealised: number | null; unrealised_derivatives: number | null; incomplete: string[]; legs: number };
  /** pnl_today_since is null when the strategies' baselines differ; pnl_today_note says why (late first sample, a strategy seen later, no sample yet) */
  pnl_today: number | null; pnl_today_partial: boolean; pnl_today_since: string | null; pnl_today_note: string | null;
  account: VAccount | null; strategies: VStrat[]; positions: VPos[]; positions_source: "exchange" | "books"; positions_note: string | null;
  /** available: KILL writes the flags (every venue that has started); ok: and the runner would act on them now. */
  kill: { ok: boolean; available: boolean; why: string | null };
};
export type VAlert = { venue: string; venue_label: string; strategy: string | null; name: string | null; kind: string; level: "crit" | "warn"; text: string };
export type VenuesDoc = {
  generated_at: number; primary: string; venues: Venue[]; alerts: VAlert[]; mode: "llm" | "manual" | "bot"; auto_on: boolean;
  allocator: {
    available: boolean; control: { enabled: boolean; interval_min: number | null };
    status: { state: string; model: string | null; last_decision_ms: number | null; next_decision_ms: number | null; last_weights_ms: number | null;
      next_weights_ms: number | null; updated_ms: number | null; weights_mode: string | null; last_error: string | null } | null;
  };
};

/** Presence as words. "LIVE" is kept for real money, so a running demo runner is UP. */
export const PRESENCE: Record<string, [string, string]> = {
  live: ["UP", "ok"], stale: ["STALE", "bad"], stopped: ["UNIT DOWN", "bad"], absent: ["NOT STARTED", "off"], error: ["CONFIG ERROR", "bad"],
};
/** Strategy state words; colour only when something is wrong. */
export const STATE: Record<string, [string, string]> = {
  running: ["RUNNING", ""], idle: ["IDLE", "idle"], paused: ["PAUSED", ""], latched: ["LATCHED", "bad"], error: ["ERROR", "bad"],
  stale: ["STALE", "bad"], runner_stopped: ["UNIT DOWN", "warn"], no_data: ["NO DATA", "idle"],
};
const ORDER = ["carry", "dated_carry", "trend", "xsmom", "basis", "longvol", "events"];

async function reload() {
  await Promise.all([loadVenues().catch(() => {}), loadStrat().catch(() => {}), loadOverview().catch(() => {})]);
}

/** What KILL does on one venue, said before it happens (and, when the runner is down or stale, that it cannot be confirmed). */
function killText(v: Venue): string {
  const n = v.counts.total;
  const unit = v.unit || "its runner unit";
  const what = v.controlled_by === "allocator"
    ? `Pauses the LLM allocator, pauses all ${n} strategies (owner-locked, so the allocator cannot switch them back on) and asks the runner (${unit}) to close every position through its books.`
    : `Pauses all ${n} strategies (owner-locked) and asks the runner (${unit}) to close every position through its books.`;
  const when = v.kill.ok ? "The runner acts within a minute." : `NOT CONFIRMABLE NOW: ${v.kill.why} The flags are written anyway and read when the runner runs again.`;
  return `KILL ${v.label}?\n\n${what} ${when} Other venues are untouched.`;
}

export async function killVenue(v: Venue) {
  if (!confirm(killText(v))) return;
  try {
    const r = await post(`/api/venues/${encodeURIComponent(v.id)}/kill`);
    alert(`${v.label}: confirmed, flattening ${r.flattening.length} strategies (${r.flattening.join(", ") || "none"})${r.already_flat?.length ? `; already flat (their runner clears the flag): ${r.already_flat.join(", ")}` : ""}. Watch the alerts until they are flat.`);
  } catch (e: any) {
    alert(e.message); // "written, NOT confirmed: ...", "partially written: ..." or "has not started"
  }
  await reload();
}

/** KILL ALL: every venue that is up is killed; the answer says, per venue, what happened or why not. */
export async function killAll(doc: VenuesDoc | null) {
  if (!doc) return; // the button is disabled until the venues have loaded
  const lines = doc.venues.map((v) => {
    const what = `${v.controlled_by === "allocator" ? "allocator paused, " : ""}${v.counts.total} strategies paused and flagged to flatten`;
    return v.presence === "absent" ? `${v.label}: not started, skipped` : !v.kill.available ? `${v.kill.why}`
      : v.kill.ok ? `${v.label}: ${what}`
        : `${v.label}: ${what}, NOT confirmable now (${PRESENCE[v.presence]?.[0] || v.presence}): ${v.kill.why}`;
  });
  const botLine = doc.mode === "bot"
    ? "PER-SYMBOL BOT (the header mode): every position on its account closed, its orders cancelled, its trading off."
    : "Per-symbol bot: AUTO-TRADE switched off (it is not trading in this mode, so nothing of it is closed on the exchange).";
  if (!confirm(`KILL ALL venues?\n\n${lines.join("\n")}\n\n${botLine}\n\nStrategy positions close through each runner's books.`)) return;
  try {
    const r = await post("/api/kill-all");
    alert(r.venues.map((x: any) => `${x.label}: ${x.ok ? `confirmed, flattening ${x.flattening.join(", ") || "none"}${x.already_flat?.length ? `; already flat: ${x.already_flat.join(", ")}` : ""}` : x.skipped && /has not started/.test(x.error) ? "not started, skipped" : x.error}`).join("\n")
      + `\n\n${r.bot?.why || "per-symbol bot"} (AUTO-TRADE off).`);
  } catch (e: any) {
    alert(e.message);
  }
  await reload();
}

const Ph = ({ code, children, right }: { code: string; children?: React.ReactNode; right?: React.ReactNode }) => (
  <div className="ph"><span className="code">{code}</span>{children}{right !== undefined && <span className="ph-r">{right}</span>}</div>
);

export function WorldView() {
  const doc = useStore((s) => s.venues) as VenuesDoc | null;
  useNow(10_000);
  if (!doc) return <main className="view on"><div className="empty">Loading the venues</div></main>;
  const crit = doc.alerts.filter((a) => a.level === "crit").length;
  return (
    <main className="view on">
      <div className="grid-world">
        <section className="pnl w-strip">
          <Ph code="VEN" right={<button className="kbtn red" onClick={() => killAll(doc)} data-tip="Pause and flag to flatten every venue that has started; the answer says, per venue, whether the runner confirms">KILL ALL</button>}>Venues</Ph>
          <div className="pb tight scroll"><VenueStrip doc={doc} /></div>
          <AllocatorLine doc={doc} />
        </section>
        <section className="pnl w-matrix">
          <Ph code="MTX" right="cell: P&L today (its baseline time when not 00:00 UTC) · applied weight · positions">Strategies × venues</Ph>
          <div className="pb tight scroll"><Matrix doc={doc} /></div>
        </section>
        <section className="pnl w-alerts">
          <Ph code="ALRT" right={doc.alerts.length ? <span className={crit ? "dn" : "amb"}>{doc.alerts.length} open{crit ? ` · ${crit} critical` : ""}</span> : "none"}>Alerts</Ph>
          <div className="pb tight scroll"><AlertList alerts={doc.alerts} /></div>
        </section>
        <section className="pnl w-pos"><PositionsPanel doc={doc} /></section>
      </div>
    </main>
  );
}

/* ---------------------------------------------------------------- venue strip */
function Presence({ v, at }: { v: Venue; at: number }) {
  const [txt, cls] = PRESENCE[v.presence] || [v.presence.toUpperCase(), "bad"];
  const stale = v.presence === "stale" && v.updated_ms ? ` · ${age(at - v.updated_ms / 1000)}` : "";
  return <span className={`vbadge ${cls}`} data-tip={v.presence_text || (v.presence === "live" ? `${v.unit || "runner"} active, status files fresh` : undefined)}>{txt}{stale}</span>;
}

function VenueStrip({ doc }: { doc: VenuesDoc }) {
  return (
    <table className="t vstrip">
      <thead><tr>
        <th>Venue</th><th>State</th>
        <th className="r" data-tip="Sum of the strategies' virtual books on this venue">Equity (books)</th>
        <th className="r" data-tip="Books now minus the harness's own first sample of the UTC day (kept on disk, survives restarts); the time of that sample is shown when it is not 00:00 UTC">P&amp;L today</th>
        <th className="r" data-tip="Sum of every book's P&L since it started (status.json pnl_usdt)">P&amp;L since start</th>
        <th>Strategies</th><th>Controlled by</th><th />
      </tr></thead>
      <tbody>{doc.venues.map((v) => {
        const off = v.presence === "absent";
        const grey = v.presence !== "live" ? "stale-row" : "";
        const c = v.counts;
        return (
          <tr key={v.id} className={grey}>
            <td className="sym" data-tip={`${v.root}${v.unit ? `\nunit ${v.unit}` : ""}`}>{v.label}</td>
            <td><Presence v={v} at={doc.generated_at} /></td>
            <td className="r" data-tip={v.books.incomplete.length ? `Leaves out ${v.books.incomplete.join(", ")}: equity or P&L unknown` : undefined}>{off ? NA : money(v.books.equity, v.ccy)}{v.books.incomplete.length > 0 && <span className="muted"> (partial)</span>}</td>
            <td className={`r ${off ? "" : ud(v.pnl_today)}`} data-tip={off ? undefined : [v.pnl_today_since || (v.pnl_today === null ? "no sample yet today" : "the strategies' baselines differ"), v.pnl_today_note].filter(Boolean).join("\n")}>
              {off ? NA : smoney(v.pnl_today, v.ccy)}
              {!off && v.pnl_today_since && v.pnl_today_since !== "since 00:00 UTC" && <span className="muted"> {v.pnl_today_since}</span>}
              {!off && v.pnl_today_partial && <span className="muted"> (partial)</span>}</td>
            <td className={`r ${off ? "" : ud(v.books.pnl)}`}>{off ? NA : smoney(v.books.pnl, v.ccy)}</td>
            <td className="wrap">{off ? <span className="muted">not started</span> : <>{c.running} running · {c.idle} idle{c.paused ? ` · ${c.paused} paused` : ""} / {c.total}{c.abnormal > 0 && <b className="amb"> · {c.abnormal} need attention</b>}</>}</td>
            <td>{v.controlled_by === "allocator" ? "LLM allocator" : "owner"}</td>
            <td className="r">{!off && (
              <span data-tip={v.kill.ok ? "Pause and flatten every strategy on this venue" : v.kill.available ? `Writes the kill flags, but cannot confirm now: ${v.kill.why}` : v.kill.why || undefined}>
                <button className="kbtn red" disabled={!v.kill.available} onClick={() => killVenue(v)}>KILL{v.kill.available && !v.kill.ok ? " (unconfirmed)" : ""}</button>
              </span>
            )}</td>
          </tr>
        );
      })}</tbody>
      <tfoot>{doc.venues.filter((v) => v.account).map((v) => <AccountRow key={v.id} v={v} />)}</tfoot>
    </table>
  );
}

/** The exchange account under the books (bybit-demo: the USDT equity apart from every other coin's collateral). */
function AccountRow({ v }: { v: Venue }) {
  const a = v.account!;
  return (
    <tr className={`acct-row ${a.stale ? "stale-row" : ""}`}>
      <td colSpan={8}>
        {a.stale && <><span className="vbadge bad" data-tip={a.sync_error || "no good exchange read in the last 60 s"}>STALE · {a.sync_age_s !== null ? age(a.sync_age_s) : "never read"}</span>{" "}</>}
        <span className="muted">{v.label} account{a.source === "account.json" ? " (runner's account.json)" : ""}{a.stale ? ", last known" : ""}:</span>{" "}
        {a.ccy} equity <b className="w">{money(a.equity, a.ccy)}</b>
        {isNum(a.collateral_usd) && <span data-tip="Bybit totalEquity: every coin in the unified account, valued in USD. Not the strategies' money."> · all collateral {money(a.collateral_usd, "USD")}</span>}
        {" · "}<span data-tip={a.source === "exchange" ? "The exchange's unrealised P&L on perps and futures (spot legs are in the coin balances)" : undefined}>{a.source === "exchange" ? "derivatives unrealised" : "unrealised"}</span> <span className={ud(a.unrealised)}>{smoney(a.unrealised, a.ccy)}</span>
        {a.source === "exchange" && <span data-tip="Like with like: the exchange's derivatives unrealised minus the books' perp and future legs' unrealised. Unknown when any leg's uPnL is unknown."> (books&apos; perp legs {smoney(a.books_unrealised, v.ccy)}, unattributed <span className={ud(a.unattributed)}>{sg(a.unattributed, 2)}</span>)</span>}
        {a.source === "account.json" && <>{" · "}P&amp;L today <span className={ud(a.pnl_today)}>{smoney(a.pnl_today, a.ccy)}</span>{a.pnl_today_since && <span className="muted"> {a.pnl_today_since}</span>}</>}
      </td>
    </tr>
  );
}

/* ---------------------------------------------------------------- allocator line */
function AllocatorLine({ doc }: { doc: VenuesDoc }) {
  const pred = useStore((s) => s.pred) as PredDoc | null;
  const a = doc.allocator, st = a?.status;
  const venue = doc.venues.find((v) => v.id === doc.primary);
  const kappa = pred?.forecast?.kappa ?? pred?.allocator?.kappa ?? null;
  const rg = pred?.regime;
  const stay = rg?.p7.find((p) => p.state === rg.state)?.p;
  const regime = rg ? <span data-tip={`state ${rg.state} on ${rg.date || NA}`}>{rg.state_help}{isNum(stay) ? `, ${(Number(stay) * 100).toFixed(0)}% chance it still is in 7 days` : ""}</span> : <span className="muted">{NA}</span>;
  const now = useNow(10_000) / 1000;
  let control: React.ReactNode;
  if (doc.mode === "bot") control = <>paused: the header mode is <b className="amb">PER-SYMBOL BOT</b></>;
  else if (doc.mode === "manual") control = <>paused: the header mode is MANUAL, the owner switches {venue?.label || "Bybit"} strategies</>;
  else if (!a?.available) control = <span className="amb">not installed in this build</span>;
  else control = (
    <>
      {isNum(kappa) && Number(kappa) > 0 ? `tilt live, live control ${(Number(kappa) * 100).toFixed(0)}%` : "shadow tilt, live control 0%"}
      {isNum(kappa) ? ` (kappa ${fx(kappa, 2)})` : ""}
      {" · "}<b className={a.control.enabled ? "w" : "amb"}>{a.control.enabled ? "AUTO-TRADE ON" : "AUTO-TRADE OFF"}</b>
    </>
  );
  return (
    <div className="alloc-line">
      <div><span className="muted" data-tip={`One decision per strategy (made on ${venue?.label || "BYBIT·DEMO"}'s books), applied on each venue under its own risk rules. The vol cap uses ${venue?.label || "BYBIT·DEMO"}'s forecast for every venue: an approximation elsewhere. The drawdown floor keeps running while the allocator is paused, but never on a book you switched by hand (owner-locked). CTB_ALLOCATOR_VENUES leaves a venue to the owner.`}>
        LLM ALLOCATOR · controls {doc.venues.filter((v) => v.controlled_by === "allocator").map((v) => v.label).join(", ") || "no venue"}:</span> {control}
        {st && doc.mode === "llm" && <>
          <span className="muted"> · weights last set </span>{st.last_weights_ms ? `${hhmm(st.last_weights_ms / 1000)} (${ago(st.last_weights_ms / 1000)})` : NA}
          <span className="muted">, next </span>{st.next_weights_ms ? hhmm(st.next_weights_ms / 1000) : NA}
          <span className="muted" data-tip="The hourly decision only refreshes the LLM's P(up 24h) answers; weights are written once a day."> · last P(up) poll </span>{st.last_decision_ms ? ago(st.last_decision_ms / 1000) : NA}
          <span className="muted"> · model </span>{st.model || NA}
          {st.last_error && <b className="dn" data-tip={st.last_error}> · LAST ERROR</b>}
        </>}
        {st?.next_weights_ms && st.next_weights_ms / 1000 < now - 3600 && doc.mode === "llm" && <b className="amb"> · weights overdue</b>}
      </div>
      <div><span className="muted">REGIME:</span> {regime}</div>
    </div>
  );
}

/* ---------------------------------------------------------------- matrix */
function Matrix({ doc }: { doc: VenuesDoc }) {
  const pred = useStore((s) => s.pred) as PredDoc | null;
  const pDec = new Map((pred?.strategies || []).map((r) => [r.id, r.p_decider]));
  const ids: string[] = [];
  const names = new Map<string, string>(), labels = new Map<string, string>();
  for (const v of doc.venues) for (const s of v.strategies) {
    if (!ids.includes(s.id)) ids.push(s.id);
    names.set(s.id, s.name);
    labels.set(s.id, s.label);
  }
  ids.sort((a, b) => (ORDER.indexOf(a) < 0 ? 99 : ORDER.indexOf(a)) - (ORDER.indexOf(b) < 0 ? 99 : ORDER.indexOf(b)) || a.localeCompare(b));
  if (!ids.length) return <div className="empty">No venue has written a strategies.json yet</div>;
  return (
    <table className="t matrix">
      <thead><tr>
        <th>Strategy</th>
        {doc.venues.map((v) => <th key={v.id} className={v.presence !== "live" ? "muted" : ""} data-tip={v.presence_text || undefined}>{v.label}{v.presence !== "live" && ` · ${(PRESENCE[v.presence]?.[0] || v.presence).toLowerCase()}`}</th>)}
      </tr></thead>
      <tbody>{ids.map((id) => (
        <tr key={id}>
          <td className="sym lbl" data-tip={`${labels.get(id)} (${id})`}>{names.get(id)}</td>
          {doc.venues.map((v) => <Cell key={v.id} v={v} s={v.strategies.find((x) => x.id === id)} pUp={v.controlled_by === "allocator" ? pDec.get(id) : undefined} />)}
        </tr>
      ))}</tbody>
    </table>
  );
}

/** pUp: the allocator's LLM P(up 24h) for this strategy, given only on the allocator venue's cells (no LLM feeds the others). */
function Cell({ v, s, pUp }: { v: Venue; s: VStrat | undefined; pUp?: number | null }) {
  if (!s) return <td className="cell-x muted">{NA}</td>;
  const [word, cls] = STATE[s.pill] || [s.pill.toUpperCase(), "bad"];
  const grey = v.presence !== "live";
  const w = isNum(s.weight) ? fx(s.weight, 2) : NA;
  const asked = isNum(s.weight_requested) && isNum(s.weight) && Math.abs(Number(s.weight_requested) - Number(s.weight)) >= 0.005 ? ` (asked ${fx(s.weight_requested, 2)})` : "";
  const tip = [
    `${s.label} on ${v.label}: ${word}`,
    `applied weight ${w}${asked}${s.set_by ? `, set by ${s.set_by === "llm" ? "the LLM allocator" : "the owner"}` : ""}${s.owner_lock ? ", owner lock" : ""}`,
    s.reason ? `reason: ${s.reason}` : "",
    `P&L today ${smoney(s.pnl_today, v.ccy)} ${s.pnl_today_since || "(no sample yet)"} · since start ${smoney(s.pnl, v.ccy)}`,
    `${s.positions_n ?? NA} positions · ${s.trades_today ?? NA} trades today${s.last_trade_ms ? ` · last trade ${ago(s.last_trade_ms / 1000)}` : " · no trade yet"}`,
    s.updated_ms ? `status written ${ago(s.updated_ms / 1000)}` : "",
    isNum(pUp) ? `LLM P(up 24h) ${(Number(pUp) * 100).toFixed(0)}%: mean of the allocator's last 24 hourly answers` : "",
    s.last_error ? `LAST ERROR: ${s.last_error}` : "",
  ].filter(Boolean).join("\n");
  return (
    <td className={`cell-m ${grey ? "stale-row" : ""}`} data-tip={tip}>
      <div><span className={grey ? "" : ud(s.pnl_today)}>{sg(s.pnl_today, 2)}</span>{(cls || s.pill !== "running") && <span className={`sbadge ${cls}`}>{word}</span>}{s.last_error && s.pill !== "error" && s.pill !== "latched" && <span className="sbadge warn">ERR</span>}</div>
      <div className="muted">w {w}{s.owner_lock ? " · lock" : ""} · {s.positions_n ?? NA} pos{isNum(pUp) ? ` · P(up) ${(Number(pUp) * 100).toFixed(0)}%` : ""}{s.pnl_today_since && s.pnl_today_since !== "since 00:00 UTC" ? ` · ${s.pnl_today_since}` : ""}</div>
    </td>
  );
}

/* ---------------------------------------------------------------- alerts */
function AlertList({ alerts }: { alerts: VAlert[] }) {
  if (!alerts.length) return <div className="empty">All clear</div>;
  const sorted = [...alerts].sort((a, b) => (a.level === b.level ? 0 : a.level === "crit" ? -1 : 1));
  return (
    <table className="t alerts"><tbody>{sorted.map((a, i) => (
      <tr key={`${a.venue}${a.strategy}${a.kind}${i}`}>
        <td><span className={`vbadge ${a.level === "crit" ? "bad" : "warn"}`}>{a.kind.replace(/_/g, " ").toUpperCase()}</span></td>
        <td className="sym">{a.venue_label}{a.name ? ` · ${a.name}` : ""}</td>
        <td className="wrap" data-tip={a.text}>{a.text}</td>
      </tr>
    ))}</tbody></table>
  );
}

/* ---------------------------------------------------------------- positions */
function PositionsPanel({ doc }: { doc: VenuesDoc }) {
  const present = doc.venues.filter((v) => v.presence !== "absent");
  const [sel, setSel] = useState<string | null>(null);
  const v = present.find((x) => x.id === sel) || present[0];
  const head = (
    <div className="ph">
      <span className="code">PST</span>
      {v ? <>Open positions {v.positions.length} (netted){v.books.legs ? ` · ${v.books.legs} strategy legs` : ""}</> : "Open positions"}
      <span className="seg">{present.map((x) => (
        <button key={x.id} className={x.id === v?.id ? "on" : ""} onClick={() => setSel(x.id)}>{x.label} {x.positions.length}</button>
      ))}</span>
      <span className={`ph-r ${v?.positions_note ? "amb" : ""}`}>{v?.positions_source === "exchange" ? `${v.positions_note || "the exchange's netted positions"}; close them from F7 (FLATTEN) or KILL` : v?.positions_note || "the books' legs netted per coin and market"}</span>
    </div>
  );
  if (!v) return <>{head}<div className="empty">No venue has started</div></>;
  return (
    <>
      {head}
      <div className="pb tight scroll">
        {v.positions.length ? (
          <table className="t">
            <thead><tr><th>Symbol</th><th className="r">uPnL ({v.ccy})</th><th>Side</th><th className="r">Size</th><th className="r">Entry</th><th className="r">Mark</th><th>Held by</th></tr></thead>
            <tbody>{v.positions.map((p) => (
              <tr key={`${p.symbol}:${p.market}:${p.source || ""}:${p.account || ""}`} className={v.presence !== "live" ? "stale-row" : ""}>
                <td className="sym">{p.symbol}{p.market && p.market !== "linear" && <span className="tag flat"> {p.market.toUpperCase()}</span>}{p.account && <span className="tag flat" data-tip="A strategy on another account: from its book, not the exchange feed"> {p.account}</span>}</td>
                <td className={`r ${ud(p.upnl)}`}>{sg(p.upnl, 2)}</td>
                <td><span className={`tag ${p.side === "long" ? "long" : p.side === "short" ? "short" : "flat"}`}>{p.side.toUpperCase()}</span></td>
                <td className="r">{isNum(p.size) ? Number(p.size).toLocaleString("en-US", { maximumFractionDigits: 6 }) : NA}</td>
                <td className="r" data-tip={p.entry === null && (p.legs || 0) > 1 ? "legs on both sides: no single entry price" : undefined}>{pf(p.entry)}</td>
                <td className="r">{pf(p.mark)}</td>
                <td className="ellip" data-tip={p.held_by.join(", ") || "no strategy book holds this coin"}>{p.held_by.length ? p.held_by.join(", ") : <span className="amb">no book</span>}</td>
              </tr>
            ))}</tbody>
          </table>
        ) : <div className="empty">{v.label}: flat, no open positions</div>}
      </div>
    </>
  );
}

/** Header list of venues with their presence ("VENUES BYBIT·DEMO up · OKX·DEMO up · DERIBIT·TEST not started"). */
export function VenueList() {
  const doc = useStore((s) => s.venues) as VenuesDoc | null;
  if (!doc) return null;
  return (
    <span className="venue">VENUES {doc.venues.map((v, i) => (
      <span key={v.id}>{i > 0 && " · "}<b className={PRESENCE[v.presence]?.[1] === "bad" ? "dn" : ""} data-tip={v.presence_text || undefined}
        onClick={() => setState({ view: "strat", stratVenue: v.id })} style={{ cursor: "pointer" }}>{v.label}</b>{" "}
        <span className={PRESENCE[v.presence]?.[1] === "bad" ? "dn" : "muted"}>{(PRESENCE[v.presence]?.[0] || v.presence).toLowerCase()}</span></span>
    ))}{doc.mode !== "bot" && <span className="muted"> · {MODE_LABEL[doc.mode]} on {doc.venues.find((v) => v.id === doc.primary)?.label}</span>}</span>
  );
}
