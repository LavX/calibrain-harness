"use client";
import { useEffect } from "react";
import { connectSocket, loadDecisions, loadMarket, loadMonitor, loadNews, loadOverview, loadPerf, loadPredictions, loadStrat, loadVenues } from "@/lib/data";
import { setState, useStore } from "@/lib/store";
import { Alerts, Clocks, CommandBar, FKeys, StatusLine } from "./TopBar";
import { BotView, MarketView } from "./Monitor";
import { WorldView } from "./Venues";
import { EvalView, LogView, NewsView, TradesView } from "./Views";
import { Modals } from "./Modals";
import { ValidationView } from "./Validation";
import { StrategiesView } from "./Strategies";
import { Tip } from "./Tip";

/** Polls only what has no push channel; everything that moves arrives over /ws. */
function every(ms: number, fn: () => Promise<unknown>) {
  const id = setInterval(() => {
    if (!document.hidden) fn().catch(() => {});
  }, ms);
  return () => clearInterval(id);
}

export default function Terminal() {
  const view = useStore((s) => s.view);
  const ready = useStore((s) => !!s.ov);

  useEffect(() => {
    let stop: (() => void)[] = [];
    (async () => {
      await loadOverview().catch(() => {});
      loadStrat().catch(() => {});
      loadVenues().catch(() => {});
      loadPredictions().catch(() => {});
      await Promise.all([loadMonitor(), loadMarket(), loadDecisions(), loadPerf(), loadNews()].map((p) => p.catch(() => {})));
      stop = [
        connectSocket(),
        every(10_000, loadStrat), // live strategies, allocator and the auto-trade mode
        every(10_000, loadVenues), // F1: every venue, the matrix, alerts and positions
        every(30_000, loadPredictions), // predictions panel and scoreboard (the server caches 30 s)
        every(10_000, loadOverview), // settings, budgets, labelled count (account itself is pushed)
        every(5_000, loadMonitor), // universe map, macro, tape, headlines, sparklines (ticks are pushed)
        every(10_000, loadMarket), // state preview + derivatives block (candles/book are pushed)
        every(30_000, loadPerf),
        every(60_000, loadNews),
        every(60_000, loadDecisions), // labels land in the background
      ];
    })();
    const onKey = (e: KeyboardEvent) => {
      const fk = ({ F1: "mon", F2: "eval", F3: "trades", F4: "news", F5: "log", F6: "valid", F7: "strat", F8: "market", F9: "bot" } as const)[e.key as "F1"];
      if (fk) {
        e.preventDefault();
        setState({ view: fk });
        return;
      }
      const tag = (document.activeElement as HTMLElement | null)?.tagName || "";
      if ((e.key === "/" || e.key === "`") && !/INPUT|SELECT|TEXTAREA/.test(tag) && !document.querySelector("dialog[open]")) {
        e.preventDefault();
        (document.getElementById("cmd") as HTMLInputElement | null)?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      stop.forEach((f) => f());
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <>
      <Clocks />
      <CommandBar />
      <FKeys />
      <Alerts />
      {ready ? (
        <>
          {view === "mon" && <WorldView />}
          {/* always mounted: the chart keeps its candles and the socket keeps watching the selected symbol */}
          <MarketView hidden={view !== "market"} />
          {view === "bot" && <BotView />}
          {view === "eval" && <EvalView />}
          {view === "valid" && <ValidationView />}
          {view === "strat" && <StrategiesView />}
          {view === "trades" && <TradesView />}
          {view === "news" && <NewsView />}
          {view === "log" && <LogView />}
        </>
      ) : (
        <div className="empty">Connecting to the local terminal service on 127.0.0.1:8800…</div>
      )}
      <StatusLine />
      <Modals />
      <Tip />
    </>
  );
}
