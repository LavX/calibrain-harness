# Strategy allocator

`ctb/allocator.py` decides which live strategies run, how much of each strategy's virtual book is
deployed, and when. It places no orders: it only writes the allocation fields of each strategy's
`control.json`, which the live runner reads every loop.

There are two ways it sets weights:

- **Computed** (when `forecast/latest.json` exists): weights come from the prediction layer's
  forecasts by a static core (inverse-volatility mix) that the forecasts may tilt once the tilt has earned
  it (it starts in shadow, so live weights start as the core), plus fixed rules, see "Computed weights" below. The decision model is one forecaster among several and still has a veto.
- **Decider only** (no forecast file): the decision model's probabilities map straight to weights,
  see "Model and question set". `status.json` says `weights_mode: "decider-only"`.

Layout under the live root (`CTB_LIVE_ROOT`, default `data/live`; on aurora4
`/home/lavx/Documents/calibrain-study/data/live`):

| File | Written by | Content |
| --- | --- | --- |
| `strategies.json`, `<id>/status.json`, `<id>/trades.jsonl` | runner | what the allocator reads |
| `<id>/control.json` | dashboard, allocator | allocator sets `enabled`, `weight`, `set_by: "llm"`, `reason`; `owner_lock: true` is never touched |
| `allocator/control.json` | dashboard | `{"enabled", "interval_min", "decide_now"}` (allocator clears `decide_now`) |
| `allocator/status.json` | allocator | `state`, `model`, `last_decision_ms`, `next_decision_ms`, `last_error`, `cost_usd_today` (plus `interval_min`, `last_reasoning`, `last_confidence`, `weights_mode`, `forecast`, `last_weights_ms`, `next_weights_ms`) |
| `allocator/decisions.jsonl` | allocator | one line per decision, risk-floor action or failed attempt |
| `allocator/state.json` | allocator | equity samples and peaks for the P&L windows and the drawdown floor, the decider's last answers (`p_hist`), `last_weights_ms` |
| `forecast/latest.json`, `regime/latest.json` | prediction layer (`research/live/forecast.py`, `regime.py`) | read only here; contract in `live-status-contract.md` revision 3 |

## Venues

The allocator lives in the bybit-demo root (its control, status, decisions and state files, and the forecast and regime
it reads) and makes one decision per strategy there. It applies that decision to every venue in `CTB_LIVE_ROOTS` that it
controls (`CTB_ALLOCATOR_VENUES`, default all) and whose runner's strategies.json lists the strategy; a strategy a venue
does not list is never written there, and a strategy no decision covers (not on bybit-demo) is held. Each venue is its own
account: the vol cap is computed over that venue's strategies, the drawdown floor, the day anchor of the daily step and
the exploration floor are per (venue, strategy), and every write checks that venue's owner lock and pending kill and the
allocator's pause. `state.json` keys peaks, history and the day anchor as `<venue>/<strategy>` (files from before are
migrated on load: their keys become `bybit-demo/<id>`, values unchanged; `keys_version` 2). The decider's answers
(`p_hist`) stay per strategy. A decision line keeps bybit-demo's fields at the top level and adds `venues`: per venue the
weights, what was written, and the owner-lock, kill-pending, paused and risk-floor skips. A risk-floor line names its venue.

## Rules enforced in code

- A weight moves at most 0.5 per decision.
- A strategy more than 15% below its book's peak gets weight 0, `enabled` false and
  `flatten_on_zero` true. This check runs every 30 s tick without the model, so a model outage
  never disables it. The peak is tracked from the runner's `equity_usdt` (and
  `start_equity_usdt`, or `peak_equity_usdt` when the runner reports one). After an intentional
  restart of a cut strategy run `python -m ctb.allocator reset-peak <id>`.
- A `control.json` with no `weight` counts as weight 0 (nothing decided yet means off, the same
  default the runner uses), so the first decision's step limit starts from 0.
- Invalid model output (schema, ranges, unknown ids) is retried once with the problem stated, then
  the current allocation is kept and the error is logged.
- `owner_lock: true` is respected, including a lock set between reading and writing.
- A model request budget (`CTB_ALLOCATOR_MAX_RPH`, default 12 per hour) survives restarts.

## Computed weights

Source of the numbers: `forecast/latest.json` (daily 23:40 UTC): per strategy `sigma_d` and the daily mean
net return `mu` of each forecaster (`base`, `regime`, `ridge`, carry also `structural`), `cov_d`, and
`weights_k` (each forecaster's weight, from the daily scoring), and `kappa` (the tilt room, see "Tilt room by skill"). The decider is not in the file: the
allocator adds it. The math is in `ctb/allocator_math.py` (pure, no I/O); the constants are in
`ctb/config.py` (`CTB_ALLOCATOR_*`, defaults below).

Schedule:

- Every hour (`interval_min`) the decider is asked the same `alloc-v1.1` questions. The answers go into
  `state.json` (`p_hist`, last 24 per strategy) and into a `decisions.jsonl` line (`weights_mode: "refresh"`,
  `answers.p`, which `forecast.py score` reads). Weights are not touched.
- Weights are set once a day at 23:50 UTC (`CTB_ALLOCATOR_WEIGHTS_UTC`), on the first run when none were
  ever set, and on DECIDE NOW or `once`. The decider is asked first, so its newest answer is in p. If the
  decider or market data fail, the stored p history is used and the weights are still set.
- A forecast older than 36 h (`CTB_ALLOCATOR_STALE_H`), or an unreadable file: weights are held, the hourly
  refresh continues, and the reason is in `status.json` `last_error` and `forecast`. The drawdown floor and
  `owner_lock` keep working. A missing file is not an error: decider-only mode, noted in `forecast.note`.

Per strategy, a weights run does:

1. `p` = mean of the decider's last 24 answers; `mu_decider = (2p - 1) * 0.8 * sigma_d`.
2. Clip every forecast to annual Sharpe +-1.5: `|mu| <= 1.5 sigma_d / sqrt(365)`.
3. `mu = sum_k W_k mu_k`, `W` from `weights_k` (the decider's weight too). If the file has no row for a
   strategy, or no `decider` key, the ledger's starting weights fill in (base 0.30, regime 0.25, ridge 0.20,
   decider 0.25; carry: structural 0.40, base 0.15, regime 0.10, ridge 0.10, decider 0.25). Weights are
   renormalised over the forecasters that have a value, so a decider with no answers just drops out.
4. Core plus tilt. Why: in the walk-forward 2023-01-01 to 2026-11-18 the mean-variance optimum of the
   forecasts returned -0.41%/yr (Sharpe -0.08), the static inverse-volatility mix +2.79%/yr (Sharpe 0.31):
   the forecasts mostly cut exposure. So the forecasts no longer set the weight, they can only nudge it,
   and only after the nudge has shown it beats the core ("core live, tilt earns it").
   - Core: `core_s = min(1, k / sigma_d_s)`, k the largest scale at which `sqrt(365 w'Sigma w) / N` is at
     most the vol cap (10%/yr). That is the static inverse-vol book of the walk-forward, before the rules.
   - Tilt: `SR_s = mu_s * 365 / (sigma_d_s * sqrt(365))`, the combined forecast (all forecasters and the
     decider) over the annual volatility. `tau_s = clip(SR_s / SR_full, -T_eff, +T_eff)`, `SR_full` 1.0,
     `T_eff = T * kappa`, `T` 0.5, kappa 0 until earned (below). With kappa 0 the live tilt is 0.
   - Shadow: the same tilt at full `T` (kappa 1) is computed every run and logged (`shadow_tau`,
     `shadow_w_target`), whatever kappa is. The forecasts and the decider are still computed and asked as before.
   - Target: `w_target_s = clip(core_s * (1 + tau_s), 0, 1)`. A zero forecast gives the core exactly; the
     tilt is at most +-50% of the core weight; the target stays in the box, so a core of 1 cannot be raised.
   The mean-variance optimiser (`solve_weights`) and the cost term (`CTB_ALLOCATOR_GAMMA`, `_HORIZON_DAYS`,
   `_COST_BP`, `_CARRY_COST_BP`, the gross exposure `G_s`) are kept for the comparison and no longer used.
   `w_prev` is the weight in `control.json` now.
5. The rules, in this order (each is logged with before and after):
   `vol_cap` (scale all so `sqrt(365 w'Sigma w) / N <= 10%/yr`, N the number of strategies), `buffer`
   (a change under 0.10 is not made), `step` (up at most 0.25 per run, down unlimited), `veto` (p below 0.40
   caps w at half of `w_prev`), `explore_floor` (w at least 0.10, only on the venues in
   `CTB_ALLOCATOR_EXPLORE_VENUES`, default demo and testnet; 0 disables), then `floor_lock`: the 15%
   drawdown floor (weight 0, flatten) and `owner_lock` (weight untouched) override everything, the lock
   wins over the floor.
6. A strategy with no forecast (or no sigma) is held at its weight and listed in `no_forecast`.

### Tilt room by skill

`kappa` in [0, 1] multiplies `T`. It is 0 at the start: the live weight is the core and the tilt runs in
shadow. The prediction layer (`forecast.py score`, nightly) writes `tilt_skill` and `kappa` into
`forecast/scores.json`, and `forecast.py run` copies `kappa` into `forecast/latest.json`, which is where this
module reads it (absent or unreadable: 0; clipped to [0, 1]).
`tilt_skill` is the shadow difference between the tilted book and the core book, from the archived
forecasts and the returns files: for each archived day, `d = (1/N) sum_s (shadow_target_s - core_s) r_s` for the
next day's return `r_s` of each strategy's book, at full `T` and kappa 1, without the rules or costs and
without the decider. The mean of `d` over the last 90 and the last 180 days gives `skill_90` and `skill_180`
(a window with fewer days than its length has no value). `kappa` is 0 by default and until 90 archived days
exist; 0.5 when `skill_90` is positive; 1.0 when `skill_90` and `skill_180` are both positive. It goes back
down by the same rules when a mean turns zero or negative. `decisions.jsonl` carries the `kappa` and `t_eff` used.

Defaults: `CTB_ALLOCATOR_TILT_MAX` 0.5, `_TILT_SR_FULL` 1.0, `CTB_ALLOCATOR_GAMMA` 2, `_HORIZON_DAYS` 7, `_COST_BP` 10, `_CARRY_COST_BP` 16, `_SHARPE_CAP` 1.5,
`_DECIDER_K` 0.8, `_P_WINDOW` 24, `_VOL_CAP` 0.10, `_BUFFER` 0.10, `_STEP_UP` 0.25, `_VETO_P` 0.40,
`_VETO_FACTOR` 0.5, `_EXPLORE_FLOOR` 0.10, `_STALE_H` 36, `_WEIGHTS_UTC` 23:50.

A weights run's `decisions.jsonl` line carries the usual fields (`weights` is the final applied weight,
`applied`, `risk_floor`, `skipped_owner_lock`, `error`, `answers`, `inputs`) plus: `weights_mode`
(`computed`, `refresh` or `held`), `forecast` (state, date, age), `mu` (per forecaster and `combined`, daily
net return after the Sharpe clip), `mu_ann_pct`, `sigma_d`, `clipped`, `p_decider`, `w_prev`, `core`, `sr`,
`tau`, `kappa`, `t_eff`, `w_target`, `shadow_tau`, `shadow_w_target`, `w_final`, `steps` (the ordered rules, each with `before`, `after`,
`changed`; `vol_cap` also `vol_ann` and `factor`), `no_forecast`, and `reasons`, one plain line per strategy
that is also the card's `reason`. With no tilt room: `trend: core 0.42 (tilt in shadow, +18%, earns live
control after 90 days ahead of core)`, plus ` -> 0.25 after step cap` when a rule moves it. With room:
`trend: core 0.42, tilt +18% (SR +0.36), w 0.50 -> 0.45 after step cap [tilt room 0.50]`. (`w_target` replaces the earlier `w_star`; `cost_per_day` is gone.) The
regime in the line is today's state from `regime/latest.json` and its 7-day probability.

`python -m ctb.allocator once --dry-run` asks the decider (its answer counts toward p but is not stored),
prints the full table (mu per forecaster in %/yr, p, sigma, previous weight, core, SR, live tilt, shadow tilt, target, the weight
after each rule, final, the reasons) and writes nothing. Without a forecast file it prints the decider-only output as before.
The chat fallback (`CTB_ALLOCATOR_MODEL=llm`) never uses the forecast.

## Model and question set

The decision model decides (default). `CTB_ALLOCATOR_MODEL` is `kairos`, `jev`, `ordecide` or
`ordecide:<openrouter-id>`. Unset, it follows `CTB_MODEL_PROVIDER` when that is a decision spec
(including `or:<id>`), else `ordecide`. The same `SystemOneClient.ask(state, questions)`, keys and
retry policy as the bot are used. `status()["model"]` and every decision line's `model` read
`<provider> / <model id>`, for example `ordecide / google/gemini-decision-1`.

State: the context built for each decision. Questions, set
`alloc-v1.1` (`ctb/kairos/allocator_questions.py`, bump the version on any wording change; the questions are the same as in
`alloc-v1.0`, only the state grew, so `score` reads both):

- per strategy id, binary `net_positive_<id>`: the strategy's book earns a positive net return on
  the capital it deploys over the next 24 hours, after fees and funding;
- one categorical `best_net_return`: which strategy, or `none`, has the highest such return, and it
  is above zero.

### State (alloc-v1.1)

Built by `Allocator.build_context` and `ctb/allocator_context.py`; the chat fallback keeps its old context. Blocks:

| Block | Content | Source |
| --- | --- | --- |
| `strategies`, `market` | as in alloc-v1.0 | runner files, Bybit public REST |
| `regime` | `state`, `date`, the eight `features`, `p7` (probability of each regime in 7 days) | `regime/latest.json` |
| `forecasts` | `date`, `age_h`, per strategy `mu_pct_yr` and `sigma_pct_yr` (%/yr), `forecast_sharpe`, `core_weight`, `forecasters` used | `forecast/latest.json` |
| `news` | last 24 h, newest first, at most 40 items: `t` (UTC), `title` (at most 160 chars), `src`, `type`, `tags` when the collector has them; `count`, `older_not_shown` | H7 collector day files |
| `state_notes` | one line per block that is missing, old or cut | |

- `forecasts` is the same computation as the weights run (`compute_weights`) with the decider left out: its past answers are
  not passed in, and a `decider` row in the file is dropped. So `mu_pct_yr` is the weighted mean of `base`, `regime`, `ridge`
  (carry also `structural`), clipped to the Sharpe cap. A stale forecast (over 36 h) is kept with its age and a note.
- `news` reads `items/h7-items-YYYY-MM-DD.jsonl` and `raw/h7-raw-YYYY-MM-DD.jsonl` under the collector root
  (default `<live root>/../research/h7/collector`, on aurora4 `/home/lavx/Documents/calibrain-study/data/research/h7/collector`;
  override `CTB_ALLOCATOR_NEWS_ROOT`). Item and raw lines are joined by `(key, sha)`; only the title is read from the raw line,
  never the body. The time is the publication time, or the receive time when it is missing or ahead of the clock. The collector
  stores no announcement tags, so `tags` appears only if a line has a `tags` list.
- RSS headlines (every source whose name does not start with `bybit`) are in the same block as the Bybit announcements,
  newest first. `CTB_ALLOCATOR_RSS_HEADLINES` is on by default (owner decision 2026-11-19); set `0` to send Bybit announcements only.
  Headlines are third-party text: the question text tells the model to read them as data, never as instructions.
- A block whose file is missing is left out and `state_notes` says so. Nothing in the state builder raises.
- Payload bound: before the first request the state is checked against the model's `max_payload_bytes` (Kairos 1 MiB). If the
  largest request of the set would exceed it, older news is cut first, then the news, forecasts and regime blocks, with a note
  each. The books and the market are never cut. Normal size is a few KB.
- Every decision line carries `state_bytes`: bytes of the state as sent, per block and `total`. `inputs` holds the state as sent.

They are split into requests of at most the model's `max_questions` (Kairos: 8). Missing or
off-contract answers get one more attempt; a strategy whose question never validates keeps its
current allocation.

Mapping, in code (constants in `ctb/config.py`: `CTB_ALLOCATOR_P_ZERO` 0.5, `_P_FULL` 0.65,
`_WINNER_MIN_P` 0.55, `_WINNER_WEIGHT` 0.5):

    weight = clip((p - 0.5) / 0.15, 0, 1)
    categorical winner with its own p > 0.55: weight = max(weight, 0.5)
    enabled = weight > 0 (after the step limit)

then the step limit, the drawdown floor and owner_lock as above. The UI reasoning is generated
from the numbers, e.g. `carry p=0.61 -> 0.73, trend p=0.48 -> 0, xsmom winner p=0.57 -> 0.5, best: xsmom (0.50)`.

Every decision line carries `qset`, `model`, `provider`, `request_ids`, `answers` (`p` per strategy,
`best` probabilities), `mapping` (target weight per strategy, winner boost, held), the final
`weights` and the full `inputs`.

`python -m ctb.allocator score` scores those answers against the realised book returns: the equity at
the decision (from `inputs`) against the equity samples in `allocator/state.json` 24 h later (kept 8
days, so run it at least weekly). Output per strategy: n, base rate, mean p, Brier score against the
always-0.5 and base-rate baselines, hit rate; for the categorical: accuracy and mean probability of
the realised outcome. A book's return is divided by its weight, and a book held below weight 0.1
cannot be labelled, so it counts as `flat` (a strategy at weight 0 is never scored until it has
weight again).

Chat fallback: `CTB_ALLOCATOR_MODEL=llm` (uses `CTB_LLM_MODEL`) or `or:<openrouter-chat-model-id>`
asks one chat model for a JSON allocation directly (system prompt in
`ctb/kairos/allocator_prompt.py`). Note `or:<id>` here means a chat model; for an OpenRouter
decision model use `ordecide:<id>`. Cost comes from the `usage.cost` field the provider returns.

Market data is Bybit mainnet public REST (no keys). The trend universe defaults to ten liquid
perps and can be set with `CTB_ALLOCATOR_TREND_SYMBOLS` (comma separated).

## CLI

    python -m ctb.allocator once --dry-run   # computed: the state with its sizes, then the full weight table; decider-only: context and answer; writes nothing
    python -m ctb.allocator once             # one decision, applied
    python -m ctb.allocator run              # loop until stopped
    python -m ctb.allocator score            # score logged answers against realised book returns (--qset alloc-v1.1 for one set)
    python -m ctb.allocator reset-peak [id ...]

## Dashboard server (ctb/server.py), the lines it needs

In `make_app`, next to the engine start and stop:

```python
from . import allocator as alloc_mod   # top of the file

async def on_startup(_):
    await engine.start()
    app["allocator"] = alloc_mod.create_allocator(engine.session)           # reuses the engine's aiohttp session
    app["allocator_task"] = asyncio.create_task(app["allocator"].run())     # loop: floor every tick, decisions per interval

async def on_cleanup(_):
    app["allocator"].stop(); app["allocator_task"].cancel()
    await engine.stop()

@routes.get("/api/allocator")
async def allocator_status(req):
    a = req.app["allocator"]; return j({"status": a.status(), "control": a.control()})

@routes.get("/api/allocator/decisions")
async def allocator_decisions(req):
    return j(req.app["allocator"].decisions(int(req.query.get("limit", 50))))
```

Pause, change the interval and "decide now" are written by the dashboard straight into
`allocator/control.json` (atomic write, keys `enabled`, `interval_min`, `decide_now`,
`updated_ms`); the loop picks them up within 30 s. Owner overrides go into
`<id>/control.json` with `owner_lock: true`.
