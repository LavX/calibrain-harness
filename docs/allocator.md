# Strategy allocator

`ctb/allocator.py` decides which live strategies run, how much of each strategy's virtual book is
deployed, and when. It places no orders: it only writes the allocation fields of each strategy's
`control.json`, which the live runner reads every loop.

There are two ways it sets weights:

- **Computed** (when `forecast/latest.json` exists): weights come from the prediction layer's
  forecasts by a mean-variance optimum plus fixed rules, see "Computed weights" below. The decision
  model is one forecaster among several and still has a veto.
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
`weights_k` (each forecaster's weight, from the daily scoring). The decider is not in the file: the
allocator adds it. The math is in `ctb/allocator_math.py` (pure, no I/O); the constants are in
`ctb/config.py` (`CTB_ALLOCATOR_*`, defaults below).

Schedule:

- Every hour (`interval_min`) the decider is asked the same `alloc-v1.0` questions. The answers go into
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
4. `w* = argmax_{0<=w<=1} sum w_s mu_s - (gamma/2) w'Sigma w - sum c_s G_s |w_s - w_prev_s| / H`, gamma 2,
   H 7 days, c 10 bp per side (carry 16 bp). Projected (proximal) gradient in pure Python, fixed start
   (`w_prev`), no randomness. Alone and unconstrained, `w = mu / (gamma sigma^2)`. `G_s` is the book's gross
   exposure per unit of capital: `status.json` positions (qty times mark) over equity, divided by the
   current weight because the runner scales targets by it; 1 when the book is flat, unreadable, or its
   weight is under 0.10. `w_prev` is the weight in `control.json` now.
5. The rules, in this order (each is logged with before and after):
   `vol_cap` (scale all so `sqrt(365 w'Sigma w) / N <= 10%/yr`, N the number of strategies), `buffer`
   (a change under 0.10 is not made), `step` (up at most 0.25 per run, down unlimited), `veto` (p below 0.40
   caps w at half of `w_prev`), `explore_floor` (w at least 0.10, only on the venues in
   `CTB_ALLOCATOR_EXPLORE_VENUES`, default demo and testnet; 0 disables), then `floor_lock`: the 15%
   drawdown floor (weight 0, flatten) and `owner_lock` (weight untouched) override everything, the lock
   wins over the floor.
6. A strategy with no forecast (or no sigma) is held at its weight and listed in `no_forecast`.

Defaults: `CTB_ALLOCATOR_GAMMA` 2, `_HORIZON_DAYS` 7, `_COST_BP` 10, `_CARRY_COST_BP` 16, `_SHARPE_CAP` 1.5,
`_DECIDER_K` 0.8, `_P_WINDOW` 24, `_VOL_CAP` 0.10, `_BUFFER` 0.10, `_STEP_UP` 0.25, `_VETO_P` 0.40,
`_VETO_FACTOR` 0.5, `_EXPLORE_FLOOR` 0.10, `_STALE_H` 36, `_WEIGHTS_UTC` 23:50.

A weights run's `decisions.jsonl` line carries the usual fields (`weights` is the final applied weight,
`applied`, `risk_floor`, `skipped_owner_lock`, `error`, `answers`, `inputs`) plus: `weights_mode`
(`computed`, `refresh` or `held`), `forecast` (state, date, age), `mu` (per forecaster and `combined`, daily
net return after the Sharpe clip), `mu_ann_pct`, `sigma_d`, `clipped`, `p_decider`, `w_prev`, `w_star`,
`w_final`, `steps` (the ordered rules, each with `before`, `after`, `changed`; `vol_cap` also `vol_ann` and
`factor`), `cost_per_day`, `no_forecast`, and `reasons`, one plain line per strategy that is also the card's
`reason`, e.g. `trend: mu +6.1%/yr (regime up_calm 0.62), sigma 18%, w* 0.41 -> 0.35 after step cap`. The
regime in the line is today's state from `regime/latest.json` and its 7-day probability.

`python -m ctb.allocator once --dry-run` asks the decider (its answer counts toward p but is not stored),
prints the full table (mu per forecaster in %/yr, p, sigma, previous weight, w*, the weight after each rule,
final, the reasons) and writes nothing. Without a forecast file it prints the decider-only output as before.
The chat fallback (`CTB_ALLOCATOR_MODEL=llm`) never uses the forecast.

## Model and question set

The decision model decides (default). `CTB_ALLOCATOR_MODEL` is `kairos`, `jev`, `ordecide` or
`ordecide:<openrouter-id>`. Unset, it follows `CTB_MODEL_PROVIDER` when that is a decision spec
(including `or:<id>`), else `ordecide`. The same `SystemOneClient.ask(state, questions)`, keys and
retry policy as the bot are used. `status()["model"]` and every decision line's `model` read
`<provider> / <model id>`, for example `ordecide / google/gemini-decision-1`.

State: the context built for each decision (strategy books and market summary). Questions, set
`alloc-v1.0` (`ctb/kairos/allocator_questions.py`, bump the version on any wording change):

- per strategy id, binary `net_positive_<id>`: the strategy's book earns a positive net return on
  the capital it deploys over the next 24 hours, after fees and funding;
- one categorical `best_net_return`: which strategy, or `none`, has the highest such return, and it
  is above zero.

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

    python -m ctb.allocator once --dry-run   # computed: the full weight table; decider-only: context and answer; writes nothing
    python -m ctb.allocator once             # one decision, applied
    python -m ctb.allocator run              # loop until stopped
    python -m ctb.allocator score            # score logged answers against realised book returns
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
