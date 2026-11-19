# LLM strategy allocator

`ctb/allocator.py` lets the language model decide which live strategies run, how much of each
strategy's virtual book is deployed, and when. It places no orders: it only writes the allocation
fields of each strategy's `control.json`, which the live runner reads every loop.

Layout under the live root (`CTB_LIVE_ROOT`, default `data/live`; on aurora4
`/home/lavx/Documents/calibrain-study/data/live`):

| File | Written by | Content |
| --- | --- | --- |
| `strategies.json`, `<id>/status.json`, `<id>/trades.jsonl` | runner | what the allocator reads |
| `<id>/control.json` | dashboard, allocator | allocator sets `enabled`, `weight`, `set_by: "llm"`, `reason`; `owner_lock: true` is never touched |
| `allocator/control.json` | dashboard | `{"enabled", "interval_min", "decide_now"}` (allocator clears `decide_now`) |
| `allocator/status.json` | allocator | `state`, `model`, `last_decision_ms`, `next_decision_ms`, `last_error`, `cost_usd_today` (plus `interval_min`, `last_reasoning`, `last_confidence`) |
| `allocator/decisions.jsonl` | allocator | one line per decision, risk-floor action or failed attempt |
| `allocator/state.json` | allocator | equity samples and peaks for the P&L windows and the drawdown floor |

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

    python -m ctb.allocator once --dry-run   # print the context and the model's answer, write nothing
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
