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
- Invalid model output (schema, ranges, unknown ids) is retried once with the problem stated, then
  the current allocation is kept and the error is logged.
- `owner_lock: true` is respected, including a lock set between reading and writing.
- A model request budget (`CTB_ALLOCATOR_MAX_RPH`, default 12 per hour) survives restarts.

## Model

Any chat model: `CTB_ALLOCATOR_MODEL` is `llm` (default, uses `CTB_LLM_MODEL`) or
`or:<openrouter-model-id>`. The key and endpoint come from the same `OPENROUTER_API_KEY` and
`OPENROUTER_BASE_URL` as the rest of the harness. Decision models (Kairos, Jev, `ordecide`)
answer typed questions and cannot write an allocation, so they are refused. Cost comes from the
`usage.cost` field the provider returns.

Market data is Bybit mainnet public REST (no keys). The trend universe defaults to ten liquid
perps and can be set with `CTB_ALLOCATOR_TREND_SYMBOLS` (comma separated).

## CLI

    python -m ctb.allocator once --dry-run   # print the context and the model's answer, write nothing
    python -m ctb.allocator once             # one decision, applied
    python -m ctb.allocator run              # loop until stopped
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
