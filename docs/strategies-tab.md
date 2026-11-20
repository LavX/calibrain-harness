# Strategies tab

The STRATEGIES screen (F7, or type `STRATEGIES`) shows and controls the live strategies that trade on the Bybit
**demo** account. The strategies are not run by this dashboard. A separate live runner (a systemd user unit) trades
them and writes plain files; the dashboard reads those files and writes only `control.json` files plus start/stop of
the runner unit. It never reads a credential and never places an order itself.

## Modes (header)

The header has a mode selector next to AUTO-TRADE. Exactly one mode trades the demo account at a time.

| Mode | Who decides | AUTO-TRADE switch |
| --- | --- | --- |
| LLM DECIDES | the LLM allocator (`ctb/allocator.py`) picks which strategies trade and their weight | on: allocator on and runner started; off: allocator paused and every strategy off (positions stay open, but the drawdown floor still closes a book 15% below its peak, except books you switched on by hand, which the floor does not touch) |
| MANUAL | you, per card (the allocator is paused) | on: the strategies ticked "in header set" are switched on (owner-locked) and the runner started; off: all off. The drawdown floor does not touch owner-locked books, so in MANUAL it protects nothing the header switch or a card turned on |
| PER-SYMBOL BOT | the older per-symbol LLM bot | the existing switch (`/api/trading`) |

With no `strategies.json` and no allocator the selector is hidden and the header behaves as before.

Changing the mode asks for confirmation and pauses the other side first:

* to PER-SYMBOL BOT: allocator and every strategy are paused and the runner unit is stopped. Refused while a strategy
  holds a position (FLATTEN it first), so the bot never inherits strategy positions on the same account.
* away from PER-SYMBOL BOT: the bot's AUTO-TRADE is turned off. Refused while the bot has open trades.
* LLM DECIDES to MANUAL: the allocator is paused, strategies stay as they are.

The guard only applies on the demo venue (the testnet venue is another host and another account). A strategy conflicts
with the bot when its `account` in `strategies.json` equals `CTB_DASHBOARD_ACCOUNT`, or when that variable is unset.
A background check every 15 s also turns the bot's AUTO-TRADE off if the runner unit is started behind the dashboard's
back (for example with `systemctl`). After a dashboard restart the mode is LLM DECIDES if the allocator is switched on,
MANUAL if strategies are configured, otherwise PER-SYMBOL BOT.

## Card controls

* **AUTO TRADE** switch: `control.json` `enabled`. On: the runner places that strategy's orders. Off: it stops placing
  new orders and keeps its positions. Turning it on also starts the runner unit if it is not running. Flipping it by hand
  sets `owner_lock: true` and `set_by: "owner"`; the allocator must not change a locked strategy. **UNLOCK TO LLM** clears
  the lock (`owner_lock: false`, `set_by: "llm"`). An owner "on" on a strategy with weight 0 sets weight 1.
* **FLATTEN** (confirm step): `flatten: true`. **RESET LATCH** (confirm step, only when latched): `reset_latch: true`.
  Both need the runner running, otherwise nobody would act on them and the flag would fire at the next start.
* The card shows state, equity, P&L (USDT and %), fees, funding, positions with unrealised P&L, targets, recent fills,
  the last and next rebalance, the LLM's weight and reason, and an equity sparkline. The sparkline lives in the
  dashboard's memory (one sample a minute, last 12 hours), so it starts empty after a dashboard restart.

## Allocator panel

State, model, last and next decision, interval (change), cost today, a DECIDE NOW button (confirm step) and the feed of
the last 15 decisions (time, weight per strategy, reasoning, strategies skipped because of an owner lock). It reads
`<root>/allocator/status.json` and `decisions.jsonl` (through the running allocator's `status()`, `control()` and
`decisions()`) and writes `<root>/allocator/control.json` (`enabled`, `interval_min`, `decide_now`). The decision loop is
`ctb.allocator` (see `docs/allocator.md`): the server creates it on the engine's session at startup and stops it at
cleanup. The loop treats a missing `allocator/control.json` as enabled, so the server writes that file from the mode
before the loop starts, and the mode switch keeps it current: **the allocator only writes strategy controls in LLM
DECIDES** (control `enabled: true`). If no allocator can be created (no model configured) the panel says "not installed"
and LLM DECIDES refuses to switch on. DECIDE NOW is picked up by the loop within its 30 s tick. The feed shows failed
attempts and risk-floor actions too.

## Predictions panel and scoreboard

Under the allocator panel, from `GET /api/predictions` (read-only, cached 30 s, `ctb/predictions.py`). It reads, under the
live root: `regime/latest.json`, `forecast/latest.json`, `forecast/scores.json`, `forecast/archive/<day>.json`,
the newest `allocator/decisions.jsonl` line that carries the `core` breakdown, `returns/<id>.csv`, and each strategy's
`status.json` and `control.json`. A missing or damaged file blanks only its own section, which then says "not computed yet".

- PREDICTIONS: the regime now with the 8 features and a one-line meaning each, the 7-day state probabilities, and per
  strategy the core weight, the shadow tilt (tau) and forecast Sharpe, the expected net return in %/yr of each forecaster
  (base, regime, ridge, structural, decider) and combined, risk in %/yr, the decider's mean p over 24 h, the live weight and
  the allocator's one-line reason. A forecast older than 36 h is flagged (the allocator holds weights then).
- SCOREBOARD: per strategy and forecaster the loss L and weight W, the decider's Brier score against the base rate and the
  number of days scored, kappa and the 90 and 180-day tilt skill, and the line "tilt live control: off until the shadow tilt
  beats the core over 90 days (n/90 days so far)". Below it live P&L (`status.json`) next to the modelled shadow return of
  `returns/<id>.csv` from the go-live day (2026-11-19, `CTB_GO_LIVE_DATE` overrides it), at weight 1 and at the live weight,
  and a chart of the cumulative return of the core book against the shadow-tilted book (full tilt, no rules, no costs, no
  decider, the series the tilt skill is judged on). The chart needs a forecast archived on or after go-live plus the next
  day's returns, so it starts empty.

## Venues (several exchanges)

One runner per exchange account, each with its own root and systemd user unit. The list comes from configuration, never
from a directory glob (a quarantined root next to a real one stays invisible):

```
CTB_LIVE_ROOTS=bybit-demo=<root>,okx-demo=<root>,deribit-test=<root>,bitget-demo=<root>
```

| Venue id | Label | Unit | Currency | Controlled by |
| --- | --- | --- | --- | --- |
| bybit-demo | BYBIT·DEMO | calibrain-live.service | USDT | the LLM allocator |
| okx-demo | OKX·DEMO | calibrain-live-okx.service | USDT | the LLM allocator |
| deribit-test | DERIBIT·TEST | calibrain-live-deribit.service | USDC | the LLM allocator |
| bitget-demo | BITGET·DEMO | calibrain-live-bitget.service | USDT | the LLM allocator |

Every venue is the LLM allocator's unless `CTB_ALLOCATOR_VENUES` (comma-separated venue ids, default every configured
venue) leaves it out; a left-out venue is the owner's. The allocator makes one decision per strategy (on bybit-demo's
books, forecasts and decider) and applies it to each venue it controls whose runner lists that strategy, under that
venue's own risk rules: the 10% vol cap over that venue's strategies, the 15% drawdown floor per (venue, strategy) book,
the +0.25 daily step from a day anchor per (venue, strategy), the exploration floor per (venue, strategy), and on every
write the owner lock, a pending kill and the allocator's pause (only risk-reducing writes while paused). The header
AUTO-TRADE switch and the mode apply to every allocator venue: OFF pauses their strategies, ON in LLM DECIDES starts
their runners (ON in MANUAL switches only the bybit-demo header set). KILL stays per venue. F7's HAND TO ALLOCATOR
(`POST /api/venues/{vid}/hand-to-allocator`, body `{"cancel_kill": true}` to cancel pending kills) clears the owner
locks on a venue's strategies so the allocator decides them at its next weights run. A strategy without an owner lock
is the allocator's from its first tick (the drawdown floor can cut it at once); see docs/allocator.md "Venues" for the
step from the day anchor and the vol cap, which uses bybit-demo's forecast for every venue (an approximation elsewhere).

bybit-demo must be listed: the allocator, the predictions panel and the per-symbol bot's guard stay bound to it, and
`/api/strategies` (and its actions) are aliases of `/api/venues/bybit-demo/...`. Unset, the dashboard has one venue,
bybit-demo at `CTB_LIVE_ROOT`, as before. Each venue has a presence: **live** (unit active, a status.json fresher than
3 min), **stale** (unit active, every status.json older), **stopped** (unit not active), **absent** (no root or no
strategies.json, an empty one, or no known unit and no status.json yet: "not started") and **error** (strategies.json unusable).
Two venues may not share a root (refused at startup, like a malformed entry). A strategy whose own unit is down while the
venue is up raises a "unit down" alert. Unit start times are read as unix seconds and shown in UTC.

Routes: `GET /api/venues` (every venue: presence, books, account, P&L today, strategies, netted positions, alerts),
`GET /api/venues/{vid}/strategies`, `POST /api/venues/{vid}/strategies/{sid}/{enable|disable|pause|unlock|flatten|reset}`,
`POST /api/venues/{vid}/runner/{start|stop}`, `POST /api/venues/{vid}/kill`, `POST /api/kill-all`.

KILL always records the owner's intent: it pauses the allocator (bybit-demo), and pauses every strategy of the venue
(owner-locked) with its flatten flag set, every strategy, also one whose status shows it flat: a status can predate
a fill or a working order (it lists no orders at all). The answer names a strategy "already flat" only when its runner is
confirmed live and its own fresh status, written after the runner saw the pause (`enabled: false`), lists positions as an
empty list and no non-zero target; the flag is still written and that runner clears it on its next loop. A venue that is
not live never answers 200. Enabling a strategy with a pending flag always needs the owner's cancel, whatever its status
shows. It then answers 200 only when every written strategy's runner unit is active and
its status.json is fresh (checked per strategy, since units can be per strategy). Otherwise it answers 409 "written, NOT
confirmed" naming the strategies nobody may be acting for yet: a hung or restarting runner reads the flags when it comes
back. A failed write answers 500 with what was and was not written. The flags are written before any systemd call (only
strategies.json and the status files are read first, to tell a venue that has not started), and a strategy with no known runner unit is never reported confirmed. Neither is a latched strategy: the runner skips a
latched strategy before it reads `flatten`, so its kill runs only after the latch is reset (the answer, the preview and
a critical alert say so). A strategy listed in
strategies.json without a directory is named as not flagged. F1's KILL button applies the same per-strategy test, so its
preview and the answer agree. Only a venue that has not started, or whose
strategies.json is unusable, is refused with nothing written. KILL ALL does every venue and reports each one (a venue
that has not started is skipped) and then switches the per-symbol bot's AUTO-TRADE off. In mode PER-SYMBOL BOT it also
kills the bot (every position on its account closed, orders cancelled); bybit-demo's strategies are killed as in every
mode, because that mode pauses them and stops their runner only when the bot trades the same account. In the other modes the bot is not touched on the exchange: its kill closes
every position on the account, which would close the strategies' netted legs behind their runner. The answer reports
the bot's part on its own. `/api/kill` (the old header
button) is the bybit-demo kill in LLM DECIDES and MANUAL.

A pending kill (`flatten: true`) ends in one of two ways only: the runner, once flat, clears the flag itself
(`research/live/runner.py` `_flatten_clear`: it writes `enabled: false, flatten: false`), or the owner cancels it. The
card switch, the MANUAL header switch and UNLOCK TO LLM answer 409 "a kill is pending" for such a strategy unless the
request carries `cancel_kill: true`, which the page sends only after a confirm that says that, if the runner has not
started closing yet, the positions will not be closed (closes already sent still land). The card also has CANCEL KILL (`/strategies/{sid}/cancel_kill`: flatten cleared, and the strategy paused and
owner-locked, also when it was an enabled strategy's FLATTEN request). The MANUAL header switch takes `cancel_kill` as the
list of ids the confirm named; any other target with a pending kill is refused, also when the kill lands while the runner
is being started (rechecked at the write).
The allocator never switches on a strategy whose control.json has `flatten: true` (it logs why and skips it). A decision
that was still running when the allocator was paused (AUTO-TRADE OFF, or a switch to MANUAL or PER-SYMBOL BOT) writes only
what reduces risk (switching off, a lower weight); the decision feed marks the rest "allocator paused during the decision". The "kill
pending" alert shows only while the book still holds positions (or is switched on); flat and paused is done.
UNLOCK TO LLM is refused on an owner-controlled venue (no allocator).

P&L today uses the harness's own 00:00 UTC baseline, one small JSON file per venue under the harness data directory
(next to the database: `data/venues/baseline-<vid>.json`). A background task samples every minute, so the file is
written within a minute of midnight while the dashboard runs and survives restarts. Every P&L today figure carries the
real time of its baseline: "since 00:00 UTC" only for a sample taken in the first 90 s of the UTC day, else "since
HH:MM UTC" (a 00:04 sample reads 00:04). A strategy first seen later in the day gets its own baseline. Only a fresh status (written
in the last 3 minutes) seeds a baseline: a runner stopped or hung across midnight would otherwise carry yesterday's
equity into today, so such a strategy is counted from its first fresh status. A venue whose strategies' baselines differ
is marked partial with the reason (the harness's first sample of the day was late, a strategy's status was stale until
later, a strategy appeared later and is counted from then, or a strategy has no fresh status yet). There is no account-level P&L today on bybit-demo: the USDT coin's
equity drops when a spot leg is bought with USDT, which is not a loss. The account line shows the USDT equity apart from
all collateral, and "unattributed" compares like with like: the exchange's derivatives unrealised P&L minus the books'
perp and future legs (markets linear, inverse and future; spot excluded) of the strategies that trade that account
(`CTB_DASHBOARD_ACCOUNT`); it is unknown when any leg's market or uPnL is unknown. The exchange's positions feed holds
USDT linear positions only, so the positions panel keeps the books' spot legs next to it, and the books' legs of
strategies on another registered account, labelled with that account. The account comes from the
dashboard's own exchange read (the per-symbol bot's executor), which keeps its last snapshot when a sync fails: on a sync
error or no good read for 60 s the line is marked STALE with its age, unattributed is not computed, the positions panel
falls back to the books' legs (and says so), and an "account stale" alert is raised. Other venues show sums over their
books until the runner publishes `<root>/account.json` (read when it exists). The MANUAL header set is kept in the same directory
(`auto-set-bybit-demo.json`), not in the browser.

## Files (the contract with the runner)

Root: `CTB_LIVE_ROOT`, shared with `ctb.allocator` (default `data/live` in the repo; on aurora4 the unit sets `/home/lavx/Documents/calibrain-study/data/live`).

| File | Written by | Read by |
| --- | --- | --- |
| `strategies.json` (id, label, unit, account) | runner, at start | dashboard |
| `<id>/status.json` (at least every 60 s) | runner | dashboard |
| `<id>/trades.jsonl` | runner | dashboard (last 12 fills) |
| `<id>/control.json` (enabled, flatten, reset_latch, weight, set_by, owner_lock, reason) | dashboard and allocator (one lock per file in the harness); the runner clears flatten and enabled when a flatten is done (not under that lock) | runner |
| `allocator/status.json`, `allocator/decisions.jsonl` | allocator | dashboard |
| `allocator/control.json` (enabled, interval_min, decide_now) | dashboard | allocator |

A status older than 3 minutes while the unit is active shows as STALE. `allocator` is reserved and cannot be a
strategy id. Writes to `control.json` are atomic (temp file and rename) and keep keys the dashboard does not know.

Runner start/stop: only unit names found in `strategies.json` (or listed in `CTB_LIVE_UNITS`, for the very first start
before the runner has written the file) are accepted, each validated as a plain `.service` name, and run as
`systemctl --user <action> -- <unit>` with an argument list and no shell. Child processes get a minimal environment
(`PATH`, `HOME`, `XDG_RUNTIME_DIR`), never the dashboard's own.

## Deploy on aurora4

The dashboard must run where the units live. aurora4 has no Node, so build the UI export on a machine with Node and
copy it (`web-next/out` is not committed).

```sh
# on the laptop: build the UI
cd web-next && npm ci && npm run build && cd ..

# put the code on aurora4 (branch lavx once this PR is merged, or feat/study-tab before)
ssh lavx@aurora4 'test -d ~/Documents/calibrain-harness || git clone https://github.com/LavX/calibrain-harness.git ~/Documents/calibrain-harness'
ssh lavx@aurora4 'cd ~/Documents/calibrain-harness && git fetch origin && git checkout lavx && git pull --ff-only'
rsync -a --delete web-next/out/ lavx@aurora4:Documents/calibrain-harness/web-next/out/

# on aurora4
ssh lavx@aurora4
cd ~/Documents/calibrain-harness
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p ~/.config/systemd/user
cp docs/deploy/calibrain-harness.service ~/.config/systemd/user/
# edit CTB_HOST (tailnet address), CTB_DASHBOARD_ACCOUNT and CTB_LIVE_UNITS in the copy, then:
systemctl --user daemon-reload
systemctl --user enable --now calibrain-harness.service
systemctl --user status calibrain-harness.service
```

Open `http://100.96.108.128:8800/` (tailnet) or tunnel with `ssh -L 8800:127.0.0.1:8800 lavx@aurora4` and open
`http://127.0.0.1:8800/`. aurora4 has `Linger=yes`, so the user unit survives logout. Keys for the per-symbol bot
go in the repo's `.env` on aurora4 (the STRATEGIES tab itself needs none).

Trading stays off: `CTB_TRADING_ENABLED=0` in the unit and the switch starts off in the UI. `CTB_VENUE` decides which
account the per-symbol bot would use (`testnet` default); the strategies always use the demo account through the runner.

## Local preview

`scripts/preview_strategies.py` copies the fixture files in `tests/fixtures/live` to a temp directory with fresh
timestamps (plus `tests/fixtures/predictions` for the predictions panel) and starts the dashboard against them (no keys, no runner, observe-only). Press F7. The runner start/stop
buttons answer "not installed" there because no such unit exists on the machine; the file-based controls work.
