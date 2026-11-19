"""Pure math of the computed allocator. No I/O, no clock, no network: numbers in, numbers out.

The prediction layer (research/live/forecast.py) writes, per strategy, a daily volatility sigma_d and one
forecast of the daily mean net return per forecaster, plus a covariance matrix and the forecaster
weights. This module turns them, together with the decider's probabilities, into target weights:

  1. every forecast is clipped to an annual Sharpe of +-sharpe_cap: |mu| <= cap * sigma_d / sqrt(365)
  2. mu_s = sum_k W_sk mu_sk (weights renormalised over the forecasters that have a value)
  3. w* = argmax over 0 <= w <= 1 of  sum_s w_s mu_s - gamma/2 w'Sigma w - sum_s k_s |w_s - w_prev_s|
     with k_s = c_s G_s / H (round-trip-free cost per side, amortised over the holding horizon H days),
     by proximal gradient (soft-threshold around w_prev, then clip to the box), deterministic
  4. the rules, in order: vol cap, buffer, step, decider veto, exploration floor, then the drawdown
     floor and owner_lock, which override everything (see apply_rules)

All returns are daily fractions (0.001 = 10 bp), all weights are fractions of a strategy's virtual book.
Units for display: annual % = daily * 365 * 100, annual sigma % = sigma_d * sqrt(365) * 100.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DAYS = 365.0
STEP_LABELS = {
    "vol_cap": "vol cap", "buffer": "buffer", "step": "step cap", "veto": "decider veto",
    "explore_floor": "exploration floor", "floor_lock": "drawdown floor or owner lock",
}


@dataclass(frozen=True)
class RuleParams:
    """The numbers the rules need. Defaults are the ledger's (ctb/config.py carries the same ones)."""
    vol_cap: float = 0.10           # portfolio volatility cap, annual
    buffer: float = 0.10            # a change smaller than this is not made
    step_up: float = 0.25           # weight may rise at most this much per day, down is unlimited
    veto_p: float = 0.40            # decider probability below this caps the weight ...
    veto_factor: float = 0.5        # ... at this fraction of the previous weight
    explore_floor: float = 0.10     # minimum weight so every strategy keeps producing fills (0 disables)


# ---------------------------------------------------------------------------------- forecasts
def sharpe_cap_mu(sigma_d: float, sharpe_cap: float) -> float:
    """The largest daily |mu| the Sharpe cap allows."""
    return sharpe_cap * sigma_d / math.sqrt(DAYS)


def clip_mu(mu: float, sigma_d: float, sharpe_cap: float = 1.5) -> tuple[float, bool]:
    """(clipped mu, whether it was clipped)."""
    cap = sharpe_cap_mu(sigma_d, sharpe_cap)
    if mu > cap:
        return cap, True
    if mu < -cap:
        return -cap, True
    return mu, False


def mean_p(answers: Sequence[float]) -> float | None:
    """The mean of the decider's last answers, None when there are none."""
    vals = [a for a in answers if isinstance(a, (int, float)) and math.isfinite(a)]
    return sum(vals) / len(vals) if vals else None


def decider_mu(p: float, sigma_d: float, k: float = 0.8) -> float:
    """The decider as a forecaster: (2p - 1) * k * sigma_d."""
    return (2.0 * p - 1.0) * k * sigma_d


def combine(mu: Mapping[str, float | None], weights: Mapping[str, float]) -> float | None:
    """sum_k W_k mu_k over the forecasters that have a value and a positive weight, renormalised to
    weights that sum to 1. None when nothing is left."""
    num = den = 0.0
    for k, v in mu.items():
        w = weights.get(k)
        if v is None or w is None or not (w > 0) or not math.isfinite(v):
            continue
        num += w * v
        den += w
    return num / den if den > 0 else None


# ---------------------------------------------------------------------------------- optimiser
def portfolio_vol_ann(w: Sequence[float], cov: Sequence[Sequence[float]], n: int | None = None) -> float:
    """sqrt(365 w'Sigma w) / N: the books are equal in size, so the portfolio is the average of the
    weighted books."""
    n = n or len(w)
    q = sum(w[i] * cov[i][j] * w[j] for i in range(len(w)) for j in range(len(w)))
    return math.sqrt(max(0.0, DAYS * q)) / n if n else 0.0


def solve_weights(mu: Sequence[float], cov: Sequence[Sequence[float]], w_prev: Sequence[float], cost: Sequence[float], *,
                  gamma: float = 2.0, iters: int = 50_000, tol: float = 1e-13) -> list[float]:
    """argmax over 0 <= w <= 1 of  mu'w - gamma/2 w'Sigma w - sum_s cost_s |w_s - w_prev_s|.

    cost_s is the per-unit-of-weight trading cost per day, c_s * G_s / H. Concave, so proximal gradient
    converges to the optimum: a gradient step on the quadratic, then the exact prox of the L1 term
    (soft-threshold around w_prev) and the box (clip). Starts at the clipped w_prev; no randomness.
    Unconstrained and without costs this is w = (gamma Sigma)^-1 mu, in one asset mu / (gamma sigma^2)."""
    n = len(mu)
    if n == 0:
        return []
    lip = gamma * max((sum(abs(x) for x in row) for row in cov), default=0.0)  # Gershgorin bound on gamma * lambda_max
    t = 1.0 / max(lip, 1e-18)
    w = [min(1.0, max(0.0, x)) for x in w_prev]
    for _ in range(iters):
        grad = [gamma * sum(cov[i][j] * w[j] for j in range(n)) - mu[i] for i in range(n)]  # of the negated smooth part
        new = []
        for i in range(n):
            v = w[i] - t * grad[i] - w_prev[i]
            thr = t * cost[i]
            v = v - thr if v > thr else (v + thr if v < -thr else 0.0)
            new.append(min(1.0, max(0.0, w_prev[i] + v)))
        delta = max(abs(a - b) for a, b in zip(new, w))
        w = new
        if delta < tol:
            break
    return w


# ---------------------------------------------------------------------------------- rules
def apply_rules(ids: Sequence[str], w_star: Mapping[str, float], w_prev: Mapping[str, float],
                cov: Sequence[Sequence[float]], p: Mapping[str, float | None], params: RuleParams, *,
                floored: Sequence[str] = (), locked: Sequence[str] = ()) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """The six rules in order. `ids` is the order of `cov`. Returns (final weights, steps); each step is
    {"step", "before", "after", "changed"} so the log shows what every rule did."""
    w = {i: float(w_star[i]) for i in ids}
    steps: list[dict[str, Any]] = []

    def record(name: str, after: dict[str, float], extra: dict[str, Any] | None = None) -> None:
        nonlocal w
        changed = [i for i in ids if abs(after[i] - w[i]) > 1e-9]
        step = {"step": name, "before": {i: round(w[i], 6) for i in ids}, "after": {i: round(after[i], 6) for i in ids}, "changed": changed}
        if extra:
            step.update(extra)
        steps.append(step)
        w = after

    # 1. portfolio volatility cap: scale everything down together
    vol = portfolio_vol_ann([w[i] for i in ids], cov, len(ids))
    factor = min(1.0, params.vol_cap / vol) if vol > 0 else 1.0
    record("vol_cap", {i: w[i] * factor for i in ids}, {"vol_ann": round(vol, 6), "factor": round(factor, 6)})
    # 2. buffer: a change under the threshold is not made
    record("buffer", {i: (w_prev[i] if abs(w[i] - w_prev[i]) < params.buffer - 1e-12 else w[i]) for i in ids})
    # 3. step: up at most step_up per day, down without limit
    record("step", {i: min(w[i], w_prev[i] + params.step_up) for i in ids})
    # 4. decider veto: p below the threshold caps the weight at a fraction of the previous one
    record("veto", {i: (min(w[i], params.veto_factor * w_prev[i]) if p.get(i) is not None and p[i] < params.veto_p else w[i]) for i in ids})
    # 5. exploration floor
    record("explore_floor", {i: (max(w[i], params.explore_floor) if params.explore_floor > 0 else w[i]) for i in ids})
    # 6. the drawdown floor and owner_lock override everything; owner_lock wins over the floor
    fl, lk = set(floored), set(locked)
    record("floor_lock", {i: (w_prev[i] if i in lk else (0.0 if i in fl else w[i])) for i in ids})
    return {i: min(1.0, max(0.0, w[i])) for i in ids}, steps


# ---------------------------------------------------------------------------------- text
def ann_pct(mu_daily: float) -> float:
    return mu_daily * DAYS * 100.0


def sigma_ann_pct(sigma_d: float) -> float:
    return sigma_d * math.sqrt(DAYS) * 100.0


def reason_line(sid: str, mu_combined: float, sigma_d: float, w_star: float, w_final: float, binding: Sequence[str],
                regime: str | None = None) -> str:
    """One plain line for the card, e.g.
    "trend: mu +6.1%/yr (regime up_calm 0.62), sigma 18%, w* 0.41 -> 0.35 after step cap"."""
    reg = f" ({regime})" if regime else ""
    line = f"{sid}: mu {ann_pct(mu_combined):+.1f}%/yr{reg}, sigma {sigma_ann_pct(sigma_d):.0f}%, w* {w_star:.2f} -> {w_final:.2f}"
    names = [STEP_LABELS[b] for b in binding]
    return line + (f" after {', '.join(names)}" if names else "")
