"""Pure math of the computed allocator. No I/O, no clock, no network: numbers in, numbers out.

The prediction layer (research/live/forecast.py) writes, per strategy, a daily volatility sigma_d and one
forecast of the daily mean net return per forecaster, plus a covariance matrix and the forecaster
weights. This module turns them, together with the decider's probabilities, into target weights:

  1. every forecast is clipped to an annual Sharpe of +-sharpe_cap: |mu| <= cap * sigma_d / sqrt(365)
  2. mu_s = sum_k W_sk mu_sk (weights renormalised over the forecasters that have a value)
  3. core plus tilt (the target): the core is the static inverse-volatility mix, w_s = min(1, k / sigma_s) with k
     the largest scale at which the portfolio volatility is within the cap. The forecast only tilts it:
     SR_s = mu_s 365 / (sigma_d_s sqrt(365)), tau_s = clip(SR_s / SR_full, -T_eff, +T_eff), T_eff = T kappa,
     target_s = clip(core_s (1 + tau_s), 0, 1). With tau = 0 the target is the core exactly.
     The tilt has to earn its room: kappa is 0 until the shadow tilted book has beaten the core book (see
     kappa_for_skill), so the live weights are the core. The tilted target at full T is still computed and
     logged as the shadow.
  4. the rules, in order: vol cap, buffer, step, decider veto, exploration floor, then the drawdown
     floor and owner_lock, which override everything (see apply_rules)
  5. the final check (final_vol_cap): the vol cap again, on the weights that will actually be held on a venue, every
     book of it included (owner-locked, held, without a forecast). The buffer, the exploration floor and the locks run
     after the first cap and can push the portfolio over it; this check only ever cuts, and only unlocked books.

solve_weights, the mean-variance optimiser of the first version (w* = argmax mu'w - gamma/2 w'Sigma w - cost
term), is kept for the comparison and no longer used for the target.

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
    veto_live: bool = True          # False: the veto changes no live weight, it is only logged as the shadow (veto_shadow)
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


# ---------------------------------------------------------------------------------- core plus tilt
@dataclass(frozen=True)
class TiltParams:
    """The tilt constants. t_eff = t_max * kappa; kappa in [0, 1] is the tilt room the forecast has earned
    (forecast/scores.json `kappa`, forwarded in forecast/latest.json; 0 at the start and whenever it is missing)."""
    sr_full: float = 1.0   # a forecast Sharpe of this much gives the full tilt
    t_max: float = 0.5     # T: the largest tilt, as a fraction of the core weight
    kappa: float = 0.0

    @property
    def t_eff(self) -> float:
        return max(0.0, self.t_max) * min(1.0, max(0.0, self.kappa))


def forecast_sharpe(mu_d: float, sigma_d: float) -> float:
    """SR = mu 365 / (sigma_d sqrt(365)): the annual forecast return over the annual volatility."""
    return mu_d * DAYS / (sigma_d * math.sqrt(DAYS)) if sigma_d > 0 else 0.0


def tilt_tau(sr: float, params: TiltParams) -> float:
    """tau = clip(SR / SR_full, -T_eff, +T_eff)."""
    t = params.t_eff
    return min(t, max(-t, sr / params.sr_full)) if params.sr_full > 0 else 0.0


def core_weights(sigma: Sequence[float], cov: Sequence[Sequence[float]], vol_cap: float, n: int | None = None) -> list[float]:
    """The static inverse-volatility mix: w_s = min(1, k / sigma_s), k the largest scale at which
    portfolio_vol_ann(w, cov, n) is at most vol_cap (all at 1 when even that is under it). Bisection, the
    volatility rises with k. Needs sigma_s > 0; `cov` is in the order of `sigma`."""
    if not sigma:
        return []

    def mix(k: float) -> list[float]:
        return [min(1.0, k / x) for x in sigma]

    hi = max(sigma)
    if portfolio_vol_ann(mix(hi), cov, n) <= vol_cap:
        return mix(hi)
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if portfolio_vol_ann(mix(mid), cov, n) <= vol_cap else (lo, mid)
    return mix(lo)


def tilted_target(core: Sequence[float], tau: Sequence[float]) -> list[float]:
    """target_s = core_s (1 + tau_s), kept inside the box 0 <= w <= 1."""
    return [min(1.0, max(0.0, c * (1.0 + t))) for c, t in zip(core, tau)]


KAPPA_STEPS = (0.0, 0.5, 1.0)


def kappa_for_skill(skill_90: float | None, skill_180: float | None) -> float:
    """The tilt room the shadow tilt skill has earned (mean daily return of the shadow tilted book minus the core book):
    1.0 when the last 90 days and the last 180 days are both positive, 0.5 when the last 90 days are positive,
    else 0. None (fewer days than the window) and an exact zero do not count as positive, so kappa is 0 until 90
    days exist. It goes back down by the same rules when the means turn negative."""
    pos90 = skill_90 is not None and skill_90 > 0
    pos180 = skill_180 is not None and skill_180 > 0
    if pos90 and pos180:
        return KAPPA_STEPS[2]
    return KAPPA_STEPS[1] if pos90 else KAPPA_STEPS[0]


# ---------------------------------------------------------------------------------- optimiser (kept for the comparison)
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
def apply_rules(ids: Sequence[str], w_target: Mapping[str, float], w_prev: Mapping[str, float],
                cov: Sequence[Sequence[float]], p: Mapping[str, float | None], params: RuleParams, *,
                floored: Sequence[str] = (), locked: Sequence[str] = ()) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """The six rules in order. `ids` is the order of `cov`. Returns (final weights, steps); each step is
    {"step", "before", "after", "changed"} so the log shows what every rule did."""
    w = {i: float(w_target[i]) for i in ids}
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
    # 4. decider veto: p below the threshold caps the weight at a fraction of the previous one. Not live: the live weights
    # pass unchanged and the step carries what the veto would have done (shadow_after, shadow_changed)
    vetoed = {i: (min(w[i], params.veto_factor * w_prev[i]) if p.get(i) is not None and p[i] < params.veto_p else w[i]) for i in ids}
    if params.veto_live:
        record("veto", vetoed, {"live": True})
    else:
        record("veto", dict(w), {"live": False, "shadow_after": {i: round(vetoed[i], 6) for i in ids},
                                 "shadow_changed": [i for i in ids if abs(vetoed[i] - w[i]) > 1e-9]})
    # 5. exploration floor
    record("explore_floor", {i: (max(w[i], params.explore_floor) if params.explore_floor > 0 else w[i]) for i in ids})
    # 6. the drawdown floor and owner_lock override everything; owner_lock wins over the floor
    fl, lk = set(floored), set(locked)
    record("floor_lock", {i: (w_prev[i] if i in lk else (0.0 if i in fl else w[i])) for i in ids})
    return {i: min(1.0, max(0.0, w[i])) for i in ids}, steps


def final_vol_cap(ids: Sequence[str], w: Mapping[str, float], cov: Sequence[Sequence[float]], locked: Sequence[str],
                  vol_cap: float, n: int | None = None) -> tuple[dict[str, float], dict[str, Any]]:
    """The vol cap on the final holdings of one venue. `ids` is the order of `cov` and covers every book of the venue;
    `n` is its number of books (the portfolio is their average, as in portfolio_vol_ann). Locked books are fixed; the
    others are scaled down together by the largest factor f in [0, 1] that brings the portfolio within the cap. Nothing
    is ever raised. When no f in [0, 1] is enough (the locked books alone are over the cap, say), the unlocked books are
    cut to the f with the lowest volatility (0 unless they hedge the locked ones) and the result says infeasible.
    Returns (weights, info): info has vol_before, vol_after, factor, cut (ids changed) and infeasible."""
    n = n or len(ids)
    lk = set(locked)
    x = [max(0.0, float(w[i])) for i in ids]
    u = [0.0 if i in lk else v for i, v in zip(ids, x)]   # the part that may move
    l = [v if i in lk else 0.0 for i, v in zip(ids, x)]   # the part that may not
    m = len(ids)

    def quad(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(a[r] * cov[r][c] * b[c] for r in range(m) for c in range(m))

    def vol(f: float) -> float:
        return portfolio_vol_ann([lv + f * uv for lv, uv in zip(l, u)], cov, n)

    before = vol(1.0)
    info: dict[str, Any] = {"vol_before": round(before, 6), "vol_after": round(before, 6), "factor": 1.0, "cut": [], "infeasible": False,
                            "vol_cap": vol_cap}
    if before <= vol_cap + 1e-12 or not any(u):
        info["infeasible"] = before > vol_cap + 1e-12
        return {i: v for i, v in zip(ids, x)}, info
    # 365 (a f^2 + 2 b f + c) / n^2 <= cap^2: the largest root in [0, 1], else the minimum of the parabola in [0, 1]
    a, b, c = quad(u, u), quad(u, l), quad(l, l)
    q = vol_cap ** 2 * n ** 2 / DAYS
    disc = b * b - a * (c - q)
    f = None
    if a > 0 and disc >= 0:
        hi = (-b + math.sqrt(disc)) / a
        lo = (-b - math.sqrt(disc)) / a
        if hi >= 0 and lo <= 1:
            f = min(1.0, hi)
    if f is None:
        f = min(1.0, max(0.0, -b / a)) if a > 0 else 0.0
        info["infeasible"] = True
    f = max(0.0, min(1.0, f))
    out = {i: (lv + f * uv) for i, lv, uv in zip(ids, l, u)}
    info.update({"vol_after": round(vol(f), 6), "factor": round(f, 6), "cut": [i for i, uv in zip(ids, u) if uv > 0 and f < 1.0]})
    return out, info


# ---------------------------------------------------------------------------------- text
def ann_pct(mu_daily: float) -> float:
    return mu_daily * DAYS * 100.0


def sigma_ann_pct(sigma_d: float) -> float:
    return sigma_d * math.sqrt(DAYS) * 100.0


def reason_line(sid: str, core: float, tau: float, sr: float, w_target: float, w_final: float, binding: Sequence[str],
                regime: str | None = None, kappa: float = 0.0, shadow_tau: float | None = None) -> str:
    """One plain line for the card. With no tilt room (kappa 0) the weight is the core and the tilt is shown as a shadow:
    "trend: core 0.42 (tilt in shadow, +18%, earns live control after 90 days ahead of core) -> 0.25 after step cap".
    With room: "trend: core 0.42, tilt +18% (SR +0.36), w 0.50 -> 0.45 after step cap [tilt room 0.50]"."""
    if kappa <= 0.0:
        st = tau if shadow_tau is None else shadow_tau
        line = f"{sid}: core {core:.2f} (tilt in shadow, {st * 100:+.0f}%, earns live control after 90 days ahead of core)"
    else:
        line = f"{sid}: core {core:.2f}, tilt {tau * 100:+.0f}% (SR {sr:+.2f}), w {w_target:.2f}"
    if abs(w_final - w_target) > 0.005 or binding:
        line += f" -> {w_final:.2f}"
        names = [STEP_LABELS[b] for b in binding]
        if names:
            line += f" after {', '.join(names)}"
    if 0.0 < kappa < 1.0:
        line += f" [tilt room {kappa:.2f}]"
    return line + (f" ({regime})" if regime else "")
