"""Versioned question set and probability-to-weight mapping for the decision-model allocator.

Same principles as ctb/kairos/questions.py: ask OBSERVABLE, LABELLABLE questions (what a strategy's
book earns over the next 24 hours), never "what should I do". Code turns the probabilities into
weights, and every answer can be scored later against the realised book return (python -m
ctb.allocator score), whether or not the weight was acted on. Each question reads the state
independently. Bump QUESTION_SET_VERSION on ANY wording change.

alloc-v1.0: per strategy id one binary question (net book return over the next 24 h positive), plus
one categorical question (which strategy, or none, has the highest net book return over 24 h).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

QUESTION_SET_VERSION = "alloc-v1.0"
HORIZON_H = 24
NONE_ID = "none"
BEST_QID = "best_net_return"

STATE_NOTE = ("The state lists each strategy's book (strategies.<id>: equity, P&L windows, drawdown, positions) and the market "
              "(BTC and ETH returns, volatility, funding, basis, trend breadth, dispersion). Judge the strategy's own rules in the "
              "market the state describes. The strategy's current weight does not change the answer: ask about the capital it "
              "deploys. If the inputs give no specific reason to lean either way, stay close to 0.5.")


def pos_qid(sid: str) -> str:
    return f"net_positive_{sid}"


def build_questions(ids: list[str], labels: dict[str, str] | None = None) -> dict[str, Any]:
    """alloc-v1.0: len(ids) binary questions and one categorical."""
    labels = labels or {}
    h = HORIZON_H
    qs: dict[str, Any] = {}
    for sid in ids:
        name = labels.get(sid) or sid
        qs[pos_qid(sid)] = {
            "type": "noul",
            "instructions": (f"Will the {name} strategy (strategies.{sid} in the state) earn a positive net return on the capital it "
                             f"deploys over the next {h} hours, after fees and funding? " + STATE_NOTE + f" (key: {pos_qid(sid)})"),
            "criteria": {
                "true": f"The net return of the {sid} book over the next {h} hours, after fees and funding, is above zero.",
                "false": f"The net return of the {sid} book over the next {h} hours, after fees and funding, is zero or below.",
            },
        }
    crit = {sid: (f"The {sid} strategy has the highest net return on its deployed capital over the next {h} hours, after fees and "
                  f"funding, and that return is above zero.") for sid in ids}
    crit[NONE_ID] = f"No strategy has a net return above zero over the next {h} hours."
    qs[BEST_QID] = {
        "type": "choice",
        "instructions": (f"Which strategy, {', '.join(ids)}, or {NONE_ID}, has the highest net return on its deployed capital over the "
                         f"next {h} hours, after fees and funding? " + STATE_NOTE + f" (key: {BEST_QID})"),
        "criteria": crit,
    }
    return qs


def batches(questions: dict[str, Any], max_questions: int | None) -> list[dict[str, Any]]:
    """The set split into requests of at most max_questions (None: one request)."""
    items = list(questions.items())
    n = max_questions or len(items)
    return [dict(items[i:i + n]) for i in range(0, len(items), n)]


@dataclass(frozen=True)
class MappingParams:
    p_zero: float = 0.5          # p at or below this: weight 0
    p_full: float = 0.65         # p at or above this: weight 1
    winner_min_p: float = 0.55   # the categorical winner is boosted only if its own p is above this
    winner_weight: float = 0.5   # ... to at least this weight

    def __post_init__(self) -> None:
        if not self.p_full > self.p_zero:
            raise ValueError(f"allocator p_full ({self.p_full}) must be above p_zero ({self.p_zero})")


def target_weight(p: float, params: MappingParams) -> float:
    return min(1.0, max(0.0, (p - params.p_zero) / (params.p_full - params.p_zero)))


def map_answers(ids: list[str], p: dict[str, float], best: dict[str, float] | None, hold: dict[str, tuple[bool, float]],
                params: MappingParams) -> tuple[dict[str, float], dict[str, bool], dict[str, dict[str, Any]], str]:
    """Probabilities to (weights, enabled, per-strategy mapping rows, short summary for the UI).
    A strategy without an answer keeps its current (enabled, weight)."""
    winner = max(best, key=lambda k: best[k]) if best else None
    weights: dict[str, float] = {}
    enabled: dict[str, bool] = {}
    rows: dict[str, dict[str, Any]] = {}
    parts: list[str] = []
    for sid in ids:
        pi = p.get(sid)
        if pi is None:
            enabled[sid], weights[sid] = hold[sid]
            rows[sid] = {"p": None, "target": weights[sid], "winner_boost": False, "held": True}
            parts.append(f"{sid} no answer, kept {weights[sid]:g}")
            continue
        w = round(target_weight(pi, params), 4)
        boosted = sid == winner and pi > params.winner_min_p and w < params.winner_weight
        if boosted:
            w = params.winner_weight
        weights[sid], enabled[sid] = w, w > 0
        rows[sid] = {"p": round(pi, 4), "target": w, "winner_boost": boosted, "held": False}
        parts.append(f"{sid} {'winner ' if boosted else ''}p={pi:.2f} -> {round(w, 2):g}")
    if best and winner is not None:
        parts.append(f"best: {winner} ({best[winner]:.2f})")
    return weights, enabled, rows, ", ".join(parts)
