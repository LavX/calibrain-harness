"""Prompt and output schema for the strategy allocator (ctb/allocator.py).

The model reads a compact JSON context (strategy books and market summary) and answers with one
strict JSON object: which strategies run, how much of each book is deployed, and why. The hard
limits (step size, drawdown floor, owner locks) are enforced in code after the answer; the prompt
states them so the model does not waste its answer on moves that would be clipped.
"""

from __future__ import annotations

import json
from typing import Any

REASONING_MAX_CHARS = 600

# What each published strategy is, so the model can match it to the market regime. Unknown ids fall
# back to their label from strategies.json.
STRATEGY_NOTES = {
    "carry": ("Funding cash-and-carry on BTC and ETH: long spot, short the perpetual, market neutral. Earns the funding "
              "rate while it stays positive. Small, steady returns. Loses when funding turns negative or the basis "
              "blows out. Barely cares about direction."),
    "trend": ("Time-series trend following (EWMAC) on 10 liquid perpetuals, rebalanced daily, long or short each coin. "
              "Pays in sustained directional moves, either way. Bleeds in choppy ranges and sharp reversals."),
    "xsmom": ("Cross-sectional momentum, market neutral: long recent winners, short recent losers. Pays when dispersion "
              "between coins is high and leaders keep leading. Suffers when winners and losers swap places quickly "
              "(momentum crashes)."),
}

SYSTEM_PROMPT = """You are the allocator of a small systematic trading book. Several published strategies trade a demo account, each on its own virtual book. Once per interval you decide which of them run and how much of each book is deployed. You do not pick trades: every strategy follows its own fixed rules.

Terms:
- weight: the fraction of that strategy's own book that is deployed, 0 to 1. Weights are independent per strategy and need not sum to 1. Weight 0 means flat.
- enabled false: the strategy places no new orders and keeps what it holds.

Limits applied in code after your answer. You cannot override them, so do not spend the answer on moves they would clip:
- A weight moves at most 0.5 per decision, up or down.
- A strategy more than 15% below its book's peak is forced to weight 0.
- A strategy with owner_lock true is set by the owner. Your values for it are ignored.

How to decide:
- These strategies are slow and every change costs fees. Keep the current allocation unless the inputs give a specific reason to move. Do not chase the last day's P&L.
- Match each strategy to the regime the inputs describe: funding and basis for carry, trend strength and volatility for trend, dispersion of returns for xsmom.
- Use the strategy's own drawdown and state. A strategy that is latched or in error should not get more weight.
- When a P&L window is null the history is too short. Say so in the reasoning instead of guessing.
- Name the inputs that drove the call.

Answer with one JSON object and nothing else:
{"weights": {"<id>": 0..1, ...}, "enabled": {"<id>": true|false, ...}, "reasoning": "<= 600 characters, plain text", "confidence": 0..1}
Include every strategy id in both weights and enabled. confidence is how sure you are that your allocation beats keeping the current one; 0.5 is a coin flip."""


def build_schema(ids: list[str]) -> dict[str, Any]:
    """Strict JSON schema for the answer, one entry per known strategy id."""
    return {
        "type": "object",
        "properties": {
            "weights": {"type": "object", "properties": {i: {"type": "number"} for i in ids},
                        "required": list(ids), "additionalProperties": False},
            "enabled": {"type": "object", "properties": {i: {"type": "boolean"} for i in ids},
                        "required": list(ids), "additionalProperties": False},
            "reasoning": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["weights", "enabled", "reasoning", "confidence"],
        "additionalProperties": False,
    }


def build_messages(context: dict[str, Any], ids: list[str], labels: dict[str, str] | None = None) -> list[dict[str, str]]:
    notes = {i: STRATEGY_NOTES.get(i) or (labels or {}).get(i) or i for i in ids}
    ctx = json.dumps(context, separators=(",", ":"), allow_nan=False, default=str)
    user = (f"STRATEGIES:\n{json.dumps(notes, indent=1, ensure_ascii=False)}\n\nCONTEXT:\n{ctx}\n\n"
            f"Strategy ids: {json.dumps(ids)}. Reply with the JSON object only.")
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def retry_messages(messages: list[dict[str, str]], previous: Any, problem: str) -> list[dict[str, str]]:
    """The same conversation plus the rejected answer and what was wrong with it."""
    out = list(messages)
    if previous is not None:
        out.append({"role": "assistant", "content": json.dumps(previous, separators=(",", ":"), default=str)[:1500]})
    out.append({"role": "user", "content": f"That answer was rejected: {problem}. Reply again with only the corrected JSON object."})
    return out
