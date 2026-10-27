"""
BudgetPilot — a minimal ad budget optimization agent.

The agent follows the policy in prompts/system_prompt.md:
  read data -> compute metrics -> propose a shift -> apply guardrails -> log why.

Each step is a small "tool" function so it can be called by the rule-based
policy (default, no API key needed) or exposed to an LLM via TOOL_SPECS.
An optional LLM advisor (--llm) can suggest the shift, but every suggestion
still passes through apply_guardrails(), which has the final say.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["Search", "Social", "Display"]
REQUIRED_COLUMNS = ["date", "channel", "spend", "impressions", "clicks", "conversions"]
SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.md"

# Column names that would suggest user-level / personal data (privacy guardrail).
PII_COLUMN_PATTERN = re.compile(
    r"(email|e_mail|phone|name|address|ip_?addr|user_?id|customer_?id|device_?id|cookie)",
    re.IGNORECASE,
)
# Words we never allow in a logged rationale (brand-tone guardrail).
BANNED_TONE_WORDS = ["guaranteed", "viral", "crushing", "insane", "explode", "skyrocket", "!!"]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class AgentConfig:
    daily_budget: float = 1500.0       # cost cap: total spend per day
    warmup_days: int = 2               # equal split while we collect data
    lookback_days: int = 7             # metrics window
    halflife_days: float = 3.0         # recency weighting inside the window
    max_daily_change: float = 0.20     # guardrail: +/-20% per channel per day
    min_share: float = 0.10            # guardrail: floor, never 0%
    max_share: float = 0.60            # guardrail: no channel above 60%
    # (minimum CPA edge vs portfolio, shift size as % of the winner's budget)
    edge_to_shift: tuple = ((0.20, 0.20), (0.10, 0.15), (0.05, 0.10))
    spike_ratio: float = 1.5           # yesterday CPA > 1.5x average = unstable
    max_zero_days: int = 2             # never 0% for more than 2 days in a row

    @property
    def floor(self) -> float:
        return round(self.min_share * self.daily_budget)

    @property
    def ceiling(self) -> float:
        return round(self.max_share * self.daily_budget)


# --------------------------------------------------------------------------
# Input checks (privacy guardrail)
# --------------------------------------------------------------------------
def load_performance_csv(path: str | Path) -> pd.DataFrame:
    """Read the CSV, enforce the schema and refuse anything that looks like PII."""
    df = pd.read_csv(path)
    check_privacy(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"])
    if (df[["spend", "impressions", "clicks", "conversions"]] < 0).any().any():
        raise ValueError("CSV contains negative values")
    return df.sort_values(["date", "channel"]).reset_index(drop=True)


def check_privacy(df: pd.DataFrame) -> None:
    """Privacy guardrail: the agent only accepts aggregate, channel-level data."""
    flagged = [c for c in df.columns if PII_COLUMN_PATTERN.search(str(c))]
    if flagged:
        raise ValueError(
            f"Privacy guardrail: refusing columns that look like personal data: {flagged}. "
            "Provide aggregate channel-level metrics only."
        )


def check_tone(text: str) -> str:
    """Brand-tone guardrail: keep rationales factual and neutral."""
    lowered = text.lower()
    for word in BANNED_TONE_WORDS:
        if word in lowered:
            text = re.sub(re.escape(word), "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


# --------------------------------------------------------------------------
# Tool 1: read performance data (only what was known before `date`)
# --------------------------------------------------------------------------
def read_performance(history: pd.DataFrame, before_date: pd.Timestamp) -> pd.DataFrame:
    """Return all rows strictly before `before_date` (no peeking at the future)."""
    return history[history["date"] < before_date].copy()


# --------------------------------------------------------------------------
# Tool 2: compute recency-weighted channel metrics
# --------------------------------------------------------------------------
def compute_metrics(history: pd.DataFrame, config: AgentConfig) -> pd.DataFrame:
    """
    CTR, CVR, CPA and conversions-per-dollar per channel over the lookback window.
    Recent days get more weight (exponential decay with `halflife_days`) — this is
    the agent's lightweight learning loop: estimates update every day as new
    results arrive.
    """
    if history.empty:
        return pd.DataFrame()

    last_date = history["date"].max()
    window = history[history["date"] > last_date - pd.Timedelta(days=config.lookback_days)].copy()
    age_days = (last_date - window["date"]).dt.days
    window["weight"] = 0.5 ** (age_days / config.halflife_days)

    rows = {}
    for channel, g in window.groupby("channel"):
        w = g["weight"]
        spend = float((g["spend"] * w).sum())
        impressions = float((g["impressions"] * w).sum())
        clicks = float((g["clicks"] * w).sum())
        conversions = float((g["conversions"] * w).sum())
        yesterday = g[g["date"] == last_date]
        y_spend = float(yesterday["spend"].sum())
        y_conv = float(yesterday["conversions"].sum())
        rows[channel] = {
            "spend_w": spend,
            "ctr": clicks / impressions if impressions else 0.0,
            "cvr": conversions / clicks if clicks else 0.0,
            "cpa": spend / conversions if conversions else math.inf,
            "conv_per_dollar": conversions / spend if spend else 0.0,
            "yesterday_cpa": y_spend / y_conv if y_conv else math.inf,
            "days_observed": int(g.loc[g["spend"] > 0, "date"].nunique()),
        }

    metrics = pd.DataFrame.from_dict(rows, orient="index")
    total_conv = sum(m["conv_per_dollar"] * m["spend_w"] for m in rows.values())
    total_spend = sum(m["spend_w"] for m in rows.values())
    metrics.attrs["portfolio_cpa"] = total_spend / total_conv if total_conv else math.inf
    return metrics


# --------------------------------------------------------------------------
# Tool 3: propose a budget shift (explore/exploit heuristic)
# --------------------------------------------------------------------------
@dataclass
class Proposal:
    action: str                      # "shift" or "hold"
    increase: str | None = None      # channel to grow
    donors: list = field(default_factory=list)  # channels to shrink, worst first
    shift_pct: float = 0.0           # % of the winner's current budget
    edge: float = 0.0                # how much better the winner's CPA is vs portfolio
    note: str = ""                   # why (for the log)


def _edge(cpa: float, portfolio_cpa: float) -> float:
    """Relative CPA advantage vs the portfolio (0.20 = 20% cheaper per conversion)."""
    if not math.isfinite(cpa) or not math.isfinite(portfolio_cpa) or portfolio_cpa == 0:
        return -math.inf
    return 1 - cpa / portfolio_cpa


def propose_budget(metrics: pd.DataFrame, current: dict, config: AgentConfig) -> Proposal:
    """Exploit the best-CPA channel that still has room; fund it from the worst."""
    portfolio_cpa = metrics.attrs["portfolio_cpa"]
    ranked = metrics.sort_values("conv_per_dollar", ascending=False).index.tolist()

    for candidate in ranked[:-1]:  # the worst channel is never a recipient
        edge = _edge(metrics.loc[candidate, "cpa"], portfolio_cpa)
        if edge < config.edge_to_shift[-1][0]:
            break  # nobody further down the list has a big enough edge either
        if current[candidate] >= config.ceiling:
            continue  # already at the 60% cap — try the next-best channel

        shift_pct = next(pct for min_edge, pct in config.edge_to_shift if edge >= min_edge)
        note = ""
        if metrics.loc[candidate, "yesterday_cpa"] > config.spike_ratio * metrics.loc[candidate, "cpa"]:
            shift_pct /= 2
            note = "CPA spiked yesterday, shift halved"

        donors = [c for c in reversed(ranked) if c != candidate
                  and metrics.loc[c, "conv_per_dollar"] < metrics.loc[candidate, "conv_per_dollar"]]
        return Proposal("shift", candidate, donors, shift_pct, edge, note)

    best = ranked[0]
    best_edge = _edge(metrics.loc[best, "cpa"], portfolio_cpa)
    if current[best] >= config.ceiling and best_edge >= config.edge_to_shift[-1][0]:
        reason = f"{best} already at the {config.max_share:.0%} cap"
    else:
        reason = f"no channel beats portfolio CPA by {config.edge_to_shift[-1][0]:.0%}+"
    return Proposal("hold", note=reason, edge=best_edge)


# --------------------------------------------------------------------------
# Tool 4: apply guardrails (final authority over the budget)
# --------------------------------------------------------------------------
def apply_guardrails(proposal: Proposal, current: dict, config: AgentConfig) -> tuple[dict, dict]:
    """
    Turn a proposal into tomorrow's budget while enforcing:
      - total = daily budget (money only moves between channels)
      - each channel changes by at most +/-max_daily_change
      - each channel stays within [floor, ceiling]
    Returns (new_budget, notes) where notes says which guardrail was binding.
    """
    new = dict(current)
    notes: dict = {}
    if proposal.action != "shift":
        return new, notes

    r = proposal.increase
    wanted = math.floor(proposal.shift_pct * current[r])
    room = min(math.floor(config.max_daily_change * current[r]), config.ceiling - current[r])
    amount = min(wanted, room)
    if amount < wanted:
        notes[r] = "capped by guardrail (+20%/day or 60% max share)"

    remaining = amount
    for d in proposal.donors:
        can_give = min(math.floor(config.max_daily_change * current[d]), current[d] - config.floor)
        give = max(0, min(remaining, can_give))
        if give < remaining:
            notes[d] = f"limited by guardrail (-20%/day or {config.min_share:.0%} floor)"
        new[d] -= give
        remaining -= give
        if remaining == 0:
            break

    new[r] += amount - remaining  # recipient only gets what donors could give
    return new, notes


def validate_budget(new: dict, previous: dict, zero_streak: dict, config: AgentConfig) -> None:
    """Post-condition check: raise if any hard guardrail is violated."""
    assert abs(sum(new.values()) - config.daily_budget) < 1e-6, "total budget not respected"
    for c, amount in new.items():
        assert amount >= config.floor - 1e-6, f"{c} below floor"
        assert amount <= config.ceiling + 1e-6, f"{c} above max share"
        if previous[c] > 0:
            change = abs(amount - previous[c]) / previous[c]
            assert change <= config.max_daily_change + 1e-6, f"{c} changed {change:.0%} in one day"
        streak = zero_streak.get(c, 0) + 1 if amount == 0 else 0
        assert streak <= config.max_zero_days, f"{c} at 0% for more than {config.max_zero_days} days"


# --------------------------------------------------------------------------
# Tool 5: log the decision with a reason string per channel
# --------------------------------------------------------------------------
def log_decision(date, day_number, previous, new, metrics, proposal, notes, config) -> list[dict]:
    """One log row per channel, each with a human-readable reason."""
    rows = []
    portfolio_cpa = metrics.attrs.get("portfolio_cpa", math.nan) if not metrics.empty else math.nan
    if metrics.empty:
        action = "warmup"
    else:
        action = "shift" if any(new[c] != previous[c] for c in CHANNELS) else "hold"

    for c in CHANNELS:
        before, after = previous[c], new[c]
        pct = (after - before) / before if before else 0.0
        m = metrics.loc[c] if (not metrics.empty and c in metrics.index) else None

        if m is None:
            reason = f"{c} held at ${after:,.0f}: warm-up, equal split to collect data on every channel"
        elif after > before:
            reason = (f"{c} up {pct:.0%} (${before:,.0f}->${after:,.0f}) due to better CPA "
                      f"${m.cpa:,.2f} vs ${portfolio_cpa:,.2f} portfolio ({proposal.edge:.0%} better), "
                      f"CVR {m.cvr:.1%}")
            reason += f"; {proposal.note}" if proposal.note else ", stable CPA"
        elif after < before:
            reason = (f"{c} down {abs(pct):.0%} (${before:,.0f}->${after:,.0f}) due to higher CPA "
                      f"${m.cpa:,.2f}; budget moved to {proposal.increase}")
            if after <= config.floor:
                reason += f"; kept at {config.min_share:.0%} floor to keep learning"
        else:
            if proposal.action == "hold":
                reason = f"{c} held at ${after:,.0f} (CPA ${m.cpa:,.2f}): {proposal.note}"
            elif c == proposal.increase:
                reason = (f"{c} held at ${after:,.0f} (CPA ${m.cpa:,.2f}): earned more budget, "
                          f"but no worse channel can give any (all at the floor)")
            elif after >= config.ceiling:
                reason = f"{c} held at ${after:,.0f} (CPA ${m.cpa:,.2f}): at {config.max_share:.0%} max share"
            elif after <= config.floor:
                reason = f"{c} held at ${after:,.0f} (CPA ${m.cpa:,.2f}): at {config.min_share:.0%} floor, kept for learning"
            else:
                reason = f"{c} held at ${after:,.0f} (CPA ${m.cpa:,.2f}): not the best or worst performer"
        if c in notes:
            reason += f" [{notes[c]}]"

        rows.append({
            "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
            "day": day_number,
            "channel": c,
            "budget_before": before,
            "budget_after": after,
            "change_pct": round(pct * 100, 1),
            "share_pct": round(after / config.daily_budget * 100, 1),
            "est_ctr": None if m is None else round(m.ctr, 4),
            "est_cvr": None if m is None else round(m.cvr, 4),
            "est_cpa": None if m is None or not math.isfinite(m.cpa) else round(m.cpa, 2),
            "action": action,
            "reason": check_tone(reason),
        })
    return rows


# --------------------------------------------------------------------------
# Optional LLM advisor (uses the system prompt). Off unless --llm is passed.
# --------------------------------------------------------------------------
class LLMAdvisor:
    """
    Sends the system prompt + today's metrics to Claude and asks for a JSON
    proposal. The advice is sanitized and then goes through the same guardrails.
    Requires ANTHROPIC_API_KEY; set AGENT_LLM_MODEL to choose a model.
    """

    def __init__(self, system_prompt: str, transport=None):
        self.system_prompt = system_prompt
        self.model = os.environ.get("AGENT_LLM_MODEL", "claude-sonnet-4-5")
        self.transport = transport or self._call_api

    def _call_api(self, payload: dict) -> dict:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers={
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # retries/timeouts: see scaling note
            return json.loads(resp.read())

    def advise(self, metrics: pd.DataFrame, current: dict, rule_proposal: Proposal) -> Proposal | None:
        table = metrics[["ctr", "cvr", "cpa", "conv_per_dollar"]].round(4).to_dict(orient="index")
        user_msg = (
            f"Current budget: {json.dumps(current)}\n"
            f"Portfolio CPA: {metrics.attrs['portfolio_cpa']:.2f}\n"
            f"Channel metrics (last 7 days, recency-weighted): {json.dumps(table)}\n"
            f"Rule-based suggestion: action={rule_proposal.action}, increase={rule_proposal.increase}, "
            f"shift_pct={rule_proposal.shift_pct}. Reply with JSON only."
        )
        payload = {"model": self.model, "max_tokens": 300, "system": self.system_prompt,
                   "messages": [{"role": "user", "content": user_msg}]}
        try:
            response = self.transport(payload)
            text = "".join(b.get("text", "") for b in response.get("content", []))
            advice = json.loads(text[text.index("{"): text.rindex("}") + 1])
        except Exception as exc:  # any failure -> fall back to the rule-based proposal
            print(f"  [llm] advisor unavailable ({exc.__class__.__name__}); using rule-based proposal")
            return None

        # Sanitize: only known channels, shift between 10% and 20%.
        if advice.get("action") == "hold" or advice.get("increase") not in current:
            return Proposal("hold", note=check_tone(advice.get("rationale", "LLM advised hold")),
                            edge=rule_proposal.edge)
        shift = float(np.clip(float(advice.get("shift_pct", 0.10)), 0.10, 0.20))
        increase = advice["increase"]
        donors = [c for c in rule_proposal.donors if c != increase] or \
                 [c for c in reversed(metrics.sort_values("conv_per_dollar").index) if c != increase]
        return Proposal("shift", increase, donors, shift, rule_proposal.edge,
                        check_tone(advice.get("rationale", "")))


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
class BudgetAgent:
    """Daily decision loop: read -> metrics -> propose -> guardrails -> validate -> log."""

    name = "BudgetPilot agent"

    def __init__(self, config: AgentConfig | None = None, use_llm: bool = False, enforce_validation=True):
        self.config = config or AgentConfig()
        self.system_prompt = SYSTEM_PROMPT_PATH.read_text()
        self.advisor = LLMAdvisor(self.system_prompt) if use_llm and os.environ.get("ANTHROPIC_API_KEY") else None
        if use_llm and self.advisor is None:
            print("  [llm] ANTHROPIC_API_KEY not set; running rule-based policy only")
        self.enforce_validation = enforce_validation
        self.budget = {c: round(self.config.daily_budget / len(CHANNELS)) for c in CHANNELS}
        self.zero_streak = {c: 0 for c in CHANNELS}
        self.log: list[dict] = []

    def decide(self, history: pd.DataFrame, date: pd.Timestamp, day_number: int) -> dict:
        known = read_performance(history, date)
        previous = dict(self.budget)

        if day_number <= self.config.warmup_days or known.empty:
            metrics, proposal, notes = pd.DataFrame(), Proposal("hold", note="warm-up"), {}
            new = previous
        else:
            metrics = compute_metrics(known, self.config)
            proposal = propose_budget(metrics, previous, self.config)
            if self.advisor is not None:
                proposal = self.advisor.advise(metrics, previous, proposal) or proposal
            new, notes = apply_guardrails(proposal, previous, self.config)

        if self.enforce_validation:
            validate_budget(new, previous, self.zero_streak, self.config)
        self.zero_streak = {c: (self.zero_streak[c] + 1 if new[c] == 0 else 0) for c in CHANNELS}
        self.log.extend(log_decision(date, day_number, previous, new, metrics, proposal, notes, self.config))
        self.budget = new
        return dict(new)


class EqualSplitBaseline:
    """Baseline: the same fixed equal split every day."""

    name = "Equal-split baseline"

    def __init__(self, daily_budget: float = 1500.0):
        self.budget = {c: round(daily_budget / len(CHANNELS)) for c in CHANNELS}

    def decide(self, history, date, day_number) -> dict:
        return dict(self.budget)


class GreedyNoGuardrails:
    """Ablation: put 100% into yesterday's best-CPA channel. No floor, no caps."""

    name = "Greedy (no guardrails)"

    def __init__(self, daily_budget: float = 1500.0):
        self.daily_budget = daily_budget
        self.budget = {c: round(daily_budget / len(CHANNELS)) for c in CHANNELS}

    def decide(self, history, date, day_number) -> dict:
        known = read_performance(history, date)
        if not known.empty:
            last = known[known["date"] == known["date"].max()]
            last = last[last["spend"] > 0].set_index("channel")
            cpa = (last["spend"] / last["conversions"].replace(0, np.nan)).dropna()
            if not cpa.empty:
                best = cpa.idxmin()
                self.budget = {c: (self.daily_budget if c == best else 0) for c in CHANNELS}
        return dict(self.budget)


# Tool specs, if you want to expose the tools to an LLM tool-calling loop.
TOOL_SPECS = [
    {"name": "read_performance", "description": "Channel-level spend/impressions/clicks/conversions before a date.",
     "input_schema": {"type": "object", "properties": {"before_date": {"type": "string"}}, "required": ["before_date"]}},
    {"name": "compute_metrics", "description": "Recency-weighted CTR, CVR, CPA per channel (last 7 days).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "propose_budget", "description": "Suggest which channel to grow/shrink and by how much.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "apply_guardrails", "description": "Enforce cost cap, +/-20%/day, 10% floor, 60% cap. Final.",
     "input_schema": {"type": "object", "properties": {"increase": {"type": "string"},
                                                        "shift_pct": {"type": "number"}}}},
    {"name": "log_decision", "description": "Record budgets and one reason string per channel.",
     "input_schema": {"type": "object", "properties": {"reason": {"type": "string"}}}},
]
