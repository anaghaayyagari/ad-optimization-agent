"""
Quick checks for the agent's guardrails and tools.
Run with:  python tests/test_agent.py      (or: python -m pytest tests/)
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import (CHANNELS, AgentConfig, BudgetAgent, LLMAdvisor, Proposal, apply_guardrails,  # noqa: E402
                   check_privacy, check_tone, compute_metrics, load_performance_csv, read_performance)
from run_evaluation import run_policy  # noqa: E402
from simulator import MarketSimulator  # noqa: E402

CSV = Path(__file__).resolve().parents[1] / "data" / "ad_performance.csv"
CONFIG = AgentConfig()


def test_csv_has_required_shape():
    df = load_performance_csv(CSV)
    assert df["channel"].nunique() == 3
    assert 14 <= df["date"].nunique() <= 30


def test_guardrails_hold_every_day():
    agent = BudgetAgent(CONFIG)
    run_policy(agent, MarketSimulator(load_performance_csv(CSV), seed=1))
    log = pd.DataFrame(agent.log)
    daily_total = log.groupby("day")["budget_after"].sum()
    assert (daily_total == CONFIG.daily_budget).all(), "cost cap broken"
    assert (log["budget_after"] >= CONFIG.floor).all(), "floor broken"
    assert (log["budget_after"] <= CONFIG.ceiling).all(), "max share broken"
    assert (log["change_pct"].abs() <= CONFIG.max_daily_change * 100 + 1e-9).all(), "+/-20% rule broken"
    assert log["reason"].str.len().gt(20).all(), "every decision needs a reason"


def test_oversized_shift_is_capped():
    current = {c: 500 for c in CHANNELS}
    proposal = Proposal("shift", "Search", ["Display", "Social"], shift_pct=0.90)
    new, notes = apply_guardrails(proposal, current, CONFIG)
    assert new["Search"] == 600 and sum(new.values()) == CONFIG.daily_budget
    assert "Search" in notes


def test_donor_never_goes_below_floor():
    current = {"Search": 900, "Social": 450, "Display": 150}
    proposal = Proposal("shift", "Social", ["Display", "Search"], shift_pct=0.20)
    new, _ = apply_guardrails(proposal, current, CONFIG)
    assert new["Display"] == 150 and new["Social"] == 540 and new["Search"] == 810


def test_no_future_data_leaks():
    df = load_performance_csv(CSV)
    day5 = sorted(df["date"].unique())[4]
    assert read_performance(df, day5)["date"].max() < day5


def test_privacy_guardrail_rejects_pii():
    bad = pd.DataFrame({"date": ["2026-10-01"], "channel": ["Search"], "user_email": ["a@b.com"]})
    try:
        check_privacy(bad)
    except ValueError:
        return
    raise AssertionError("PII column was accepted")


def test_tone_guardrail():
    assert "guaranteed" not in check_tone("Search up 20%, guaranteed win").lower()


def test_llm_advice_is_sanitized_and_falls_back():
    df = load_performance_csv(CSV)
    metrics = compute_metrics(df[df["date"] < sorted(df["date"].unique())[7]], CONFIG)
    current = {c: 500 for c in CHANNELS}
    rule = Proposal("shift", "Search", ["Display", "Social"], 0.2, 0.3)

    greedy_llm = LLMAdvisor("prompt", transport=lambda p: {"content": [{"type": "text", "text":
        '{"action":"shift","increase":"Search","shift_pct":0.9,"rationale":"Search guaranteed!!"}'}]})
    advice = greedy_llm.advise(metrics, current, rule)
    assert advice.shift_pct == 0.20 and "guaranteed" not in advice.note.lower()

    def broken(_):
        raise TimeoutError

    assert LLMAdvisor("prompt", transport=broken).advise(metrics, current, rule) is None


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
