"""
Run the BudgetPilot agent on the CSV and compare it with an equal-split baseline.

Usage:
    python run_evaluation.py                      # default: data/ad_performance.csv
    python run_evaluation.py --csv my_data.csv --budget 2000 --seed 7
    python run_evaluation.py --llm                # optional, needs ANTHROPIC_API_KEY

Writes to outputs/:
    decision_log.csv / decision_log.jsonl   one row per channel per day with a reason
    daily_results.csv                       simulated daily results for every policy
    summary.json                            totals, CPA, CTR, uplift, robustness
    budget_allocation.png                   how the agent moved money over time
    cumulative_conversions.png              agent vs baseline (vs greedy ablation)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from agent import CHANNELS, AgentConfig, BudgetAgent, EqualSplitBaseline, GreedyNoGuardrails, load_performance_csv
from simulator import MarketSimulator

ROOT = Path(__file__).parent

# Chart styling (validated categorical palette, light surface)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CHANNEL_COLORS = {"Search": "#2a78d6", "Social": "#eb6834", "Display": "#1baf7a"}
POLICY_STYLES = {
    "BudgetPilot agent": {"color": "#2a78d6", "ls": "-"},
    "Equal-split baseline": {"color": "#8a8984", "ls": "--"},
    "Greedy (no guardrails)": {"color": "#eb6834", "ls": ":"},
}


# --------------------------------------------------------------------------
# Simulation loop
# --------------------------------------------------------------------------
def run_policy(policy, sim: MarketSimulator) -> pd.DataFrame:
    """Each morning the policy picks budgets; the simulator returns that day's results."""
    history = pd.DataFrame(columns=["date", "channel", "spend", "impressions", "clicks", "conversions"])
    for day_number, date in enumerate(sim.dates, start=1):
        budget = policy.decide(history, pd.Timestamp(date), day_number)
        results = sim.run_day(date, budget)
        history = results if history.empty else pd.concat([history, results], ignore_index=True)
    history["policy"] = policy.name
    return history


def summarize(results: pd.DataFrame) -> dict:
    t = results[["spend", "impressions", "clicks", "conversions"]].sum()
    return {
        "spend": round(float(t.spend), 2),
        "impressions": int(t.impressions),
        "clicks": int(round(t.clicks)),
        "conversions": int(round(t.conversions)),
        "cpa": round(float(t.spend / t.conversions), 2),
        "ctr_pct": round(float(t.clicks / t.impressions * 100), 3),
        "cvr_pct": round(float(t.conversions / t.clicks * 100), 3),
    }


def robustness(data: pd.DataFrame, config: AgentConfig, elasticity: float, n_seeds: int) -> dict:
    """Repeat the comparison over many noise seeds to make sure the win isn't luck."""
    uplifts = []
    for seed in range(n_seeds):
        sim = MarketSimulator(data, elasticity=elasticity, seed=1000 + seed)
        agent_conv = run_policy(BudgetAgent(config), sim)["conversions"].sum()
        base_conv = run_policy(EqualSplitBaseline(config.daily_budget), sim)["conversions"].sum()
        uplifts.append((agent_conv - base_conv) / base_conv * 100)
    s = pd.Series(uplifts)
    return {"seeds": n_seeds, "mean_uplift_pct": round(s.mean(), 2), "min_uplift_pct": round(s.min(), 2),
            "max_uplift_pct": round(s.max(), 2), "agent_wins_pct": round((s > 0).mean() * 100, 1)}


def sensitivity(data: pd.DataFrame, config: AgentConfig, seed: int) -> list[dict]:
    """How much does the result depend on the diminishing-returns assumption?"""
    rows = []
    for e in [0.20, 0.35, 0.50]:
        sim = MarketSimulator(data, elasticity=e, seed=seed)
        a = run_policy(BudgetAgent(config), sim)["conversions"].sum()
        b = run_policy(EqualSplitBaseline(config.daily_budget), sim)["conversions"].sum()
        rows.append({"elasticity": e, "agent_conversions": int(a), "baseline_conversions": int(b),
                     "uplift_pct": round((a - b) / b * 100, 2)})
    return rows


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_budget_allocation(log: pd.DataFrame, path: Path, refresh_day: int | None = None):
    share = log.pivot(index="day", columns="channel", values="budget_after")[CHANNELS]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    ax.stackplot(share.index, [share[c] for c in CHANNELS], labels=CHANNELS,
                 colors=[CHANNEL_COLORS[c] for c in CHANNELS], edgecolor=SURFACE, linewidth=2, alpha=0.95)
    _style(ax, "Agent's daily budget by channel ($1,500/day cap)", "Daily budget ($)")
    ax.set_xlabel("Day", color=INK_2, fontsize=10)
    ax.set_xlim(share.index.min(), share.index.max())
    ax.set_xticks([1, 7, 14, 21, 28])
    ax.set_ylim(0, share.sum(axis=1).max())
    # Direct labels at the right edge, centred in each band
    last = share.iloc[-1]
    bottom = 0
    for c in CHANNELS:
        ax.text(share.index.max() - 0.3, bottom + last[c] / 2, f"{c} ${last[c]:,.0f}",
                ha="right", va="center", fontsize=9, color="white", fontweight="bold")
        bottom += last[c]
    if refresh_day:
        ax.axvline(refresh_day, color=INK, linewidth=1, linestyle="--")
        ax.text(refresh_day + 0.2, share.sum(axis=1).max() * 0.97, "Social creative refresh",
                fontsize=9, color=INK, va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_cumulative(results: pd.DataFrame, path: Path):
    daily = results.groupby(["policy", "date"])["conversions"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    for policy, g in daily.groupby("policy", sort=False):
        cum = g["conversions"].cumsum().to_numpy()
        days = range(1, len(cum) + 1)
        st = POLICY_STYLES.get(policy, {"color": INK_2, "ls": "-"})
        ax.plot(days, cum, color=st["color"], linestyle=st["ls"], linewidth=2, label=policy)
        ax.text(len(cum) + 0.3, cum[-1], f"{int(cum[-1]):,}", color=INK, fontsize=9, va="center")
    _style(ax, "Cumulative conversions over 28 days", "Conversions")
    ax.set_xlabel("Day", color=INK_2, fontsize=10)
    ax.set_xlim(1, daily["date"].nunique() + 2.5)
    ax.set_xticks([1, 7, 14, 21, 28])
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ad budget optimization agent — evaluation")
    parser.add_argument("--csv", default=str(ROOT / "data" / "ad_performance.csv"))
    parser.add_argument("--out", default=str(ROOT / "outputs"))
    parser.add_argument("--budget", type=float, default=1500.0, help="total daily budget (cost cap)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--elasticity", type=float, default=0.35, help="diminishing-returns strength")
    parser.add_argument("--seeds", type=int, default=100, help="noise seeds for the robustness check")
    parser.add_argument("--llm", action="store_true", help="let an LLM advisor propose shifts (optional)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_performance_csv(args.csv)
    config = AgentConfig(daily_budget=args.budget)
    print(f"Loaded {len(data)} rows: {data['date'].nunique()} days x {data['channel'].nunique()} channels")

    # 1) Run all policies on the same simulated market (same seed = fair comparison)
    sim = MarketSimulator(data, elasticity=args.elasticity, seed=args.seed)
    agent = BudgetAgent(config, use_llm=args.llm)
    policies = [agent, EqualSplitBaseline(args.budget), GreedyNoGuardrails(args.budget)]
    results = pd.concat([run_policy(p, sim) for p in policies], ignore_index=True)

    # 2) Decision log
    log = pd.DataFrame(agent.log)
    log.to_csv(out / "decision_log.csv", index=False)
    with open(out / "decision_log.jsonl", "w") as f:
        for row in agent.log:
            f.write(json.dumps(row) + "\n")
    results.assign(date=results["date"].astype(str)).to_csv(out / "daily_results.csv", index=False)

    # 3) Evaluation
    summary = {p.name: summarize(results[results["policy"] == p.name]) for p in policies}
    base, ag = summary["Equal-split baseline"], summary["BudgetPilot agent"]
    comparison = {
        "conversion_uplift_pct": round((ag["conversions"] - base["conversions"]) / base["conversions"] * 100, 2),
        "extra_conversions": ag["conversions"] - base["conversions"],
        "cpa_change_pct": round((ag["cpa"] - base["cpa"]) / base["cpa"] * 100, 2),
    }
    print("Running robustness check over", args.seeds, "noise seeds ...")
    robust = robustness(data, config, args.elasticity, args.seeds)
    sens = sensitivity(data, config, args.seed)
    by_channel = (results.groupby(["policy", "channel"])[["spend", "conversions"]].sum()
                  .assign(cpa=lambda d: (d.spend / d.conversions).round(2)).reset_index())
    report = {"config": config.__dict__ | {"edge_to_shift": list(config.edge_to_shift)},
              "seed": args.seed, "elasticity": args.elasticity, "totals": summary,
              "agent_vs_baseline": comparison, "robustness": robust, "elasticity_sensitivity": sens,
              "by_channel": by_channel.to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=str))

    # 4) Charts
    plot_budget_allocation(log, out / "budget_allocation.png", refresh_day=15)
    plot_cumulative(results, out / "cumulative_conversions.png")

    # 5) Console report
    print("\n=== Sample decision log (first 6 decision days after warm-up) ===")
    for _, r in log[(log.day >= 3) & (log.day <= 8)].iterrows():
        print(f"{r.date}  {r.reason}")
    print("\n=== Results (28 days, $%s/day) ===" % f"{args.budget:,.0f}")
    print(pd.DataFrame(summary).T[["spend", "conversions", "cpa", "ctr_pct", "cvr_pct"]].to_string())
    print(f"\nAgent vs baseline: {comparison['conversion_uplift_pct']:+.1f}% conversions "
          f"({comparison['extra_conversions']:+d}), CPA {comparison['cpa_change_pct']:+.1f}%")
    print(f"Robustness ({robust['seeds']} seeds): mean uplift {robust['mean_uplift_pct']:+.1f}%, "
          f"range {robust['min_uplift_pct']:+.1f}% to {robust['max_uplift_pct']:+.1f}%, "
          f"agent wins {robust['agent_wins_pct']:.0f}% of runs")
    print("Elasticity sensitivity:", ", ".join(f"e={s['elasticity']}: {s['uplift_pct']:+.1f}%" for s in sens))
    print(f"\nOutputs written to {out}/")


if __name__ == "__main__":
    main()
