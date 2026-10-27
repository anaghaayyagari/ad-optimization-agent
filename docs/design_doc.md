# Design Doc — BudgetPilot Ad Optimization Agent

**Group:** Anagha Ayyagari, Shayan, Arshdeep · BUS4-118S Sec 02

BudgetPilot splits a fixed **$1,500/day** budget across Search, Social and Display to **maximize total
conversions** while keeping cost per acquisition (CPA) low.

**Agent role.** A cautious marketing-ops agent. Each morning it reads yesterday's results, re-estimates each
channel's efficiency, and moves 10–20% of budget toward the channel with the cheapest conversions. It changes
budgets only, never ad copy, bids or targeting.

| Item | Details |
|---|---|
| Inputs | CSV with `date, channel, spend, impressions, clicks, conversions` (aggregate, one row per channel per day), plus the daily budget and guardrail settings |
| Outputs | Tomorrow's budget per channel in $ (always sums to $1,500); one decision-log row per channel per day: budget before/after, % change, estimated CTR/CVR/CPA, action, reason string |

**Decision loop (runs once a day).**

1. `read_performance` loads only days before today, so there is no look-ahead.
2. `compute_metrics` estimates CTR, CVR and CPA over the last 7 days, weighting recent days more (3-day half-life). Estimates update every day; this is the learning loop.
3. `propose_budget` holds an equal split for a 2-day warm-up. After that, if the best channel's CPA beats the portfolio CPA by 5% / 10% / 20%, it grows that channel by 10% / 15% / 20%, funded by the worst channel first. The shift is halved if yesterday's CPA spiked more than 50%. Otherwise it holds.
4. `apply_guardrails` enforces the limits below. Its output is final, even over the optional LLM advisor.
5. `log_decision` writes a reason per channel, e.g. "Search up 20% ($500->$600) due to better CPA $28.08 vs $47.33 portfolio, stable CPA."

**Guardrails.**

| Guardrail | Rule |
|---|---|
| Cost cap | Total spend is exactly $1,500/day |
| Max daily change | ±20% per channel per day |
| Minimum floor | At least 10% ($150) per channel, so no channel is ever at 0% and every channel keeps producing data to learn from |
| Max share | At most 60% ($900) per channel |
| Privacy | Aggregate channel-level data only; CSV columns that look like personal data (email, user ID, IP, phone) are rejected |
| Brand tone | Budget changes only; reason strings stay neutral and hype words ("guaranteed", "viral") are stripped |
| LLM safety | Optional LLM advice is clamped to a 10–20% shift and re-checked; on any error the rules decide |

**Evaluation metric.** Primary: total conversions over 28 days compared with an equal-split baseline at the
same $42,000 spend. Secondary: average CPA, CTR and CVR. Every policy is replayed on the same simulated market
(each day's observed rates, with diminishing returns as spend grows), and the comparison is repeated over
100 random seeds.
