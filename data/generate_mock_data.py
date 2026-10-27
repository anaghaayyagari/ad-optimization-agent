"""
Generate 28 days of mock ad performance data for 3 channels.

Output columns: date, channel, spend, impressions, clicks, conversions

The history is an EQUAL-SPLIT campaign ($500/channel/day, $1,500/day total),
which is also our evaluation baseline. Each channel has a different "true"
performance profile so the agent has something real to learn:

  Search  - expensive impressions, high CTR and CVR  -> best CPA in weeks 1-2
  Social  - cheap impressions, average CVR; a creative refresh on day 15
            lifts CTR/CVR                              -> best CPA in weeks 3-4
  Display - very cheap impressions, low CTR and CVR    -> worst CPA throughout

Run:  python data/generate_mock_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 10
START_DATE = "2026-09-28"
NUM_DAYS = 28
DAILY_SPEND_PER_CHANNEL = 500.0  # equal split of a $1,500 daily budget
CREATIVE_REFRESH_DAY = 15        # Social gets a new creative on this day (1-indexed)

# "True" market parameters per channel (unknown to the agent).
#   cpm = cost per 1,000 impressions, ctr = clicks / impressions,
#   cvr = conversions / clicks
CHANNEL_PROFILES = {
    "Search":  {"cpm": 70.0, "ctr": 0.030,  "cvr": 0.070},
    "Social":  {"cpm": 9.0,  "ctr": 0.009,  "cvr": 0.024},
    "Display": {"cpm": 3.0,  "ctr": 0.0025, "cvr": 0.013},
}
SOCIAL_AFTER_REFRESH = {"ctr": 0.011, "cvr": 0.038}

# Weekends convert a little worse for Search, a little better for Social.
WEEKEND_CVR_MULTIPLIER = {"Search": 0.90, "Social": 1.08, "Display": 1.00}


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=NUM_DAYS, freq="D")
    rows = []

    for day_index, date in enumerate(dates, start=1):
        is_weekend = date.dayofweek >= 5
        for channel, profile in CHANNEL_PROFILES.items():
            ctr, cvr = profile["ctr"], profile["cvr"]
            if channel == "Social" and day_index >= CREATIVE_REFRESH_DAY:
                ctr, cvr = SOCIAL_AFTER_REFRESH["ctr"], SOCIAL_AFTER_REFRESH["cvr"]
            if is_weekend:
                cvr *= WEEKEND_CVR_MULTIPLIER[channel]

            # Day-to-day auction noise on price and engagement.
            cpm = profile["cpm"] * rng.lognormal(0, 0.08)
            ctr_today = ctr * rng.lognormal(0, 0.06)
            cvr_today = cvr * rng.lognormal(0, 0.10)

            spend = DAILY_SPEND_PER_CHANNEL
            impressions = int(spend / cpm * 1000)
            clicks = int(rng.binomial(impressions, min(ctr_today, 1.0)))
            conversions = int(rng.binomial(clicks, min(cvr_today, 1.0)))

            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "channel": channel,
                "spend": round(spend, 2),
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_path = Path(__file__).with_name("ad_performance.csv")
    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({df['date'].nunique()} days x "
          f"{df['channel'].nunique()} channels) to {out_path}")
    summary = df.groupby("channel")[["spend", "clicks", "conversions"]].sum()
    summary["CPA"] = (summary["spend"] / summary["conversions"]).round(2)
    print(summary)
