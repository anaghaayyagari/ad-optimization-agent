"""
Market simulator used for offline evaluation.

The CSV tells us what each channel delivered on each day at its historical
spend. To ask "what would have happened with a different budget?" we replay
each day using that day's observed CPM, CTR and CVR, plus one assumption:

  Diminishing returns — doubling a channel's spend does not double its
  impressions. impressions = hist_impressions * (spend / hist_spend) ** (1 - elasticity)

Clicks and conversions are then drawn with binomial noise (seeded, so runs
are reproducible). Every policy is evaluated on the same simulator and seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketSimulator:
    def __init__(self, data: pd.DataFrame, elasticity: float = 0.35, seed: int = 42, noise: bool = True):
        self.data = data.set_index(["date", "channel"]).sort_index()
        self.dates = sorted(data["date"].unique())
        self.channel_ids = {c: i for i, c in enumerate(sorted(data["channel"].unique()))}
        self.elasticity = elasticity
        self.seed = seed
        self.noise = noise

    def run_day(self, date, budget: dict) -> pd.DataFrame:
        rows = []
        for channel, spend in budget.items():
            hist = self.data.loc[(date, channel)]
            ctr = hist.clicks / hist.impressions if hist.impressions else 0.0
            cvr = hist.conversions / hist.clicks if hist.clicks else 0.0

            if spend <= 0 or hist.spend <= 0:
                impressions = 0
            else:
                scale = (spend / hist.spend) ** (1 - self.elasticity)
                impressions = int(round(hist.impressions * scale))

            if self.noise:
                # One RNG per (seed, day, channel) -> common random numbers across policies.
                rng = np.random.default_rng([self.seed, pd.Timestamp(date).dayofyear,
                                             self.channel_ids[channel]])
                clicks = int(rng.binomial(impressions, ctr))
                conversions = int(rng.binomial(clicks, cvr))
            else:
                clicks = impressions * ctr
                conversions = clicks * cvr

            rows.append({"date": date, "channel": channel, "spend": float(spend),
                         "impressions": impressions, "clicks": clicks, "conversions": conversions})
        return pd.DataFrame(rows)
