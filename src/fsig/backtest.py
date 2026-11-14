"""Point-in-time evaluation of filing signals.

Timing rule, used everywhere:
    d0    = first trading day on or after the filing date
    entry = close of d0 + entry_lag_days
A signal is never used before its entry date, and returns are measured from the
entry close. With the default lag of 1 the strategy gives the market a full
day to react before it trades, which is deliberately conservative.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_forward_returns(signals: pd.DataFrame, prices: pd.DataFrame, bt: dict) -> pd.DataFrame:
    """Add entry_date plus raw and market-adjusted forward returns per filing."""
    idx = prices.index
    lag, horizons = bt["entry_lag_days"], bt["horizons"]
    bench = prices[bt["benchmark"]].to_numpy()
    df = signals.copy()
    pos = idx.searchsorted(df["filing_date"].to_numpy(), side="left") + lag
    entry_date = pd.Series(idx[np.minimum(pos, len(idx) - 1)], index=df.index)
    df["entry_date"] = entry_date.where(pos < len(idx))
    for h in horizons:
        df[f"abn_{h}"] = np.nan

    for ticker, rows in df.groupby("ticker").groups.items():
        if ticker not in prices.columns:
            continue
        px = prices[ticker].to_numpy()
        p = pos[df.index.get_indexer(rows)]
        for h in horizons:
            ok_h = p + h < len(idx)
            a, b = np.where(ok_h, p, 0), np.where(ok_h, p + h, 0)
            stock = px[b] / px[a] - 1
            market = bench[b] / bench[a] - 1
            df.loc[rows, f"abn_{h}"] = np.where(ok_h, stock - market, np.nan)
    return df


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return x.rank().corr(y.rank())


def information_coefficients(events: pd.DataFrame, signal_cols: list[str], horizons: list[int],
                             min_names: int = 20) -> pd.DataFrame:
    """Rank IC per calendar quarter, then mean and t-stat across quarters."""
    quarter = events["filing_date"].dt.to_period("Q")
    rows = []
    for sig in signal_cols:
        for h in horizons:
            ics = []
            for _, g in events.groupby(quarter):
                g = g[[sig, f"abn_{h}"]].dropna()
                if len(g) >= min_names and g[sig].nunique() > 1:
                    ics.append(_spearman(g[sig], g[f"abn_{h}"]))
            ics = pd.Series(ics, dtype=float).dropna()
            n = len(ics)
            t = ics.mean() / (ics.std(ddof=1) / math.sqrt(n)) if n > 1 and ics.std(ddof=1) > 0 else np.nan
            rows.append({"signal": sig, "horizon": h, "mean_ic": ics.mean() if n else np.nan,
                         "t_stat": t, "quarters": n})
    return pd.DataFrame(rows)


def long_short(events: pd.DataFrame, prices: pd.DataFrame, signal: str, bt: dict) -> pd.DataFrame:
    """Monthly-rebalanced, equal-weighted top-minus-bottom quantile portfolio.

    At each month-end t every company carries the signal from its latest filing
    whose entry date is <= t and which is younger than max_signal_age_days.
    Positions are held to the next month-end. Costs are charged on turnover.
    """
    q, max_age = bt["quantiles"], pd.Timedelta(days=bt["max_signal_age_days"])
    cost_rate = bt["cost_bps"] / 1e4
    month_ends = prices.index.to_series().groupby(prices.index.to_period("M")).max().to_list()
    ev = events.dropna(subset=[signal, "entry_date"]).sort_values("filing_date")
    ev = ev[ev["ticker"].isin(prices.columns)]

    rows, prev_w = [], pd.Series(dtype=float)
    for t, t_next in zip(month_ends[:-1], month_ends[1:]):
        live = ev[(ev["entry_date"] <= t) & (ev["filing_date"] >= t - max_age)]
        s = live.groupby("ticker")[signal].last().sort_index()
        s = s[prices.loc[t, s.index].notna().to_numpy()]
        if len(s) < 2 * q:
            prev_w = pd.Series(dtype=float)
            continue
        bucket = ((s.rank(method="first") - 1) * q // len(s)).astype(int)
        longs, shorts = s.index[bucket == q - 1], s.index[bucket == 0]
        ret = prices.loc[t_next, s.index] / prices.loc[t, s.index] - 1
        gross = ret[longs].mean() - ret[shorts].mean()

        w = pd.concat([pd.Series(1 / len(longs), index=longs), pd.Series(-1 / len(shorts), index=shorts)])
        turnover = w.sub(prev_w, fill_value=0).abs().sum()
        cost = turnover * cost_rate
        rows.append({"date": t_next, "gross": gross, "cost": cost, "net": gross - cost,
                     "turnover": turnover, "n_long": len(longs), "n_short": len(shorts)})
        prev_w = w
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["gross", "cost", "net", "turnover", "n_long", "n_short"])


def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {"months": n, "ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "t_stat": np.nan, "hit_rate": np.nan, "max_drawdown": np.nan}
    mean, sd = r.mean(), r.std(ddof=1)
    curve = (1 + r).cumprod()
    return {
        "months": n,
        "ann_return": mean * 12,
        "ann_vol": sd * math.sqrt(12),
        "sharpe": mean / sd * math.sqrt(12) if sd > 0 else np.nan,
        "t_stat": mean / (sd / math.sqrt(n)) if sd > 0 else np.nan,
        "hit_rate": (r > 0).mean(),
        "max_drawdown": (curve / curve.cummax() - 1).min(),
    }


def split_stats(monthly: pd.DataFrame, cutoff: str, col: str = "net") -> pd.DataFrame:
    """Performance for the full sample and either side of the model's training cutoff.

    Months after the cutoff are the only ones where the LLM cannot have seen
    what happened next, so that row is the one to trust.
    """
    c = pd.Timestamp(cutoff)
    parts = {"full": monthly, "pre_cutoff": monthly[monthly.index <= c], "post_cutoff": monthly[monthly.index > c]}
    return pd.DataFrame({k: perf_stats(v[col]) for k, v in parts.items()}).T
