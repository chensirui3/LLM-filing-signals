"""Point-in-time evaluation of filing signals.

Timing rule, used everywhere:
    d0    = first trading day on or after the filing date
    entry = close of d0 + entry_lag_days
A signal is never used before its entry date, and returns are measured from the
entry close. With the default lag of 1 the market has a full trading day to
react before the strategy trades, which is a deliberately conservative choice.
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


def add_context_returns(events: pd.DataFrame, prices: pd.DataFrame, bt: dict, lookback: int = 63) -> pd.DataFrame:
    """Add market-adjusted returns before and around each filing.

    abn_before  return over the `lookback` trading days that end at the close
                before d0, i.e. the period the filing reports on
    abn_filing  return from the close before d0 to the entry close: the
                market's reaction to the filing and to any earnings release
                that accompanies it. The strategy enters after this window and
                does not capture it.
    """
    idx, lag = prices.index, bt["entry_lag_days"]
    bench = prices[bt["benchmark"]].to_numpy()
    df = events.copy()
    d0 = idx.searchsorted(df["filing_date"].to_numpy(), side="left")
    windows = {"abn_before": (d0 - 1 - lookback, d0 - 1), "abn_filing": (d0 - 1, d0 + lag)}
    for col in windows:
        df[col] = np.nan
    for ticker, rows in df.groupby("ticker").groups.items():
        if ticker not in prices.columns:
            continue
        px, loc = prices[ticker].to_numpy(), df.index.get_indexer(rows)
        for col, (a, b) in windows.items():
            a, b = a[loc], b[loc]
            ok = (a >= 0) & (b < len(idx))
            a, b = np.where(ok, a, 0), np.where(ok, b, 0)
            df.loc[rows, col] = np.where(ok, px[b] / px[a] - 1 - (bench[b] / bench[a] - 1), np.nan)
    return df


def _spearman(x: pd.Series, y: pd.Series) -> float:
    return x.rank().corr(y.rank())


# Multiple of the standard error that an effect must reach to be detected with
# 80% power in a two-sided test at the 5% level (1.96 + 0.84).
DETECTABLE = 2.8


def _ic_stats(events: pd.DataFrame, sig: str, ret: str, min_names: int) -> dict:
    """Rank IC of `sig` against `ret` per calendar quarter, summarised across quarters."""
    ics = []
    for _, g in events.groupby(events["filing_date"].dt.to_period("Q")):
        g = g[[sig, ret]].dropna()
        if len(g) >= min_names and g[sig].nunique() > 1:
            ics.append(_spearman(g[sig], g[ret]))
    ics = pd.Series(ics, dtype=float).dropna()
    n = len(ics)
    se = ics.std(ddof=1) / math.sqrt(n) if n > 1 else np.nan
    return {"mean_ic": ics.mean() if n else np.nan, "t_stat": ics.mean() / se if n > 1 and se > 0 else np.nan,
            "quarters": n, "se": se, "min_detectable_ic": DETECTABLE * se}


def information_coefficients(events: pd.DataFrame, signal_cols: list[str], horizons: list[int],
                             min_names: int = 20) -> pd.DataFrame:
    """Rank IC per calendar quarter, then mean, t-stat and precision across quarters.

    min_detectable_ic is the smallest true IC the sample would detect with 80%
    power at the 5% level. A null result rules out effects larger than this,
    not smaller ones.
    """
    return pd.DataFrame([{"signal": sig, "horizon": h, **_ic_stats(events, sig, f"abn_{h}", min_names)}
                         for sig in signal_cols for h in horizons])


def timing_profile(events: pd.DataFrame, signal_cols: list[str], cutoff: str, horizon: int,
                   min_names: int = 20) -> pd.DataFrame:
    """Rank IC of each signal against returns before, around and after the filing.

    Needs the columns from add_context_returns and add_forward_returns. A signal
    that reflects information the market already has correlates with the first
    two windows but not with the third. Each cell is "IC (t-statistic)".
    """
    c = pd.Timestamp(cutoff)
    periods = {"full": events, "pre_cutoff": events[events["filing_date"] <= c],
               "post_cutoff": events[events["filing_date"] > c]}
    windows = {"before_filing": "abn_before", "filing_window": "abn_filing", "after_entry": f"abn_{horizon}"}
    rows = []
    for sig in signal_cols:
        for period, ev in periods.items():
            row = {"signal": sig, "period": period}
            for name, col in windows.items():
                s = _ic_stats(ev, sig, col, min_names)
                row[name] = f"{s['mean_ic']:.3f} ({s['t_stat']:.1f})" if s["quarters"] > 1 else ""
            rows.append(row)
    return pd.DataFrame(rows)


def quantile_weights(s: pd.Series, q: int) -> pd.Series:
    """Long-short weights: the top 1/q of names long and the bottom 1/q short.

    Names with equal signal values share a rank range, so a group of tied names
    that straddles a quantile boundary is split evenly across it rather than
    ordered arbitrarily. Many LLM scores are tied (llm_caution takes 13
    values), and any fixed tie-break such as ticker order would decide most of
    each leg. Long weights sum to 1 and short weights to -1; if every name is
    tied, all weights are zero.
    """
    n = len(s)
    k = n / q
    below, upto = s.rank(method="min") - 1, s.rank(method="max")  # the group occupies ranks [below, upto)
    size = upto - below
    top = (np.maximum(upto, n - k) - np.maximum(below, n - k)).clip(lower=0) / size
    bottom = (np.minimum(upto, k) - np.minimum(below, k)).clip(lower=0) / size
    return (top - bottom) / k


def _leg_return(w: pd.Series, ret: pd.Series) -> float:
    """Weighted return of one leg, renormalised over the names that have a return."""
    ok = ret.notna() & (w > 0)
    return float((w[ok] * ret[ok]).sum() / w[ok].sum()) if ok.any() else 0.0


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
        w = quantile_weights(s, q)
        ret = prices.loc[t_next, s.index] / prices.loc[t, s.index] - 1
        gross = _leg_return(w.clip(lower=0), ret) - _leg_return((-w).clip(lower=0), ret)

        turnover = w.sub(prev_w, fill_value=0).abs().sum()
        cost = turnover * cost_rate
        rows.append({"date": t_next, "gross": gross, "cost": cost, "net": gross - cost,
                     "turnover": turnover, "n_long": int((w > 0).sum()), "n_short": int((w < 0).sum())})
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

    Months after the cutoff are the only ones in which the LLM cannot have seen
    subsequent events, so only that row is free of possible look-ahead.
    """
    c = pd.Timestamp(cutoff)
    parts = {"full": monthly, "pre_cutoff": monthly[monthly.index <= c], "post_cutoff": monthly[monthly.index > c]}
    return pd.DataFrame({k: perf_stats(v[col]) for k, v in parts.items()}).T
