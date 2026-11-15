"""Sensitivity of the headline results to methodological choices.

sensitivity         long-short Sharpe ratio for other entry lags and cost levels
variant_comparison  raw versus anonymised LLM scores for the same filing, and
                    whether the difference between them predicts returns
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .backtest import (_ic_stats, add_context_returns, add_forward_returns, information_coefficients,
                       long_short, perf_stats)
from .extract import load_scores
from .signals import SIGNALS

LLM_SIGNALS = ["llm_score", "llm_score_chg", "llm_caution"]


def sensitivity(signals: pd.DataFrame, prices: pd.DataFrame, bt: dict,
                lags: tuple[int, ...] = (0, 1, 2, 5), costs: tuple[int, ...] = (0, 10, 25)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-sample net Sharpe of each long-short portfolio by entry lag and by cost.

    Lag 0 trades at the close of the filing day. Many filings are released
    after the close, so lag 0 is not always achievable; it is reported as an
    upper bound.
    """
    sig_cols = [s for s in SIGNALS if s in signals.columns and signals[s].notna().any()]
    by_lag, by_cost = {}, {}
    for lag in lags:
        cfg = {**bt, "entry_lag_days": lag}
        events = add_forward_returns(signals, prices, cfg)
        by_lag[f"lag_{lag}"] = {s: perf_stats(long_short(events, prices, s, cfg)["net"])["sharpe"] for s in sig_cols}
    events = add_forward_returns(signals, prices, bt)
    for cost in costs:
        cfg = {**bt, "cost_bps": cost}
        by_cost[f"{cost}_bps"] = {s: perf_stats(long_short(events, prices, s, cfg)["net"])["sharpe"] for s in sig_cols}
    return (pd.DataFrame(by_lag).rename_axis("signal"), pd.DataFrame(by_cost).rename_axis("signal"))


def full_text_check(signals: pd.DataFrame, llm: dict, prices: pd.DataFrame, bt: dict) -> tuple[int, pd.DataFrame]:
    """Does scoring only the first chunks of each MD&A change the result?

    Uses the filings whose whole MD&A has been scored (`extract --all-chunks`)
    and is longer than the part the signals read. For those filings it compares
    the signal built from the first llm['max_chunks'] chunks with the same
    signal built from every chunk: their correlation, and the rank IC of each
    against returns before the filing and after entry. Returns (number of
    filings, table).
    """
    full = load_scores(llm, "raw", all_chunks=True)
    full = full[full["n_chunks"] > llm["max_chunks"]].copy()
    full["llm_score_full"] = full[["tone", "outlook", "demand", "margins"]].mean(axis=1)
    full["llm_caution_full"] = -(full["liquidity_stress"] + full["uncertainty"])
    df = signals.merge(full[["accession", "llm_score_full", "llm_caution_full"]], on="accession")
    df = df.sort_values(["ticker", "filing_date"])
    df["llm_score_chg_full"] = df.groupby("ticker")["llm_score_full"].diff()
    df["llm_score_chg"] = df.groupby("ticker")["llm_score"].diff()  # previous filing within this subset
    horizon = max(bt["horizons"])
    ev = add_context_returns(add_forward_returns(df, prices, bt), prices, bt)
    rows = []
    for s in LLM_SIGNALS:
        for label, col in (("first chunks", s), ("whole section", f"{s}_full")):
            before = _ic_stats(ev, col, "abn_before", 4 * bt["quantiles"])
            after = _ic_stats(ev, col, f"abn_{horizon}", 4 * bt["quantiles"])
            rows.append({"signal": s, "text_scored": label,
                         "correlation_with_whole": ev[s].corr(ev[f"{s}_full"]) if label == "first chunks" else 1.0,
                         "ic_before_filing": before["mean_ic"], "t_before": before["t_stat"],
                         "ic_after_entry": after["mean_ic"], "t_after": after["t_stat"]})
    return len(df), pd.DataFrame(rows)


def variant_comparison(raw: pd.DataFrame, anon: pd.DataFrame, prices: pd.DataFrame, bt: dict,
                       cutoff: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw-text and anonymised-text scores filing by filing.

    Returns (agreement, gap_ic).

    agreement: correlation between the two scores and the mean of raw - anon
    on either side of the cutoff. If knowing the company changed the model's
    assessment, the scores would diverge.

    gap_ic: rank IC of (raw - anon) llm_score against forward returns. This is
    the direct leakage test: if the model shifted its score towards outcomes
    it had memorised, the gap would predict returns before the cutoff and not
    after it.
    """
    c = pd.Timestamp(cutoff)
    m = raw.merge(anon[["accession"] + LLM_SIGNALS], on="accession", suffixes=("", "_anon"))
    pre = m["filing_date"] <= c

    rows = []
    for s in LLM_SIGNALS:
        d = m.dropna(subset=[s, f"{s}_anon"])
        row = {"signal": s, "n": len(d), "correlation": d[s].corr(d[f"{s}_anon"])}
        for name, mask in (("pre", pre), ("post", ~pre)):
            gap = (d[s] - d[f"{s}_anon"])[mask.loc[d.index]]
            row[f"gap_{name}"] = gap.mean()
            sd = gap.std(ddof=1) if len(gap) > 1 else 0.0
            row[f"t_{name}"] = gap.mean() / (sd / math.sqrt(len(gap))) if sd > 0 else np.nan
        rows.append(row)
    agreement = pd.DataFrame(rows).set_index("signal")

    m["gap"] = m["llm_score"] - m["llm_score_anon"]
    events = add_forward_returns(m, prices, bt)
    parts = []
    for name, mask in (("pre_cutoff", pre), ("post_cutoff", ~pre)):
        ic = information_coefficients(events[mask.to_numpy()], ["gap"], bt["horizons"], min_names=4 * bt["quantiles"])
        parts.append(ic.drop(columns="signal").assign(period=name))
    gap_ic = pd.concat(parts, ignore_index=True)[["period", "horizon", "mean_ic", "t_stat", "quarters"]]
    return agreement, gap_ic
