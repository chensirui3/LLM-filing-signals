"""Daily adjusted close prices from Yahoo Finance, cached to parquet."""
from __future__ import annotations

import pandas as pd

from .config import DATA

PRICES = DATA / "prices.parquet"


def download_prices(tickers: list[str], start: str, benchmark: str) -> pd.DataFrame:
    import yfinance as yf

    symbols = sorted(set(tickers) | {benchmark})
    # Start a little early so the first filings have a price history.
    begin = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    # threads=False: yfinance's on-disk cache is SQLite and locks under parallel writes.
    raw = yf.download(symbols, start=begin, auto_adjust=True, progress=False, threads=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].set_axis(symbols, axis=1)
    for sym in [s for s in symbols if s not in close.columns or close[s].isna().all()]:
        retry = yf.download(sym, start=begin, auto_adjust=True, progress=False, threads=False)
        if not retry.empty:
            close[sym] = retry["Close"].squeeze()
    close = close.dropna(how="all").sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.columns = [str(c) for c in close.columns]
    DATA.mkdir(parents=True, exist_ok=True)
    close.to_parquet(PRICES)
    return close


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PRICES)
