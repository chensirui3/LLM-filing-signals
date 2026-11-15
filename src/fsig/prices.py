"""Daily adjusted close prices from Yahoo Finance and monthly factor returns
from Kenneth French's data library, cached to parquet."""
from __future__ import annotations

import io
import re
import zipfile

import pandas as pd
import requests

from .config import DATA

PRICES = DATA / "prices.parquet"
FACTORS = DATA / "factors.parquet"
FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
FACTOR_COLS = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]


def download_prices(tickers: list[str], start: str, benchmark: str) -> pd.DataFrame:
    import yfinance as yf

    symbols = sorted(set(tickers) | {benchmark})
    # Start slightly before the sample so the earliest filings have a price history.
    begin = (pd.Timestamp(start) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    # threads=False: yfinance's on-disk cache is SQLite and locks under parallel writes.
    raw = yf.download(symbols, start=begin, auto_adjust=True, progress=False, threads=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].set_axis(symbols, axis=1)
    # Retry any symbol that came back empty from the batch request.
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


def _french_monthly(name: str) -> pd.DataFrame:
    """Monthly rows of one CSV from Kenneth French's data library, as decimal returns."""
    r = requests.get(FRENCH_URL.format(name=name), timeout=120)
    r.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(r.content))
    lines = archive.read(archive.namelist()[0]).decode("latin-1").splitlines()
    header = next(l for l in lines if l.startswith(","))
    cols = [c.strip() for c in header.split(",")[1:]]
    rows = {}
    for l in lines:  # monthly rows are dated YYYYMM; the annual rows that follow are dated YYYY
        parts = [p.strip() for p in l.split(",")]
        if re.fullmatch(r"\d{6}", parts[0]) and len(parts) == len(cols) + 1:
            rows[pd.Period(parts[0], "M")] = [float(p) / 100 for p in parts[1:]]
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols)


def download_factors() -> pd.DataFrame:
    """Fama-French five factors and momentum, monthly, indexed by month.

    Missing values in the source (-99.99) are left out.
    """
    df = _french_monthly("F-F_Research_Data_5_Factors_2x3").join(_french_monthly("F-F_Momentum_Factor"), how="inner")
    df = df.rename(columns={"Mkt-RF": "MKT", "Mom": "MOM"})[FACTOR_COLS]
    df = df[(df > -0.99).all(axis=1)]
    df.index = df.index.astype(str)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FACTORS)
    return df


def get_factors() -> pd.DataFrame | None:
    """The factor table with a monthly PeriodIndex, downloaded on first use.

    Returns None if the data library cannot be reached; the factor regression
    is optional and the rest of the pipeline does not depend on it.
    """
    if not FACTORS.exists():
        try:
            download_factors()
        except (requests.RequestException, zipfile.BadZipFile, StopIteration, KeyError):
            return None
    df = pd.read_parquet(FACTORS)
    df.index = pd.PeriodIndex(df.index, freq="M")
    return df
