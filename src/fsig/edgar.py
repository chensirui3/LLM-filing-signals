"""SEC EDGAR access: universe, filing index and raw filing downloads.

All responses are cached on disk, so each step can be re-run and resumes where
it stopped.
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from .config import DATA, edgar_user_agent

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"

RAW = DATA / "raw"
FILINGS_INDEX = DATA / "filings.parquet"
UNIVERSE = DATA / "universe.parquet"

_COLS = ("accessionNumber", "filingDate", "reportDate", "form", "primaryDocument")


class EdgarClient:
    """Throttled, retrying HTTP client. The SEC allows 10 requests per second; this client uses 5."""

    def __init__(self, min_interval: float = 0.2):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": edgar_user_agent(), "Accept-Encoding": "gzip, deflate"}
        )
        self.min_interval = min_interval
        self._last = 0.0

    def get(self, url: str) -> requests.Response:
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.session.get(url, timeout=90)
            except requests.RequestException:
                time.sleep(2 * 2**attempt)
                continue
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 * 2**attempt)
                continue
            if r.status_code == 403:
                raise RuntimeError(
                    f"SEC returned 403 for {url}. Check EDGAR_USER_AGENT in .env, "
                    "or wait about 10 minutes if the client has been rate-limited."
                )
            r.raise_for_status()
        raise RuntimeError(f"Request failed after 5 attempts: {url}")

    def get_json(self, url: str, cache: Path | None = None, refresh: bool = False):
        if cache and cache.exists() and not refresh:
            return json.loads(cache.read_text(encoding="utf-8"))
        data = self.get(url).json()
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data), encoding="utf-8")
        return data


def _frame(block: dict) -> pd.DataFrame:
    return pd.DataFrame({k: block[k] for k in _COLS})


def filings_for_cik(client: EdgarClient, cik: int, start_date: str, forms: list[str]) -> pd.DataFrame:
    """All filings of the given forms since start_date for one company.

    The submissions API returns only the most recent 1,000 or so filings in the
    main file. Frequent filers such as banks exceed that within weeks, so older
    pages are fetched as well; each page's date range is used to skip pages
    that end before start_date.
    """
    sub = RAW / "submissions"
    main = client.get_json(
        SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json"), sub / f"CIK{cik:010d}.json"
    )
    frames = [_frame(main["filings"]["recent"])]
    for page in main["filings"].get("files", []):
        if page["filingTo"] >= start_date:
            data = client.get_json(SUBMISSIONS_URL.format(name=page["name"]), sub / page["name"])
            frames.append(_frame(data))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["form"].isin(forms) & (df["filingDate"] >= start_date)]
    df = df[df["primaryDocument"].str.len() > 0]
    df = df.drop_duplicates("accessionNumber").sort_values("filingDate")
    return df.rename(
        columns={
            "accessionNumber": "accession",
            "filingDate": "filing_date",
            "reportDate": "report_date",
            "primaryDocument": "primary_doc",
        }
    ).assign(cik=cik)


def sic_code(client: EdgarClient, cik: int) -> str:
    """The company's four-digit SIC industry code from its submissions file ('' if none)."""
    name = f"CIK{cik:010d}.json"
    return str(client.get_json(SUBMISSIONS_URL.format(name=name), RAW / "submissions" / name).get("sic") or "")


def industries() -> dict[str, str]:
    """Ticker -> two-digit SIC major group, for companies that have a SIC code.

    The code is the company's current classification, not its classification
    at the time of each filing.
    """
    uni = pd.read_parquet(UNIVERSE)
    if "sic" not in uni.columns:
        return {}
    return {t: s[:2] for t, s in zip(uni["ticker"], uni["sic"]) if s}


def build_universe(client: EdgarClient, size: int, start_date: str, forms: list[str]):
    """Pick the first `size` companies in SEC's ticker file that file 10-Ks.

    The SEC file is ordered by current market capitalisation. Selecting the
    universe by current size introduces look-ahead and survivorship bias; see
    the Limitations section of the README.
    """
    raw = client.get_json(TICKERS_URL, RAW / "company_tickers.json")
    candidates = pd.DataFrame(list(raw.values())).drop_duplicates("cik_str")

    members, all_filings = [], []
    bar = tqdm(total=size, desc="universe")
    for row in candidates.itertuples():
        if len(members) >= size:
            break
        cik = int(row.cik_str)
        try:
            f = filings_for_cik(client, cik, start_date, forms)
        except Exception as e:  # one failing company should not abort the run
            tqdm.write(f"skip {row.ticker}: {e}")
            continue
        if (f["form"] == "10-K").sum() == 0:  # foreign filers (20-F), funds and similar
            continue
        f = f.assign(ticker=row.ticker, name=row.title)
        members.append({"cik": cik, "ticker": row.ticker, "name": row.title, "sic": sic_code(client, cik)})
        all_filings.append(f)
        bar.update(1)
    bar.close()

    universe = pd.DataFrame(members)
    filings = pd.concat(all_filings, ignore_index=True)
    filings["filing_date"] = pd.to_datetime(filings["filing_date"])
    filings["report_date"] = pd.to_datetime(filings["report_date"], errors="coerce")
    DATA.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(UNIVERSE, index=False)
    filings.to_parquet(FILINGS_INDEX, index=False)
    return universe, filings


def filing_path(cik: int, accession: str) -> Path:
    return RAW / "filings" / str(cik) / f"{accession}.htm.gz"


def download_filings(client: EdgarClient, filings: pd.DataFrame) -> int:
    """Download each filing's primary document (gzip-compressed). Returns new downloads."""
    new = 0
    for row in tqdm(list(filings.itertuples()), desc="download"):
        path = filing_path(row.cik, row.accession)
        if path.exists():
            continue
        url = ARCHIVE_URL.format(
            cik=row.cik, acc=row.accession.replace("-", ""), doc=row.primary_doc
        )
        try:
            content = client.get(url).content
        except Exception as e:
            tqdm.write(f"failed {row.ticker} {row.accession}: {e}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wb") as f:
            f.write(content)
        tmp.replace(path)
        new += 1
    return new


def load_filings() -> pd.DataFrame:
    return pd.read_parquet(FILINGS_INDEX)
