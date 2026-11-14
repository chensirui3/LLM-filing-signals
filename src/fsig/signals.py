"""Assemble one row per filing with the LLM signals and a non-LLM baseline."""
from __future__ import annotations

import math
import re
from collections import Counter

import pandas as pd
from tqdm import tqdm

from .config import DATA
from .extract import load_scores
from .sections import load_section

_WORD = re.compile(r"[a-z]{2,}")

# Signals the backtest evaluates. Higher value = expected higher return.
SIGNALS = {
    "llm_score": "LLM: mean of tone, outlook, demand, margins (level)",
    "llm_score_chg": "LLM: change in llm_score vs the company's previous filing",
    "llm_caution": "LLM: minus (liquidity stress + uncertainty)",
    "sim_yoy": "Baseline: word-count similarity to the same filing a year earlier (Lazy Prices)",
    "emb_sim_yoy": "Embeddings: whole-document similarity to the same filing a year earlier",
    "emb_align_yoy": "Embeddings: passage-level match with the same filing a year earlier",
}


def bag(text: str) -> Counter:
    return Counter(_WORD.findall(text.lower()))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return float("nan")
    small, large = (a, b) if len(a) < len(b) else (b, a)
    dot = sum(v * large.get(k, 0) for k, v in small.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def similarity_signal(filings: pd.DataFrame) -> pd.Series:
    """Cosine similarity of each MD&A with the same form filed about a year earlier.

    This replicates the idea of Cohen, Malloy & Nguyen (2020), "Lazy Prices":
    companies that change their filing language a lot tend to underperform. It
    needs no LLM, so it is the bar the LLM signal has to clear.
    """
    out = {}
    for _, grp in tqdm(filings.sort_values("filing_date").groupby(["ticker", "form"]), desc="similarity"):
        history: list[tuple[pd.Timestamp, Counter]] = []
        for r in grp.itertuples():
            text = load_section(r.accession, "mdna")
            if not text:
                continue
            cur = bag(text)
            prior = [(abs((r.filing_date - d).days - 365), b) for d, b in history
                     if 320 <= (r.filing_date - d).days <= 410]
            if prior:
                out[r.accession] = cosine(cur, min(prior, key=lambda x: x[0])[1])
            history.append((r.filing_date, cur))
    return pd.Series(out, name="sim_yoy", dtype=float)


def build_signals(filings: pd.DataFrame, llm: dict, variant: str = "raw", emb: dict | None = None) -> pd.DataFrame:
    df = filings.merge(load_scores(llm, variant), on="accession", how="left")
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

    df["llm_score"] = df[["tone", "outlook", "demand", "margins"]].mean(axis=1)
    df["llm_caution"] = -(df["liquidity_stress"] + df["uncertainty"])
    # Change vs the previous *scored* filing of the same company.
    scored = df.dropna(subset=["llm_score"])
    df["llm_score_chg"] = scored.groupby("ticker")["llm_score"].diff()

    df = df.merge(similarity_signal(filings).rename_axis("accession").reset_index(), on="accession", how="left")
    if emb:  # NaN columns if the embed step has not been run
        from .embed import embedding_signals

        df = df.merge(embedding_signals(filings, emb), on="accession", how="left")
    df.to_parquet(DATA / f"signals_{variant}.parquet", index=False)
    return df


def load_signals(variant: str = "raw") -> pd.DataFrame:
    return pd.read_parquet(DATA / f"signals_{variant}.parquet")
