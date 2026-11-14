"""Tests for whether the LLM 'predicts' returns by remembering them.

An LLM trained on text up to date C has read the news about what happened to
each company before C. A backtest that feeds it pre-C filings can therefore
look profitable for the wrong reason. Three checks:

1. memory probe   - give the model NO filing text, only two company names and
                    a date, and ask which stock did better over the next three
                    months. Accuracy above chance before C but not after C is
                    direct evidence of memorised outcomes. (Asking about one
                    stock vs the market does not work: the model answers
                    "underperform" ~95% of the time. A randomly ordered pair
                    makes any such fixed bias score exactly 50%.)
2. identity probe - give the model anonymised MD&A text and ask which company
                    and year it is. If it can tell, anonymisation does not
                    protect the backtest.
3. cutoff split   - backtest.split_stats reports performance before and after
                    C separately. Only the post-cutoff row is clean.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from .extract import anonymise, cache_key, chunk_text, company_core_name, db, ollama_chat
from .sections import load_section

MEMORY_SYSTEM = (
    "You are a financial historian with an excellent memory for stock market history. "
    "Answer from memory. You must choose one of the two answers."
)
MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["A", "B"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["answer", "confidence"],
}

IDENTITY_SYSTEM = (
    "You will read an excerpt from a company's SEC filing in which the company name and "
    "the years have been masked. Guess which company wrote it and the fiscal year it covers."
)
IDENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string"},
        "ticker": {"type": "string"},
        "fiscal_year": {"type": "integer"},
    },
    "required": ["company_name", "ticker", "fiscal_year"],
}


def _run_probe(kind: str, rows: list[tuple[str, str]], llm: dict, system: str, schema: dict) -> dict[str, dict]:
    """rows: (id, user_prompt). Returns id -> response. Cached and resumable."""
    con = db()
    out, todo = {}, []
    for rid, user in rows:
        key = cache_key(llm["model"], kind, user)
        hit = con.execute("SELECT response FROM probes WHERE key=?", (key,)).fetchone()
        if hit:
            out[rid] = json.loads(hit[0])
        else:
            todo.append((rid, key, user))

    def work(task):
        try:
            return task, ollama_chat(llm, system, task[2], schema)
        except (ValueError, KeyError):
            return task, None

    with ThreadPoolExecutor(max_workers=llm["workers"]) as pool:
        for (rid, key, _), resp in tqdm(pool.map(work, todo), total=len(todo), desc=kind):
            if resp is None:
                continue
            con.execute("INSERT OR REPLACE INTO probes VALUES (?,?,?,?,?)",
                        (key, kind, rid, llm["model"], json.dumps(resp)))
            con.commit()
            out[rid] = resp
    return out


def month_end_forward_returns(prices: pd.DataFrame, tickers: list[str], horizon: int) -> pd.DataFrame:
    fwd = prices[tickers].shift(-horizon) / prices[tickers] - 1
    month_ends = prices.index.to_series().groupby(prices.index.to_period("M")).max()
    return fwd.loc[month_ends].dropna(how="all")


def memory_sample(prices: pd.DataFrame, names: dict[str, str], llm: dict, benchmark: str,
                  horizon: int = 63, per_period: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ask the text-free 'which stock did better?' question for random pairs.

    names maps ticker -> company name. Pairs and their A/B order are random.
    Returns (one row per question with the model's guess, month-end forward returns).
    """
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp(llm["training_cutoff"])
    tickers = [t for t in prices.columns if t != benchmark and t in names]
    fwd = month_end_forward_returns(prices, tickers, horizon)

    rows = []
    for period, dates in (("pre_cutoff", fwd.index[fwd.index <= cutoff]), ("post_cutoff", fwd.index[fwd.index > cutoff])):
        if len(dates) == 0:
            continue
        for i in rng.integers(0, len(dates), size=per_period):
            d = dates[i]
            r = fwd.loc[d].dropna()
            if len(r) < 2:
                continue
            a, b = (r.index[j] for j in rng.choice(len(r), size=2, replace=False))
            rows.append({"id": f"{d:%Y%m%d}-{a}-{b}", "period": period, "date": pd.Timestamp(d),
                         "a": a, "b": b, "truth": "A" if r[a] > r[b] else "B"})
    sample = pd.DataFrame(rows).drop_duplicates("id")

    prompts = [
        (r.id,
         f"Date: {r.date:%B %d, %Y}.\nStock A: {names[r.a]} ({r.a})\nStock B: {names[r.b]} ({r.b})\n"
         "Which stock had the higher return over the three months after this date?")
        for r in sample.itertuples()
    ]
    resp = _run_probe("memory_pair", prompts, llm, MEMORY_SYSTEM, MEMORY_SCHEMA)
    sample = sample[sample["id"].isin(resp.keys())].copy()
    sample["guess"] = [resp[i]["answer"] for i in sample["id"]]
    sample["correct"] = sample["guess"] == sample["truth"]
    sample["pick"] = np.where(sample["guess"] == "A", sample["a"], sample["b"])
    sample["other"] = np.where(sample["guess"] == "A", sample["b"], sample["a"])
    return sample, fwd


def _z(acc: float, n: int) -> float:
    return (acc - 0.5) / math.sqrt(0.25 / n) if n else float("nan")  # 50% null


def memory_summary(sample: pd.DataFrame) -> pd.DataFrame:
    out = []
    for period, g in sample.groupby("period"):
        n, acc = len(g), g["correct"].mean()
        out.append({"period": period, "n": n, "accuracy": acc, "z_vs_50pct": _z(acc, n),
                    "share_guess_A": (g["guess"] == "A").mean()})
    return pd.DataFrame(out).set_index("period")


def memory_probe(prices: pd.DataFrame, names: dict[str, str], llm: dict, benchmark: str,
                 horizon: int = 63, per_period: int = 300, seed: int = 0) -> pd.DataFrame:
    """Accuracy of text-free guesses, before vs after the cutoff."""
    return memory_summary(memory_sample(prices, names, llm, benchmark, horizon, per_period, seed)[0])


def memory_controls(sample: pd.DataFrame, fwd: pd.DataFrame, size_rank: dict[str, int],
                    cutoff: str) -> pd.DataFrame:
    """Is the memory probe's accuracy date-specific memory, or a taste for certain names?

    Above-chance accuracy before the cutoff has two possible sources:
      (a) the model recalls what happened after that particular date, or
      (b) the model simply prefers some companies (the big, famous ones), and
          those happened to do well over the sample.
    Three controls tell them apart, each reported before and after the cutoff:

      picks_larger       how often the model picks the company that is larger
                         today (size_rank: ticker -> rank, 0 = largest)
      larger_wins        accuracy of the rule "always pick the larger company"
      acc_when_smaller   model accuracy on the questions where it picked the
                         smaller company, i.e. went against that rule
      placebo_accuracy   the model's pick scored against what the same pair did
                         after every OTHER month-end in the same period. If
                         this is as high as the real accuracy, the model is not
                         using the date: it just likes the right names.
    """
    c = pd.Timestamp(cutoff)
    rows = []
    for period, g in sample.groupby("period"):
        dates = fwd.index[fwd.index <= c] if period == "pre_cutoff" else fwd.index[fwd.index > c]
        larger = np.array([size_rank[p] < size_rank[o] for p, o in zip(g["pick"], g["other"])])
        truth_pick = np.where(g["truth"] == "A", g["a"], g["b"])
        larger_wins = np.array([size_rank[t] < size_rank[o] for t, o in
                                zip(truth_pick, np.where(g["truth"] == "A", g["b"], g["a"]))])
        placebo = []
        for r in g.itertuples():
            other = fwd.loc[dates.difference([r.date]), [r.pick, r.other]].dropna()
            if len(other):
                placebo.append((other[r.pick] > other[r.other]).mean())
        n_small = int((~larger).sum())
        acc_small = g["correct"].to_numpy()[~larger].mean() if n_small else float("nan")
        rows.append({
            "period": period, "n": len(g), "accuracy": g["correct"].mean(),
            "picks_larger": larger.mean(), "larger_wins": larger_wins.mean(),
            "n_picked_smaller": n_small, "acc_when_smaller": acc_small, "z_when_smaller": _z(acc_small, n_small),
            "placebo_accuracy": float(np.mean(placebo)) if placebo else float("nan"),
        })
    return pd.DataFrame(rows).set_index("period")


def identity_probe(filings: pd.DataFrame, llm: dict, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """How often the model names the company / year from anonymised MD&A text."""
    sample = filings.sample(min(n, len(filings)), random_state=seed)
    prompts, meta = [], {}
    for r in sample.itertuples():
        text = load_section(r.accession, "mdna")
        if not text:
            continue
        fy = r.report_date.year if pd.notna(r.report_date) else r.filing_date.year
        chunk = chunk_text(anonymise(text, r.name, r.ticker, fy), llm["chunk_chars"], 1)[0]
        prompts.append((r.accession, f"Masked filing excerpt:\n\n{chunk}"))
        meta[r.accession] = (r.ticker, company_core_name(r.name).lower(), fy)
    resp = _run_probe("identity", prompts, llm, IDENTITY_SYSTEM, IDENTITY_SCHEMA)

    rows = []
    for acc, r in resp.items():
        ticker, core, fy = meta[acc]
        guess_name = str(r.get("company_name", "")).lower()
        rows.append({
            "accession": acc,
            "company_correct": str(r.get("ticker", "")).upper().strip() == ticker or (len(core) >= 3 and core in guess_name),
            "year_correct": r.get("fiscal_year") == fy,
            "year_within_1": isinstance(r.get("fiscal_year"), int) and abs(r["fiscal_year"] - fy) <= 1,
        })
    return pd.DataFrame(rows)
