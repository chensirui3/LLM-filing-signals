"""Tests of whether the LLM's apparent predictive power reflects memorised outcomes.

An LLM trained on text up to date C has been exposed to news about how each
company performed before C. A backtest that gives it pre-C filings can
therefore appear profitable because of memorised outcomes rather than
information in the text. This module provides three checks:

1. memory probe    - the model receives no filing text, only two company names
                     and a date, and is asked which stock had the higher return
                     over the following three months. Accuracy above chance
                     before C but not after C indicates memorised outcomes.
                     Every pair is asked in both orders and only answers that
                     name the same company both times are scored, so a
                     preference for one position cannot affect the result.
                     Running the probe on models with different cutoffs shows
                     whether the boundary moves with C.

                     Two question formats were tried and rejected. Asking about
                     one stock against the market is uninformative: the model
                     answers "underperform" about 95% of the time. Asking for a
                     label ("A" or "B") wastes most questions: the model's own
                     preferred continuation is the ticker, so the forced label
                     is close to arbitrary and only about 15% of answers
                     survive a swap of the order, against about 63% when the
                     model answers with the ticker.
2. memory controls - test whether the probe's accuracy depends on the date or
                     reflects a preference for particular companies.
3. identity probe  - the model receives anonymised MD&A text and is asked to
                     name the company and fiscal year. If it can, anonymisation
                     does not conceal the filer.

The cutoff split (backtest.split_stats) and the comparison of raw and
anonymised scores (robustness.variant_comparison) complete the set.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .extract import anonymise, cache_key, chunk_text, company_core_name, db, ollama_chat
from .sections import load_section

MEMORY_SYSTEM = (
    "You are a financial historian with an excellent memory for stock market history. "
    "Answer from memory. You must choose one of the two answers."
)


def memory_schema(first: str, second: str) -> dict:
    """The answer must be one of the two tickers in the question."""
    return {"type": "object", "properties": {"answer": {"type": "string", "enum": [first, second]}},
            "required": ["answer"]}


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

PERIODS = ("pre_cutoff", "post_cutoff")


def installed_models(host: str) -> set[str]:
    """Names of the models available on the local Ollama server."""
    r = requests.get(f"{host}/api/tags", timeout=30)
    r.raise_for_status()
    return {m["name"] for m in r.json().get("models", [])}


def _run_probe(kind: str, rows: list[tuple[str, str]], llm: dict, system: str, schema: dict | None = None,
               schemas: dict[str, dict] | None = None) -> dict[str, dict]:
    """rows: (id, user_prompt). Returns id -> response. Cached and resumable.

    Pass `schema` when every row shares one output schema, or `schemas`
    (id -> schema) when each row has its own.
    """
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
            return task, ollama_chat(llm, system, task[2], schemas[task[0]] if schemas else schema)
        except (ValueError, KeyError, requests.RequestException):
            return task, None

    failed = 0
    with ThreadPoolExecutor(max_workers=llm["workers"]) as pool:
        results = tqdm(pool.map(work, todo), total=len(todo), desc=f"{kind}[{llm['model']}]")
        for n, ((rid, key, _), resp) in enumerate(results):
            if resp is None:
                failed += 1
                continue
            con.execute("INSERT OR REPLACE INTO probes VALUES (?,?,?,?,?)",
                        (key, kind, rid, llm["model"], json.dumps(resp)))
            out[rid] = resp
            if n % 50 == 0:
                con.commit()
    con.commit()
    if failed:
        print(f"{kind}[{llm['model']}]: {failed} of {len(todo)} requests failed and are not cached")
    return out


# ----------------------------------------------------------------- memory probe

def month_end_forward_returns(prices: pd.DataFrame, tickers: list[str], horizon: int) -> pd.DataFrame:
    """Return of each ticker over the `horizon` trading days after every month-end."""
    fwd = prices[tickers].shift(-horizon) / prices[tickers] - 1
    month_ends = prices.index.to_series().groupby(prices.index.to_period("M")).max()
    return fwd.loc[month_ends].dropna(how="all")


def memory_pairs(prices: pd.DataFrame, tickers: list[str], horizon: int, per_year: int,
                 start: str | None = None, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Random pairs of stocks at random month-ends, the same number in every calendar year.

    Returns (pairs, fwd, window_end).
      pairs       one row per pair: id, date, window_end, year, x, y and winner,
                  the ticker with the higher return over the `horizon` trading
                  days after `date`. x and y are in alphabetical order.
      fwd         forward return of every ticker from every month-end
      window_end  for every month-end, the date on which its return window ends
    """
    fwd = month_end_forward_returns(prices, tickers, horizon)
    if start:
        fwd = fwd[fwd.index >= pd.Timestamp(start)]
    window_end = pd.Series(prices.index, index=prices.index).shift(-horizon).loc[fwd.index]
    rng = np.random.default_rng(seed)
    rows = []
    for year in sorted(set(fwd.index.year)):
        dates = fwd.index[fwd.index.year == year]
        for i in rng.integers(0, len(dates), size=per_year):
            d = dates[i]
            r = fwd.loc[d].dropna()
            if len(r) < 2:
                continue
            x, y = sorted(r.index[j] for j in rng.choice(len(r), size=2, replace=False))
            if r[x] == r[y]:
                continue
            rows.append({"id": f"{d:%Y%m%d}-{x}-{y}", "date": d, "window_end": window_end[d], "year": year,
                         "x": x, "y": y, "winner": x if r[x] > r[y] else y})
    pairs = pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)
    return pairs, fwd, window_end


def memory_answers(pairs: pd.DataFrame, names: dict[str, str], llm: dict) -> pd.DataFrame:
    """Ask the model about every pair in both orders and score the answers.

    names maps ticker -> company name. Adds to `pairs`:
      pick_xy, pick_yx  the ticker chosen with x listed first and with y listed first
      consistent        the model chose the same company in both orders
      pick, other       that company and the one it was preferred to (when consistent)
      score             share of the two answers that were correct: 1, 0.5 or 0. A
                        pair answered by position alone scores exactly 0.5.
      first_listed      share of the two answers that chose the stock listed first
    """
    def ask(date, a, b):
        return (f"Date: {date:%B %d, %Y}.\nStocks: {names[a]} ({a}) and {names[b]} ({b}).\n"
                "Which of the two had the higher return over the three months after this date? "
                "Answer with its ticker.")

    prompts, schemas = [], {}
    for r in pairs.itertuples():
        for order, (a, b) in (("xy", (r.x, r.y)), ("yx", (r.y, r.x))):
            prompts.append((f"{r.id}|{order}", ask(r.date, a, b)))
            schemas[f"{r.id}|{order}"] = memory_schema(a, b)
    resp = _run_probe("memory_ticker", prompts, llm, MEMORY_SYSTEM, schemas=schemas)

    xy = [resp.get(f"{i}|xy", {}).get("answer") for i in pairs["id"]]
    yx = [resp.get(f"{i}|yx", {}).get("answer") for i in pairs["id"]]
    valid = np.array([a in (x, y) and b in (x, y) for a, b, x, y in zip(xy, yx, pairs["x"], pairs["y"])])
    out = pairs[valid].copy()
    out["pick_xy"] = np.array(xy, dtype=object)[valid]
    out["pick_yx"] = np.array(yx, dtype=object)[valid]
    out["consistent"] = out["pick_xy"] == out["pick_yx"]
    out["pick"] = out["pick_xy"].where(out["consistent"])
    out["other"] = pd.Series(np.where(out["pick_xy"] == out["x"], out["y"], out["x"]),
                             index=out.index).where(out["consistent"])
    out["score"] = ((out["pick_xy"] == out["winner"]).astype(float) + (out["pick_yx"] == out["winner"])) / 2
    out["first_listed"] = ((out["pick_xy"] == out["x"]).astype(float) + (out["pick_yx"] == out["y"])) / 2
    return out


def with_periods(answers: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Label each pair relative to a training cutoff.

    pre_cutoff    the whole return window ends on or before the cutoff
    post_cutoff   the question date is after the cutoff
    spans_cutoff  the window starts before the cutoff and ends after it; these
                  pairs are left out of the before-and-after comparison
    """
    c = pd.Timestamp(cutoff)
    period = np.where(answers["window_end"] <= c, "pre_cutoff",
                      np.where(answers["date"] > c, "post_cutoff", "spans_cutoff"))
    return answers.assign(period=period)


def _z(acc: float, n: int) -> float:
    return (acc - 0.5) / math.sqrt(0.25 / n) if n else float("nan")  # 50% null


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (Wilson score interval)."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def _accuracy(g: pd.DataFrame) -> dict:
    """Accuracy over the pairs on which the model gave the same answer in both orders."""
    c = g[g["consistent"]]
    n, k = len(c), int((c["pick"] == c["winner"]).sum())
    lo, hi = _wilson(k, n)
    return {"pairs": len(g), "order_independent": g["consistent"].mean() if len(g) else float("nan"),
            "n": n, "accuracy": k / n if n else float("nan"), "ci_low": lo, "ci_high": hi,
            "z_vs_50pct": _z(k / n, n) if n else float("nan"), "accuracy_all_pairs": g["score"].mean()}


def _by_period(answers: pd.DataFrame):
    """(period, rows) in chronological order, skipping empty periods."""
    for period in PERIODS:
        g = answers[answers["period"] == period]
        if len(g):
            yield period, g


def memory_summary(answers: pd.DataFrame) -> pd.DataFrame:
    """Probe accuracy before and after the cutoff (see _accuracy for the columns)."""
    rows = [{"period": period, **_accuracy(g)} for period, g in _by_period(answers)]
    return pd.DataFrame(rows).set_index("period").drop(columns=["ci_low", "ci_high"])


def memory_by_year(answers: pd.DataFrame) -> pd.DataFrame:
    """Probe accuracy by calendar year of the question date, with a 95% interval."""
    rows = [{"year": int(year), **_accuracy(g)} for year, g in answers.groupby("year")]
    return pd.DataFrame(rows).set_index("year")


def memory_controls(answers: pd.DataFrame, fwd: pd.DataFrame, window_end: pd.Series,
                    size_rank: dict[str, int], cutoff: str) -> pd.DataFrame:
    """Controls that separate date-specific memory from a preference for certain companies.

    Above-chance accuracy before the cutoff has two possible sources:
      (a) the model recalls what happened after that particular date, or
      (b) the model prefers some companies (large, well-known ones) that
          happened to outperform over the sample.
    The following controls distinguish the two, each reported before and after
    the cutoff and computed on the order-independent answers:

      picks_larger       how often the model picks the company that is larger
                         today (size_rank: ticker -> rank, 0 = largest)
      larger_wins        accuracy of the rule "always pick the larger company",
                         over all pairs in the period
      acc_when_smaller   model accuracy on the pairs in which it picked the
                         smaller company, contrary to that rule
      placebo_accuracy   the model's pick scored against the same pair's
                         outcome after every other month-end in the same
                         period. If this is as high as the actual accuracy,
                         the model is not using the date and source (b) applies.
    """
    c = pd.Timestamp(cutoff)
    period_dates = {"pre_cutoff": fwd.index[(window_end <= c).to_numpy()], "post_cutoff": fwd.index[fwd.index > c]}
    rows = []
    for period, g in _by_period(answers):
        loser = np.where(g["winner"] == g["x"], g["y"], g["x"])
        larger_wins = np.mean([size_rank[w] < size_rank[l] for w, l in zip(g["winner"], loser)])
        sel = g[g["consistent"]]
        larger = np.array([size_rank[p] < size_rank[o] for p, o in zip(sel["pick"], sel["other"])], dtype=bool)
        correct = (sel["pick"] == sel["winner"]).to_numpy()
        placebo = []
        for r in sel.itertuples():
            other = fwd.loc[period_dates[period].difference([r.date]), [r.pick, r.other]].dropna()
            if len(other):
                placebo.append((other[r.pick] > other[r.other]).mean())
        n_small = int((~larger).sum())
        acc_small = correct[~larger].mean() if n_small else float("nan")
        rows.append({
            "period": period, "n": len(sel), "accuracy": correct.mean() if len(sel) else float("nan"),
            "picks_larger": larger.mean() if len(sel) else float("nan"), "larger_wins": larger_wins,
            "n_picked_smaller": n_small, "acc_when_smaller": acc_small, "z_when_smaller": _z(acc_small, n_small),
            "placebo_accuracy": float(np.mean(placebo)) if placebo else float("nan"),
        })
    return pd.DataFrame(rows).set_index("period")


def run_memory_probe(prices: pd.DataFrame, names: dict[str, str], llm: dict, lk: dict, benchmark: str,
                     size_rank: dict[str, int], start: str | None = None) -> dict[str, dict]:
    """Run the memory probe on every configured model that is installed.

    lk is the `leakage` block of the configuration. The same pairs are put to
    every model, and each model's answers are split at its own training cutoff.
    Returns {model: {"cutoff", "answers", "summary", "by_year", "controls"}},
    ordered by cutoff.
    """
    tickers = [t for t in prices.columns if t != benchmark and t in names]
    pairs, fwd, window_end = memory_pairs(prices, tickers, lk["memory_horizon"], lk["memory_pairs_per_year"], start)
    models = {**lk["memory_models"], llm["model"]: llm["training_cutoff"]}
    have = installed_models(llm["host"])
    out = {}
    for model, cutoff in sorted(models.items(), key=lambda kv: str(kv[1])):
        if model not in have and f"{model}:latest" not in have:
            print(f"memory probe: {model} is not installed in Ollama and is skipped (ollama pull {model})")
            continue
        answers = with_periods(memory_answers(pairs, names, {**llm, "model": model}), str(cutoff))
        out[model] = {"cutoff": str(cutoff), "answers": answers, "summary": memory_summary(answers),
                      "by_year": memory_by_year(answers),
                      "controls": memory_controls(answers, fwd, window_end, size_rank, str(cutoff))}
    return out


# ----------------------------------------------------------------- identity probe

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
