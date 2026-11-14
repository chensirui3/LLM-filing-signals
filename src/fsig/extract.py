"""Structured signal extraction from MD&A text with a local Ollama model.

Every model response is cached in SQLite, keyed by a hash of
(model, prompt version, variant, text), so runs are resumable and a prompt
change never silently mixes with old results.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

from .config import DATA
from .sections import load_section

CACHE = DATA / "llm_cache.sqlite"
PROMPT_VERSION = "v1"

SYSTEM = """You are a careful equity analyst. You will read an excerpt from the \
Management's Discussion and Analysis (MD&A) section of a company's SEC filing and \
rate what management says about the business.

Rules:
- Rate ONLY what this excerpt says. Do not use outside knowledge about the company, \
its industry, or anything that happened after the filing.
- Boilerplate, accounting policy descriptions and legal language are neutral (0).
- If the excerpt does not discuss a dimension, give it 0.

Scales:
- tone, outlook, demand, margins: integer from -2 (clearly negative / deteriorating) \
to 2 (clearly positive / improving), 0 = neutral or mixed.
  - tone: management's description of results in the reported period.
  - outlook: forward-looking statements about future periods.
  - demand: revenue, volumes, orders, customer demand.
  - margins: profitability, costs, pricing power.
- liquidity_stress: 0 = none, 1 = some concern, 2 = serious concern about cash, debt or covenants.
- uncertainty: 0 = confident and specific, 1 = some hedging, 2 = heavily hedged or vague.
- evidence: one short phrase from the excerpt that best supports your tone rating."""

_SCORE = {"type": "integer", "minimum": -2, "maximum": 2}
_LEVEL = {"type": "integer", "minimum": 0, "maximum": 2}
SCHEMA = {
    "type": "object",
    "properties": {
        "tone": _SCORE,
        "outlook": _SCORE,
        "demand": _SCORE,
        "margins": _SCORE,
        "liquidity_stress": _LEVEL,
        "uncertainty": _LEVEL,
        "evidence": {"type": "string"},
    },
    "required": ["tone", "outlook", "demand", "margins", "liquidity_stress", "uncertainty", "evidence"],
}
FIELDS = ["tone", "outlook", "demand", "margins", "liquidity_stress", "uncertainty"]
_BOUNDS = {f: (-2, 2) for f in FIELDS[:4]} | {f: (0, 2) for f in FIELDS[4:]}


# ----------------------------------------------------------------- text prep

def chunk_text(text: str, chunk_chars: int, max_chunks: int) -> list[str]:
    """First `max_chunks` chunks of about `chunk_chars`, split on line breaks."""
    chunks, cur, size = [], [], 0
    for line in text.split("\n"):
        if size + len(line) > chunk_chars and cur:
            chunks.append("\n".join(cur))
            if len(chunks) == max_chunks:
                return chunks
            cur, size = [], 0
        cur.append(line[:chunk_chars])
        size += len(line) + 1
    if cur and len(chunks) < max_chunks:
        chunks.append("\n".join(cur))
    return chunks


_SUFFIX = re.compile(
    r"[\s,\.]*(/[a-z]{2,3}/?|\b(inc|incorporated|corp|corporation|co|company|ltd|plc|llc|lp|"
    r"holdings?|group|n\.?v|s\.?a|the)\b)[\s,\.]*$",
    re.I,
)
_YEAR = re.compile(r"\b(19|20)\d{2}\b")


def company_core_name(name: str) -> str:
    """'Apple Inc.' -> 'Apple'; 'JPMORGAN CHASE & CO' -> 'JPMORGAN CHASE'."""
    core = name.strip()
    while True:
        new = _SUFFIX.sub("", core).strip(" ,.&")
        if new == core or not new:
            break
        core = new
    return core


def anonymise(text: str, name: str, ticker: str, fiscal_year: int) -> str:
    """Mask the company name, ticker and calendar years.

    Years become offsets from the fiscal year ('Y+0', 'Y-1') so period-over-period
    comparisons still read correctly but the model can't tell which year it is.
    Product and brand names are NOT masked; the identification probe in
    leakage.py measures how much that matters.
    """
    core = company_core_name(name)
    if len(core) >= 3:
        text = re.sub(rf"\b{re.escape(core)}\b(\s*,?\s*(inc|corp|corporation|co|company|ltd|plc)\b\.?)?",
                      "the Company", text, flags=re.I)
    if len(ticker) >= 3:
        text = re.sub(rf"\b{re.escape(ticker)}\b", "the Company", text)
    return _YEAR.sub(lambda m: f"Y{int(m.group()) - fiscal_year:+d}", text)


# ----------------------------------------------------------------- model call

def ollama_chat(llm: dict, system: str, user: str, schema: dict) -> dict:
    r = requests.post(
        f"{llm['host']}/api/chat",
        json={
            "model": llm["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": llm["num_ctx"]},
        },
        timeout=600,
    )
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def validate(resp: dict) -> dict:
    out = {}
    for f in FIELDS:
        lo, hi = _BOUNDS[f]
        out[f] = max(lo, min(hi, int(resp[f])))
    out["evidence"] = str(resp.get("evidence", ""))[:300]
    return out


# ----------------------------------------------------------------- cache

def db() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE)
    con.execute(
        """CREATE TABLE IF NOT EXISTS chunks(
            key TEXT PRIMARY KEY, accession TEXT, variant TEXT, chunk_idx INTEGER,
            model TEXT, prompt_version TEXT, n_chars INTEGER, response TEXT)"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS probes(
            key TEXT PRIMARY KEY, kind TEXT, accession TEXT, model TEXT, response TEXT)"""
    )
    return con


def cache_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


# ----------------------------------------------------------------- pipeline

def _tasks(filings: pd.DataFrame, llm: dict, variant: str):
    for r in filings.itertuples():
        text = load_section(r.accession, "mdna")
        if not text:
            continue
        if variant == "anon":
            fy = r.report_date.year if pd.notna(r.report_date) else r.filing_date.year
            text = anonymise(text, r.name, r.ticker, fy)
        for i, chunk in enumerate(chunk_text(text, llm["chunk_chars"], llm["max_chunks"])):
            yield cache_key(llm["model"], PROMPT_VERSION, variant, chunk), r.accession, i, chunk


def run_extraction(filings: pd.DataFrame, llm: dict, variant: str = "raw") -> dict:
    """Score every MD&A chunk not already cached. variant is 'raw' or 'anon'."""
    con = db()
    done = {k for (k,) in con.execute("SELECT key FROM chunks")}
    todo = [t for t in _tasks(filings, llm, variant) if t[0] not in done]
    stats = {"cached": len(done), "todo": len(todo), "ok": 0, "failed": 0}
    if not todo:
        return stats

    def work(task):
        key, accession, idx, chunk = task
        user = f"MD&A excerpt:\n\n{chunk}"
        for _ in range(2):  # one retry on a malformed answer
            try:
                return task, validate(ollama_chat(llm, SYSTEM, user, SCHEMA))
            except (KeyError, ValueError, TypeError):
                continue
        return task, None

    with ThreadPoolExecutor(max_workers=llm["workers"]) as pool:
        futures = [pool.submit(work, t) for t in todo]
        for n, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"extract[{variant}]")):
            (key, accession, idx, chunk), resp = fut.result()
            if resp is None:
                stats["failed"] += 1
                continue
            con.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                (key, accession, variant, idx, llm["model"], PROMPT_VERSION, len(chunk), json.dumps(resp)),
            )
            stats["ok"] += 1
            if n % 20 == 0:
                con.commit()
    con.commit()
    return stats


def load_scores(llm: dict, variant: str = "raw") -> pd.DataFrame:
    """One row per filing: mean of each field across its chunks."""
    con = db()
    rows = con.execute(
        "SELECT accession, response FROM chunks WHERE model=? AND prompt_version=? AND variant=?",
        (llm["model"], PROMPT_VERSION, variant),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["accession", *FIELDS, "n_chunks"])
    df = pd.DataFrame([{"accession": a, **{f: json.loads(r)[f] for f in FIELDS}} for a, r in rows])
    out = df.groupby("accession")[FIELDS].mean()
    out["n_chunks"] = df.groupby("accession").size()
    return out.reset_index()
