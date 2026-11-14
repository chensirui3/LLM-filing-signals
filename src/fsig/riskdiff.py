"""What changed in a company's Risk Factors since last year's 10-K?

Risk Factors run to 50-200 thousand characters and are mostly copied forward
from the previous year, so the informative part is the small share that is
new, dropped or reworded. A plain text diff finds edits but breaks down when a
paragraph is moved or rewritten. This tool matches paragraphs by meaning first
(embeddings), then classifies each match by how much of the wording survived:

    unchanged   same words (word-level similarity >= 0.98)
    edited      same paragraph with changes (>= 0.60); shown as a word diff
    rewritten   different wording, same content (embedding cosine >= 0.93)
    added       no counterpart in last year's filing
    removed     in last year's filing, no counterpart this year

A local LLM then writes a one-line plain-English summary of each added and
removed risk. Everything is computed from the two filings alone.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from .config import REPORTS
from .embed import embed_texts
from .extract import ollama_chat
from .sections import load_section

MIN_PARA_CHARS = 150   # shorter lines are headings, bullets from the summary, page furniture
# TOPIC was set by reading matched pairs: with nomic-embed-text, two unrelated
# risk paragraphs from the same company already score about 0.72, pairs between
# 0.80 and 0.92 share a theme but say different things, and genuine rewrites of
# the same paragraph start at about 0.93.
SAME, EDITED, TOPIC = 0.98, 0.60, 0.93
_NOT_HEADING = re.compile(r"(?:[;,:]|\band|\bor)$")
_BULLET = re.compile(r"^[\s•·\-–—*]+")
_WORDS = re.compile(r"\S+")

SUMMARY_SYSTEM = (
    "You will read one paragraph from the Risk Factors section of a company's annual report. "
    "State the specific risk it describes in one plain sentence of at most 25 words. "
    "Do not start with 'The company' or 'This paragraph'."
)
SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}


def paragraphs(text: str) -> list[dict]:
    """Body paragraphs, each with the nearest heading above it."""
    out, heading = [], ""
    for ln in text.split("\n"):
        ln = _BULLET.sub("", ln).strip()
        if len(ln) >= MIN_PARA_CHARS:
            out.append({"heading": heading, "text": ln})
        elif (15 <= len(ln) and ln[0].isupper() and not _NOT_HEADING.search(ln)
              and not ln.lower().startswith(("table of contents", "item 1a"))):
            heading = ln  # list items (lower-case start, or ending in ';' / 'and') are not headings
    return out


def word_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _WORDS.findall(a), _WORDS.findall(b), autojunk=False).ratio()


def classify(old: list[str], new: list[str], old_vecs: np.ndarray, new_vecs: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Returns (one record per new paragraph, one record per removed old paragraph)."""
    sim = new_vecs @ old_vecs.T
    new_rows, matched_old = [], set()
    for j, text in enumerate(new):
        i = int(sim[j].argmax())
        cos, ratio = float(sim[j, i]), word_ratio(old[i], text)
        status = ("unchanged" if ratio >= SAME else "edited" if ratio >= EDITED
                  else "rewritten" if cos >= TOPIC else "added")
        if status != "added":
            matched_old.add(i)
        new_rows.append({"new": j, "old": i, "cos": cos, "ratio": ratio, "status": status})
    removed = []
    for i, text in enumerate(old):
        if i in matched_old:
            continue
        j = int(sim[:, i].argmax())
        cos = float(sim[j, i])
        if cos < TOPIC and word_ratio(text, new[j]) < EDITED:
            removed.append({"old": i, "cos": cos})
    return new_rows, removed


def inline_diff(a: str, b: str, context: int = 10) -> str:
    """Word diff as markdown: ~~removed~~ **added**, with unchanged runs shortened."""
    wa, wb = _WORDS.findall(a), _WORDS.findall(b)
    ops = SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
    parts = []
    for n, (op, i1, i2, j1, j2) in enumerate(ops):
        if op == "equal":
            run = wa[i1:i2]
            if len(run) > 2 * context + 3:
                head = [] if n == 0 else run[:context]
                tail = [] if n == len(ops) - 1 else run[-context:]
                run = head + ["…"] + tail
            parts.append(" ".join(run))
        else:
            if i2 > i1:
                parts.append("~~" + " ".join(wa[i1:i2]) + "~~")
            if j2 > j1:
                parts.append("**" + " ".join(wb[j1:j2]) + "**")
    return " ".join(p for p in parts if p)


def _embed(emb: dict, texts: list[str]) -> np.ndarray:
    return np.vstack([embed_texts(emb, texts[i : i + emb["batch"]]) for i in range(0, len(texts), emb["batch"])])


def _summaries(llm: dict, texts: list[str]) -> list[str]:
    def one(t: str) -> str:
        try:
            return str(ollama_chat(llm, SUMMARY_SYSTEM, t[:4000], SUMMARY_SCHEMA).get("summary", "")).strip()
        except Exception:
            return ""
    with ThreadPoolExecutor(max_workers=llm["workers"]) as pool:
        return list(pool.map(one, texts))


def pick_filings(filings: pd.DataFrame, ticker: str, year: int | None) -> tuple:
    """(previous 10-K, chosen 10-K) rows. year is the filing year; default latest."""
    k = filings[(filings["ticker"] == ticker.upper()) & (filings["form"] == "10-K")].sort_values("filing_date")
    k = k[[load_section(a, "risk") is not None for a in k["accession"]]].reset_index(drop=True)
    if year is not None:
        hit = k.index[k["filing_date"].dt.year == year]
        if len(hit) == 0:
            raise SystemExit(f"No {ticker} 10-K with a parsed Risk Factors section filed in {year}. "
                             f"Available: {sorted(k['filing_date'].dt.year)}")
        pos = int(hit[-1])
    else:
        pos = len(k) - 1
    if pos < 1:
        raise SystemExit(f"No earlier {ticker} 10-K to compare with.")
    return k.iloc[pos - 1], k.iloc[pos]


def _excerpt(text: str, n: int = 600) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " …"


def risk_diff(filings: pd.DataFrame, ticker: str, year: int | None, emb: dict, llm: dict | None,
              max_items: int = 12) -> str:
    """Write reports/riskdiff_<TICKER>_<year>.md and return its path."""
    prev, cur = pick_filings(filings, ticker, year)
    old_p, new_p = (paragraphs(load_section(r.accession, "risk")) for r in (prev, cur))
    old, new = [p["text"] for p in old_p], [p["text"] for p in new_p]
    rows, removed = classify(old, new, _embed(emb, old), _embed(emb, new))

    by = {s: [r for r in rows if r["status"] == s] for s in ("unchanged", "edited", "rewritten", "added")}
    # Most novel first, favouring full paragraphs over one-line list items.
    by["added"].sort(key=lambda r: -(1 - r["cos"]) * min(len(new[r["new"]]), 600))
    removed.sort(key=lambda r: -(1 - r["cos"]) * min(len(old[r["old"]]), 600))
    by["edited"].sort(key=lambda r: r["ratio"])
    added, gone = by["added"][:max_items], removed[:max_items]
    notes = _summaries(llm, [new[r["new"]] for r in added] + [old[r["old"]] for r in gone]) if llm else []
    notes += [""] * (len(added) + len(gone) - len(notes))

    chars = lambda rs, src, key: sum(len(src[r[key]]) for r in rs)
    total = sum(map(len, new)) or 1
    out = [f"# {cur['name']} ({cur.ticker}): what changed in Risk Factors",
           "",
           f"10-K filed {cur.filing_date:%Y-%m-%d} compared with the 10-K filed {prev.filing_date:%Y-%m-%d}.",
           f"Paragraphs: {len(old)} last year, {len(new)} this year.",
           "",
           "| | paragraphs | share of this year's text |",
           "|---|---|---|"]
    for s in ("unchanged", "edited", "rewritten", "added"):
        out.append(f"| {s} | {len(by[s])} | {chars(by[s], new, 'new') / total:.0%} |")
    out += [f"| removed (from last year) | {len(removed)} | |", "",
            "Matching is by embedding similarity (" + emb["model"] + "); one-line summaries are written by "
            + (llm["model"] if llm else "no model") + " and can be wrong. Read the quoted text.", ""]

    def block(title: str, items: list[dict], src: list[str], meta: list[dict], key: str, summaries: list[str], total_n: int):
        out.extend([f"## {title} ({total_n})", ""])
        if not items:
            out.extend(["None.", ""])
        for r, note in zip(items, summaries):
            head = note or meta[r[key]]["heading"] or "(no heading)"
            out.extend([f"**{head}**", ""])
            if note and meta[r[key]]["heading"]:
                out.extend([f"*Under: {meta[r[key]]['heading']}*", ""])
            out.extend(["> " + _excerpt(src[r[key]]), ""])
        if total_n > len(items):
            out.extend([f"…and {total_n - len(items)} more.", ""])

    block("New this year", added, new, new_p, "new", notes[: len(added)], len(by["added"]))
    block("Removed since last year", gone, old, old_p, "old", notes[len(added):], len(removed))

    out += [f"## Rewritten: same topic, new wording ({len(by['rewritten'])})", ""]
    for r in sorted(by["rewritten"], key=lambda r: r["ratio"])[:max_items]:
        out += [f"**{new_p[r['new']]['heading'] or '(no heading)'}**", "",
                "> *Last year:* " + _excerpt(old[r["old"]], 400), ">", "> *This year:* " + _excerpt(new[r["new"]], 400), ""]
    if not by["rewritten"]:
        out += ["None.", ""]

    out += [f"## Edited paragraphs ({len(by['edited'])})", "",
            "Most changed first. ~~Struck~~ words were removed, **bold** words added; … marks unchanged text.", ""]
    for r in by["edited"][:max_items]:
        out += [f"**{new_p[r['new']]['heading'] or '(no heading)'}** ({1 - r['ratio']:.0%} of words changed)", "",
                "> " + inline_diff(old[r["old"]], new[r["new"]]), ""]
    if len(by["edited"]) > max_items:
        out += [f"…and {len(by['edited']) - max_items} more.", ""]

    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"riskdiff_{cur.ticker}_{cur.filing_date.year}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)
