"""Year-over-year comparison of the Risk Factors section of a company's 10-K.

A Risk Factors section runs to 50-200 thousand characters and is largely
carried forward from the previous year, so the informative part is the small
share that is new, removed or reworded. A plain text diff finds edits but
fails when a paragraph is moved or rewritten. This tool first matches
paragraphs by meaning (embeddings), then classifies each match by how much of
the wording is retained:

    unchanged   same words (word-level similarity >= 0.98)
    edited      same paragraph with changes (>= 0.60); shown as a word diff
    rewritten   different wording, same content (embedding cosine >= 0.93)
    added       no counterpart in the previous filing
    removed     present in the previous filing, no counterpart in the current one

A local LLM then writes a one-line summary of each added and removed risk.
All output is derived from the two filings alone.
"""
from __future__ import annotations

import gzip
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from .config import REPORTS
from .embed import embed_texts
from .extract import ollama_chat
from .edgar import filing_path
from .sections import emphasised_lines, load_section

MIN_PARA_CHARS = 150   # shorter lines are headings, summary bullets, or page headers and footers
# TOPIC was chosen by inspecting matched pairs. With nomic-embed-text, two
# unrelated risk paragraphs from the same company score about 0.72, pairs
# between 0.80 and 0.92 share a theme but differ in content, and rewrites of
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


MAX_HEADING_CHARS = 800  # some filers (Boeing, for example) use a full bold paragraph as the heading


def paragraphs(text: str, emphasised: set[str] | None = None) -> list[dict]:
    """Body paragraphs, each with the nearest heading above it.

    `emphasised` is the set of lines the filing prints in bold, italic or
    underline (sections.emphasised_lines). Risk-factor headings are often full
    sentences of 150-400 characters, so length does not distinguish them from
    body text, whereas formatting does. If no formatting is supplied, any short
    capitalised line is treated as a heading.
    """
    lines = [(ln.strip(), _BULLET.sub("", ln).strip()) for ln in text.split("\n")]
    emphasised = emphasised or set()
    marked = [raw in emphasised and len(ln) <= MAX_HEADING_CHARS for raw, ln in lines]
    # Use the formatting only if it marks a plausible share of lines: none means
    # it was not detected, and more than half means the whole section is bold.
    use_format = 0 < sum(marked) <= 0.5 * len(lines)

    out, heading = [], ""
    for (_, ln), is_marked in zip(lines, marked):
        if ln.lower().startswith(("table of contents", "item 1a")):
            continue
        if use_format and is_marked and len(ln) >= 4:
            heading = _excerpt(ln, 220)
        elif len(ln) < 15:
            continue
        elif len(ln) >= MIN_PARA_CHARS:
            out.append({"heading": heading, "text": ln})
        elif not use_format and ln[0].isupper() and not _NOT_HEADING.search(ln):
            heading = ln  # list items (lower-case start, or ending in ';' or 'and') are excluded
    return out


def load_paragraphs(cik: int, accession: str) -> list[dict]:
    with gzip.open(filing_path(cik, accession), "rb") as f:
        emphasised = emphasised_lines(f.read())
    return paragraphs(load_section(accession, "risk"), emphasised)


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
    """(previous 10-K, selected 10-K) rows. year is the filing year; the default is the latest."""
    k = filings[(filings["ticker"] == ticker.upper()) & (filings["form"] == "10-K")].sort_values("filing_date")
    if k.empty:
        raise SystemExit(f"{ticker} has no 10-K in the filing index (it may not be in the universe).")
    k = k[[load_section(a, "risk") is not None for a in k["accession"]]].reset_index(drop=True)
    if k.empty:
        raise SystemExit(f"No {ticker} 10-K has a parsed Risk Factors section.")
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
    old_p, new_p = (load_paragraphs(r.cik, r.accession) for r in (prev, cur))
    old, new = [p["text"] for p in old_p], [p["text"] for p in new_p]
    rows, removed = classify(old, new, _embed(emb, old), _embed(emb, new))

    by = {s: [r for r in rows if r["status"] == s] for s in ("unchanged", "edited", "rewritten", "added")}
    # Least similar to the other filing first, with full paragraphs ranked above one-line list items.
    by["added"].sort(key=lambda r: -(1 - r["cos"]) * min(len(new[r["new"]]), 600))
    removed.sort(key=lambda r: -(1 - r["cos"]) * min(len(old[r["old"]]), 600))
    by["edited"].sort(key=lambda r: r["ratio"])
    added, gone = by["added"][:max_items], removed[:max_items]
    notes = _summaries(llm, [new[r["new"]] for r in added] + [old[r["old"]] for r in gone]) if llm else []
    notes += [""] * (len(added) + len(gone) - len(notes))

    chars = lambda rs, src, key: sum(len(src[r[key]]) for r in rs)
    total = sum(map(len, new)) or 1
    out = [f"# {cur['name']} ({cur.ticker}): changes in Risk Factors",
           "",
           f"10-K filed {cur.filing_date:%Y-%m-%d} compared with the 10-K filed {prev.filing_date:%Y-%m-%d}. "
           f"Paragraphs compared: {len(old)} in the previous filing, {len(new)} in the current filing.",
           "",
           "| Status | Paragraphs | Share of current text |",
           "|---|---|---|"]
    for s in ("unchanged", "edited", "rewritten", "added"):
        out.append(f"| {s} | {len(by[s])} | {chars(by[s], new, 'new') / total:.0%} |")
    note = f"Paragraphs are matched by embedding similarity ({emb['model']})."
    if llm:
        note += (f" The one-line summaries are generated by {llm['model']} and may be inaccurate; "
                 "the quoted filing text is authoritative.")
    out += [f"| removed (from the previous filing) | {len(removed)} | |", "", note, ""]

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

    block("Added", added, new, new_p, "new", notes[: len(added)], len(by["added"]))
    block("Removed", gone, old, old_p, "old", notes[len(added):], len(removed))

    out += [f"## Rewritten: same content, new wording ({len(by['rewritten'])})", ""]
    for r in sorted(by["rewritten"], key=lambda r: r["ratio"])[:max_items]:
        out += [f"**{new_p[r['new']]['heading'] or '(no heading)'}**", "",
                "> *Previous:* " + _excerpt(old[r["old"]], 400), ">", "> *Current:* " + _excerpt(new[r["new"]], 400), ""]
    if not by["rewritten"]:
        out += ["None.", ""]
    elif len(by["rewritten"]) > max_items:
        out += [f"…and {len(by['rewritten']) - max_items} more.", ""]

    out += [f"## Edited paragraphs ({len(by['edited'])})", "",
            "Ordered from most to least changed. ~~Struck~~ words were removed and **bold** words were "
            "added; … marks omitted unchanged text.", ""]
    for r in by["edited"][:max_items]:
        out += [f"**{new_p[r['new']]['heading'] or '(no heading)'}** ({1 - r['ratio']:.0%} of words changed)", "",
                "> " + inline_diff(old[r["old"]], new[r["new"]]), ""]
    if len(by["edited"]) > max_items:
        out += [f"…and {len(by['edited']) - max_items} more.", ""]

    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"riskdiff_{cur.ticker}_{cur.filing_date.year}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)
