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

compare_filings returns a comparison as plain data; to_markdown formats it as
a report, and export_demo precomputes comparisons for the static demo page in
docs/.
"""
from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import HOME, REPORTS, ROOT
from .embed import embed_texts
from .extract import ollama_chat
from .edgar import ARCHIVE_URL, filing_path
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


_SENTENCE_END = re.compile(r"(?:[.!?:;][\"'”’)\]]*|;\s*(?:and|or))$")  # '; and' closes a list item
MAX_FURNITURE_LINES, MAX_FURNITURE_CHARS = 3, 80


def _join_page_breaks(lines: list[tuple[str, str]], is_heading: list[bool]) -> tuple[list[tuple[str, str]], list[bool]]:
    """Rejoin paragraphs that a page break split in two.

    A body paragraph that stops without sentence-ending punctuation is joined
    to the next line that starts with a lower-case letter. Up to
    MAX_FURNITURE_LINES short lines in between (page numbers, running headers
    such as 'Table of Contents' or the company name) are dropped. Bulleted
    lines and headings are never joined, so list items stay separate.
    """
    out, out_heading, i = [], [], 0
    while i < len(lines):
        raw, ln = lines[i]
        while len(ln) >= MIN_PARA_CHARS and not is_heading[i] and not _SENTENCE_END.search(ln):
            nxt, skipped = None, 0
            for k in range(i + 1, len(lines)):
                cand = lines[k][0]
                if not cand:
                    continue
                if cand[0].islower() and not is_heading[k]:
                    nxt = k
                elif cand != lines[k][1]:  # a bulleted line: the paragraph introduces a list
                    pass
                elif len(cand) <= MAX_FURNITURE_CHARS and skipped < MAX_FURNITURE_LINES:
                    skipped += 1
                    continue
                break
            if nxt is None:
                break
            ln, i = f"{ln} {lines[nxt][1]}", nxt
        out.append((raw, ln))
        out_heading.append(is_heading[i] and ln == lines[i][1])
        i += 1
    return out, out_heading


def _reflow(lines: list[tuple[str, str]], is_heading: list[bool]) -> tuple[list[tuple[str, str]], list[bool]]:
    """Rebuild the paragraphs of a hard-wrapped section, in which every line is short.

    Lines are joined until one ends a sentence well short of the wrap width
    (estimated as the 90th percentile of line lengths), which marks the end
    of a paragraph. A new paragraph also starts at a blank or bulleted line,
    where heading formatting starts or stops, and after a short line without
    final punctuation that is followed by a capitalised line (a heading).
    """
    lengths = sorted(len(ln) for _, ln in lines if ln)
    if not lengths:
        return lines, is_heading
    width = lengths[int(0.9 * (len(lengths) - 1))]
    out, out_heading, raw_buf, buf = [], [], [], []
    buf_heading = False

    def flush() -> None:
        if buf:
            out.append((" ".join(raw_buf), " ".join(buf)))
            out_heading.append(buf_heading)
        raw_buf.clear()
        buf.clear()

    for (raw, ln), heading in zip(lines, is_heading):
        if not ln:
            flush()
            continue
        if buf:
            prev = buf[-1]
            if (heading != buf_heading or raw != ln
                    or (_SENTENCE_END.search(prev) and len(prev) < 0.85 * width)
                    or (not re.search(r"[.!?:;,]$", prev) and len(prev) < 0.6 * width and ln[0].isupper())):
                flush()
        if not buf:
            buf_heading = heading
        raw_buf.append(raw)
        buf.append(ln)
    flush()
    return out, out_heading


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
    marked = [use_format and m for m in marked]
    if max((len(ln) for _, ln in lines), default=0) < MIN_PARA_CHARS:
        lines, marked = _reflow(lines, marked)  # hard-wrapped: no line is long enough to be a paragraph
    lines, marked = _join_page_breaks(lines, marked)

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


def diff_segments(a: str, b: str, context: int = 10) -> list[list[str]]:
    """Word diff from a to b as [op, text] pairs: '=' unchanged, '-' removed, '+' added.

    Unchanged runs longer than 2 * context + 3 words keep `context` words on
    each side of a '…' (one side only at the start or end of the paragraph).
    """
    wa, wb = _WORDS.findall(a), _WORDS.findall(b)
    ops = SequenceMatcher(None, wa, wb, autojunk=False).get_opcodes()
    out = []
    for n, (op, i1, i2, j1, j2) in enumerate(ops):
        if op == "equal":
            run = wa[i1:i2]
            if len(run) > 2 * context + 3:
                head = [] if n == 0 else run[:context]
                tail = [] if n == len(ops) - 1 else run[-context:]
                run = head + ["…"] + tail
            out.append(["=", " ".join(run)])
        else:
            if i2 > i1:
                out.append(["-", " ".join(wa[i1:i2])])
            if j2 > j1:
                out.append(["+", " ".join(wb[j1:j2])])
    return [s for s in out if s[1]]


_DIFF_MARK = {"=": "{}", "-": "~~{}~~", "+": "**{}**"}


def _diff_markdown(segments: list[list[str]]) -> str:
    return " ".join(_DIFF_MARK[op].format(text) for op, text in segments)


def inline_diff(a: str, b: str, context: int = 10) -> str:
    """Word diff as markdown: ~~removed~~ **added**, with unchanged runs shortened."""
    return _diff_markdown(diff_segments(a, b, context))


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


# ----------------------------------------------------------------- threshold validation

VALIDATION = ROOT / "validation" / "riskdiff_pairs.csv"
COS_BINS = [0.0, 0.85, 0.90, 0.93, 0.96, 1.01]


def sample_pairs(filings: pd.DataFrame, emb: dict, specs: list[tuple[str, int]], n: int, seed: int = 0) -> pd.DataFrame:
    """Paragraph pairs for checking the TOPIC threshold.

    For each (ticker, filing year) in specs, takes every current paragraph whose
    wording differs from its closest previous paragraph (word ratio below
    EDITED), which are the cases the embedding threshold has to decide. Returns
    n pairs spread evenly over the cosine ranges in COS_BINS, with an empty
    `label` column to be filled in: 1 if the current paragraph is a rewrite of
    the previous one (same specific content), 0 if it is new content.
    """
    rows = []
    for ticker, year in specs:
        try:
            prev, cur = pick_filings(filings, ticker, year)
        except SystemExit as e:
            print(f"skipped {ticker} {year}: {e}")
            continue
        old = [p["text"] for p in load_paragraphs(prev.cik, prev.accession)]
        new = [p["text"] for p in load_paragraphs(cur.cik, cur.accession)]
        sim = _embed(emb, new) @ _embed(emb, old).T
        for j, text in enumerate(new):
            i = int(sim[j].argmax())
            ratio = word_ratio(old[i], text)
            if ratio < EDITED:
                rows.append({"id": f"{ticker}-{year}-{j}", "ticker": ticker, "year": year, "cos": float(sim[j, i]),
                             "ratio": ratio, "previous": old[i], "current": text})
    pool = pd.DataFrame(rows)
    pool["bin"] = pd.cut(pool["cos"], COS_BINS, right=False)
    per_bin = n // (len(COS_BINS) - 1)
    picked = [g.sample(min(per_bin, len(g)), random_state=seed) for _, g in pool.groupby("bin", observed=True)]
    return pd.concat(picked).drop(columns="bin").assign(label="").sort_values("id").reset_index(drop=True)


def threshold_table(pairs: pd.DataFrame, thresholds: tuple[float, ...] = (0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94,
                                                                         0.95, 0.96)) -> pd.DataFrame:
    """Accuracy of "cosine >= threshold means rewritten" against the labels.

    precision  share of pairs called rewritten that are rewrites
    recall     share of rewrites that are called rewritten
    """
    y, rows = pairs["label"].astype(int).to_numpy() == 1, []
    for t in thresholds:
        pred = pairs["cos"].to_numpy() >= t
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        rows.append({"threshold": t, "called_rewritten": int(pred.sum()), "precision": precision, "recall": recall,
                     "f1": 2 * precision * recall / (precision + recall) if precision + recall else float("nan"),
                     "accuracy": float((pred == y).mean())})
    return pd.DataFrame(rows)


def validation_report(table_writer) -> str:
    """Write reports/riskdiff_validation.md from the labelled pairs and return its path.

    table_writer turns a DataFrame into a markdown table (report._md).
    """
    pairs = pd.read_csv(VALIDATION)
    clear = pairs[pairs["borderline"] == 0]
    by_range = (pairs.assign(cosine=pd.cut(pairs["cos"], COS_BINS, right=False).astype(str))
                .groupby("cosine")["label"].agg(pairs="size", share_rewrites="mean").reset_index())
    at = threshold_table(pairs, (TOPIC,)).iloc[0]
    out = [
        "# Risk Factors comparison: validation of the similarity threshold", "",
        f"The comparison tool calls a paragraph *rewritten* rather than *added* when its wording differs from the "
        f"closest paragraph in the previous filing but their embedding cosine is at least {TOPIC}. That threshold "
        "was chosen by inspecting one company (NVIDIA). This report tests it on other companies.", "",
        f"**Sample.** {len(pairs)} paragraph pairs from {pairs['ticker'].nunique()} companies "
        f"({', '.join(sorted(pairs['ticker'].unique()))}), none of them NVIDIA. Each pair is a current paragraph "
        f"and its closest previous paragraph, restricted to pairs whose wording differs (word-level similarity "
        f"below {EDITED}), which are the cases the threshold decides. Equal numbers were drawn from five cosine "
        "ranges, so borderline cases are over-represented relative to a real filing.", "",
        "**Labels.** Each pair was labelled 1 if the current paragraph restates the specific content of the "
        "previous one (including a condensed or expanded version, or a fragment of it) and 0 if it is new "
        "content, even on the same broad theme. The labels were assigned by an LLM assistant (Claude) reading the "
        "pairs in random order without the cosine values. The author spot-checked a sample of them; they have "
        f"not been labelled independently by a second annotator. {int(pairs['borderline'].sum())} close calls are flagged `borderline` in "
        "`validation/riskdiff_pairs.csv`.", "",
        f"**Result at the threshold in use ({TOPIC}).** Precision {at['precision']:.0%}, recall "
        f"{at['recall']:.0%}: of the paragraphs the tool calls rewritten, {at['precision']:.0%} are rewrites, and "
        f"it finds {at['recall']:.0%} of all rewrites. The remaining rewrites are reported as added.", "",
        "## Accuracy by threshold", "", table_writer(threshold_table(pairs)), "",
        "Excluding the borderline pairs:", "", table_writer(threshold_table(clear)), "",
        "## Share of pairs labelled as rewrites, by cosine range", "", table_writer(by_range), "",
        "## Reading the result", "",
        "- Cosine similarity orders the pairs well: almost none below 0.85 are rewrites and all of those above "
        "0.96 are.",
        f"- A threshold of 0.92 scores slightly higher on this sample than {TOPIC}. The difference is within the "
        f"sampling error of {len(pairs)} pairs, and adopting it would mean tuning on the test set, so the "
        "threshold is unchanged.",
        "- Many low-similarity pairs in the sample are fragments: the parser splits a paragraph where the "
        "filing has a page break, and the fragment is then compared with a whole paragraph. Joining paragraphs "
        "across page breaks would remove most of these.", ""]
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "riskdiff_validation.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return str(path)


def risk_diff(filings: pd.DataFrame, ticker: str, year: int | None, emb: dict, llm: dict | None,
              max_items: int = 12) -> str:
    """Write reports/riskdiff_<TICKER>_<year>.md and return its path."""
    name, text = render(filings, ticker, year, emb, llm, max_items)
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def available(filings: pd.DataFrame) -> dict[str, list[int]]:
    """Tickers that can be compared, each with the filing years that have an earlier 10-K to compare with."""
    k = filings[filings["form"] == "10-K"].sort_values("filing_date")
    k = k[[load_section(a, "risk") is not None for a in k["accession"]]]
    years = k.groupby("ticker")["filing_date"].apply(lambda d: sorted(d.dt.year)[1:])
    return {t: y for t, y in years.items() if y}


# ----------------------------------------------------------------- comparison

STATUSES = ("unchanged", "edited", "rewritten", "added")


def _filing_meta(row) -> dict:
    return {"date": f"{row.filing_date:%Y-%m-%d}", "accession": row.accession,
            "url": ARCHIVE_URL.format(cik=row.cik, acc=row.accession.replace("-", ""), doc=row.primary_doc)}


def _prepared(row, emb: dict, cache: dict | None) -> tuple[list[dict], np.ndarray]:
    """A filing's Risk Factors paragraphs and their embeddings, reused from `cache` if present."""
    if cache is not None and row.accession in cache:
        return cache[row.accession]
    paras = load_paragraphs(row.cik, row.accession)
    out = (paras, _embed(emb, [p["text"] for p in paras]) if paras else np.zeros((0, 0)))
    if cache is not None:
        cache[row.accession] = out
    return out


def compare_filings(prev, cur, emb: dict, llm: dict | None, max_items: int = 12,
                    cache: dict | None = None) -> dict:
    """Compare the Risk Factors of two 10-K filings (rows of the filing index).

    Returns plain data: paragraph counts and shares of the current text by
    status, and up to `max_items` paragraphs of each kind, with one-line
    summaries of the added and removed ones if `llm` is given. `cache` (a
    dict) keeps each filing's paragraphs and embeddings, so a filing compared
    twice is read and embedded once.
    """
    old_p, old_vecs = _prepared(prev, emb, cache)
    new_p, new_vecs = _prepared(cur, emb, cache)
    if not old_p or not new_p:
        raise ValueError(f"no Risk Factors paragraphs found in {(cur if old_p else prev).accession}")
    old, new = [p["text"] for p in old_p], [p["text"] for p in new_p]
    rows, removed = classify(old, new, old_vecs, new_vecs)

    by = {s: [r for r in rows if r["status"] == s] for s in STATUSES}
    # Least similar to the other filing first, with full paragraphs ranked above one-line list items.
    by["added"].sort(key=lambda r: -(1 - r["cos"]) * min(len(new[r["new"]]), 600))
    removed.sort(key=lambda r: -(1 - r["cos"]) * min(len(old[r["old"]]), 600))
    by["edited"].sort(key=lambda r: r["ratio"])
    added, gone = by["added"][:max_items], removed[:max_items]
    notes = _summaries(llm, [new[r["new"]] for r in added] + [old[r["old"]] for r in gone]) if llm else []
    notes += [""] * (len(added) + len(gone) - len(notes))

    total = sum(map(len, new)) or 1
    return {
        "ticker": cur.ticker, "name": cur["name"],
        "previous": _filing_meta(prev), "current": _filing_meta(cur),
        "paragraphs": {"previous": len(old), "current": len(new)},
        "counts": {s: len(by[s]) for s in STATUSES} | {"removed": len(removed)},
        "share": {s: sum(len(new[r["new"]]) for r in by[s]) / total for s in STATUSES},
        "embed_model": emb["model"], "llm_model": llm["model"] if llm else None,
        "added": [{"heading": new_p[r["new"]]["heading"], "summary": note, "text": _excerpt(new[r["new"]])}
                  for r, note in zip(added, notes[: len(added)])],
        "removed": [{"heading": old_p[r["old"]]["heading"], "summary": note, "text": _excerpt(old[r["old"]])}
                    for r, note in zip(gone, notes[len(added):])],
        "rewritten": [{"heading": new_p[r["new"]]["heading"], "previous": _excerpt(old[r["old"]], 400),
                       "current": _excerpt(new[r["new"]], 400)}
                      for r in sorted(by["rewritten"], key=lambda r: r["ratio"])[:max_items]],
        "edited": [{"heading": new_p[r["new"]]["heading"], "changed": 1 - r["ratio"],
                    "diff": diff_segments(old[r["old"]], new[r["new"]])}
                   for r in by["edited"][:max_items]],
    }


def compare(filings: pd.DataFrame, ticker: str, year: int | None, emb: dict, llm: dict | None,
            max_items: int = 12) -> dict:
    """Compare a company's 10-K filed in `year` (default: the latest) with the one before it."""
    prev, cur = pick_filings(filings, ticker, year)
    return compare_filings(prev, cur, emb, llm, max_items)


def to_markdown(c: dict) -> str:
    """A comparison from compare_filings as a markdown report."""
    counts = c["counts"]
    out = [f"# {c['name']} ({c['ticker']}): changes in Risk Factors",
           "",
           f"10-K filed {c['current']['date']} compared with the 10-K filed {c['previous']['date']}. "
           f"Paragraphs compared: {c['paragraphs']['previous']} in the previous filing, "
           f"{c['paragraphs']['current']} in the current filing.",
           "",
           "| Status | Paragraphs | Share of current text |",
           "|---|---|---|"]
    for s in STATUSES:
        out.append(f"| {s} | {counts[s]} | {c['share'][s]:.0%} |")
    note = f"Paragraphs are matched by embedding similarity ({c['embed_model']})."
    if c["llm_model"]:
        note += (f" The one-line summaries are generated by {c['llm_model']} and may be inaccurate; "
                 "the quoted filing text is authoritative.")
    out += [f"| removed (from the previous filing) | {counts['removed']} | |", "", note, ""]

    def more(shown: int, total_n: int) -> None:
        if total_n > shown:
            out.extend([f"…and {total_n - shown} more.", ""])

    for title, kind in (("Added", "added"), ("Removed", "removed")):
        out.extend([f"## {title} ({counts[kind]})", ""])
        if not c[kind]:
            out.extend(["None.", ""])
        for it in c[kind]:
            out.extend([f"**{it['summary'] or it['heading'] or '(no heading)'}**", ""])
            if it["summary"] and it["heading"]:
                out.extend([f"*Under: {it['heading']}*", ""])
            out.extend(["> " + it["text"], ""])
        more(len(c[kind]), counts[kind])

    out += [f"## Rewritten: same content, new wording ({counts['rewritten']})", ""]
    for it in c["rewritten"]:
        out += [f"**{it['heading'] or '(no heading)'}**", "",
                "> *Previous:* " + it["previous"], ">", "> *Current:* " + it["current"], ""]
    if not counts["rewritten"]:
        out += ["None.", ""]
    more(len(c["rewritten"]), counts["rewritten"])

    out += [f"## Edited paragraphs ({counts['edited']})", "",
            "Ordered from most to least changed. ~~Struck~~ words were removed and **bold** words were "
            "added; … marks omitted unchanged text.", ""]
    for it in c["edited"]:
        out += [f"**{it['heading'] or '(no heading)'}** ({it['changed']:.0%} of words changed)", "",
                "> " + _diff_markdown(it["diff"]), ""]
    more(len(c["edited"]), counts["edited"])
    return "\n".join(out)


def render(filings: pd.DataFrame, ticker: str, year: int | None, emb: dict, llm: dict | None,
           max_items: int = 12) -> tuple[str, str]:
    """Compare a 10-K's Risk Factors with the previous year's. Returns (report name, markdown)."""
    c = compare(filings, ticker, year, emb, llm, max_items)
    return f"riskdiff_{c['ticker']}_{c['current']['date'][:4]}", to_markdown(c)


# ----------------------------------------------------------------- static demo

DEMO = HOME / "docs" / "data"


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def _rounded(c: dict) -> dict:
    """A comparison with its fractions rounded, which keeps the demo files small."""
    return {**c, "share": {k: round(v, 4) for k, v in c["share"].items()},
            "edited": [{**e, "changed": round(e["changed"], 3)} for e in c["edited"]]}


def export_demo(filings: pd.DataFrame, tickers: list[str], emb: dict, llm: dict | None,
                max_items: int = 12) -> dict:
    """Precompute the comparisons shown by the static demo page, docs/index.html.

    For each ticker, every 10-K with a parsed Risk Factors section is compared
    with the one before it; for a year with two 10-Ks the later one is used, as
    in `pick_filings`. Writes docs/data/<TICKER>.json (one comparison per
    filing year) and docs/data/index.json. Comparisons already in a company's
    file are kept, so an interrupted run resumes; delete docs/data to compute
    everything again. Returns the number of comparisons computed, kept and
    failed.
    """
    DEMO.mkdir(parents=True, exist_ok=True)
    stats = {"computed": 0, "kept": 0, "failed": 0}
    companies = []
    tenk = filings[(filings["form"] == "10-K") & filings["ticker"].isin(tickers)].sort_values("filing_date")
    for ticker in tqdm(tickers, desc="demo"):
        k = tenk[tenk["ticker"] == ticker]
        k = k[[load_section(a, "risk") is not None for a in k["accession"]]].reset_index(drop=True)
        pairs = {int(k.loc[i, "filing_date"].year): (k.iloc[i - 1], k.iloc[i]) for i in range(1, len(k))}
        path = DEMO / f"{ticker}.json"
        done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        done = {y: v for y, v in done.items() if int(y) in pairs}
        cache: dict = {}
        for year, (prev, cur) in pairs.items():
            if done.get(str(year), {}).get("current", {}).get("accession") == cur.accession:
                stats["kept"] += 1
                continue
            try:
                done[str(year)] = _rounded(compare_filings(prev, cur, emb, llm, max_items, cache))
            except Exception as e:  # one unreadable filing should not stop the run
                tqdm.write(f"skip {ticker} {year}: {e}")
                stats["failed"] += 1
                continue
            stats["computed"] += 1
            _write_json(path, dict(sorted(done.items())))
        if done:
            companies.append({"ticker": ticker, "name": k.iloc[-1]["name"], "years": sorted(map(int, done))})
    names = filings.drop_duplicates("ticker").set_index("ticker")["name"]
    shown = {c["ticker"] for c in companies}
    _write_json(DEMO / "index.json", {
        "generated": date.today().isoformat(), "embed_model": emb["model"],
        "llm_model": llm["model"] if llm else None, "max_items": max_items, "selected": len(tickers),
        "companies": companies,
        # fewer than two 10-Ks with a Risk Factors section that could be read
        "missing": [{"ticker": t, "name": names.get(t, "")} for t in tickers if t not in shown]})
    return stats
