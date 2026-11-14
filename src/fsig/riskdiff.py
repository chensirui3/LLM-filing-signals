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

from .config import REPORTS, ROOT
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
        "pairs in random order without the cosine values, and have not been independently checked by a second "
        f"annotator. {int(pairs['borderline'].sum())} close calls are flagged `borderline` in "
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


def render(filings: pd.DataFrame, ticker: str, year: int | None, emb: dict, llm: dict | None,
           max_items: int = 12) -> tuple[str, str]:
    """Compare a 10-K's Risk Factors with the previous year's. Returns (report name, markdown)."""
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

    return f"riskdiff_{cur.ticker}_{cur.filing_date.year}", "\n".join(out)
