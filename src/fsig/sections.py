"""Turn a raw 10-K / 10-Q HTML document into clean MD&A and Risk Factors text."""
from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from lxml import etree
from lxml import html as lxml_html
from tqdm import tqdm

from .config import DATA
from .edgar import filing_path

SECTIONS = DATA / "sections"

_BLOCK = {
    "p", "div", "tr", "br", "li", "ul", "ol", "table", "hr", "title", "section",
    "h1", "h2", "h3", "h4", "h5", "h6", "center", "blockquote", "pre",
}
_HIDDEN = re.compile(r"display\s*:\s*none", re.I)
# Old filings write curly quotes and dashes as Windows-1252 numeric entities
# (&#146;), which arrive as control characters. Map them to the real characters.
_C1 = str.maketrans({0x91: "‘", 0x92: "’", 0x93: "“", 0x94: "”",
                     0x95: "•", 0x96: "–", 0x97: "—"})
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>")
_SPACES = re.compile(r"[ \t  -​ 　]+")


_EMPH_TAGS = {"b", "strong", "i", "em", "u"}
_EMPH_STYLE = re.compile(r"font-weight\s*:\s*(?:bold|[6-9]00)|font-style\s*:\s*italic|text-decoration\s*:\s*underline", re.I)


def _walk(raw: bytes) -> list[tuple[str, bool]]:
    """Visible text pieces in document order, each flagged if bold/italic/underlined.

    Line breaks are emitted only at block boundaries. Inline elements are joined
    without a separator because filings often split a single word across
    <span>s; table cells get a space.
    """
    # Decode ourselves: without a charset declaration lxml assumes Latin-1 and
    # mangles curly apostrophes, which breaks the "Management's" header match.
    try:
        doc = raw.decode("utf-8")
    except UnicodeDecodeError:
        doc = raw.decode("cp1252", errors="replace")
    doc = _XML_DECL.sub("", doc, count=1)  # lxml rejects str input that declares an encoding
    tree = lxml_html.fromstring(doc)
    drop = []
    for el in tree.iter():
        if not isinstance(el.tag, str):
            continue
        tag = el.tag.lower()
        # ix:header holds hidden XBRL metadata; display:none hides the rest of it
        if tag in ("script", "style", "head", "ix:header") or _HIDDEN.search(el.get("style") or ""):
            drop.append(el)
    for el in drop:
        if el.getparent() is not None:
            el.drop_tree()

    parts: list[tuple[str, bool]] = []
    depth = 0  # how many enclosing elements are emphasised
    for event, el in etree.iterwalk(tree, events=("start", "end")):
        tag = el.tag.lower() if isinstance(el.tag, str) else ""
        emph = tag in _EMPH_TAGS or (bool(tag) and bool(_EMPH_STYLE.search(el.get("style") or "")))
        if event == "start":
            depth += emph
            if tag in _BLOCK:
                parts.append(("\n", False))
            if tag and el.text:
                parts.append((el.text, depth > 0))
        else:
            depth -= emph
            if tag in _BLOCK:
                parts.append(("\n", False))
            elif tag in ("td", "th"):
                parts.append((" ", False))
            if el.tail:
                parts.append((el.tail, depth > 0))
    return parts


def html_to_text(raw: bytes) -> str:
    """Visible text, one line per block element."""
    text = "".join(t for t, _ in _walk(raw)).replace("\r", "\n").translate(_C1)
    lines = (_SPACES.sub(" ", ln).strip() for ln in text.split("\n"))
    return "\n".join(ln for ln in lines if ln)


def emphasised_lines(raw: bytes, min_share: float = 0.9) -> set[str]:
    """Lines of html_to_text(raw) that are almost entirely bold, italic or underlined.

    Filings mark headings with formatting rather than heading tags, so this is
    how a heading is told apart from a short paragraph.
    """
    parts = _walk(raw)
    text = "".join(t for t, _ in parts).replace("\r", "\n").translate(_C1)
    mask = "".join(("1" if e else "0") * len(t) for t, e in parts)
    out, pos = set(), 0
    for ln in text.split("\n"):
        flags = [m for ch, m in zip(ln, mask[pos : pos + len(ln)]) if not ch.isspace()]
        pos += len(ln) + 1
        if flags and flags.count("1") / len(flags) >= min_share:
            out.add(_SPACES.sub(" ", ln).strip())
    return out


_SEP = r"[\s\.\:\-–—]*"


def _hdr(item: str, title: str) -> re.Pattern:
    # Headers must start a line, which filters out most in-sentence cross-references.
    return re.compile(rf"^(?:part\s+i+\s*[,\.\-–—]?\s*)?item\s*{item}\b{_SEP}{title}", re.I | re.M)


def _title(title: str) -> re.Pattern:
    """A short line that starts with the title: a heading, not a sentence."""
    return re.compile(rf"^{title}[^\n]{{0,110}}$", re.I | re.M)


_MDNA = r"management\W{0,2}s?\s+discussion"
_PATTERNS = {
    "10-K": {
        "mdna": (
            [_hdr("7", _MDNA)],
            [_hdr("7a", "quantitative"), _hdr("8", r"(?:consolidated\s+)?financial\s+statements")],
        ),
        "risk": (
            [_hdr("1a", r"risk\s+factors")],
            [_hdr("1b", "unresolved"), _hdr("1c", "cybersecurity"), _hdr("2", "properties")],
        ),
    },
    "10-Q": {
        "mdna": (
            [_hdr("2", _MDNA)],
            [_hdr("3", "quantitative"), _hdr("4", "controls")],
        ),
        "risk": (
            [_hdr("1a", r"risk\s+factors")],
            [_hdr("2", "unregistered"), _hdr("5", "other"), _hdr("6", "exhibits")],
        ),
    },
}

# Fallbacks for filings that do not label the MD&A with an "Item" number in the
# body (banks, Intel, Chevron...): the item numbers only appear in an index
# that points to pages. The section then runs from an MD&A heading to the first
# heading that normally follows it.
_MDNA_TITLE_START = [_title(_MDNA)]
# Banks often skip the MD&A heading too and open with one of these.
_MDNA_ALT_START = [_title(r"executive\s+(?:overview|summary)\s*$"), _title(r"financial\s+review\s*$")]
_INCOME_STMT = _title(r"(?:condensed\s+)?consolidated\s+(?:condensed\s+)?statements?\s+of\s+(?:income|operations|earnings)")
_MDNA_TITLE_END = {
    "10-K": [
        _title(r"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market"),
        _title(r"(?:consolidated\s+)?financial\s+statements\s+and\s+supplementa"),
        _title(r"management\W{0,2}s?\s+report\s+on\s+internal\s+control"),
        _title(r"reports?\s+of\s+independent\s+registered"),
        _INCOME_STMT,
    ],
    "10-Q": [
        _title(r"quantitative\s+and\s+qualitative\s+disclosures?\s+about\s+market"),
        _title(r"controls\s+and\s+procedures"),
        _title(r"part\s+ii\b"),
        _title(r"legal\s+proceedings"),
        _INCOME_STMT,
    ],
}
# A fallback span that swallows another item's header started in the table of
# contents, not in the body.
_OTHER_ITEM = re.compile(r"^item\s*(?:1a?|2|3|5|8|9a?)\b[\s\.\:\-–—]*[a-z]", re.I | re.M)

MIN_SECTION_CHARS = 2000
BRIDGE_CHARS = 6000


def find_section(text: str, starts: list[re.Pattern], ends: list[re.Pattern],
                 reject: re.Pattern | None = None, bridge: int = 0) -> str | None:
    """Longest span from a start header to the next end header.

    Each header appears at least twice (table of contents and body). The table
    of contents span is a few lines; the real section is the longest. Spans
    containing a `reject` match are skipped.

    bridge > 0 handles filings that repeat the section title as a running page
    header: if another start header follows an end header within `bridge`
    characters, the section has not really ended, so the span continues.
    """
    end_pos = sorted(m.start() for p in ends for m in p.finditer(text))
    start_matches = sorted((m.start(), m.end()) for p in starts for m in p.finditer(text))
    start_pos = [s for s, _ in start_matches]
    best = None
    for s0, s0_end in start_matches:
        nxt = next((e for e in end_pos if e > s0_end), None)
        while nxt is not None and bridge:
            resume = next((s for s in start_pos if nxt < s <= nxt + bridge), None)
            if resume is None:
                break
            nxt = next((e for e in end_pos if e > resume), None)
        if nxt is None or (best is not None and nxt - s0 <= best[1] - best[0]):
            continue
        if reject is not None and reject.search(text, s0_end, nxt):
            continue
        best = (s0, nxt)
    if best is None or best[1] - best[0] < MIN_SECTION_CHARS:
        return None
    return text[best[0] : best[1]]


def find_mdna(text: str, form: str) -> str | None:
    """MD&A by Item header; else by title; else by a bank-style opening heading."""
    starts, ends = _PATTERNS[form]["mdna"]
    loose_ends = ends + _MDNA_TITLE_END[form]
    return (
        find_section(text, starts, ends)
        or find_section(text, _MDNA_TITLE_START, loose_ends, reject=_OTHER_ITEM, bridge=BRIDGE_CHARS)
        or find_section(text, _MDNA_ALT_START, loose_ends, reject=_OTHER_ITEM)
    )


def clean_section(section: str) -> str:
    """Drop lines that are mostly numbers (flattened financial tables)."""
    kept = []
    for ln in section.split("\n"):
        dense = ln.replace(" ", "")
        letters = sum(c.isalpha() for c in dense)
        if dense and letters / len(dense) >= 0.6:
            kept.append(ln)
    return "\n".join(kept)


def extract_sections(raw: bytes, form: str) -> dict[str, str | None]:
    text = html_to_text(raw)
    out = {}
    for name, (starts, ends) in _PATTERNS[form].items():
        sec = find_mdna(text, form) if name == "mdna" else find_section(text, starts, ends)
        out[name] = clean_section(sec) if sec else None
    return out


def section_path(accession: str) -> Path:
    return SECTIONS / f"{accession}.json"


def load_section(accession: str, name: str = "mdna") -> str | None:
    p = section_path(accession)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(name)


def _process(args: tuple[int, str, str]) -> tuple[str, bool]:
    cik, accession, form = args
    src = filing_path(cik, accession)
    try:
        with gzip.open(src, "rb") as f:
            out = extract_sections(f.read(), form)
    except Exception:
        out = {"mdna": None, "risk": None}
    dst = section_path(accession)
    tmp = dst.with_suffix(".tmp")  # atomic: other steps may be reading sections
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(dst)
    return accession, out["mdna"] is not None


def parse_all(filings: pd.DataFrame, workers: int = 6, retry_failed: bool = False) -> pd.DataFrame:
    """Parse every downloaded filing not yet parsed. Returns coverage per form.

    retry_failed also re-parses filings where no MD&A was found, which is what
    you want after improving the parser. Filings that parsed are never touched,
    so their cached LLM scores and embeddings stay valid.
    """
    SECTIONS.mkdir(parents=True, exist_ok=True)

    def needed(accession: str) -> bool:
        p = section_path(accession)
        if not p.exists():
            return True
        return retry_failed and json.loads(p.read_text(encoding="utf-8"))["mdna"] is None

    todo = [
        (r.cik, r.accession, r.form)
        for r in filings.itertuples()
        if filing_path(r.cik, r.accession).exists() and needed(r.accession)
    ]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(tqdm(pool.map(_process, todo, chunksize=4), total=len(todo), desc="parse"))
    return coverage(filings)


def coverage(filings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in filings.itertuples():
        p = section_path(r.accession)
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            rows.append({"form": r.form, "mdna": s["mdna"] is not None, "risk": s["risk"] is not None})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby("form").agg(filings=("mdna", "size"), mdna=("mdna", "mean"), risk=("risk", "mean"))
