"""Embedding-based year-over-year similarity: Lazy Prices with a modern text model.

The word-count baseline (signals.similarity_signal) only sees which words are
used. Two paragraphs that say the same thing in different words look different
to it, and a paragraph that reuses the same vocabulary to say something new
looks unchanged. Embeddings compare meaning instead.

Each MD&A is cut into ~1,500-character chunks and every chunk is embedded with
a local model through Ollama. Two signals compare a filing with the same form
filed about a year earlier:

  emb_sim_yoy    cosine similarity of the two documents' mean chunk vectors
                 (one number for "is the overall content the same?")
  emb_align_yoy  for each chunk, the best-matching chunk in the other filing;
                 averaged in both directions. Low values mean passages were
                 added or removed, even if the document's overall topic mix is
                 unchanged.

Higher = less change = expected higher return, same sign as sim_yoy.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .config import DATA
from .extract import chunk_text
from .sections import load_section

EMB = DATA / "embeddings"


def emb_path(model: str, accession: str, section: str = "mdna") -> Path:
    return EMB / model.replace(":", "_") / f"{accession}_{section}.npy"


def embed_texts(emb: dict, texts: list[str]) -> np.ndarray:
    """Unit-length embedding per text."""
    r = requests.post(
        f"{emb['host']}/api/embed",
        json={"model": emb["model"], "input": [emb["prefix"] + t for t in texts], "truncate": True},
        timeout=600,
    )
    r.raise_for_status()
    v = np.asarray(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed_all(filings: pd.DataFrame, emb: dict, section: str = "mdna") -> dict:
    """Embed every parsed section that has no saved embedding yet."""
    todo = [a for a in filings["accession"] if not emb_path(emb["model"], a, section).exists()]
    stats = {"cached": len(filings) - len(todo), "embedded": 0, "no_text": 0}

    def work(accession: str) -> bool:
        text = load_section(accession, section)
        if not text:
            return False
        chunks = chunk_text(text, emb["chunk_chars"], 10**9)
        vecs = np.vstack([embed_texts(emb, chunks[i : i + emb["batch"]])
                          for i in range(0, len(chunks), emb["batch"])])
        path = emb_path(emb["model"], accession, section)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, vecs.astype(np.float16))  # half precision: plenty for cosine, half the disk
        tmp.replace(path)
        return True

    with ThreadPoolExecutor(max_workers=emb["workers"]) as pool:
        for ok in tqdm(pool.map(work, todo), total=len(todo), desc=f"embed[{section}]"):
            stats["embedded" if ok else "no_text"] += 1
    return stats


def prior_year_map(filings: pd.DataFrame, has) -> dict[str, str]:
    """accession -> accession of the same form filed 320-410 days earlier (closest to 365).

    `has(accession)` says whether a filing is usable. Same matching rule as the
    word-count baseline so the two signals are compared on equal terms.
    """
    out = {}
    for _, grp in filings.sort_values("filing_date").groupby(["ticker", "form"]):
        history: list[tuple[pd.Timestamp, str]] = []
        for r in grp.itertuples():
            if not has(r.accession):
                continue
            prior = [(abs((r.filing_date - d).days - 365), a) for d, a in history
                     if 320 <= (r.filing_date - d).days <= 410]
            if prior:
                out[r.accession] = min(prior)[1]
            history.append((r.filing_date, r.accession))
    return out


def doc_similarity(cur: np.ndarray, prev: np.ndarray) -> float:
    a, b = cur.mean(axis=0), prev.mean(axis=0)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def alignment(cur: np.ndarray, prev: np.ndarray) -> float:
    """Mean best-match cosine, averaged over both directions (additions and deletions)."""
    sim = cur @ prev.T
    return float((sim.max(axis=1).mean() + sim.max(axis=0).mean()) / 2)


def embedding_signals(filings: pd.DataFrame, emb: dict, section: str = "mdna") -> pd.DataFrame:
    """One row per filing that has a prior-year match: emb_sim_yoy, emb_align_yoy."""
    model = emb["model"]
    pairs = prior_year_map(filings, lambda a: emb_path(model, a, section).exists())
    cache: dict[str, np.ndarray] = {}

    def load(a: str) -> np.ndarray:
        if a not in cache:
            cache[a] = np.load(emb_path(model, a, section)).astype(np.float32)
        return cache[a]

    rows = []
    for cur, prev in tqdm(pairs.items(), desc="embedding signals"):
        c, p = load(cur), load(prev)
        rows.append({"accession": cur, "emb_sim_yoy": doc_similarity(c, p), "emb_align_yoy": alignment(c, p)})
    return pd.DataFrame(rows, columns=["accession", "emb_sim_yoy", "emb_align_yoy"])
