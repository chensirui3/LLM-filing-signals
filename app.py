"""Browser front end for the Risk Factors comparison.

    .venv\\Scripts\\python -m pip install -e .[app]
    .venv\\Scripts\\python -m streamlit run app.py

Needs the downloaded and parsed filings (the `download` and `parse` steps) and a
running Ollama server, because paragraphs are embedded on demand.
"""
import streamlit as st

from fsig import riskdiff
from fsig.config import load_config
from fsig.edgar import load_filings

st.set_page_config(page_title="Risk Factors comparison", layout="wide")
cfg = load_config()


@st.cache_data(show_spinner="Reading the filing index")
def choices() -> tuple[dict[str, list[int]], dict[str, str]]:
    filings = load_filings()
    names = dict(zip(filings["ticker"], filings["name"]))
    return riskdiff.available(filings), names


@st.cache_data(show_spinner="Comparing the two filings")
def compare(ticker: str, year: int, summaries: bool, max_items: int) -> tuple[str, str]:
    return riskdiff.render(load_filings(), ticker, year, cfg["embed"], cfg["llm"] if summaries else None, max_items)


years_by_ticker, names = choices()
with st.sidebar:
    st.header("Risk Factors comparison")
    ticker = st.selectbox("Company", sorted(years_by_ticker), format_func=lambda t: f"{t}  {names[t]}",
                          index=sorted(years_by_ticker).index("NVDA") if "NVDA" in years_by_ticker else 0)
    year = st.selectbox("10-K filed in", years_by_ticker[ticker][::-1])
    summaries = st.checkbox("One-line summaries of added and removed risks", value=True,
                            help=f"Written by {cfg['llm']['model']}. They may be inaccurate; the quoted text is authoritative.")
    max_items = st.slider("Paragraphs shown per section", 5, 40, 12)
    st.caption("Each paragraph of the selected 10-K is matched to the closest paragraph of the previous year's "
               "10-K by meaning, then classified by how much of its wording changed.")

name, text = compare(ticker, year, summaries, max_items)
title, _, body = text.partition("\n")
st.title(title.lstrip("# "))
intro, *sections = body.split("\n## ")
st.markdown(intro)
for tab, section in zip(st.tabs([s.partition("\n")[0] for s in sections]), sections):
    tab.markdown(section.partition("\n")[2])
st.download_button("Download as Markdown", text, file_name=f"{name}.md")
