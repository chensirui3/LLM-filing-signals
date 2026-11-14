# Filing signals

Do signals that a local LLM extracts from SEC filings predict stock returns, and
is the answer real or an artefact of the model remembering the future?

The pipeline downloads 10-K and 10-Q filings from SEC EDGAR, has a local model
(Llama 3.1 8B through Ollama) score the Management's Discussion and Analysis
section, and evaluates the scores with a point-in-time backtest. Everything runs
on one machine with free data.

## Why this is harder than it looks

An LLM trained on text up to December 2023 has read the news about what happened
to every large company before then. Hand it Tesla's 2019 filing and it may score
the text optimistically because it knows how 2020 went. Most published "LLM
predicts returns" backtests do not separate this from real skill. This project
does, in three ways:

| Check | What it does | Where |
|---|---|---|
| Cutoff split | Reports every result separately before and after the model's training cutoff. Only the post-cutoff numbers are clean. | `backtest.split_stats` |
| Memory probe | Gives the model no filing text, just two company names and a date, and asks which stock did better over the next three months. Above-chance accuracy before the cutoff only is direct evidence of memorised outcomes. Pairs are in random order, so a model that always picks the same answer scores 50%. | `leakage.memory_probe` |
| Identity probe | Masks company name and years, then asks the model to name the company. Measures whether anonymising the text would actually protect a backtest. | `leakage.identity_probe` |
| Memory controls | Checks whether the memory probe's accuracy is date-specific or just a preference for big names: how often the model picks the larger company, and how accurate its picks are when scored against the wrong date. | `leakage.memory_controls` |
| Raw vs anonymised | Scores every filing twice, with and without the company masked, and tests whether the difference predicts returns before the cutoff. | `robustness.variant_comparison` |

## Results

100 largest US filers, 4,349 scored filings, January 2015 to November 2026,
Llama 3.1 8B (training cutoff December 2023). Full tables are in
`reports/report_raw.md` and `reports/report_anon.md`.

**No signal predicts returns.** None of the six signals reaches |t| = 2 at any
horizon. The pre-declared primary signal has a net long-short Sharpe of -0.23
(-0.50 before the cutoff, 0.49 after, neither significant). The Lazy Prices
similarity signals have the wrong sign in this large-cap sample. Changing the
entry lag (0 to 5 days) or costs (0 to 25 bps) does not change any conclusion.

**The model does remember outcomes.** With no filing text, it picks the better
of two stocks 59% of the time before its cutoff (n = 300, z = 3.2) and 47%
after. This is date-specific memory, not a taste for famous names: it picks the
larger company only 51% of the time, and when its picks are scored against the
same pair at other dates accuracy falls to 51%.

**Anonymising would not hide the company.** From text with name, ticker and
years masked, the model names the right company 82% of the time.

**But the memory does not reach the filing scores.** Raw and anonymised scores
correlate 0.94, and the gap between them does not predict returns before the
cutoff (rank IC between -0.02 and -0.03, t between -1.2 and -1.8). Reading a
filing and being asked for its tone does not trigger recall of what happened
next, at least for this model and prompt.

## Method

1. **Universe**: the largest US companies in the SEC ticker file that file 10-Ks.
2. **Text**: the MD&A section, located by its `Item` headers, with numeric table
   rows removed. The first 3 chunks of about 3,000 tokens are scored.
3. **LLM scores**: for each chunk the model returns JSON constrained by a schema:
   tone, outlook, demand, margins (-2 to 2), liquidity stress and uncertainty
   (0 to 2). Temperature 0, fixed seed, responses cached by content hash.
4. **Signals**
   - `llm_score`: mean of tone, outlook, demand, margins
   - `llm_score_chg`: change in `llm_score` from the company's previous filing
     (**primary signal**, declared in `config.yaml` before any results)
   - `llm_caution`: minus (liquidity stress + uncertainty)
   - `sim_yoy`: non-LLM baseline, the word-count similarity of this MD&A to the
     same filing a year earlier (Cohen, Malloy & Nguyen 2020, "Lazy Prices")
   - `emb_sim_yoy`, `emb_align_yoy`: the same year-over-year comparison using
     embeddings from a local model (`nomic-embed-text`), at whole-document and
     passage level. This tests whether comparing meaning beats comparing words.
5. **Timing**: a signal becomes tradable at the close of the first trading day
   *after* the filing date. Nothing is used earlier.
6. **Evaluation**
   - Rank IC against market-adjusted returns at 5, 21 and 63 trading days,
     computed per calendar quarter then averaged, with a t-statistic.
   - Monthly-rebalanced top-minus-bottom quintile portfolio, equal-weighted,
     10 bps one-way costs charged on turnover.

## Run it

```bash
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
copy .env.example .env        # then put your contact details in .env
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

```bash
.venv\Scripts\python -m fsig all          # universe -> download -> parse -> prices -> extract -> signals -> report
.venv\Scripts\python -m fsig leakage      # memory + identity probes, appended to the report
.venv\Scripts\python -m fsig extract --variant anon   # optional: re-score anonymised text (as long as the first scoring run)
.venv\Scripts\python -m fsig signals --variant anon
.venv\Scripts\python -m fsig robustness   # lag and cost sensitivity; raw vs anon comparison if both exist
.venv\Scripts\python -m pytest
```

Every step caches to `data/` and resumes if interrupted. Results land in
`reports/`.

Rough timings on an RTX 4090 for the default 100 companies from 2015: download
about 70 minutes (EDGAR rate limit), LLM scoring about 3 hours (0.75 s per
chunk with 4 parallel requests), everything else a few minutes.

## Known limitations

State these in any write-up; an interviewer will ask.

- **Universe selection bias.** Companies are chosen by *today's* market cap, so
  the sample is tilted to past winners and excludes delisted firms. A long-short
  portfolio partly cancels this, but not fully. The fix is point-in-time index
  membership and delisting returns (CRSP), which are not free.
- **Short clean sample.** The post-cutoff period is under three years. Expect
  wide confidence intervals; a null result there is a legitimate finding.
- **Section parsing.** MD&A is found in 93% of 10-Ks and 99% of 10-Qs. The
  parser tries the `Item` header first, then the MD&A title alone (handling
  titles repeated as page headers), then bank-style opening headings. What is
  still missed is mostly companies that put the MD&A in a separate annual-report
  exhibit (IBM, Eaton, Progressive, Wells Fargo 10-Ks), which is not downloaded.
  For a few filers found by title (Citi, GE 10-Qs) the extracted text starts at
  the table of contents, so it carries a few thousand characters of headings
  before the real section. Coverage is printed by the `parse` step.
- **Truncation.** Only the first ~9,000 tokens of each MD&A are scored.
- **Returns.** Market-adjusted (stock minus SPY), not factor-adjusted. No
  borrow costs or market impact beyond the flat cost assumption.
- **Multiple testing.** Six signals at three horizons is eighteen tests. That is
  why one primary signal is declared up front.
- **Memory probe.** One random draw of 300 pairs per period, one prompt, one
  model. Company size in the controls is today's rank, not size at the time.

## Layout

```
config.yaml          experiment definition
src/fsig/edgar.py    universe, filing index, downloads
src/fsig/sections.py HTML -> MD&A / risk factor text
src/fsig/extract.py  LLM scoring, anonymisation, response cache
src/fsig/embed.py    chunk embeddings and year-over-year embedding signals
src/fsig/signals.py  per-filing signal table, Lazy Prices baseline
src/fsig/backtest.py forward returns, rank IC, long-short portfolio
src/fsig/leakage.py  memory and identity probes, memory controls
src/fsig/robustness.py lag and cost sensitivity, raw vs anonymised comparison
src/fsig/report.py   markdown report and charts
tests/               timing and parsing tests
```
