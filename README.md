# Filing signals

Do signals that a local LLM extracts from SEC filings predict stock returns, and
can an apparent result be separated from the model recalling outcomes it saw
during training?

The pipeline downloads 10-K and 10-Q filings from SEC EDGAR, scores the
Management's Discussion and Analysis (MD&A) section with a local model
(Llama 3.1 8B through Ollama), and evaluates the scores with a point-in-time
backtest. It runs on a single machine and uses only free data.

## Look-ahead bias in LLM backtests

A language model trained on text up to December 2023 has been exposed to news
about how large companies performed before that date. Given a 2019 filing, it
may rate the text favourably because it has seen how the company performed in
2020, not because of anything in the filing. A backtest over the pre-cutoff
period cannot by itself distinguish this look-ahead effect from genuine
predictive content. This project tests for the effect directly with five
checks:

| Check | What it does | Where |
|---|---|---|
| Cutoff split | Reports every result separately before and after the model's training cutoff. Only the post-cutoff results are free of possible look-ahead. | `backtest.split_stats` |
| Memory probe | Gives the model no filing text, only two company names and a date, and asks which stock had the higher return over the following three months. Accuracy above chance before the cutoff but not after it indicates memorised outcomes. Pair order is randomised, so a model that always gives the same answer scores 50%. | `leakage.memory_probe` |
| Memory controls | Tests whether the memory probe's accuracy is specific to the date or reflects a general preference for larger, better-known companies: how often the model picks the larger company, and how accurate its picks are when scored against other dates. | `leakage.memory_controls` |
| Identity probe | Masks the company name and years, then asks the model to name the company. Measures whether anonymisation prevents the model from identifying the filer. | `leakage.identity_probe` |
| Raw vs anonymised | Scores every filing twice, with and without the company masked, and tests whether the difference between the two scores predicts returns before the cutoff. | `robustness.variant_comparison` |

## Results

Sample: the 100 largest US filers, 4,349 scored filings from January 2015 to
November 2026. Model: Llama 3.1 8B, training cutoff December 2023. Full tables
are in [reports/report_raw.md](reports/report_raw.md) and
[reports/report_anon.md](reports/report_anon.md).

**No signal predicts returns.** None of the six signals reaches |t| = 2 at any
horizon. The pre-declared primary signal has a net long-short Sharpe ratio of
-0.23 (-0.50 before the cutoff and 0.49 after, neither significant). The Lazy
Prices similarity signals have the opposite sign to the published effect in
this large-cap sample. Varying the entry lag (0 to 5 days) or trading costs
(0 to 25 bps) does not change any conclusion.

**The model has memorised outcomes.** Given no filing text, it picks the better
performer of two stocks 59% of the time before its cutoff (n = 300, z = 3.2)
and 47% after. The effect is specific to the date rather than a preference for
well-known companies: the model picks the larger company only 51% of the time,
and when its picks are scored against the same pair at other dates, accuracy
falls to 51%.

**Anonymisation does not conceal the company.** From text with the name,
ticker and years masked, the model names the correct company 82% of the time.

**The memorised outcomes do not affect the filing scores.** Raw and anonymised
scores have a correlation of 0.94, and the gap between them does not predict
returns before the cutoff (rank IC between -0.02 and -0.03, t between -1.2 and
-1.8). Scoring the tone of a filing does not appear to draw on the model's
knowledge of subsequent events, at least for this model and prompt.

## Risk-factor comparison tool

The same pipeline supports a second tool, which reports what changed in a
company's Risk Factors section since the previous year's 10-K.

```powershell
.venv\Scripts\python -m fsig riskdiff --ticker NVDA --year 2023
```

Paragraphs are matched by meaning using embeddings, then classified as
unchanged, edited (shown as a word-level diff), rewritten, added or removed.
The local LLM writes a one-line summary of each added and removed risk. Each
paragraph is shown under its risk-factor heading; headings are recognised by
their bold or italic formatting in the filing, because many are full sentences
as long as a short paragraph. Examples:

- [NVIDIA, 10-K filed February 2023](reports/riskdiff_NVDA_2023.md): the new
  US export licence requirements for China and the transfer of operations out
  of China appear as added risks.
- [Boeing, 10-K filed January 2020](reports/riskdiff_BA_2020.md): the 737 MAX
  grounding and production suspension appear as added risks.
- [Meta, 10-K filed February 2022](reports/riskdiff_META_2022.md): the
  metaverse strategy and blockchain-based payment initiatives appear as added
  risks.

The similarity threshold that separates "rewritten" from "added" (0.93) was
chosen by inspecting matched pairs for one company and has not been validated
on labelled data. The LLM summaries may be inaccurate; the quoted filing text
is authoritative.

## Method

1. **Universe**: the largest US companies in the SEC ticker file that file 10-Ks.
2. **Text**: the MD&A section, located by its `Item` headers, with numeric table
   rows removed. The first 3 chunks of about 3,000 tokens each are scored.
3. **LLM scores**: for each chunk the model returns JSON constrained by a schema:
   tone, outlook, demand and margins (-2 to 2), liquidity stress and uncertainty
   (0 to 2). Temperature 0, fixed seed, responses cached by content hash.
4. **Signals**
   - `llm_score`: mean of tone, outlook, demand and margins
   - `llm_score_chg`: change in `llm_score` from the company's previous filing
     (**primary signal**, declared in `config.yaml` before any results were
     computed)
   - `llm_caution`: minus (liquidity stress + uncertainty)
   - `sim_yoy`: non-LLM baseline, the word-count similarity of the MD&A to the
     same filing a year earlier (Cohen, Malloy and Nguyen, "Lazy Prices",
     Journal of Finance, 2020)
   - `emb_sim_yoy`, `emb_align_yoy`: the same year-over-year comparison using
     embeddings from a local model (`nomic-embed-text`), at whole-document and
     passage level. These test whether semantic comparison adds information
     beyond word counts.
5. **Timing**: a signal becomes tradable at the close of the first trading day
   *after* the filing date. No signal is used before that point.
6. **Evaluation**
   - Rank IC against market-adjusted returns at 5, 21 and 63 trading days,
     computed per calendar quarter and then averaged, with a t-statistic.
   - Monthly-rebalanced top-minus-bottom quintile portfolio, equal-weighted,
     with 10 bps one-way costs charged on turnover.

## Usage

Requires Python 3.11 or later and [Ollama](https://ollama.com). The commands
below are written for Windows; on macOS or Linux, create the environment with
`python3 -m venv .venv` and replace `.venv\Scripts\python` with
`.venv/bin/python`.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
copy .env.example .env        # then set EDGAR_USER_AGENT in .env
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

```powershell
.venv\Scripts\python -m fsig all          # universe, download, parse, prices, extract, embed, signals, report
.venv\Scripts\python -m fsig leakage      # memory and identity probes, added to the report
.venv\Scripts\python -m fsig extract --variant anon   # optional: score the anonymised text (takes as long as the first scoring run)
.venv\Scripts\python -m fsig signals --variant anon
.venv\Scripts\python -m fsig robustness   # lag and cost sensitivity; raw vs anonymised comparison if both variants exist
.venv\Scripts\python -m pytest
```

Every step caches its output in `data/` and resumes if interrupted. Results
are written to `reports/`.

Approximate run times on an RTX 4090 for the default 100 companies from 2015:
download about 70 minutes (EDGAR rate limit), LLM scoring about 3 hours (0.75 s
per chunk with 4 parallel requests), embedding about 2 hours when sharing the
GPU with scoring, and a few minutes for the remaining steps.

## Limitations

- **Universe selection bias.** Companies are selected by *current* market
  capitalisation, so the sample is tilted towards past winners and excludes
  delisted firms. A long-short portfolio offsets this only in part. Removing
  the bias requires point-in-time index membership and delisting returns (for
  example from CRSP), which are not freely available.
- **Short post-cutoff sample.** The post-cutoff period is under three years
  (35 monthly observations), so estimates there are imprecise and the tests
  have low power.
- **Section parsing.** The MD&A is found in 93% of 10-Ks and 99% of 10-Qs. The
  parser tries the `Item` header first, then the MD&A title alone (handling
  titles repeated as page headers), then the opening headings used by banks.
  The remaining failures are mostly companies that place the MD&A in a separate
  annual-report exhibit (IBM, Eaton, Progressive and Wells Fargo 10-Ks), which
  the pipeline does not download. For a few filers located by title (Citi and
  GE 10-Qs) the extracted text begins at the table of contents and therefore
  includes a few thousand characters of headings before the section itself.
  Coverage is printed by the `parse` step.
- **Truncation.** Only the first 9,000 tokens (approximately) of each MD&A are
  scored.
- **Returns.** Returns are market-adjusted (stock minus SPY), not
  factor-adjusted. Borrow costs and market impact are not modelled beyond the
  flat cost assumption.
- **Multiple testing.** Six signals at three horizons give eighteen tests. For
  this reason a single primary signal was declared before any results were
  computed.
- **Memory probe.** The probe uses one random draw of 300 pairs per period,
  one prompt and one model. Company size in the controls is the current rank,
  not size at the time of the question.

## Repository layout

```
config.yaml             experiment definition
src/fsig/edgar.py       universe, filing index, downloads
src/fsig/sections.py    HTML to MD&A and Risk Factors text
src/fsig/extract.py     LLM scoring, anonymisation, response cache
src/fsig/embed.py       chunk embeddings and year-over-year embedding signals
src/fsig/signals.py     per-filing signal table, Lazy Prices baseline
src/fsig/backtest.py    forward returns, rank IC, long-short portfolio
src/fsig/leakage.py     memory and identity probes, memory controls
src/fsig/robustness.py  lag and cost sensitivity, raw vs anonymised comparison
src/fsig/riskdiff.py    year-over-year Risk Factors comparison
src/fsig/report.py      markdown reports and charts
tests/                  unit tests for parsing, timing, signals and probes
reports/                generated reports, charts and example comparisons
```
