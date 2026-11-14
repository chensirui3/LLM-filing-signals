# Filing signals

[![tests](https://github.com/chensirui3/LLM-filing-signals/actions/workflows/tests.yml/badge.svg)](https://github.com/chensirui3/LLM-filing-signals/actions/workflows/tests.yml)

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
| Memory probe | Gives the model no filing text, only two company names and a date, and asks which stock had the higher return over the following three months. Accuracy above chance before the cutoff but not after it would indicate memorised outcomes. Each pair is asked in both orders and only order-independent answers are scored. The probe is run on three Llama versions with different training cutoffs. | `leakage.run_memory_probe` |
| Memory controls | Tests whether the memory probe's accuracy is specific to the date or reflects a general preference for certain companies: how often the model picks the larger company, and how accurate its picks are when scored against other dates. | `leakage.memory_controls` |
| Identity probe | Masks the company name and years, then asks the model to name the company. Measures whether anonymisation prevents the model from identifying the filer. | `leakage.identity_probe` |
| Raw vs anonymised | Scores every filing twice, with and without the company masked, and tests whether the difference between the two scores predicts returns before the cutoff. | `robustness.variant_comparison` |

## Results

Sample: the 100 largest US filers, 4,349 scored filings from January 2015 to
November 2026. Model: Llama 3.1 8B, training cutoff December 2023. Full tables
are in [reports/report_raw.md](reports/report_raw.md) and
[reports/report_anon.md](reports/report_anon.md).

**No signal predicts returns.** None of the six signals reaches |t| = 2 at any
horizon. The pre-declared primary signal has a net long-short Sharpe ratio of
-0.20 (-0.45 before the cutoff and 0.41 after, neither significant). The Lazy
Prices similarity signals have the opposite sign to the published effect in
this large-cap sample. Varying the entry lag (0 to 5 days) or trading costs
(0 to 25 bps) does not change any conclusion. The sample can only rule out
large effects: the smallest rank IC it would reliably detect is between 0.04
and 0.07, depending on the signal.

![Long-short cumulative returns](reports/long_short_raw.png)

**The scores carry real information, but the market already has it.** The LLM
tone score has a rank IC of 0.089 (t = 4.2) with the stock's market-adjusted
return over the 63 trading days *before* the filing, and this holds after the
training cutoff (0.080, t = 2.8). Against the 63 days after entry the IC is
0.016 (t = 0.8). The model reads the reported quarter correctly; by the time
the filing can be traded, that information is in the price.

**The memory probe gives little evidence of memorised outcomes.** With no
filing text, Llama 3.1 picks the better performer of two stocks 57.2% of the
time before its cutoff (n = 2,067, z = 6.6), but also 55.0% of the time after
it (n = 669, z = 2.6); the fall of 2.2 points is not significant (z = 1.0).
Most of the accuracy survives when the date is wrong: scored against the same
pair at other dates, the model's picks are still right 54.8% of the time. The
model therefore has a stable preference for companies that tended to
outperform, and that preference kept working after the cutoff. The boundary
does not move with the cutoff either: for questions dated 2023, Llama 3 8B
(cutoff March 2023) scores 57.6% and Llama 3.1 (cutoff December 2023) scores
58.8%. Llama 3 8B shows a larger fall after its cutoff (56.9% to 52.8%,
z = 2.2), which is the only sign of date-specific memory. Llama 2 is close to
chance throughout. An earlier version of this probe, with 300 pairs per period
and a single question order, appeared to show a clear effect (59% against
47%); it did not hold at this scale.

![Memory probe accuracy by year](reports/memory_probe.png)

**Anonymisation does not conceal the company.** From text with the name,
ticker and years masked, the model names the correct company 82% of the time.

**There is no evidence that memorised outcomes inflate the backtest.** If the
scores drew on the model's knowledge of subsequent events, they would predict
returns before the cutoff. They do not: the three LLM-scored signals have
pre-cutoff net Sharpe ratios of -0.01, -0.45 and -0.14. Raw and anonymised
scores also agree closely (correlation 0.94), and the gap between them does
not predict returns before the cutoff (rank IC between -0.02 and -0.03, t
between -1.2 and -1.8). This second comparison is weak evidence on its own,
because the identity probe shows that anonymisation does not conceal the
company. One pattern remains unexplained: before the cutoff the tone score is
correlated with the return around the filing date (IC 0.070, t = 4.1), and
after the cutoff it is not (-0.022, t = -0.6, 12 quarters). The post-cutoff
sample is too short to tell whether this is sampling variation or the model
recalling how filings were received. These results apply to this model and
prompt.

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

A browser front end lets you pick any company and year in the universe. It
needs the downloaded filings and a running Ollama server:

```powershell
.venv\Scripts\python -m pip install -e .[app]
.venv\Scripts\python -m streamlit run app.py
```

The similarity threshold that separates "rewritten" from "added" (0.93) was
chosen by inspecting matched pairs for NVIDIA. It was then tested on 120
labelled paragraph pairs from 16 other companies: of the paragraphs the tool
calls rewritten, 90% are rewrites, and it finds 68% of all rewrites; the rest
are reported as added. The sample deliberately over-represents hard cases, and
the labels were assigned by an LLM assistant without a second annotator. See
[reports/riskdiff_validation.md](reports/riskdiff_validation.md); the pairs and
labels are in `validation/riskdiff_pairs.csv`, and `python -m fsig riskeval`
rebuilds the report. The LLM summaries may be inaccurate; the quoted filing
text is authoritative.

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
     with 10 bps one-way costs charged on turnover. Companies with equal
     scores are split evenly across a quintile boundary, so the result does
     not depend on how ties are ordered.
   - Rank IC against returns before and around the filing, to show whether a
     signal reflects information the market already has.
7. **Memory probe**: 400 random stock pairs per calendar year (4,775 in all),
   each asked in both orders, with the model answering by ticker. Only pairs
   on which the model names the same company in both orders are scored. A pair
   counts as pre-cutoff when its whole three-month window ends before the
   model's cutoff. The cutoffs are those in Meta's model cards: September 2022
   for Llama 2, March 2023 for Llama 3 8B and December 2023 for Llama 3.1.

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
ollama pull llama3:8b         # optional: the two older models used only by the memory probe
ollama pull llama2:7b
```

```powershell
.venv\Scripts\python -m fsig all          # universe, download, parse, prices, extract, embed, signals, report
.venv\Scripts\python -m fsig leakage      # memory probe on every installed model, identity probe; added to the reports
.venv\Scripts\python -m fsig extract --variant anon   # optional: score the anonymised text (takes as long as the first scoring run)
.venv\Scripts\python -m fsig signals --variant anon
.venv\Scripts\python -m fsig robustness   # lag and cost sensitivity; raw vs anonymised comparison if both variants exist
.venv\Scripts\python -m pytest
```

Every step caches its output in `data/` and resumes if interrupted. Results
are written to `reports/`.

To install the exact package versions behind the published results, use
`requirements-lock.txt` in place of the second command above:

```powershell
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e . --no-deps
```

### Trial run

A small run exercises every step in 10 to 15 minutes. Setting `FSIG_HOME`
keeps its data and reports in a separate folder, so it does not overwrite the
full results:

```powershell
$env:FSIG_HOME = "quickstart"
.venv\Scripts\python -m fsig all --size 12 --start 2022-01-01
$env:FSIG_HOME = $null
```

With 12 companies the statistics are not meaningful; the run only shows that
the pipeline works.

### Rebuilding the reports without a GPU

The repository includes the filing index, the per-filing signal tables and the
cache of every LLM response (`data/`), which is all the backtest needs apart
from prices. With no Ollama installation and no filings downloaded:

```powershell
.venv\Scripts\python -m fsig prices       # about a minute; downloads from Yahoo Finance
.venv\Scripts\python -m fsig report
.venv\Scripts\python -m fsig report --variant anon
.venv\Scripts\python -m fsig robustness
```

Because prices are downloaded afresh and Yahoo revises adjusted prices, the
rebuilt tables can differ from the published ones in the third decimal place.
Price data is not redistributed here. The leakage step still needs Ollama and
the downloaded filings.

## Limitations

- **Universe selection bias.** Companies are selected by *current* market
  capitalisation, so the sample is tilted towards past winners and excludes
  delisted firms. A long-short portfolio offsets this only in part. Removing
  the bias requires point-in-time index membership and delisting returns (for
  example from CRSP), which are not freely available.
- **Low power.** With 100 companies and about 45 quarters, the smallest rank
  IC the study would detect with 80% power is 0.04 to 0.07. A smaller real
  effect would not be found. The post-cutoff period is under three years
  (35 monthly observations), so estimates there are less precise still.
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
  scored, which is a median of 47% of the section in a 10-K and 67% in a 10-Q.
  The liquidity discussion usually comes late in the section, so the
  `llm_caution` signal is the most affected.
- **Price data.** Prices are Yahoo Finance adjusted closes and have not been
  checked against a second source. The losses of the two similarity
  portfolios are concentrated in April and May 2026, when several
  semiconductor stocks in their short legs rose by 60% to 110% in a month.
- **Returns.** Returns are market-adjusted (stock minus SPY), not
  factor-adjusted. Borrow costs and market impact are not modelled beyond the
  flat cost assumption.
- **Multiple testing.** Six signals at three horizons give eighteen tests. For
  this reason a single primary signal was declared before any results were
  computed.
- **Memory probe.** The probe uses one prompt and three models from one
  family. Its above-chance accuracy cannot be separated from the universe
  selection bias: companies chosen for their current size outperformed over
  the sample, so a model that prefers them is right more often than not,
  before and after its cutoff. Company size in the controls is the current
  rank, not size at the time of the question. Llama 2's fine-tuning data runs
  to July 2023, later than its pretraining cutoff.

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
app.py                  browser front end for the Risk Factors comparison
validation/             labelled paragraph pairs for the comparison threshold
tests/                  unit tests for parsing, timing, signals and probes
reports/                generated reports, charts and example comparisons
data/                   filing index, signal tables and LLM response cache (other data is not tracked)
requirements-lock.txt   exact package versions used for the published results
.github/workflows/      runs the unit tests on every push
```

## License

Released under the [MIT License](LICENSE). This project is for research and
educational purposes and is not investment advice.
