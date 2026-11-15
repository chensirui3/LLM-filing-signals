# Filing signals report (anon text, llama3.1:8b)

- Filings: 4466, with LLM score: 4349, companies: 100
- Sample: 2015-01-09 to 2026-11-12
- Model training cutoff: 2023-12-31
- Primary signal (declared in the configuration before any results were computed): `llm_score_chg`
- Entry lag: 1 trading day; one-way cost: 10 bps; quantiles: 5
- Share of each MD&A read by the LLM (first 3 chunks of 12,000 characters), median: 47% for 10-K, 67% for 10-Q

## Signals

- `llm_score`: LLM: mean of tone, outlook, demand, margins (level)
- `llm_score_chg`: LLM: change in llm_score vs the company's previous filing
- `llm_caution`: LLM: minus (liquidity stress + uncertainty)
- `sim_yoy`: Baseline: word-count similarity to the same filing a year earlier (Lazy Prices)
- `emb_sim_yoy`: Embeddings: whole-document similarity to the same filing a year earlier
- `emb_align_yoy`: Embeddings: passage-level match with the same filing a year earlier

## Rank IC against market-adjusted forward returns

Spearman correlation between the signal and the return, computed within each calendar quarter and then averaged across quarters. A |t| above about 2 is conventionally treated as significant. `se` is the standard error of the mean IC and `min_detectable_ic` (2.8 standard errors) is the smallest true IC this sample would detect with 80% power at the 5% level; a null result rules out larger effects, not smaller ones.

| signal | horizon | mean_ic | t_stat | quarters | se | min_detectable_ic |
|---|---|---|---|---|---|---|
| llm_score | 5 | -0.005 | -0.261 | 48 | 0.018 | 0.051 |
| llm_score | 21 | -0.007 | -0.364 | 47 | 0.020 | 0.057 |
| llm_score | 63 | 0.019 | 0.896 | 47 | 0.021 | 0.058 |
| llm_score_chg | 5 | 0.004 | 0.300 | 47 | 0.014 | 0.040 |
| llm_score_chg | 21 | -0.011 | -0.585 | 46 | 0.018 | 0.051 |
| llm_score_chg | 63 | 0.011 | 0.636 | 46 | 0.017 | 0.048 |
| llm_caution | 5 | 0.001 | 0.055 | 48 | 0.017 | 0.047 |
| llm_caution | 21 | 0.000 | 0.025 | 47 | 0.015 | 0.041 |
| llm_caution | 63 | -0.019 | -1.245 | 47 | 0.015 | 0.042 |
| sim_yoy | 5 | -0.006 | -0.354 | 44 | 0.018 | 0.049 |
| sim_yoy | 21 | -0.004 | -0.193 | 43 | 0.020 | 0.056 |
| sim_yoy | 63 | -0.003 | -0.133 | 43 | 0.025 | 0.069 |
| emb_sim_yoy | 5 | -0.008 | -0.489 | 44 | 0.017 | 0.049 |
| emb_sim_yoy | 21 | -0.026 | -1.167 | 43 | 0.022 | 0.062 |
| emb_sim_yoy | 63 | -0.031 | -1.337 | 43 | 0.023 | 0.065 |
| emb_align_yoy | 5 | -0.012 | -0.680 | 44 | 0.017 | 0.049 |
| emb_align_yoy | 21 | -0.001 | -0.070 | 43 | 0.018 | 0.051 |
| emb_align_yoy | 63 | 0.006 | 0.321 | 43 | 0.019 | 0.054 |

## Timing: returns before, around and after the filing

Rank IC of each signal against the market-adjusted return in three windows, shown as IC (t-statistic). `before_filing` is the 63 trading days that end the day before the filing; `filing_window` runs from the close before the filing to the entry close; `after_entry` is the 63 trading days after entry, the only window the strategy can earn. A signal that reflects information the market already has correlates with the first two windows and not the third.

| signal | period | before_filing | filing_window | after_entry |
|---|---|---|---|---|
| llm_score | full | 0.094 (4.3) | 0.050 (3.0) | 0.019 (0.9) |
| llm_score | pre_cutoff | 0.104 (3.8) | 0.075 (4.5) | 0.012 (0.5) |
| llm_score | post_cutoff | 0.063 (2.5) | -0.023 (-0.6) | 0.039 (0.9) |
| llm_score_chg | full | 0.067 (3.5) | 0.063 (4.0) | 0.011 (0.6) |
| llm_score_chg | pre_cutoff | 0.089 (3.8) | 0.077 (4.3) | 0.021 (1.0) |
| llm_score_chg | post_cutoff | 0.003 (0.1) | 0.021 (0.7) | -0.021 (-0.8) |
| llm_caution | full | 0.026 (1.8) | 0.012 (0.8) | -0.019 (-1.2) |
| llm_caution | pre_cutoff | 0.032 (1.8) | 0.028 (1.8) | -0.019 (-1.1) |
| llm_caution | post_cutoff | 0.006 (0.3) | -0.036 (-1.2) | -0.018 (-0.6) |
| sim_yoy | full | -0.003 (-0.1) | -0.006 (-0.4) | -0.003 (-0.1) |
| sim_yoy | pre_cutoff | -0.016 (-0.8) | -0.012 (-0.6) | -0.026 (-1.0) |
| sim_yoy | post_cutoff | 0.033 (1.1) | 0.009 (0.3) | 0.063 (1.1) |
| emb_sim_yoy | full | -0.011 (-0.6) | -0.003 (-0.2) | -0.031 (-1.3) |
| emb_sim_yoy | pre_cutoff | -0.032 (-1.6) | -0.002 (-0.1) | -0.056 (-2.2) |
| emb_sim_yoy | post_cutoff | 0.045 (1.1) | -0.005 (-0.2) | 0.042 (0.9) |
| emb_align_yoy | full | 0.030 (1.9) | 0.012 (0.6) | 0.006 (0.3) |
| emb_align_yoy | pre_cutoff | 0.032 (1.7) | 0.000 (0.0) | -0.009 (-0.4) |
| emb_align_yoy | post_cutoff | 0.023 (0.7) | 0.042 (1.4) | 0.049 (1.3) |

## Long-short portfolios (monthly, net of costs)

Top-minus-bottom quantile portfolios. Companies with equal scores are split evenly across a quantile boundary.

### `llm_score`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 142 | 0.032 | 0.113 | 0.285 | 0.979 | 0.535 | -0.267 |
| pre_cutoff | 107 | 0.000 | 0.109 | 0.004 | 0.011 | 0.514 | -0.267 |
| post_cutoff | 35 | 0.129 | 0.123 | 1.052 | 1.797 | 0.600 | -0.061 |

Average monthly turnover: 0.87 (2.0 = both legs fully replaced).

### `llm_score_chg`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 139 | -0.017 | 0.096 | -0.179 | -0.608 | 0.468 | -0.404 |
| pre_cutoff | 104 | -0.034 | 0.098 | -0.347 | -1.020 | 0.462 | -0.404 |
| post_cutoff | 35 | 0.032 | 0.091 | 0.356 | 0.608 | 0.486 | -0.160 |

Average monthly turnover: 1.49 (2.0 = both legs fully replaced).

### `llm_caution`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 142 | -0.036 | 0.075 | -0.483 | -1.662 | 0.415 | -0.444 |
| pre_cutoff | 107 | -0.035 | 0.066 | -0.533 | -1.591 | 0.430 | -0.327 |
| post_cutoff | 35 | -0.039 | 0.098 | -0.397 | -0.679 | 0.371 | -0.174 |

Average monthly turnover: 0.76 (2.0 = both legs fully replaced).

### `sim_yoy`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 130 | -0.048 | 0.146 | -0.329 | -1.082 | 0.492 | -0.585 |
| pre_cutoff | 95 | -0.043 | 0.107 | -0.403 | -1.133 | 0.484 | -0.419 |
| post_cutoff | 35 | -0.062 | 0.222 | -0.278 | -0.474 | 0.514 | -0.465 |

Average monthly turnover: 0.77 (2.0 = both legs fully replaced).

### `emb_sim_yoy`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 130 | -0.075 | 0.137 | -0.545 | -1.794 | 0.431 | -0.657 |
| pre_cutoff | 95 | -0.063 | 0.114 | -0.559 | -1.573 | 0.400 | -0.507 |
| post_cutoff | 35 | -0.105 | 0.188 | -0.557 | -0.952 | 0.514 | -0.434 |

Average monthly turnover: 0.78 (2.0 = both legs fully replaced).

### `emb_align_yoy`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 130 | -0.008 | 0.114 | -0.070 | -0.231 | 0.531 | -0.375 |
| pre_cutoff | 95 | 0.001 | 0.091 | 0.008 | 0.024 | 0.526 | -0.166 |
| post_cutoff | 35 | -0.032 | 0.164 | -0.195 | -0.333 | 0.543 | -0.375 |

Average monthly turnover: 0.79 (2.0 = both legs fully replaced).

![Long-short cumulative returns](long_short_anon.png)

## Leakage: memory probe (no filing text given)

Each model is given two company names and a date, with no filing text, and asked which stock had the higher return over the following three months. The same 4,799 pairs, an equal number from every calendar year, are put to every model in both orders. `order_independent` is the share of pairs on which the model named the same company both times; `accuracy` is measured on those pairs (`n`), so a preference for one position cannot affect it. `accuracy_all_pairs` counts a pair with conflicting answers as half right. A pair is pre-cutoff when its whole three-month window ends before the model's training cutoff and post-cutoff when the question date is after it.

| model | cutoff | period | pairs | order_independent | n | accuracy | z_vs_50pct | accuracy_all_pairs |
|---|---|---|---|---|---|---|---|---|
| llama2:7b | 2022-09-30 | pre_cutoff | 3000 | 0.626 | 1878 | 0.482 | -1.523 | 0.489 |
| llama2:7b | 2022-09-30 | post_cutoff | 1692 | 0.589 | 996 | 0.489 | -0.697 | 0.493 |
| llama3:8b | 2023-03-31 | pre_cutoff | 3167 | 0.696 | 2205 | 0.512 | 1.129 | 0.508 |
| llama3:8b | 2023-03-31 | post_cutoff | 1489 | 0.763 | 1136 | 0.533 | 2.196 | 0.525 |
| llama3.1:8b | 2023-12-31 | pre_cutoff | 3514 | 0.527 | 1852 | 0.515 | 1.255 | 0.508 |
| llama3.1:8b | 2023-12-31 | post_cutoff | 1200 | 0.519 | 623 | 0.483 | -0.841 | 0.491 |

Memorised outcomes would show as accuracy above 50% before a model's cutoff that falls to 50% after it, with the fall occurring at a different date for each model.

![Memory probe accuracy by year](memory_probe.png)

Accuracy on order-independent answers by year of the question, with the number of such answers in brackets:

| year | llama2:7b | llama3:8b | llama3.1:8b |
|---|---|---|---|
| 2015 | 0.435 (253) | 0.460 (278) | 0.409 (220) |
| 2016 | 0.457 (267) | 0.495 (289) | 0.532 (231) |
| 2017 | 0.536 (233) | 0.610 (277) | 0.614 (202) |
| 2018 | 0.504 (260) | 0.560 (252) | 0.553 (199) |
| 2019 | 0.496 (248) | 0.517 (263) | 0.554 (204) |
| 2020 | 0.456 (241) | 0.531 (303) | 0.559 (195) |
| 2021 | 0.508 (250) | 0.462 (299) | 0.439 (237) |
| 2022 | 0.487 (232) | 0.479 (267) | 0.477 (195) |
| 2023 | 0.510 (243) | 0.542 (310) | 0.543 (208) |
| 2024 | 0.425 (240) | 0.503 (308) | 0.464 (224) |
| 2025 | 0.500 (224) | 0.557 (307) | 0.526 (190) |
| 2026 | 0.494 (241) | 0.535 (299) | 0.464 (209) |

The controls separate date-specific memory from a general preference for certain companies. `picks_larger` is the share of answers that chose the company that is larger today; `larger_wins` is the accuracy of always choosing the larger company; `acc_when_smaller` is the accuracy when the model chose the smaller company; `placebo_accuracy` scores the model's choice against the same pair's outcome at every other month-end in the period, and so shows how much accuracy remains when the date is wrong.

| model | period | accuracy | placebo_accuracy | picks_larger | larger_wins | n_picked_smaller | acc_when_smaller | z_when_smaller |
|---|---|---|---|---|---|---|---|---|
| llama2:7b | pre_cutoff | 0.482 | 0.494 | 0.549 | 0.505 | 847 | 0.484 | -0.928 |
| llama2:7b | post_cutoff | 0.489 | 0.496 | 0.543 | 0.518 | 455 | 0.477 | -0.984 |
| llama3:8b | pre_cutoff | 0.512 | 0.524 | 0.500 | 0.505 | 1102 | 0.494 | -0.422 |
| llama3:8b | post_cutoff | 0.533 | 0.498 | 0.504 | 0.514 | 564 | 0.509 | 0.421 |
| llama3.1:8b | pre_cutoff | 0.515 | 0.529 | 0.488 | 0.510 | 948 | 0.498 | -0.130 |
| llama3.1:8b | post_cutoff | 0.483 | 0.496 | 0.475 | 0.510 | 327 | 0.474 | -0.940 |

## Leakage: identity probe on anonymised text (n=293)

Share of anonymised MD&A excerpts for which the model named the correct company and fiscal year.

| metric | share |
|---|---|
| company_correct | 0.734 |
| year_correct | 0.212 |
| year_within_1 | 0.386 |

## Leakage: raw versus anonymised scores

Each filing was scored twice: as filed, and with the company name, ticker and years masked. `gap` is the mean of the raw score minus the anonymised score (`llm_score` ranges from -2 to 2).

| signal | n | correlation | gap_pre | t_pre | gap_post | t_post |
|---|---|---|---|---|---|---|
| llm_score | 4349 | 0.944 | -0.009 | -2.490 | 0.000 | 0.029 |
| llm_score_chg | 4249 | 0.858 | 0.000 | -0.074 | 0.005 | 0.618 |
| llm_caution | 4349 | 0.890 | 0.001 | 0.234 | 0.001 | 0.117 |

Rank IC of the gap in `llm_score` against market-adjusted forward returns. If the model had shifted its scores towards outcomes it had memorised, the IC would be positive before the cutoff and close to zero after it.

| period | horizon | mean_ic | t_stat | quarters |
|---|---|---|---|---|
| pre_cutoff | 5 | -0.027 | -1.822 | 36 |
| pre_cutoff | 21 | -0.032 | -1.582 | 36 |
| pre_cutoff | 63 | -0.018 | -1.179 | 36 |
| post_cutoff | 5 | 0.012 | 0.399 | 12 |
| post_cutoff | 21 | 0.037 | 1.274 | 11 |
| post_cutoff | 63 | 0.033 | 1.065 | 11 |

## Robustness: scoring the whole MD&A

The signals read only the first 3 chunks of each MD&A. For 3,559 filings with a longer MD&A, every chunk was scored as well. The table compares the signal built from the first chunks with the same signal built from the whole section, on those filings: their correlation, and the rank IC of each against the 63-day return before the filing and the return after entry. In this table `llm_score_chg` is the change from the previous filing within this set of filings.

| signal | text_scored | correlation_with_whole | ic_before_filing | t_before | ic_after_entry | t_after |
|---|---|---|---|---|---|---|
| llm_score | first chunks | 0.831 | 0.084 | 3.936 | 0.019 | 0.906 |
| llm_score | whole section | 1 | 0.073 | 3.079 | 0.013 | 0.561 |
| llm_score_chg | first chunks | 0.744 | 0.052 | 2.506 | 0.015 | 0.860 |
| llm_score_chg | whole section | 1 | 0.057 | 2.579 | -0.006 | -0.395 |
| llm_caution | first chunks | 0.650 | 0.021 | 1.329 | -0.014 | -1.028 |
| llm_caution | whole section | 1 | -0.005 | -0.319 | -0.012 | -0.796 |

## Robustness: entry lag and trading cost

Full-sample net Sharpe ratio of each long-short portfolio. Lag 0 trades at the close of the filing day, which is not achievable for filings released after the close, so it is an upper bound.

By entry lag (trading days after the filing date, 10 bps cost):

| signal | lag_0 | lag_1 | lag_2 | lag_5 |
|---|---|---|---|---|
| llm_score | 0.290 | 0.285 | 0.327 | 0.376 |
| llm_score_chg | -0.057 | -0.179 | -0.171 | -0.040 |
| llm_caution | -0.451 | -0.483 | -0.623 | -0.630 |
| sim_yoy | -0.362 | -0.329 | -0.389 | -0.378 |
| emb_sim_yoy | -0.559 | -0.545 | -0.601 | -0.616 |
| emb_align_yoy | -0.077 | -0.070 | -0.100 | -0.102 |

By one-way cost (entry lag 1):

| signal | 0_bps | 10_bps | 25_bps |
|---|---|---|---|
| llm_score | 0.377 | 0.285 | 0.146 |
| llm_score_chg | 0.007 | -0.179 | -0.455 |
| llm_caution | -0.360 | -0.483 | -0.668 |
| sim_yoy | -0.267 | -0.329 | -0.421 |
| emb_sim_yoy | -0.478 | -0.545 | -0.645 |
| emb_align_yoy | 0.013 | -0.070 | -0.194 |
