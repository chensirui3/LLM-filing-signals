# Filing signals report (raw text, llama3.1:8b)

- Filings: 21467, with LLM score: 20993, companies: 500
- Sample: 2015-01-07 to 2026-11-12
- Model training cutoff: 2023-12-31
- Primary signal (declared in the configuration before any results were computed): `llm_score_chg`
- Entry lag: 1 trading day; one-way cost: 10 bps; quantiles: 5
- Share of each MD&A read by the LLM (first 3 chunks of 12,000 characters), median: 47% for 10-K, 68% for 10-Q

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
| llm_score | 5 | 0.003 | 0.219 | 48 | 0.012 | 0.032 |
| llm_score | 21 | -0.009 | -0.604 | 47 | 0.015 | 0.042 |
| llm_score | 63 | 0.005 | 0.356 | 47 | 0.015 | 0.043 |
| llm_score_chg | 5 | 0.009 | 1.095 | 47 | 0.008 | 0.023 |
| llm_score_chg | 21 | 0.003 | 0.325 | 46 | 0.009 | 0.025 |
| llm_score_chg | 63 | 0.003 | 0.318 | 46 | 0.008 | 0.024 |
| llm_caution | 5 | -0.006 | -0.684 | 48 | 0.009 | 0.024 |
| llm_caution | 21 | -0.012 | -1.414 | 47 | 0.008 | 0.023 |
| llm_caution | 63 | -0.015 | -1.884 | 47 | 0.008 | 0.022 |
| sim_yoy | 5 | 0.008 | 1.112 | 44 | 0.007 | 0.021 |
| sim_yoy | 21 | -0.004 | -0.445 | 43 | 0.009 | 0.025 |
| sim_yoy | 63 | -0.001 | -0.077 | 43 | 0.011 | 0.032 |
| emb_sim_yoy | 5 | 0.005 | 0.652 | 44 | 0.008 | 0.021 |
| emb_sim_yoy | 21 | -0.009 | -0.894 | 43 | 0.010 | 0.029 |
| emb_sim_yoy | 63 | -0.006 | -0.460 | 43 | 0.012 | 0.034 |
| emb_align_yoy | 5 | 0.007 | 0.768 | 44 | 0.009 | 0.024 |
| emb_align_yoy | 21 | 0.002 | 0.268 | 43 | 0.008 | 0.023 |
| emb_align_yoy | 63 | 0.005 | 0.395 | 43 | 0.012 | 0.034 |

## Timing: returns before, around and after the filing

Rank IC of each signal against the market-adjusted return in three windows, shown as IC (t-statistic). `before_filing` is the 63 trading days that end the day before the filing; `filing_window` runs from the close before the filing to the entry close; `after_entry` is the 63 trading days after entry, the only window the strategy can earn. A signal that reflects information the market already has correlates with the first two windows and not the third.

| signal | period | before_filing | filing_window | after_entry |
|---|---|---|---|---|
| llm_score | full | 0.083 (5.2) | 0.036 (4.3) | 0.005 (0.4) |
| llm_score | pre_cutoff | 0.089 (4.3) | 0.040 (4.0) | 0.002 (0.1) |
| llm_score | post_cutoff | 0.066 (3.6) | 0.025 (1.7) | 0.018 (0.7) |
| llm_score_chg | full | 0.065 (5.7) | 0.044 (6.5) | 0.003 (0.3) |
| llm_score_chg | pre_cutoff | 0.072 (4.9) | 0.049 (5.9) | 0.004 (0.4) |
| llm_score_chg | post_cutoff | 0.043 (4.8) | 0.030 (3.0) | -0.002 (-0.2) |
| llm_caution | full | 0.020 (2.1) | 0.003 (0.5) | -0.015 (-1.9) |
| llm_caution | pre_cutoff | 0.028 (2.5) | 0.011 (1.6) | -0.006 (-0.7) |
| llm_caution | post_cutoff | -0.002 (-0.1) | -0.019 (-1.5) | -0.042 (-4.0) |
| sim_yoy | full | 0.002 (0.2) | 0.008 (0.9) | -0.001 (-0.1) |
| sim_yoy | pre_cutoff | -0.001 (-0.1) | 0.006 (0.6) | -0.008 (-0.6) |
| sim_yoy | post_cutoff | 0.009 (0.5) | 0.012 (0.7) | 0.019 (0.7) |
| emb_sim_yoy | full | 0.003 (0.2) | 0.002 (0.3) | -0.006 (-0.5) |
| emb_sim_yoy | pre_cutoff | -0.003 (-0.2) | 0.005 (0.5) | -0.012 (-0.9) |
| emb_sim_yoy | post_cutoff | 0.017 (0.8) | -0.004 (-0.2) | 0.014 (0.5) |
| emb_align_yoy | full | 0.018 (1.7) | 0.005 (0.6) | 0.005 (0.4) |
| emb_align_yoy | pre_cutoff | 0.022 (1.8) | 0.004 (0.4) | 0.006 (0.5) |
| emb_align_yoy | post_cutoff | 0.006 (0.3) | 0.007 (0.5) | 0.000 (0.0) |

## Long-short portfolios (monthly, net of costs)

Top-minus-bottom quantile portfolios. Companies with equal scores are split evenly across a quantile boundary.

### `llm_score`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 142 | 0.000 | 0.089 | 0.004 | 0.015 | 0.549 | -0.287 |
| pre_cutoff | 107 | -0.019 | 0.089 | -0.209 | -0.624 | 0.542 | -0.287 |
| post_cutoff | 35 | 0.058 | 0.089 | 0.653 | 1.116 | 0.571 | -0.101 |

Average monthly turnover: 0.80 (2.0 = both legs fully replaced).

### `llm_score_chg`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 139 | -0.021 | 0.053 | -0.390 | -1.328 | 0.460 | -0.292 |
| pre_cutoff | 104 | -0.030 | 0.055 | -0.543 | -1.598 | 0.452 | -0.272 |
| post_cutoff | 35 | 0.006 | 0.047 | 0.131 | 0.223 | 0.486 | -0.079 |

Average monthly turnover: 1.39 (2.0 = both legs fully replaced).

### `llm_caution`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 142 | -0.043 | 0.052 | -0.830 | -2.857 | 0.437 | -0.394 |
| pre_cutoff | 107 | -0.028 | 0.055 | -0.508 | -1.518 | 0.505 | -0.232 |
| post_cutoff | 35 | -0.091 | 0.042 | -2.170 | -3.706 | 0.229 | -0.236 |

Average monthly turnover: 0.72 (2.0 = both legs fully replaced).

### `sim_yoy`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 130 | -0.039 | 0.090 | -0.437 | -1.438 | 0.477 | -0.480 |
| pre_cutoff | 95 | -0.037 | 0.079 | -0.472 | -1.329 | 0.474 | -0.333 |
| post_cutoff | 35 | -0.044 | 0.115 | -0.385 | -0.658 | 0.486 | -0.280 |

Average monthly turnover: 0.75 (2.0 = both legs fully replaced).

### `emb_sim_yoy`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 130 | -0.047 | 0.085 | -0.545 | -1.794 | 0.446 | -0.489 |
| pre_cutoff | 95 | -0.046 | 0.074 | -0.626 | -1.760 | 0.432 | -0.388 |
| post_cutoff | 35 | -0.047 | 0.112 | -0.422 | -0.721 | 0.486 | -0.253 |

Average monthly turnover: 0.76 (2.0 = both legs fully replaced).

### `emb_align_yoy`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 130 | -0.032 | 0.072 | -0.443 | -1.458 | 0.477 | -0.385 |
| pre_cutoff | 95 | -0.018 | 0.069 | -0.261 | -0.735 | 0.484 | -0.224 |
| post_cutoff | 35 | -0.070 | 0.080 | -0.871 | -1.488 | 0.457 | -0.241 |

Average monthly turnover: 0.74 (2.0 = both legs fully replaced).

![Long-short cumulative returns](long_short_raw.png)

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

## Robustness: industry and factor adjustment

Net Sharpe ratio and t-statistic of each long-short portfolio as reported, and when each company's signal is measured relative to the mean of its industry (two-digit SIC group, 53 groups; industries with fewer than 3 companies at a rebalance are left out). The within-industry portfolio takes no position on industries.

| signal | period | sharpe (as reported) | t (as reported) | sharpe (within industry) | t (within industry) |
|---|---|---|---|---|---|
| llm_score | full | 0.004 | 0.015 | -0.212 | -0.730 |
| llm_score | post_cutoff | 0.653 | 1.116 | 0.515 | 0.879 |
| llm_score_chg | full | -0.390 | -1.328 | -0.576 | -1.961 |
| llm_score_chg | post_cutoff | 0.131 | 0.223 | -0.487 | -0.831 |
| llm_caution | full | -0.830 | -2.857 | -0.881 | -3.031 |
| llm_caution | post_cutoff | -2.170 | -3.706 | -2.003 | -3.421 |
| sim_yoy | full | -0.437 | -1.438 | -0.396 | -1.299 |
| sim_yoy | post_cutoff | -0.385 | -0.658 | -0.027 | -0.046 |
| emb_sim_yoy | full | -0.545 | -1.794 | -0.474 | -1.553 |
| emb_sim_yoy | post_cutoff | -0.422 | -0.721 | 0.007 | 0.011 |
| emb_align_yoy | full | -0.443 | -1.458 | -0.364 | -1.193 |
| emb_align_yoy | post_cutoff | -0.871 | -1.488 | -0.279 | -0.477 |

Regression of each portfolio's monthly net return on the Fama-French five factors and momentum (Kenneth French's data library). `alpha_ann` is the annualised return the factors do not explain and `t_alpha` its t-statistic; the remaining columns are the factor loadings. The post-cutoff regressions have few months for seven coefficients and are imprecise.

| signal | period | portfolio | months | alpha_ann | t_alpha | MKT | SMB | HML | RMW | CMA | MOM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_score | full | as reported | 140 | -0.008 | -0.365 | 0.050 | -0.075 | -0.163 | -0.062 | -0.059 | 0.273 |
| llm_score | full | within industry | 140 | -0.010 | -0.624 | -0.038 | 0.003 | -0.116 | -0.063 | 0.004 | 0.178 |
| llm_score | post_cutoff | as reported | 33 | 0.083 | 1.657 | 0.015 | 0.120 | -0.406 | -0.055 | -0.001 | 0.197 |
| llm_score | post_cutoff | within industry | 33 | 0.048 | 1.358 | -0.086 | 0.071 | -0.251 | -0.076 | 0.174 | 0.188 |
| llm_score_chg | full | as reported | 137 | -0.023 | -1.445 | -0.005 | -0.022 | -0.029 | -0.010 | 0.108 | 0.081 |
| llm_score_chg | full | within industry | 137 | -0.030 | -1.824 | -0.044 | 0.046 | -0.006 | 0.120 | 0.026 | 0.059 |
| llm_score_chg | post_cutoff | as reported | 33 | 0.004 | 0.131 | 0.105 | -0.024 | -0.141 | 0.199 | 0.027 | 0.018 |
| llm_score_chg | post_cutoff | within industry | 33 | -0.024 | -0.649 | 0.073 | 0.016 | -0.080 | 0.217 | 0.064 | 0.044 |
| llm_caution | full | as reported | 140 | -0.041 | -2.756 | -0.046 | -0.049 | -0.093 | 0.035 | 0.127 | 0.057 |
| llm_caution | full | within industry | 140 | -0.036 | -2.593 | -0.036 | 0.035 | -0.106 | 0.027 | 0.147 | 0.049 |
| llm_caution | post_cutoff | as reported | 33 | -0.110 | -3.388 | 0.067 | -0.084 | 0.045 | 0.054 | -0.106 | -0.019 |
| llm_caution | post_cutoff | within industry | 33 | -0.078 | -2.192 | -0.052 | 0.016 | -0.117 | 0.081 | 0.056 | 0.089 |
| sim_yoy | full | as reported | 128 | -0.031 | -1.204 | -0.151 | -0.015 | 0.199 | 0.246 | -0.134 | -0.031 |
| sim_yoy | full | within industry | 127 | -0.021 | -1.078 | -0.064 | -0.066 | 0.009 | 0.124 | 0.001 | 0.050 |
| sim_yoy | post_cutoff | as reported | 33 | 0.004 | 0.066 | -0.290 | -0.002 | 0.108 | 0.251 | -0.222 | -0.280 |
| sim_yoy | post_cutoff | within industry | 33 | 0.025 | 0.564 | -0.122 | -0.003 | -0.083 | 0.209 | 0.100 | 0.068 |
| emb_sim_yoy | full | as reported | 128 | -0.033 | -1.350 | -0.162 | -0.013 | 0.232 | 0.160 | -0.204 | -0.077 |
| emb_sim_yoy | full | within industry | 127 | -0.023 | -1.320 | -0.061 | -0.021 | 0.009 | 0.055 | -0.124 | 0.037 |
| emb_sim_yoy | post_cutoff | as reported | 33 | -0.003 | -0.048 | -0.199 | 0.099 | 0.147 | 0.208 | -0.421 | -0.327 |
| emb_sim_yoy | post_cutoff | within industry | 33 | -0.015 | -0.320 | 0.045 | -0.043 | 0.033 | 0.062 | -0.095 | -0.022 |
| emb_align_yoy | full | as reported | 128 | -0.032 | -1.514 | -0.072 | 0.061 | 0.010 | 0.264 | -0.282 | 0.113 |
| emb_align_yoy | full | within industry | 127 | -0.018 | -1.058 | -0.065 | 0.029 | -0.038 | 0.172 | -0.078 | 0.076 |
| emb_align_yoy | post_cutoff | as reported | 33 | -0.063 | -1.051 | -0.025 | 0.048 | -0.077 | 0.223 | -0.240 | -0.071 |
| emb_align_yoy | post_cutoff | within industry | 33 | 0.002 | 0.048 | -0.032 | 0.087 | -0.099 | 0.121 | 0.018 | -0.093 |

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
| llm_score | 0.016 | 0.004 | -0.010 | -0.046 |
| llm_score_chg | -0.418 | -0.390 | -0.460 | -0.418 |
| llm_caution | -0.804 | -0.830 | -0.851 | -0.802 |
| sim_yoy | -0.458 | -0.437 | -0.433 | -0.533 |
| emb_sim_yoy | -0.528 | -0.545 | -0.549 | -0.620 |
| emb_align_yoy | -0.467 | -0.443 | -0.456 | -0.547 |

By one-way cost (entry lag 1):

| signal | 0_bps | 10_bps | 25_bps |
|---|---|---|---|
| llm_score | 0.112 | 0.004 | -0.157 |
| llm_score_chg | -0.076 | -0.390 | -0.854 |
| llm_caution | -0.664 | -0.830 | -1.077 |
| sim_yoy | -0.338 | -0.437 | -0.584 |
| emb_sim_yoy | -0.439 | -0.545 | -0.703 |
| emb_align_yoy | -0.320 | -0.443 | -0.627 |
