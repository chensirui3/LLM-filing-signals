# Filing signals report (anon text, llama3.1:8b)

- Filings: 4466, with LLM score: 4349, companies: 100
- Sample: 2015-01-09 to 2026-11-12
- Model training cutoff: 2023-12-31
- Primary signal (declared in the configuration before any results were computed): `llm_score_chg`
- Entry lag: 1 trading day; one-way cost: 10 bps; quantiles: 5

## Signals

- `llm_score`: LLM: mean of tone, outlook, demand, margins (level)
- `llm_score_chg`: LLM: change in llm_score vs the company's previous filing
- `llm_caution`: LLM: minus (liquidity stress + uncertainty)
- `sim_yoy`: Baseline: word-count similarity to the same filing a year earlier (Lazy Prices)
- `emb_sim_yoy`: Embeddings: whole-document similarity to the same filing a year earlier
- `emb_align_yoy`: Embeddings: passage-level match with the same filing a year earlier

## Rank IC against market-adjusted forward returns

Spearman correlation between the signal and the return, computed within each calendar quarter and then averaged across quarters. A |t| above about 2 is conventionally treated as significant.

| signal | horizon | mean_ic | t_stat | quarters |
|---|---|---|---|---|
| llm_score | 5 | -0.005 | -0.261 | 48 |
| llm_score | 21 | -0.007 | -0.364 | 47 |
| llm_score | 63 | 0.019 | 0.895 | 47 |
| llm_score_chg | 5 | 0.004 | 0.303 | 47 |
| llm_score_chg | 21 | -0.011 | -0.585 | 46 |
| llm_score_chg | 63 | 0.011 | 0.635 | 46 |
| llm_caution | 5 | 0.001 | 0.056 | 48 |
| llm_caution | 21 | 0.000 | 0.025 | 47 |
| llm_caution | 63 | -0.019 | -1.247 | 47 |
| sim_yoy | 5 | -0.006 | -0.361 | 44 |
| sim_yoy | 21 | -0.004 | -0.193 | 43 |
| sim_yoy | 63 | -0.003 | -0.132 | 43 |
| emb_sim_yoy | 5 | -0.009 | -0.499 | 44 |
| emb_sim_yoy | 21 | -0.026 | -1.167 | 43 |
| emb_sim_yoy | 63 | -0.031 | -1.337 | 43 |
| emb_align_yoy | 5 | -0.012 | -0.689 | 44 |
| emb_align_yoy | 21 | -0.001 | -0.070 | 43 |
| emb_align_yoy | 63 | 0.006 | 0.322 | 43 |

## Long-short portfolios (monthly, net of costs)

### `llm_score`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 142 | 0.035 | 0.114 | 0.308 | 1.060 | 0.528 | -0.261 |
| pre_cutoff | 107 | 0.003 | 0.111 | 0.030 | 0.088 | 0.505 | -0.261 |
| post_cutoff | 35 | 0.133 | 0.122 | 1.089 | 1.860 | 0.600 | -0.064 |

Average monthly turnover: 0.88 (2.0 = both legs fully replaced).

### `llm_score_chg`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 139 | -0.017 | 0.095 | -0.179 | -0.609 | 0.482 | -0.412 |
| pre_cutoff | 104 | -0.036 | 0.097 | -0.374 | -1.100 | 0.462 | -0.412 |
| post_cutoff | 35 | 0.040 | 0.089 | 0.454 | 0.776 | 0.543 | -0.143 |

Average monthly turnover: 1.49 (2.0 = both legs fully replaced).

### `llm_caution`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 142 | -0.037 | 0.103 | -0.363 | -1.249 | 0.458 | -0.519 |
| pre_cutoff | 107 | -0.053 | 0.089 | -0.589 | -1.758 | 0.486 | -0.431 |
| post_cutoff | 35 | 0.009 | 0.137 | 0.068 | 0.115 | 0.371 | -0.137 |

Average monthly turnover: 0.78 (2.0 = both legs fully replaced).

### `sim_yoy`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 130 | -0.052 | 0.146 | -0.355 | -1.170 | 0.492 | -0.598 |
| pre_cutoff | 95 | -0.047 | 0.107 | -0.437 | -1.228 | 0.484 | -0.436 |
| post_cutoff | 35 | -0.066 | 0.222 | -0.297 | -0.506 | 0.514 | -0.464 |

Average monthly turnover: 0.77 (2.0 = both legs fully replaced).

### `emb_sim_yoy`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 130 | -0.080 | 0.138 | -0.576 | -1.896 | 0.454 | -0.671 |
| pre_cutoff | 95 | -0.065 | 0.114 | -0.575 | -1.619 | 0.432 | -0.516 |
| post_cutoff | 35 | -0.118 | 0.191 | -0.618 | -1.056 | 0.514 | -0.452 |

Average monthly turnover: 0.78 (2.0 = both legs fully replaced).

### `emb_align_yoy`

| period | months | ann_return | ann_vol | sharpe | t_stat | hit_rate | max_drawdown |
|---|---|---|---|---|---|---|---|
| full | 130 | -0.010 | 0.115 | -0.088 | -0.290 | 0.523 | -0.356 |
| pre_cutoff | 95 | -0.007 | 0.092 | -0.077 | -0.218 | 0.516 | -0.195 |
| post_cutoff | 35 | -0.018 | 0.163 | -0.112 | -0.191 | 0.543 | -0.356 |

Average monthly turnover: 0.80 (2.0 = both legs fully replaced).

![Long-short cumulative returns](long_short_anon.png)

## Leakage: memory probe (no filing text given)

The model is given two company names and a date, with no filing text, and asked which stock had the higher return over the following three months. Accuracy above 50% before the cutoff but not after it indicates that the model has memorised outcomes.

| period | n | accuracy | z_vs_50pct | share_guess_A |
|---|---|---|---|---|
| pre_cutoff | 300 | 0.593 | 3.233 | 0.167 |
| post_cutoff | 300 | 0.473 | -0.924 | 0.183 |

The controls below separate date-specific memory from a general preference for certain companies. `picks_larger` is the share of questions in which the model chose the company that is larger today; `larger_wins` is the accuracy of always choosing the larger company; `acc_when_smaller` is the model's accuracy when it chose the smaller company; `placebo_accuracy` scores the model's choice against the same pair's outcome at every other month-end in the period, and so shows how much accuracy remains when the date is wrong.

| period | picks_larger | larger_wins | n_picked_smaller | acc_when_smaller | z_when_smaller | placebo_accuracy |
|---|---|---|---|---|---|---|
| pre_cutoff | 0.510 | 0.530 | 147 | 0.565 | 1.567 | 0.506 |
| post_cutoff | 0.493 | 0.553 | 152 | 0.421 | -1.947 | 0.491 |

## Leakage: identity probe on anonymised text (n=290)

Share of anonymised MD&A excerpts for which the model named the correct company and fiscal year.

| metric | share |
|---|---|
| company_correct | 0.824 |
| year_correct | 0.224 |
| year_within_1 | 0.448 |

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
| pre_cutoff | 63 | -0.018 | -1.178 | 36 |
| post_cutoff | 5 | 0.012 | 0.402 | 12 |
| post_cutoff | 21 | 0.037 | 1.274 | 11 |
| post_cutoff | 63 | 0.033 | 1.065 | 11 |

## Robustness: entry lag and trading cost

Full-sample net Sharpe ratio of each long-short portfolio. Lag 0 trades at the close of the filing day, which is not achievable for filings released after the close, so it is an upper bound.

By entry lag (trading days after the filing date, 10 bps cost):

| signal | lag_0 | lag_1 | lag_2 | lag_5 |
|---|---|---|---|---|
| llm_score | 0.346 | 0.308 | 0.366 | 0.412 |
| llm_score_chg | -0.107 | -0.179 | -0.152 | -0.043 |
| llm_caution | -0.359 | -0.363 | -0.503 | -0.578 |
| sim_yoy | -0.389 | -0.355 | -0.432 | -0.390 |
| emb_sim_yoy | -0.578 | -0.576 | -0.624 | -0.634 |
| emb_align_yoy | -0.069 | -0.088 | -0.103 | -0.150 |

By one-way cost (entry lag 1):

| signal | 0_bps | 10_bps | 25_bps |
|---|---|---|---|
| llm_score | 0.400 | 0.308 | 0.170 |
| llm_score_chg | 0.009 | -0.179 | -0.458 |
| llm_caution | -0.272 | -0.363 | -0.500 |
| sim_yoy | -0.293 | -0.355 | -0.449 |
| emb_sim_yoy | -0.509 | -0.576 | -0.676 |
| emb_align_yoy | -0.005 | -0.088 | -0.212 |
