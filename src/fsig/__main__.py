"""Command line: python -m fsig <step> [options]

Steps, in order:
  universe   pick companies and build the filing index
  download   download filing documents from EDGAR
  parse      extract MD&A / risk sections
  prices     download daily prices
  extract    score MD&A chunks with the local LLM  (--variant raw|anon)
  embed      embed MD&A chunks with the local embedding model
  signals    assemble the per-filing signal table
  report     run the backtest and write reports/report_<variant>.md
  leakage    run the memory and identity probes, append them to the report
  robustness entry-lag and cost sensitivity; raw vs anonymised score comparison
  riskdiff   what changed in a company's Risk Factors vs last year  (--ticker NVDA [--year 2023])
  riskeval   accuracy of the riskdiff similarity threshold on the labelled pairs in validation/
  demo       precompute Risk Factors comparisons for the static demo page in docs/
  all        universe -> report
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import backtest, edgar, embed, extract, leakage, prices, report, riskdiff, robustness, sections, signals
from .config import load_config


def rebuild_reports(cfg: dict, px: pd.DataFrame) -> None:
    """Rebuild the report of every variant that has a signal table."""
    for v in ("raw", "anon"):
        if (edgar.DATA / f"signals_{v}.parquet").exists():
            print("wrote", report.make_report(signals.load_signals(v), px, cfg, v))


def main() -> None:
    ap = argparse.ArgumentParser(prog="fsig", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["universe", "download", "parse", "prices", "extract", "embed",
                                     "signals", "report", "leakage", "robustness", "riskdiff", "riskeval", "demo",
                                     "all"])
    ap.add_argument("--ticker", help="riskdiff: company ticker")
    ap.add_argument("--year", type=int, help="riskdiff: filing year of the 10-K (default: latest)")
    ap.add_argument("--no-summary", action="store_true", help="riskdiff, demo: skip the LLM one-line summaries")
    ap.add_argument("--variant", choices=["raw", "anon"], default="raw")
    ap.add_argument("--all-chunks", action="store_true",
                    help="extract: score the whole MD&A, not only the first chunks (for the truncation check)")
    ap.add_argument("--size", type=int, help="override universe_size")
    ap.add_argument("--start", help="override start_date (YYYY-MM-DD)")
    ap.add_argument("--quantiles", type=int, help="override backtest.quantiles")
    ap.add_argument("--retry-failed", action="store_true", help="parse: re-parse filings with no MD&A found")
    args = ap.parse_args()

    cfg = load_config()
    if args.size:
        cfg["universe_size"] = args.size
    if args.start:
        cfg["start_date"] = args.start
    if args.quantiles:
        cfg["backtest"]["quantiles"] = args.quantiles
    llm, bt = cfg["llm"], cfg["backtest"]
    run = lambda s: args.step in (s, "all")

    if run("universe"):
        uni, f = edgar.build_universe(edgar.EdgarClient(), cfg["universe_size"], cfg["start_date"], cfg["forms"])
        print(f"{len(uni)} companies, {len(f)} filings\n{f['form'].value_counts().to_string()}")
    if run("download"):
        n = edgar.download_filings(edgar.EdgarClient(), edgar.load_filings())
        print(f"downloaded {n} new filings")
    if run("parse"):
        print(sections.parse_all(edgar.load_filings(), retry_failed=args.retry_failed).to_string())
    if run("prices"):
        f = edgar.load_filings()
        px = prices.download_prices(sorted(f["ticker"].unique()), cfg["start_date"], bt["benchmark"])
        missing = sorted(set(f["ticker"]) - set(px.columns[px.notna().any()]))
        print(f"prices: {px.shape[0]} days x {px.shape[1]} symbols; no data for: {missing or 'none'}")
    if run("extract"):
        print(extract.run_extraction(edgar.load_filings(), llm, args.variant, args.all_chunks))
    if run("embed"):
        print(embed.embed_all(edgar.load_filings(), cfg["embed"]))
    if run("signals"):
        s = signals.build_signals(edgar.load_filings(), llm, args.variant, cfg.get("embed"))
        print(s[list(signals.SIGNALS)].describe().T.to_string())
    if run("report"):
        print("wrote", report.make_report(signals.load_signals(args.variant), prices.load_prices(), cfg, args.variant))
    if args.step == "leakage":
        sig, px, lk = signals.load_signals(args.variant), prices.load_prices(), cfg["leakage"]
        names = dict(zip(sig["ticker"], sig["name"]))
        uni = pd.read_parquet(edgar.UNIVERSE)  # ordered largest first
        results = leakage.run_memory_probe(px, names, llm, lk, bt["benchmark"],
                                           {t: i for i, t in enumerate(uni["ticker"])}, cfg["start_date"])
        report.save_section("all:memory", "Leakage: memory probe (no filing text given)",
                            report.memory_section(results, report.memory_chart(results)))
        ident = leakage.identity_probe(edgar.load_filings(), llm, n=lk["identity_sample"])
        ident_summary = ident.drop(columns="accession").mean().to_frame("share").rename_axis("metric")
        report.save_section(
            "all:identity", f"Leakage: identity probe on anonymised text (n={len(ident)})",
            "Share of anonymised MD&A excerpts for which the model named the correct company "
            "and fiscal year.\n\n" + report._md(ident_summary))
        for model, r in results.items():
            print(f"\n{model} (cutoff {r['cutoff']})\n{r['summary'].to_string()}\n{r['controls'].to_string()}")
        print(ident_summary.to_string())
        rebuild_reports(cfg, px)
    if args.step == "riskeval":
        print("wrote", riskdiff.validation_report(report._md))
    if args.step == "demo":
        demo = cfg["demo"]
        uni = pd.read_parquet(edgar.UNIVERSE)  # ordered largest first
        print(riskdiff.export_demo(edgar.load_filings(), list(uni["ticker"][: demo["companies"]]), cfg["embed"],
                                   None if args.no_summary else llm, demo["max_items"]))
    if args.step == "riskdiff":
        if not args.ticker:
            ap.error("riskdiff needs --ticker")
        print("wrote", riskdiff.risk_diff(edgar.load_filings(), args.ticker, args.year, cfg["embed"],
                                          None if args.no_summary else llm))
    if args.step == "robustness":
        px = prices.load_prices()
        sig = signals.load_signals(args.variant)
        by_lag, by_cost = robustness.sensitivity(sig, px, bt)
        report.save_section(
            f"{args.variant}:sensitivity", "Robustness: entry lag and trading cost",
            "Full-sample net Sharpe ratio of each long-short portfolio. Lag 0 trades at the close "
            "of the filing day, which is not achievable for filings released after the close, so "
            "it is an upper bound.\n\nBy entry lag (trading days after the filing date, "
            f"{bt['cost_bps']} bps cost):\n\n" + report._md(by_lag)
            + f"\n\nBy one-way cost (entry lag {bt['entry_lag_days']}):\n\n" + report._md(by_cost))
        print(by_lag.to_string(), "\n", by_cost.to_string())
        groups = edgar.industries()
        if groups:
            factors = prices.get_factors()
            if factors is None:
                print("factor data could not be downloaded; the factor regression is skipped")
            by_ind, alpha = robustness.adjusted_returns(sig, px, bt, llm["training_cutoff"], groups, factors)
            text = ("Net Sharpe ratio and t-statistic of each long-short portfolio as reported, and when each "
                    "company's signal is measured relative to the mean of its industry (two-digit SIC group, "
                    f"{len(set(groups.values()))} groups; industries with fewer than {backtest.MIN_GROUP} companies "
                    "at a rebalance are left out). The within-industry portfolio takes no position on "
                    "industries.\n\n" + report._md(by_ind))
            if alpha is not None:
                text += ("\n\nRegression of each portfolio's monthly net return on the Fama-French five factors "
                         "and momentum (Kenneth French's data library). `alpha_ann` is the annualised return "
                         "the factors do not explain and `t_alpha` its t-statistic; the remaining columns are "
                         "the factor loadings. The post-cutoff regressions have few months for seven "
                         "coefficients and are imprecise.\n\n" + report._md(alpha))
            report.save_section(f"{args.variant}:adjusted", "Robustness: industry"
                                + (" and factor adjustment" if alpha is not None else " adjustment"), text)
            print(by_ind.to_string(), "\n", "" if alpha is None else alpha.to_string())
        if args.variant == "raw":
            n_full, full = robustness.full_text_check(sig, llm, px, bt)
            if n_full:
                report.save_section(
                    "all:fulltext", "Robustness: scoring the whole MD&A",
                    f"The signals read only the first {llm['max_chunks']} chunks of each MD&A. For {n_full:,} "
                    "filings with a longer MD&A, every chunk was scored as well. The table compares the "
                    "signal built from the first chunks with the same signal built from the whole section, "
                    "on those filings: their correlation, and the rank IC of each against the 63-day return "
                    "before the filing and the return after entry. In this table `llm_score_chg` is the "
                    "change from the previous filing within this set of filings.\n\n" + report._md(full))
                print(full.to_string())
        if all((edgar.DATA / f"signals_{v}.parquet").exists() for v in ("raw", "anon")):
            agree, gap_ic = robustness.variant_comparison(
                signals.load_signals("raw"), signals.load_signals("anon"), px, bt, llm["training_cutoff"])
            report.save_section(
                "all:variants", "Leakage: raw versus anonymised scores",
                "Each filing was scored twice: as filed, and with the company name, ticker and "
                "years masked. `gap` is the mean of the raw score minus the anonymised score "
                "(`llm_score` ranges from -2 to 2).\n\n"
                + report._md(agree)
                + "\n\nRank IC of the gap in `llm_score` against market-adjusted forward returns. "
                "If the model had shifted its scores towards outcomes it had memorised, the IC "
                "would be positive before the cutoff and close to zero after it.\n\n"
                + report._md(gap_ic))
            print(agree.to_string(), "\n", gap_ic.to_string())
        rebuild_reports(cfg, px)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
