"""Write reports/report_<variant>.md with tables and charts for every signal."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import (DETECTABLE, add_context_returns, add_forward_returns, information_coefficients,
                       long_short, split_stats, timing_profile)
from .config import REPORTS
from .signals import SIGNALS

# Chart styling. Series colours are assigned in this fixed order; text and
# axes use the neutral inks, never a series colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def _axes(figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _md(df: pd.DataFrame, floats: str = "{:.3f}") -> str:
    """Small markdown table writer (avoids the optional 'tabulate' dependency)."""
    df = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    def fmt(v):
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            if v.is_integer() and abs(v) >= 1:
                return str(int(v))
            s = floats.format(v)
            return s.lstrip("-") if float(s) == 0 else s  # print 0.000, not -0.000
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# Sections produced by the leakage and robustness steps, in the order they
# appear after the backtest results.
SECTION_ORDER = ("memory", "identity", "variants", "adjusted", "fulltext", "sensitivity")


def _sections_path():
    return REPORTS / "sections.json"


def load_sections() -> dict[str, dict]:
    p = _sections_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_section(key: str, title: str, body: str) -> None:
    """Store a section produced by a slow step (probes, robustness) so that every
    later report rebuild includes it. The key is '<scope>:<name>', where name is
    one of SECTION_ORDER and scope is 'all' (included in every report) or a
    variant (included only in that variant's report)."""
    REPORTS.mkdir(exist_ok=True)
    s = load_sections()
    s[key] = {"title": title, "body": body}
    _sections_path().write_text(json.dumps(s, indent=1), encoding="utf-8")


# ----------------------------------------------------------------- memory probe

def memory_chart(results: dict[str, dict], name: str = "memory_probe.png") -> str:
    """Probe accuracy by year for each model, with 95% intervals and each model's cutoff."""
    fig, ax = _axes((9, 4.6))
    years = sorted({y for r in results.values() for y in r["by_year"].index})
    ax.axhline(0.5, color=AXIS, lw=1.2)
    ax.text(1.005, 0.5, "chance", transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED, va="center")
    n = len(results)
    for i, (model, r) in enumerate(results.items()):
        by, colour = r["by_year"], SERIES[i]
        x = by.index.to_numpy() + (i - (n - 1) / 2) * 0.12  # offset so the intervals do not overlap
        ax.errorbar(x, by["accuracy"], yerr=[by["accuracy"] - by["ci_low"], by["ci_high"] - by["accuracy"]],
                    color=colour, lw=2, elinewidth=1, capsize=0, marker=MARKERS[i], ms=6,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=f"{model} (cutoff {r['cutoff'][:7]})")
        c = pd.Timestamp(r["cutoff"])
        cx = c.year + (c.dayofyear / 365.25) - 0.5  # year labels sit at mid-year
        ax.plot([cx, cx], [0.335, 0.36], color=colour, lw=2)
        ax.plot(cx, 0.36, marker=MARKERS[i], ms=6, color=colour, markeredgecolor=SURFACE, markeredgewidth=1)
    ax.text(years[0] - 0.45, 0.345, "training cutoffs", fontsize=8, color=MUTED, va="center")
    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)
    ax.set_ylim(0.32, 0.72)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Memory probe: accuracy on order-independent answers, by year of the question",
                 fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=9, frameon=False, labelcolor=MUTED, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncols=n)
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return name


def memory_section(results: dict[str, dict], chart: str) -> str:
    """Markdown for the memory probe: summary and controls per model, accuracy by year."""
    summary = pd.concat([r["summary"].assign(model=m, cutoff=r["cutoff"]).reset_index() for m, r in results.items()],
                        ignore_index=True)
    controls = pd.concat([r["controls"].assign(model=m).reset_index() for m, r in results.items()], ignore_index=True)
    by_year = pd.DataFrame({m: r["by_year"]["accuracy"].map("{:.3f}".format) + " (" + r["by_year"]["n"].astype(str) + ")"
                            for m, r in results.items()}).rename_axis("year")
    pairs = int(next(iter(results.values()))["answers"].shape[0])
    return "\n\n".join([
        f"Each model is given two company names and a date, with no filing text, and asked which stock had the "
        f"higher return over the following three months. The same {pairs:,} pairs, an equal number from every "
        "calendar year, are put to every model in both orders. `order_independent` is the share of pairs on which "
        "the model named the same company both times; `accuracy` is measured on those pairs (`n`), so a "
        "preference for one position cannot affect it. `accuracy_all_pairs` counts a pair with conflicting "
        "answers as half right. A pair is pre-cutoff when its whole three-month window ends before the model's "
        "training cutoff and post-cutoff when the question date is after it.",
        _md(summary[["model", "cutoff", "period", "pairs", "order_independent", "n", "accuracy", "z_vs_50pct",
                     "accuracy_all_pairs"]]),
        "Memorised outcomes would show as accuracy above 50% before a model's cutoff that falls to 50% after it, "
        "with the fall occurring at a different date for each model.",
        f"![Memory probe accuracy by year]({chart})",
        "Accuracy on order-independent answers by year of the question, with the number of such answers in "
        "brackets:",
        _md(by_year),
        "The controls separate date-specific memory from a general preference for certain companies. "
        "`picks_larger` is the share of answers that chose the company that is larger today; `larger_wins` is the "
        "accuracy of always choosing the larger company; `acc_when_smaller` is the accuracy when the model chose "
        "the smaller company; `placebo_accuracy` scores the model's choice against the same pair's outcome at "
        "every other month-end in the period, and so shows how much accuracy remains when the date is wrong.",
        _md(controls[["model", "period", "accuracy", "placebo_accuracy", "picks_larger", "larger_wins",
                      "n_picked_smaller", "acc_when_smaller", "z_when_smaller"]]),
    ])


# ----------------------------------------------------------------- main report

def make_report(signals: pd.DataFrame, prices: pd.DataFrame, cfg: dict, variant: str = "raw") -> str:
    bt, llm = cfg["backtest"], cfg["llm"]
    REPORTS.mkdir(exist_ok=True)
    events = add_context_returns(add_forward_returns(signals, prices, bt), prices, bt)
    sig_cols = [s for s in SIGNALS if s in events.columns and events[s].notna().any()]
    min_names = 4 * bt["quantiles"]

    out = [f"# Filing signals report ({variant} text, {llm['model']})", ""]
    out += [f"- Filings: {len(events)}, with LLM score: {events['llm_score'].notna().sum()}, "
            f"companies: {events['ticker'].nunique()}",
            f"- Sample: {events['filing_date'].min():%Y-%m-%d} to {events['filing_date'].max():%Y-%m-%d}",
            f"- Model training cutoff: {llm['training_cutoff']}",
            f"- Primary signal (declared in the configuration before any results were computed): "
            f"`{bt['primary_signal']}`",
            f"- Entry lag: {bt['entry_lag_days']} trading day{'' if bt['entry_lag_days'] == 1 else 's'}; "
            f"one-way cost: {bt['cost_bps']} bps; quantiles: {bt['quantiles']}"]
    if "scored_share" in events.columns:
        share = events.groupby("form")["scored_share"].median()
        out += [f"- Share of each MD&A read by the LLM (first {llm['max_chunks']} chunks of "
                f"{llm['chunk_chars']:,} characters), median: "
                + ", ".join(f"{v:.0%} for {form}" for form, v in share.items())]
    out += [""]

    out += ["## Signals", ""] + [f"- `{k}`: {v}" for k, v in SIGNALS.items() if k in sig_cols] + [""]

    out += ["## Rank IC against market-adjusted forward returns", "",
            "Spearman correlation between the signal and the return, computed within each "
            "calendar quarter and then averaged across quarters. A |t| above about 2 is "
            "conventionally treated as significant. `se` is the standard error of the mean IC and "
            f"`min_detectable_ic` ({DETECTABLE} standard errors) is the smallest true IC this sample "
            "would detect with 80% power at the 5% level; a null result rules out larger effects, "
            "not smaller ones.", ""]
    ic = information_coefficients(events, sig_cols, bt["horizons"], min_names=min_names)
    out += [_md(ic), ""]

    horizon = max(bt["horizons"])
    out += ["## Timing: returns before, around and after the filing", "",
            "Rank IC of each signal against the market-adjusted return in three windows, shown as "
            "IC (t-statistic). `before_filing` is the 63 trading days that end the day before the "
            "filing; `filing_window` runs from the close before the filing to the entry close; "
            f"`after_entry` is the {horizon} trading days after entry, the only window the strategy "
            "can earn. A signal that reflects information the market already has correlates with "
            "the first two windows and not the third.", "",
            _md(timing_profile(events, sig_cols, llm["training_cutoff"], horizon, min_names)), ""]

    out += ["## Long-short portfolios (monthly, net of costs)", "",
            "Top-minus-bottom quantile portfolios. Companies with equal scores are split evenly "
            "across a quantile boundary.", ""]
    fig, ax = _axes((9, 4.6))
    cutoff = pd.Timestamp(llm["training_cutoff"])
    for i, sig in enumerate(sig_cols):
        monthly = long_short(events, prices, sig, bt)
        out += [f"### `{sig}`", ""]
        if monthly.empty:
            out += ["Too few companies to form portfolios.", ""]
            continue
        stats = split_stats(monthly, llm["training_cutoff"])
        out += [_md(stats.rename_axis("period")), "",
                f"Average monthly turnover: {monthly['turnover'].mean():.2f} "
                f"(2.0 = both legs fully replaced).", ""]
        curve = (1 + monthly["net"]).cumprod()
        ax.plot(curve.index, curve, color=SERIES[i % len(SERIES)], lw=2.4 if sig == bt["primary_signal"] else 1.4,
                label=sig + (" (primary)" if sig == bt["primary_signal"] else ""))
    ax.axhline(1, color=AXIS, lw=1.2)
    ax.axvline(cutoff, color=MUTED, ls="--", lw=1)
    ax.text(cutoff, ax.get_ylim()[1], " model cutoff", va="top", fontsize=8, color=MUTED)
    ax.set_title("Long-short cumulative return, net of costs (growth of 1)", fontsize=11, color=INK, loc="left")
    ax.legend(fontsize=9, frameon=False, labelcolor=MUTED, ncols=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(REPORTS / f"long_short_{variant}.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    out += [f"![Long-short cumulative returns](long_short_{variant}.png)", ""]

    saved = load_sections()
    for name in SECTION_ORDER:
        for key in (f"all:{name}", f"{variant}:{name}"):
            if key in saved:
                out += [f"## {saved[key]['title']}", "", saved[key]["body"], ""]

    path = REPORTS / f"report_{variant}.md"
    path.write_text("\n".join(out), encoding="utf-8")
    events.to_parquet(REPORTS / f"events_{variant}.parquet", index=False)
    return str(path)
