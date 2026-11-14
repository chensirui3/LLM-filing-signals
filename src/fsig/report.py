"""Write reports/report_<variant>.md with tables and a chart for every signal."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import add_forward_returns, information_coefficients, long_short, split_stats
from .config import REPORTS
from .signals import SIGNALS


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
SECTION_ORDER = ("memory", "identity", "variants", "sensitivity")


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


def make_report(signals: pd.DataFrame, prices: pd.DataFrame, cfg: dict, variant: str = "raw") -> str:
    bt, llm = cfg["backtest"], cfg["llm"]
    REPORTS.mkdir(exist_ok=True)
    events = add_forward_returns(signals, prices, bt)
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
            f"one-way cost: {bt['cost_bps']} bps; quantiles: {bt['quantiles']}", ""]

    out += ["## Signals", ""] + [f"- `{k}`: {v}" for k, v in SIGNALS.items() if k in sig_cols] + [""]

    out += ["## Rank IC against market-adjusted forward returns", "",
            "Spearman correlation between the signal and the return, computed within each "
            "calendar quarter and then averaged across quarters. A |t| above about 2 is "
            "conventionally treated as significant.", ""]
    ic = information_coefficients(events, sig_cols, bt["horizons"], min_names=min_names)
    out += [_md(ic), ""]

    out += ["## Long-short portfolios (monthly, net of costs)", ""]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cutoff = pd.Timestamp(llm["training_cutoff"])
    for sig in sig_cols:
        monthly = long_short(events, prices, sig, bt)
        out += [f"### `{sig}`", ""]
        if monthly.empty:
            out += ["Too few companies to form portfolios.", ""]
            continue
        stats = split_stats(monthly, llm["training_cutoff"])
        out += [_md(stats.rename_axis("period")), "",
                f"Average monthly turnover: {monthly['turnover'].mean():.2f} "
                f"(2.0 = both legs fully replaced).", ""]
        (1 + monthly["net"]).cumprod().plot(ax=ax, label=sig, lw=2 if sig == bt["primary_signal"] else 1)
    ax.axvline(cutoff, color="grey", ls="--", lw=1)
    ax.text(cutoff, ax.get_ylim()[1], " model cutoff", va="top", fontsize=8, color="grey")
    ax.set_title("Long-short cumulative return, net of costs")
    ax.set_ylabel("Growth of 1")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / f"long_short_{variant}.png", dpi=130)
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
