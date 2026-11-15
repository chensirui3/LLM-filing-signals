import numpy as np
import pandas as pd

from fsig.backtest import add_forward_returns, long_short, perf_stats
from fsig.extract import anonymise, chunk_text, company_core_name
from fsig.sections import extract_sections
from fsig.signals import bag, cosine

BT = {"entry_lag_days": 1, "horizons": [5], "quantiles": 2, "max_signal_age_days": 100,
      "cost_bps": 10, "benchmark": "SPY"}


def _doc(body: str) -> bytes:
    toc = "<p>Item 7. Management's Discussion and Analysis</p><p>Item 7A. Quantitative Disclosures</p>"
    return (f"<html><body>{toc}<div>Item 1A. Risk Factors</div><p>{'Risk text. ' * 300}</p>"
            f"<div>Item 1B. Unresolved Staff Comments</div>"
            f"<div><span>Item 7.</span> <span>Management’s Discussion and Analysis</span></div>"
            f"<p>{body}</p><table><tr><td>1,234</td><td>5,678</td></tr></table>"
            f"<div>Item 7A. Quantitative and Qualitative Disclosures</div><p>Market risk.</p>"
            f"</body></html>").encode()


def test_sections_skip_table_of_contents_and_numeric_rows():
    out = extract_sections(_doc("Revenue grew strongly this year. " * 200), "10-K")
    assert out["mdna"].count("Revenue grew strongly") == 200
    assert "Market risk" not in out["mdna"] and "1,234" not in out["mdna"]
    assert out["risk"].startswith("Item 1A")


def test_anonymise_masks_name_ticker_and_years():
    text = "Apple Inc. revenue rose in 2023 versus 2022. AAPL shares, apple products."
    out = anonymise(text, "Apple Inc.", "AAPL", 2023)
    assert "pple" not in out and "AAPL" not in out and "2023" not in out
    assert "Y+0" in out and "Y-1" in out
    assert company_core_name("JPMORGAN CHASE & CO") == "JPMORGAN CHASE"


def test_chunking_and_cosine():
    chunks = chunk_text("\n".join(["x" * 90] * 100), 1000, 3)
    assert len(chunks) == 3 and all(len(c) <= 1000 for c in chunks)
    assert cosine(bag("alpha beta"), bag("alpha beta")) > 0.999
    assert cosine(bag("alpha beta"), bag("gamma delta")) == 0


def _prices():
    days = pd.bdate_range("2024-01-01", periods=120)
    up = pd.Series(np.linspace(100, 160, 120), index=days)
    flat = pd.Series(100.0, index=days)
    return pd.DataFrame({"A": up, "B": up, "C": flat, "D": flat, "SPY": flat})


def test_forward_returns_never_start_before_the_filing():
    px = _prices()
    # Filed on a Saturday: d0 is Monday, entry is Tuesday's close.
    sig = pd.DataFrame({"ticker": ["A"], "filing_date": [pd.Timestamp("2024-01-06")], "s": [1.0]})
    ev = add_forward_returns(sig, px, BT)
    assert ev.loc[0, "entry_date"] == pd.Timestamp("2024-01-09")
    i = px.index.get_loc(pd.Timestamp("2024-01-09"))
    assert np.isclose(ev.loc[0, "abn_5"], px["A"].iloc[i + 5] / px["A"].iloc[i] - 1)


def test_long_short_uses_only_signals_already_tradable():
    px = _prices()
    sig = pd.DataFrame({"ticker": list("ABCD"), "filing_date": pd.Timestamp("2024-01-10"),
                        "s": [2.0, 1.5, -1.0, -2.0]})
    ev = add_forward_returns(sig, px, BT)
    monthly = long_short(ev, px, "s", BT)
    assert monthly.index.min() > pd.Timestamp("2024-01-31")  # first holding period is February
    assert (monthly["gross"] > 0).all()                      # long the risers, short the flat names
    assert monthly["turnover"].iloc[0] == 2.0 and monthly["turnover"].iloc[1] == 0.0
    assert perf_stats(monthly["net"])["months"] == len(monthly)


def test_embedding_signals_detect_added_passages():
    from fsig.embed import alignment, doc_similarity, prior_year_map

    rng = np.random.default_rng(0)
    base = rng.normal(size=(6, 16)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    extra = rng.normal(size=(2, 16)).astype(np.float32)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    assert np.isclose(alignment(base, base[::-1]), 1.0, atol=1e-5)   # reordering is not a change
    assert np.isclose(doc_similarity(base, base[::-1]), 1.0, atol=1e-5)
    assert alignment(np.vstack([base, extra]), base) < 0.95          # added passages lower the alignment

    f = pd.DataFrame({"ticker": "A", "form": ["10-K", "10-K", "10-Q", "10-K"],
                      "accession": ["k1", "k2", "q1", "k3"],
                      "filing_date": pd.to_datetime(["2022-02-01", "2023-02-03", "2023-05-01", "2024-02-02"])})
    assert prior_year_map(f, lambda a: True) == {"k2": "k1", "k3": "k2"}
    assert prior_year_map(f, lambda a: a != "k2") == {}              # no usable prior year: no signal


def _page(*blocks: str) -> bytes:
    return ("<html><body>" + "".join(f"<p>{b}</p>" for b in blocks) + "</body></html>").encode()


def test_mdna_found_by_title_when_item_numbers_are_only_in_an_index():
    from fsig.sections import find_mdna, html_to_text

    body = "Net interest income rose on higher rates. " * 80
    doc = _page(
        "Management's Discussion and Analysis", "Quantitative and Qualitative Disclosures About Market Risk",  # contents
        "Item 1. Business", "We are a bank. " * 200,
        "Management&#146;s discussion and analysis", body,           # old-style apostrophe entity
        "Consolidated statements of income", "Not MD&A. " * 300,     # page break inside the MD&A...
        "Management's discussion and analysis", body,                # ...running header resumes it
        "Report of Independent Registered Public Accounting Firm", "Audit opinion. " * 300,
        "Item 7. Management's Discussion and Analysis: see pages 40-90", "Item 7A. Quantitative: see page 91",
    )
    sec = find_mdna(html_to_text(doc), "10-K")
    assert sec.count("Net interest income rose") == 160   # both halves, bridged across the running header
    assert "We are a bank" not in sec and "Audit opinion" not in sec


def test_mdna_found_by_bank_style_opening_heading():
    from fsig.sections import find_mdna, html_to_text

    doc = _page("Executive Overview", "Controls and Procedures",                 # contents
                "EXECUTIVE OVERVIEW", "Loan growth was solid this quarter. " * 100,
                "Consolidated statements of income (unaudited)", "Numbers. " * 400,
                "Item 4 Controls and Procedures")
    sec = find_mdna(html_to_text(doc), "10-Q")
    assert sec.startswith("EXECUTIVE OVERVIEW") and sec.count("Loan growth") == 100 and "Numbers." not in sec


def test_memory_controls_separate_date_memory_from_name_preference():
    from fsig.leakage import memory_controls

    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    # BIG outperforms SMALL in alternate months and underperforms in the others.
    fwd = pd.DataFrame({"BIG": [0.1, -0.1] * 3, "SMALL": [0.0] * 6}, index=dates)
    window_end = pd.Series(dates + pd.Timedelta(days=90), index=dates)
    rank = {"BIG": 0, "SMALL": 1}

    def answers(picks):
        winner = ["BIG" if fwd.loc[d, "BIG"] > 0 else "SMALL" for d in dates]
        return pd.DataFrame({"period": "pre_cutoff", "date": dates, "x": "BIG", "y": "SMALL", "winner": winner,
                             "consistent": True, "pick": picks,
                             "other": ["SMALL" if p == "BIG" else "BIG" for p in picks]})

    # A model that knows each date's outcome is always correct, but below chance against other dates.
    knows = memory_controls(answers(["BIG", "SMALL"] * 3), fwd, window_end, rank, "2023-12-31").loc["pre_cutoff"]
    assert knows["accuracy"] == 1.0 and knows["placebo_accuracy"] < 0.5 and knows["picks_larger"] == 0.5
    # A model that always picks the larger company is correct half the time, whatever the date.
    always_larger = memory_controls(answers(["BIG"] * 6), fwd, window_end, rank, "2023-12-31").loc["pre_cutoff"]
    assert always_larger["picks_larger"] == 1.0 and always_larger["accuracy"] == 0.5
    assert abs(always_larger["placebo_accuracy"] - 0.5) < 0.11


def test_memory_probe_scores_only_order_independent_answers(monkeypatch):
    from fsig import leakage

    days = pd.bdate_range("2021-01-01", periods=260)
    px = pd.DataFrame({"UP": np.linspace(100, 200, 260), "DOWN": np.linspace(100, 60, 260), "SPY": 100.0}, index=days)
    pairs, fwd, window_end = leakage.memory_pairs(px, ["UP", "DOWN"], 63, per_year=20)
    assert set(pairs["winner"]) == {"UP"} and (pairs["x"] == "DOWN").all()      # x and y are alphabetical
    assert (pairs["window_end"] > pairs["date"]).all() and pairs["id"].is_unique

    def fake(model):
        def run(kind, rows, llm, system, schema=None, schemas=None):
            out = {}
            for rid, user in rows:
                first, second = schemas[rid]["properties"]["answer"]["enum"]
                out[rid] = {"answer": {"knows": "UP", "second": second, "wrong": "DOWN"}[model]}
            return out
        return run

    names = {"UP": "Up Corp", "DOWN": "Down Corp"}
    for model, consistent, score in (("knows", 1.0, 1.0), ("second", 0.0, 0.5), ("wrong", 1.0, 0.0)):
        monkeypatch.setattr(leakage, "_run_probe", fake(model))
        a = leakage.memory_answers(pairs, names, {"model": model})
        assert a["consistent"].mean() == consistent and a["score"].mean() == score
    # A model that always names the second-listed stock contributes no scored answers.
    assert leakage.memory_summary(leakage.with_periods(a.assign(consistent=False), "2030-01-01")).loc["pre_cutoff", "n"] == 0

    labelled = leakage.with_periods(pairs, str(pairs["date"].sort_values().iloc[len(pairs) // 2].date()))
    assert set(labelled["period"]) == {"pre_cutoff", "spans_cutoff", "post_cutoff"}


def test_tied_scores_are_split_evenly_across_quantile_boundaries():
    from fsig.backtest import quantile_weights

    s = pd.Series([1.0] * 4 + [0.0] * 6, index=list("abcdefghij"))
    w = quantile_weights(s, 5)                      # two long slots shared by four names, two short by six
    assert np.allclose(w[:4], 0.25) and np.allclose(w[4:], -1 / 6)
    assert np.isclose(w[w > 0].sum(), 1) and np.isclose(w[w < 0].sum(), -1)
    shuffled = s.sample(frac=1, random_state=1)     # the result does not depend on the order of the names
    assert np.allclose(quantile_weights(shuffled, 5).reindex(s.index), w)
    assert np.allclose(quantile_weights(pd.Series(np.arange(10.0)), 5), [-.5, -.5, 0, 0, 0, 0, 0, 0, .5, .5])
    assert np.allclose(quantile_weights(pd.Series([3.0] * 10), 5), 0)   # no information, no position


def test_variant_gap_ic_detects_scores_shifted_towards_outcomes():
    from fsig.robustness import variant_comparison

    rng = np.random.default_rng(0)
    days = pd.bdate_range("2021-01-01", periods=400)
    tickers = [f"T{i}" for i in range(40)]
    rets = pd.DataFrame(rng.normal(0, 0.02, (len(days), 40)), index=days, columns=tickers)
    prices = (1 + rets).cumprod()
    prices["SPY"] = 1.0
    bt = {"entry_lag_days": 1, "horizons": [21], "benchmark": "SPY", "quantiles": 5}
    rows = []
    for q, d in enumerate(days[[20, 85, 150, 215]]):
        for t in tickers:
            rows.append({"accession": f"{t}-{q}", "ticker": t, "filing_date": d})
    raw = pd.DataFrame(rows)
    fwd = add_forward_returns(raw, prices, bt)["abn_21"]
    anon = raw[["accession"]].assign(llm_score=rng.normal(size=len(raw)), llm_score_chg=0.0, llm_caution=0.0)
    raw = raw.assign(llm_score=anon["llm_score"] + 5 * fwd, llm_score_chg=0.0, llm_caution=0.0)  # leaks the outcome
    agree, gap_ic = variant_comparison(raw, anon, prices, bt, "2030-01-01")
    assert gap_ic.loc[gap_ic["period"] == "pre_cutoff", "mean_ic"].iloc[0] > 0.9
    assert agree.loc["llm_score", "correlation"] > 0.5


def test_risk_diff_classifies_paragraphs():
    from fsig.riskdiff import classify, inline_diff, paragraphs

    base = "We depend on a small number of suppliers for key components and any disruption could delay shipments and harm revenue and margins. " * 2
    cyber = "A breach of our information systems could expose customer data and lead to regulatory fines, litigation and reputational damage for us. " * 2
    export = "New export licensing requirements restrict sales of our fastest processors to certain countries and may reduce our revenue there. " * 2
    old = [base, cyber]
    new = [base, cyber.replace("regulatory fines", "large regulatory fines and penalties"), export]
    e = np.eye(3, dtype=np.float32)
    rows, removed = classify(old, new, e[:2], e)  # paragraph 3 is orthogonal to everything old
    assert [r["status"] for r in rows] == ["unchanged", "edited", "added"] and removed == []
    rows, removed = classify(new, old, e, e[:2])  # reversed: the export paragraph was removed
    assert [r["old"] for r in removed] == [2]

    d = inline_diff("could lead to regulatory fines and litigation", "could lead to large regulatory fines and penalties")
    assert "**large**" in d and "~~litigation~~" in d

    text = "Risks Related to Supply\n" + base + "\nmanufacturing yields; and\n" + cyber
    paras = paragraphs(text)
    assert len(paras) == 2 and paras[1]["heading"] == "Risks Related to Supply"  # the list item is not a heading


def test_headings_come_from_formatting_not_length():
    from fsig.riskdiff import paragraphs
    from fsig.sections import emphasised_lines, html_to_text

    long_heading = ("Product, system security, and data breaches and cyber-attacks could disrupt our operations, reduce "
                    "our expected revenue, increase our expenses, and significantly harm our business and reputation.")
    body = "Security breaches, computer malware and phishing attacks have become more prevalent and sophisticated. " * 3
    raw = _page("<p><b>Risks Related to Our Global Operating Business</b></p>",
                "<p>" + body.replace("Security", "Economic") + "</p>",
                '<p><span style="font-weight:700">Product, system security, and data breaches</span>'
                '<span style="font-weight:700">' + long_heading[len("Product, system security, and data breaches"):] + "</span></p>",
                "<p>" + body + "</p>")
    assert len(long_heading) > 150
    em = emphasised_lines(raw)
    assert long_heading in em and body.strip() not in em

    text = html_to_text(raw)
    with_fmt = paragraphs(text, em)
    assert [p["heading"] for p in with_fmt] == ["Risks Related to Our Global Operating Business", long_heading]
    # Without formatting, the long heading is treated as a paragraph and the previous heading is kept.
    without = paragraphs(text)
    assert len(without) == 3 and without[2]["heading"] == "Risks Related to Our Global Operating Business"


def test_threshold_table_measures_precision_and_recall():
    from fsig.riskdiff import threshold_table

    pairs = pd.DataFrame({"cos": [0.99, 0.95, 0.94, 0.91, 0.80], "label": [1, 1, 0, 1, 0]})
    t = threshold_table(pairs, (0.93, 0.90)).set_index("threshold")
    assert t.loc[0.93, "called_rewritten"] == 3 and np.isclose(t.loc[0.93, "precision"], 2 / 3)
    assert np.isclose(t.loc[0.93, "recall"], 2 / 3) and np.isclose(t.loc[0.90, "recall"], 1.0)
    assert np.isclose(t.loc[0.90, "accuracy"], 0.8)


def test_within_industry_portfolio_takes_no_industry_position():
    days = pd.bdate_range("2024-01-01", "2024-06-28")
    up = 100 * 1.01 ** np.arange(len(days))
    px = pd.DataFrame({t: up for t in "ABC"} | {t: 100.0 for t in ["D", "E", "F", "SPY"]}, index=days)
    sig = pd.DataFrame({"ticker": list("ABCDEF"), "filing_date": pd.Timestamp("2024-01-10"),
                        "s": [3.0, 2.5, 2.0, -1.0, -1.5, -2.0]})
    ev = add_forward_returns(sig, px, BT)
    groups = {"A": "x", "B": "x", "C": "x", "D": "y", "E": "y", "F": "y"}
    assert (long_short(ev, px, "s", BT)["gross"] > 0).all()                    # long industry x, short industry y
    assert np.allclose(long_short(ev, px, "s", BT, groups)["gross"], 0)        # ranked against peers: no industry bet
    assert long_short(ev, px, "s", BT, {"A": "x", "B": "x"}).empty             # too few names with a usable industry


def test_factor_alpha_recovers_intercept_and_loading():
    from fsig.robustness import factor_alpha

    rng = np.random.default_rng(1)
    months = pd.period_range("2015-01", periods=120, freq="M")
    factors = pd.DataFrame(rng.normal(0, 0.03, size=(120, 2)), index=months, columns=["MKT", "SMB"])
    y = 0.004 + 0.5 * factors["MKT"] - 0.2 * factors["SMB"] + rng.normal(0, 0.001, 120)
    out = factor_alpha(pd.Series(y.to_numpy(), index=months.to_timestamp("M")), factors)
    assert out["months"] == 120 and abs(out["alpha_ann"] - 0.048) < 0.003 and out["t_alpha"] > 10
    assert abs(out["MKT"] - 0.5) < 0.01 and abs(out["SMB"] + 0.2) < 0.01


def test_paragraphs_split_by_a_page_break_are_rejoined():
    from fsig.riskdiff import paragraphs

    start = "Our suppliers may fail to deliver components on time, which " * 3 + "could force us to delay"
    rest = "shipments to customers and reduce our revenue."
    listing = "We face several risks in our international operations, including the following " * 2 + "items; and"
    text = "\n".join([start, "Table of Contents", "47", rest, listing, "• currency fluctuations in the markets we serve",
                      "A second paragraph that ends properly. " * 5])
    out = [p["text"] for p in paragraphs(text)]
    assert out[0] == f"{start} {rest}"                    # page furniture dropped, halves joined
    assert out[1] == listing and len(out) == 3            # a list lead-in is not joined to its bullets
