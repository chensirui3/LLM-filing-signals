# Risk Factors comparison: validation of the similarity threshold

The comparison tool calls a paragraph *rewritten* rather than *added* when its wording differs from the closest paragraph in the previous filing but their embedding cosine is at least 0.93. That threshold was chosen by inspecting one company (NVIDIA). This report tests it on other companies.

**Sample.** 120 paragraph pairs from 16 companies (AAPL, AMZN, BA, COST, CVX, GS, HD, JPM, KO, META, MSFT, ORCL, PFE, TSLA, UNH, WMT), none of them NVIDIA. Each pair is a current paragraph and its closest previous paragraph, restricted to pairs whose wording differs (word-level similarity below 0.6), which are the cases the threshold decides. Equal numbers were drawn from five cosine ranges, so borderline cases are over-represented relative to a real filing.

**Labels.** Each pair was labelled 1 if the current paragraph restates the specific content of the previous one (including a condensed or expanded version, or a fragment of it) and 0 if it is new content, even on the same broad theme. The labels were assigned by an LLM assistant (Claude) reading the pairs in random order without the cosine values, and have not been independently checked by a second annotator. 16 close calls are flagged `borderline` in `validation/riskdiff_pairs.csv`.

**Result at the threshold in use (0.93).** Precision 90%, recall 68%: of the paragraphs the tool calls rewritten, 90% are rewrites, and it finds 68% of all rewrites. The remaining rewrites are reported as added.

## Accuracy by threshold

| threshold | called_rewritten | precision | recall | f1 | accuracy |
|---|---|---|---|---|---|
| 0.850 | 96 | 0.646 | 0.984 | 0.780 | 0.708 |
| 0.880 | 84 | 0.714 | 0.952 | 0.816 | 0.775 |
| 0.900 | 72 | 0.750 | 0.857 | 0.800 | 0.775 |
| 0.910 | 64 | 0.797 | 0.810 | 0.803 | 0.792 |
| 0.920 | 57 | 0.860 | 0.778 | 0.817 | 0.817 |
| 0.930 | 48 | 0.896 | 0.683 | 0.775 | 0.792 |
| 0.940 | 42 | 0.952 | 0.635 | 0.762 | 0.792 |
| 0.950 | 32 | 1 | 0.508 | 0.674 | 0.742 |
| 0.960 | 24 | 1 | 0.381 | 0.552 | 0.675 |

Excluding the borderline pairs:

| threshold | called_rewritten | precision | recall | f1 | accuracy |
|---|---|---|---|---|---|
| 0.850 | 81 | 0.667 | 0.982 | 0.794 | 0.731 |
| 0.880 | 71 | 0.732 | 0.945 | 0.825 | 0.788 |
| 0.900 | 62 | 0.774 | 0.873 | 0.821 | 0.798 |
| 0.910 | 55 | 0.818 | 0.818 | 0.818 | 0.808 |
| 0.920 | 50 | 0.880 | 0.800 | 0.838 | 0.837 |
| 0.930 | 44 | 0.886 | 0.709 | 0.788 | 0.798 |
| 0.940 | 38 | 0.947 | 0.655 | 0.774 | 0.798 |
| 0.950 | 30 | 1 | 0.545 | 0.706 | 0.760 |
| 0.960 | 23 | 1 | 0.418 | 0.590 | 0.692 |

## Share of pairs labelled as rewrites, by cosine range

| cosine | pairs | share_rewrites |
|---|---|---|
| [0.0, 0.85) | 24 | 0.042 |
| [0.85, 0.9) | 24 | 0.333 |
| [0.9, 0.93) | 24 | 0.458 |
| [0.93, 0.96) | 24 | 0.792 |
| [0.96, 1.01) | 24 | 1 |

## Reading the result

- Cosine similarity orders the pairs well: almost none below 0.85 are rewrites and all of those above 0.96 are.
- A threshold of 0.92 scores slightly higher on this sample than 0.93. The difference is within the sampling error of 120 pairs, and adopting it would mean tuning on the test set, so the threshold is unchanged.
- Many low-similarity pairs in the sample are fragments: the parser splits a paragraph where the filing has a page break, and the fragment is then compared with a whole paragraph. Joining paragraphs across page breaks would remove most of these.
