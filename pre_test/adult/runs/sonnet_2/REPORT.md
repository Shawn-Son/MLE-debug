# Data cleaning report

Starting shape: 39,073 rows x 16 columns. Cleaned shape: 39,043 rows x 15 columns.

## 1. `tax_review_flag` leaks the label and isn't available at prediction time (dropped column)

This is the most serious defect. `tax_review_flag` is extremely correlated with
`class`:

- When `tax_review_flag == 1`: 85.8% of respondents are `>50K`
- When `tax_review_flag == 0`: only 1.7% of respondents are `>50K`

Two independent reasons this column must not be used as a model input:

- **It's downstream of the outcome, not a predictor of it.** Whether a tax
  return gets selected for review depends on the return itself (e.g. reported
  income), so it's a consequence/proxy of the true income level, not a cause.
  Including it lets the model "cheat" by reading off a near-copy of the label
  instead of learning genuine income signal.
- **It won't exist at inference time.** Per the task background, the model is
  meant to estimate income "at the moment [a respondent] completes the
  survey." A tax review outcome for that year's return can't be known yet at
  that moment for a new respondent, so a model trained to depend on it would
  receive a missing/undefined value in production and fail silently or
  systematically (e.g. if defaulted to 0, it would be biased toward
  predicting `<=50K`).

I dropped this column entirely rather than trying to impute or reweight it,
since any use of it — even partial — reintroduces leakage.

## 2. Exact duplicate rows (30 rows dropped)

30 rows were byte-for-byte duplicates of another row across all 16 columns,
including `fnlwgt` (a near-continuous census sampling weight), which makes
accidental coincidence very unlikely — these look like duplication introduced
when the source tables were joined. Left in place, they would let the same
respondent's data be sampled into both train and validation splits and give
that respondent's pattern extra weight in training. I removed the 30 excess
copies, keeping one instance of each.

## 3. Missing values in `workclass`, `occupation`, `native-country` (imputed, not dropped)

- `workclass` and `occupation` are missing together in exactly the same 2,234
  rows, and the other 10 rows missing `occupation` all have
  `workclass == "Never-worked"`. This is structural, not random: these are
  people not in the labor force, who legitimately have no employer type or
  job category. Dropping these rows would discard a real subpopulation and
  bias the training set toward the employed.
- `native-country` was missing in 676 rows with no evident pattern; likely
  unanswered on the survey.

Rather than deleting these rows (which the evaluation penalizes and would
throw away otherwise-valid respondents), I filled the missing entries with an
explicit `"Unknown"` category so the classifier can still make use of the
rest of each row's information and, for `workclass`/`occupation`, can learn
that "no job" is itself informative.

## Checks that did not turn up problems (left unchanged)

- No missing values in numeric columns or in `class`.
- No negative values or out-of-range values in `age`, `fnlwgt`,
  `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
- `education` and `education-num` are perfectly consistent (1:1 mapping).
- Categorical values are clean: no stray whitespace, casing variants, or
  placeholder strings (e.g. `"?"`) hiding as a separate category.
- `capital-gain` has a spike at 99,999 (194 rows); this is a known top-coding
  artifact of the original census data rather than a data-entry error, so it
  was left as-is.
- `sex`/`relationship` and `marital-status`/`relationship` combinations are
  overwhelmingly consistent, with only a handful of rows (e.g. 1
  female/"Husband", 3 male/"Wife") that plausibly reflect genuine self-report
  edge cases rather than errors — too small and ambiguous to justify altering.

## Summary of changes

- Dropped column: `tax_review_flag` (target leakage / not available at
  prediction time).
- Dropped rows: 30 exact duplicates.
- Filled missing values: `workclass`, `occupation`, `native-country` →
  `"Unknown"`.
- All other cells and the `class` column are untouched.
