# Cleaning report

Started from `dirty.csv` (39,073 rows, 16 columns). Investigated with pandas:
value counts per categorical column, describe() per numeric column, duplicate
checks, and cross-tabulations between related columns. Two real defects were
found and fixed; everything else checked out as normal, expected variation
for this dataset.

## 1. `tax_review_flag` leaks the target — dropped the column

Cross-tabulating `tax_review_flag` against `class`:

| tax_review_flag | <=50K | >50K | P(>50K) |
|---|---|---|---|
| 0 | 28,254 | 480 | 1.7% |
| 1 | 1,470 | 8,869 | 85.8% |

This is a far sharper split than any legitimate income signal in the data
(e.g. `capital-gain > 0` only moves P(>50K) from 21% to 61%). A flag for
"selected for audit" should not, on its own, near-perfectly separate the two
income classes — the only realistic explanation is that audit selection (or
this field as recorded) is downstream of the respondent's actual/reported
income, i.e. it encodes information from the outcome we're trying to predict.

It also fails the deployment test described in the task: the model scores
respondents "at the moment they complete the survey," but a tax return can
only be selected for review after it has been filed and evaluated for the
survey year — that determination doesn't exist yet at survey time. A field
that isn't actually available at prediction time but is highly correlated
with the label is a classic case of target leakage, so it was dropped
entirely rather than kept or imputed.

## 2. Exact duplicate rows — dropped 30 rows

30 rows were byte-for-byte duplicates of another row across all 16 columns,
including `fnlwgt` (a near-continuous per-record sampling weight, so an
identical value alongside 14 other identical fields is not plausible
coincidence). Given the file was assembled by joining several internal
tables, this looks like row fan-out from that join rather than genuinely
repeated respondents. Left in, they would double-count and over-weight a
small number of respondents during training, so they were removed with
`drop_duplicates()`, taking the file from 39,073 to 39,043 rows.

## Checked and left alone

- **Missing values** in `workclass` (2,234), `occupation` (2,244), and
  `native-country` (676): left as-is (not imputed or dropped). They line up
  with real-world patterns already in the data (e.g. `workclass ==
  "Never-worked"` rows also have missing `occupation`), so they look like
  genuine nulls rather than corruption, and filling or dropping them risks
  losing/distorting information without a demonstrated defect.
- `education` / `education-num`: perfectly 1:1 consistent, no conflicts.
- `capital-gain` / `capital-loss`: never both positive in the same row;
  max capital-gain of 99,999 is the known census top-code, not an outlier
  error.
- Numeric ranges (age, hours-per-week, fnlwgt) are all sane, no negative or
  zero values, no impossible values.
- Categorical spellings: no case/whitespace variants of the same category
  (e.g. no "Private" vs "private" duplicates).
- `relationship` vs `sex` (e.g. `Husband`/`Wife` vs `Male`/`Female`): a
  handful (4 rows) don't match the "expected" pairing. This is a known,
  real characteristic of self-reported census data (same-sex spouses), not
  an error, so left untouched.

## Result

`cleaned.csv`: 39,043 rows x 15 columns (`tax_review_flag` removed,
30 duplicate rows removed). `class` retained unchanged.
