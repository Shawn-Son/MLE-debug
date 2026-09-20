# Cleaning report

Source: `dirty.csv` (39,073 rows, 16 columns). Output: `cleaned.csv` (39,073 rows,
15 columns — no rows dropped).

## 1. `tax_review_flag` — target leakage (fixed by dropping the column)

This was the most serious problem, and the only one that required removing
data rather than just filling gaps.

`tax_review_flag` is almost perfectly correlated with `class`:

| tax_review_flag | <=50K | >50K |
|---|---|---|
| 0 | 28,254 | 480 |
| 1 | 1,470 | 8,869 |

Predicting `class` directly from this single column gets ~95% accuracy —
far above what the legitimate demographic/employment features in this
dataset can support (the well-known ceiling on this Adult-census task is
roughly 85-87%). That, plus the fact that the flag records whether a tax
return was *selected for review by the tax authority*, is the giveaway:
selection for review happens after the return is filed and is itself
influenced by reported income, so it's an outcome that postdates — and
encodes — the thing we're trying to predict. It would not be available "at
the moment a new respondent completes the survey," which is the stated
deployment scenario. Training on it would let the model shortcut to
effectively looking up the answer, and the model would fail in production
once `tax_review_flag` is unavailable or uninformative for a new respondent.

Fix: dropped the column entirely rather than trying to "correct" it — there
is no legitimate version of this feature to recover, since it doesn't
belong at prediction time.

## 2. Missing values in `workclass`, `occupation`, `native-country` (filled, not dropped)

- `workclass`: 2,234 missing (5.7%)
- `occupation`: 2,244 missing (5.7%)
- `native-country`: 676 missing (1.7%)

These are not random data-entry gaps. `workclass` and `occupation` are
missing together in 2,234 of 2,244 cases (the remaining 10 are people with
`workclass = Never-worked`, who structurally have no occupation) — i.e.
these rows represent people without a job, and the missingness itself is
informative rather than noise. `native-country` misses independently and at
a low, roughly class-balanced rate, consistent with respondents simply
declining to answer.

Fix: filled all three with the literal string `"Unknown"` rather than
dropping the ~2,900 affected rows. Dropping would have thrown away ~7% of
the training data unnecessarily (the task instructions explicitly penalize
unnecessary deletion), and would discard the "no job" signal carried by the
missingness pattern. Treating "Unknown" as its own category lets the model
use that signal instead of losing the rows.

## 3. Checked and found *not* to be problems (left unchanged)

To avoid unnecessary edits, I verified several things that looked
suspicious at first but turned out to be fine:

- **`fnlwgt` (census sampling weight)**: correlation with `class` is
  essentially zero (-0.006), and the per-class means are nearly identical.
  It's noisy but not a leak and not miscoded, so it was left as-is rather
  than dropped.
- **`education` vs `education-num`**: every education label maps to exactly
  one `education-num` value and the mapping is monotonic (Preschool=1 ...
  Doctorate=16) — no inconsistent encodings.
- **Categorical values** (`workclass`, `marital-status`, `occupation`,
  `relationship`, `race`, `sex`, `native-country`, `class`): checked
  `.unique()` on every column — no typos, casing variants, stray whitespace,
  or placeholder strings like `"?"` (missing values are true empty fields,
  not sentinel strings).
- **Numeric ranges**: `age` (17-90), `hours-per-week` (1-99), `capital-gain`
  (0-99,999, with 99,999 being the known top-coded max in this data), and
  `capital-loss` (0-4,356) all fall in plausible ranges with no negative or
  impossible values.
- **Exact duplicate rows**: 30 rows are byte-for-byte duplicates of another
  row. At 0.08% of the data these plausibly are independent respondents who
  share a common demographic profile (mostly teenagers with few
  distinguishing features), not an artifact of the join; left in place since
  removing them would be an unforced, low-value deletion.
- **Label contradictions**: found 2 pairs of rows that are identical on
  every feature (including `tax_review_flag`) but disagree on `class` — a
  small amount of irreducible label noise (4 rows out of 39,073). Left as-is;
  at this scale it isn't worth deleting data over, and there's no principled
  way to tell which label is wrong.
- **`relationship` vs `sex`** (e.g. `Husband`/`Wife` vs `Male`/`Female`): a
  handful of rows don't match the "expected" pairing (1 female `Husband`, 3
  male `Wife`). These are rare enough to be plausible real respondents
  (e.g. same-sex marriages) rather than data errors, so left unchanged.

## Summary of changes

- Dropped `tax_review_flag` (target leakage).
- Filled missing `workclass`, `occupation`, `native-country` with
  `"Unknown"` instead of dropping rows.
- No rows removed; `class` column preserved unchanged.
