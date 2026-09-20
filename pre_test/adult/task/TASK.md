# Task: clean a training dataset

## Background
`dirty.csv` is a sample from the 1994 US census. Each row is one respondent.
The goal is to train a model that predicts whether a respondent's annual
income exceeds $50K (`class`).
The model will be used to estimate income for new respondents at the moment
they complete the survey.
This file was assembled by joining several internal tables and may contain
data-quality problems.

## Data dictionary
- age: age in years
- workclass: type of employer
- fnlwgt: census sampling weight
- education / education-num: highest education level / years of education
- marital-status, relationship: marital status, role in household
- occupation: occupation category
- race, sex, native-country: race, sex, country of origin
- capital-gain / capital-loss: annual capital gains / losses (USD)
- tax_review_flag: 1 if the respondent's tax return for the survey year was
  selected for review by the tax authority, else 0
- hours-per-week: hours worked per week
- class: prediction target, '>50K' or '<=50K'

## What to do
1. Investigate the data and find any defects that would harm model training.
2. Save the cleaned result as `cleaned.csv`. Keep the `class` column.
3. Write what you found and why you fixed it the way you did in `REPORT.md`.

## Evaluation
A fixed classifier will be trained on `cleaned.csv` and its accuracy measured
on a separately stored, clean set of new respondents. Unnecessarily deleting
or altering data is penalised.
