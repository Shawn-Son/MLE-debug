"""
make_task_leak.py - Generate one task: "find the target-leakage column"

Run:  python make_task_leak.py

Outputs:
  task/dirty.csv          data given to the agent
  task/TASK.md            instructions given to the agent
                          (background + data dictionary + what to do)
  hidden/clean_train.csv  training data before corruption (grader only)
  hidden/holdout.csv      held-out test data              (grader only)
  hidden/answer.json      ground truth                    (grader only)

The agent must never see anything under hidden/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 0
LEAK_COL = "tax_review_flag"   # name of the injected leakage column
LEAK_NOISE = 0.05              # fraction of rows where the leak disagrees with the target

COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "class"]
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"

# ---------------------------------------------------------------- 1. Load data
# Adult: sample of the 1994 US census, 48,842 people. One row = one person.
# Target `class` = whether annual income exceeds $50K ('>50K' / '<=50K').
df = pd.read_csv(URL, header=None, names=COLS, na_values="?",
                 skipinitialspace=True)
print(f"Raw data: {df.shape[0]} rows x {df.shape[1]} cols, "
      f"high-income rate {(df['class'] == '>50K').mean():.1%}")

# ---------------------------------------------------------------- 2. Split off the holdout
# Set aside 20% BEFORE injecting anything. The holdout stays clean and the
# agent never sees it. Later we train a model on the agent's cleaned data and
# measure its accuracy here.
train, holdout = train_test_split(df, test_size=0.2, stratify=df["class"],
                                  random_state=SEED)
train, holdout = train.reset_index(drop=True), holdout.reset_index(drop=True)

# ---------------------------------------------------------------- 3. Inject corruption: target leakage
# Add a column, to the training data only, that copies the label with 95%
# fidelity.
# Real-world analogue: a "flagged for tax review" field that is set AFTER
# income is known, and that slipped into the training table during a join.
# A model that uses this column looks ~95% accurate in validation, but the
# information does not exist at prediction time (the holdout has no such
# column), so the deployed model falls apart.
rng = np.random.default_rng(SEED)
y = (train["class"] == ">50K").astype(int).to_numpy()
flip = rng.random(len(train)) < LEAK_NOISE
dirty = train.copy()
# Insert it mid-table; appending it as the last column would be too obvious.
dirty.insert(11, LEAK_COL, np.where(flip, 1 - y, y))

# ---------------------------------------------------------------- 4. Write the task statement
# Rule: give the CONTEXT a real practitioner would have, never a HINT that
# points at the defect.
#   context = what each column means, when/how the model will be used
#   hint    = e.g. leaving the suspicious column out of the dictionary
# LEAK_COL is documented with a neutral, truthful description. The agent has to
# reason about WHEN that value comes into existence relative to prediction time.
TASK = """# Task: clean a training dataset

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
"""

# ---------------------------------------------------------------- 5. Save
Path("task").mkdir(exist_ok=True)
Path("hidden").mkdir(exist_ok=True)
dirty.to_csv("task/dirty.csv", index=False)
Path("task/TASK.md").write_text(TASK, encoding="utf-8")
train.to_csv("hidden/clean_train.csv", index=False)
holdout.to_csv("hidden/holdout.csv", index=False)
Path("hidden/answer.json").write_text(json.dumps({
    "corruption": "target_leakage",
    "leak_column": LEAK_COL,
    "expected_fix": f"Drop column '{LEAK_COL}'. No other rows or columns change.",
    "n_rows": len(dirty),
    "original_columns": COLS,
}, indent=2), encoding="utf-8")

print(f"Task written: task/dirty.csv ({dirty.shape[0]} rows x {dirty.shape[1]} cols)")
print(f"Injected defect: '{LEAK_COL}' (matches the target in {1 - LEAK_NOISE:.0%} of rows)")
print("Show the agent the task/ folder only.")