"""
score_leak.py - Grade submissions for the target-leakage task.

Run from the folder that contains task/, hidden/ and runs/:
  python score_leak.py                          # grades every runs/*/cleaned.csv
  python score_leak.py runs/haiku_1/cleaned.csv # grades specific files

A submission PASSES only if all four checks hold:
  1. the leakage column was removed
  2. every original column is still present
  3. at least 98% of the rows were kept
  4. holdout accuracy is within 1 point of the clean baseline

Other scripts can reuse the logic:  from score_leak import grade
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

MIN_ROW_RETENTION = 0.98
MAX_ACC_DROP = 0.01
TARGET = "class"

ROOT = Path(__file__).resolve().parent      # paths work from any cwd
HID = ROOT / "hidden"
ANSWER = json.loads((HID / "answer.json").read_text(encoding="utf-8"))
LEAK = ANSWER["leak_column"]
ORIG_COLS = ANSWER["original_columns"]
N_ROWS = ANSWER["n_rows"]


def _norm(s: pd.Series) -> pd.Series:
    # Do not punish harmless string normalisation (case, whitespace).
    return s.astype("string").str.strip().str.lower()


def holdout_accuracy(train: pd.DataFrame, holdout: pd.DataFrame) -> float:
    """Train a fixed model on `train`, return accuracy on the clean holdout."""
    train = train[train[TARGET].notna()]
    feats = [c for c in train.columns if c != TARGET]
    Xtr = train[feats].copy()
    # A column the model was trained on but that does not exist at prediction
    # time is filled with 0 ("nobody has been flagged yet").
    Xte = holdout.reindex(columns=feats, fill_value=0).copy()
    for c in feats:
        if not (pd.api.types.is_numeric_dtype(Xtr[c])
                and pd.api.types.is_numeric_dtype(Xte[c])):
            a, b = _norm(Xtr[c]), _norm(Xte[c])
            cats = pd.Index(a.dropna().unique())
            Xtr[c] = pd.Series(pd.Categorical(a, categories=cats).codes,
                               index=Xtr.index).replace(-1, np.nan)
            Xte[c] = pd.Series(pd.Categorical(b, categories=cats).codes,
                               index=Xte.index).replace(-1, np.nan)
    ytr = _norm(train[TARGET]).str.contains(">").astype(int)
    yte = _norm(holdout[TARGET]).str.contains(">").astype(int)
    model = HistGradientBoostingClassifier(random_state=0).fit(Xtr, ytr)
    return float(model.score(Xte, yte))


@lru_cache(maxsize=1)
def baselines() -> dict:
    """Reference accuracies. Cached because they never change within a run."""
    holdout = pd.read_csv(HID / "holdout.csv")
    return {
        "clean": holdout_accuracy(pd.read_csv(HID / "clean_train.csv"), holdout),
        "dirty": holdout_accuracy(pd.read_csv(ROOT / "task" / "dirty.csv"), holdout),
    }


def grade(path) -> dict:
    """Grade one cleaned.csv. Returns a flat dict that is easy to log."""
    path = Path(path)
    out = {"file": str(path), "leak_removed": False, "cols_kept": False,
           "row_retention": 0.0, "accuracy": None, "passed": False,
           "missing_cols": "", "new_cols": "", "error": ""}
    if not path.exists():
        out["error"] = "no cleaned.csv produced"
        return out
    try:
        sub = pd.read_csv(path)
    except Exception as e:
        out["error"] = f"unreadable csv: {type(e).__name__}"
        return out

    missing = [c for c in ORIG_COLS if c not in sub.columns]
    extra = [c for c in sub.columns if c not in ORIG_COLS and c != LEAK]
    out["leak_removed"] = LEAK not in sub.columns
    out["cols_kept"] = not missing
    out["row_retention"] = round(len(sub) / N_ROWS, 4)
    out["missing_cols"] = ";".join(missing)
    out["new_cols"] = ";".join(extra)

    if TARGET in sub.columns:
        holdout = pd.read_csv(HID / "holdout.csv")
        try:
            # New columns the agent invented cannot be computed for the holdout.
            out["accuracy"] = round(
                holdout_accuracy(sub.drop(columns=extra), holdout), 4)
        except Exception as e:
            out["error"] = f"could not train on submission: {type(e).__name__}"

    out["passed"] = bool(
        out["leak_removed"] and out["cols_kept"]
        and out["row_retention"] >= MIN_ROW_RETENTION
        and out["accuracy"] is not None
        and out["accuracy"] >= baselines()["clean"] - MAX_ACC_DROP)
    return out


def _print(r: dict) -> None:
    mark = lambda ok: "ok" if ok else "XX"
    acc_ok = (r["accuracy"] is not None
              and r["accuracy"] >= baselines()["clean"] - MAX_ACC_DROP)
    print(f"\n=== {r['file']} ===")
    if r["error"]:
        print(f"  error: {r['error']}")
    print(f"  [{mark(r['leak_removed'])}] 1. leak column '{LEAK}' removed")
    print(f"  [{mark(r['cols_kept'])}] 2. all original columns kept")
    print(f"  [{mark(r['row_retention'] >= MIN_ROW_RETENTION)}] "
          f"3. row retention >= {MIN_ROW_RETENTION:.0%}  ({r['row_retention']:.1%})")
    print(f"  [{mark(acc_ok)}] 4. accuracy within {MAX_ACC_DROP:.0%} of clean "
          f"baseline  ({r['accuracy']})")
    if r["missing_cols"]:
        print(f"  missing original columns: {r['missing_cols']}")
    if r["new_cols"]:
        print(f"  new columns (ignored for accuracy): {r['new_cols']}")
    print(f"  VERDICT: {'PASS' if r['passed'] else 'FAIL'}")


def main():
    b = baselines()
    print(f"baseline, clean training data : {b['clean']:.4f}  (upper reference)")
    print(f"baseline, dirty data untouched: {b['dirty']:.4f}  (lower reference)")
    paths = sys.argv[1:] or sorted((ROOT / "runs").glob("*/cleaned.csv"))
    if not paths:
        print("\nnothing to grade: no runs/*/cleaned.csv found")
    for p in paths:
        _print(grade(p))


if __name__ == "__main__":
    main()