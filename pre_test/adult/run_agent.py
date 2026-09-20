"""
run_agent.py - Run the agent on the task N times, save everything, grade, log.

Run from anywhere (with the project venv active so the agent has pandas):
  python run_agent.py --model haiku            # one run
  python run_agent.py --model haiku --n 3      # three runs
  python run_agent.py --model sonnet --n 3
  python run_agent.py --model claude-haiku-4-5 # full model names also work

For every run this script:
  1. creates a fresh workspace in a temp folder OUTSIDE the project and copies
     only task/dirty.csv and task/TASK.md into it
  2. runs Claude Code there in non-interactive mode (no human in the loop)
  3. saves the agent's files and its full trajectory to runs/<model>_<k>/
  4. grades cleaned.csv and appends one row to runs/results.csv

NOTE ON ISOLATION: the temp workspace means hidden/ is not next to the agent,
but the agent still has a shell and could reach any path on this machine.
This is "honour system plus audit", not a sandbox. Every trajectory is scanned
for references to grader files and flagged in results.csv. Real isolation
needs a container; that is the next step.
"""
import argparse
import csv
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

from score_leak import LEAK, ROOT, grade

PROMPT = "Read TASK.md and complete the task."
TASK_FILES = ["dirty.csv", "TASK.md"]
ALLOWED_TOOLS = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
MAX_TURNS = 60
TIMEOUT_S = 30 * 60
# If any of these strings shows up in the trajectory, the agent touched (or
# tried to touch) grader-only material.
FORBIDDEN = ["answer.json", "holdout.csv", "clean_train.csv", "score_leak",
             "make_task_leak", str(ROOT)]

RUNS = ROOT / "runs"
RESULTS = RUNS / "results.csv"
FIELDS = ["timestamp", "run", "model", "passed", "leak_removed", "cols_kept",
          "row_retention", "accuracy", "report_mentions_leak",
          "touched_grader_files", "duration_s", "exit_code", "missing_cols",
          "new_cols", "error"]


def next_run_dir(model: str) -> Path:
    tag = model.replace("/", "-")
    k = 1
    while (RUNS / f"{tag}_{k}").exists():
        k += 1
    d = RUNS / f"{tag}_{k}"
    d.mkdir(parents=True)
    return d


def run_once(model: str) -> dict:
    run_dir = next_run_dir(model)
    work = Path(tempfile.mkdtemp(prefix="cleanbench_"))
    for f in TASK_FILES:
        shutil.copy(ROOT / "task" / f, work / f)

    cmd = ["claude", "-p", PROMPT,
           "--model", model,
           "--output-format", "stream-json", "--verbose",
           "--max-turns", str(MAX_TURNS),
           "--allowedTools", *ALLOWED_TOOLS]

    print(f"[{run_dir.name}] running in {work} ...", flush=True)
    t0 = time.time()
    traj_path = run_dir / "trajectory.jsonl"
    with open(traj_path, "w", encoding="utf-8") as traj:
        try:
            proc = subprocess.run(cmd, cwd=work, stdout=traj,
                                  stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code = "timeout"
    duration = round(time.time() - t0)

    # Keep everything the agent produced; skip the big input file.
    for p in work.rglob("*"):
        if p.is_file() and p.name != "dirty.csv":
            dest = run_dir / p.relative_to(work)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
    shutil.rmtree(work, ignore_errors=True)

    result = grade(run_dir / "cleaned.csv")
    report = run_dir / "REPORT.md"
    trajectory = traj_path.read_text(encoding="utf-8", errors="ignore")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run": run_dir.name,
        "model": model,
        "report_mentions_leak": report.exists() and LEAK in report.read_text(
            encoding="utf-8", errors="ignore"),
        "touched_grader_files": ";".join(s for s in FORBIDDEN if s in trajectory),
        "duration_s": duration,
        "exit_code": exit_code,
        **{k: result[k] for k in ["passed", "leak_removed", "cols_kept",
                                  "row_retention", "accuracy", "missing_cols",
                                  "new_cols", "error"]},
    }

    new_file = not RESULTS.exists()
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)

    print(f"[{run_dir.name}] {'PASS' if row['passed'] else 'FAIL'}  "
          f"leak_removed={row['leak_removed']}  rows={row['row_retention']:.1%}  "
          f"acc={row['accuracy']}  {duration}s"
          + (f"  !! touched: {row['touched_grader_files']}"
             if row["touched_grader_files"] else ""), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="alias (haiku, sonnet, opus) or a full model name")
    ap.add_argument("--n", type=int, default=1, help="number of runs")
    args = ap.parse_args()

    if shutil.which("claude") is None:
        raise SystemExit("`claude` not found on PATH")
    RUNS.mkdir(exist_ok=True)
    rows = [run_once(args.model) for _ in range(args.n)]
    passed = sum(r["passed"] for r in rows)
    print(f"\n{args.model}: {passed}/{len(rows)} passed. "
          f"Details in {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()