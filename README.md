# MLE-Debug

**Can AI coding agents find what is silently wrong in an ML pipeline, and leave alone what is not?**

MLE-Debug is a benchmark and (work in progress) RL environment for AI agents on ML engineering tasks where nothing crashes and the metrics look fine, but the result is wrong: target leakage, bad splits, corrupted training data. Tasks are generated programmatically, run without a human in the loop, and graded automatically against a hidden ground truth.

> Status: early prototype. One task family, one dataset, a handful of runs. The numbers below are preliminary and the caveats are listed next to them.

## Why

Existing ML-agent benchmarks mostly measure how well an agent can push a score up on a clean, well-posed problem. A large share of real ML engineering is the opposite: noticing that a great validation score is too good, that a column could not have existed at prediction time, that a "cleaning" step quietly deleted 6% of the data.

These defects share three properties that make them a good test of judgment rather than coding:

- **No error is raised.** The code runs and the metrics often improve.
- **They are not visible in the data alone.** A leaked column is indistinguishable from an excellent feature unless you reason about when it comes into existence relative to prediction time.
- **The right action is often to do nothing.** An agent that "fixes" real missing values or plausible duplicates does damage.

## How a task works

```
make_task  ->  task/    dirty.csv + TASK.md        (all the agent sees)
               hidden/  clean data, holdout, answer (grader only)

run_agent  ->  fresh workspace outside the repo, agent runs headless,
               outputs + full trajectory saved to runs/<model>_<k>/

score      ->  pass/fail on four checks, one row appended to runs/results.csv
```

The agent receives a dataset and a task statement containing what a real practitioner would have: where the data came from, what each column means, and when the model will be used. It is told the data may have quality problems. It is **not** told which, where, or how many.

### Current task: target leakage (`tracks/leakage`)

- Base data: UCI Adult (48,842 rows). 80/20 stratified split made *before* any corruption.
- Injected defect: a column `tax_review_flag` that matches the label in 95% of training rows and does not exist at prediction time. It is documented in the data dictionary with a neutral, truthful description.
- Effect: holdout accuracy drops from 0.876 (clean) to 0.819 (leak left in).

A submission passes only if **all four** hold:

1. the leakage column is removed
2. every original column is still present
3. at least 98% of rows are kept
4. holdout accuracy is within 1 point of the clean baseline

Checks 2 and 3 exist because fixing the defect is half the job. Not breaking everything else is the other half.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas scikit-learn

cd tracks/leakage
python make_task_leak.py                   # writes task/ and hidden/
python run_agent.py --model sonnet --n 3   # requires Claude Code CLI on PATH
python score_leak.py                       # re-grades everything under runs/
```

`run_agent.py` currently drives Claude Code in non-interactive mode. Support for a standard, model-agnostic harness is the next milestone (see Roadmap).

## Preliminary results

| Model | Runs | Passed | Leak removed | Rows kept | Holdout acc. |
|---|---|---|---|---|---|
| Claude Sonnet | 3 | 3/3 | 3/3 | 99.9 to 100% | 0.877 to 0.878 |
| Claude Haiku 4.5 | 1* | 0/1 | 0/1 | 94.2% | 0.817 |

\* Run interactively under different conditions from the Sonnet runs. Not a like-for-like comparison yet.

What the reports showed:

- **The failing run never considered leakage.** It ran a textbook checklist (duplicates, missing values, ranges), deleted 2,203 rows of genuine survey non-response on the grounds that they were "data corruption", and ended with the same accuracy as doing nothing.
- **The passing runs reasoned about time.** They found the statistical anomaly, then cited the deployment setting from the task statement: a tax review can only happen after income is filed, so the value cannot exist when the survey is completed.
- **The passing runs also showed restraint.** They inspected the same missing values, judged them real, and left them alone.

## Known limitations

- Tiny sample: one task, two models from one vendor, four runs.
- The task was drafted with the help of a Claude model and then solved by Claude models. Same-family bias is possible and untested.
- Too easy at the top: a mid-tier model solves it 3/3, so it cannot separate frontier models as is.
- Isolation is a temp directory, not a sandbox. Trajectories are scanned for access to grader files, which detects but does not prevent.
- Runs used a personally configured agent CLI, so conditions are not fully controlled.
- Baselines shift slightly across machines (0.8764 vs 0.8771 with identical code and seed). Pinned containers will fix this.

## Design notes

Lessons from building the first task, which shape everything that follows:

1. **Downstream accuracy alone is a poor reward.** In an earlier version with three defects, removing the leak alone restored full accuracy; duplicates and unit errors barely moved the metric. Grading needs restoration checks against ground truth, not just a model score.
2. **Give context, never hints.** The first draft left the leaked column out of the data dictionary, which made the task solvable by diffing column names. Conversely, giving only X and y makes leakage undecidable. The rule: include what a practitioner would know, exclude anything that points at the defect.
3. **Some corruptions are unrecoverable.** Dividing weekly hours by 5 turns 40 into 8, which is also a legitimate value. Tasks must have an achievable optimum, and the grader must know what it is.
4. **The base dataset has its own quirks.** Adult contains natural near-duplicates. Thresholds have to tolerate defensible choices while still catching over-deletion.

## Roadmap

- [ ] Port tasks to a standard agent-eval harness with containerised isolation
- [ ] Oracle and no-op checks for every task (reference solution passes, untouched input fails)
- [ ] Like-for-like reruns, plus at least one non-Anthropic model
- [ ] **No-defect control**: clean data, where the correct action is to change nothing
- [ ] Difficulty ladder: weaker leaks, decoy columns that look suspicious but are legitimate, leaks disguised as continuous features, under-specified settings where the right move is to ask
- [ ] Further tracks: split errors, dirty data, training bugs
- [ ] A domain-specific track with defects that require domain knowledge to recognise
- [ ] Task generator with seeds and difficulty parameters, usable as an RL environment

## Related work

Agent benchmarks for ML engineering: MLE-bench, MLE-Dojo, MLAgentBench, MLGym, RE-Bench, PaperBench.
Data cleaning and preparation: CleanML, REIN, dcbench, and more recent agent-focused work including *Exploring LLM Agents for Cleaning Tabular Machine Learning Datasets* (arXiv:2503.06664), PrepBench, CDR-Bench and DataGovBench.
Task and harness design: SWE-bench and SWE-bench Verified, SWE-smith, Terminal-Bench.

A proper comparison with the data-cleaning line of work is pending; this section will be updated once that reading is done.

## Layout

```
tracks/leakage/
  make_task_leak.py   task generator
  score_leak.py       grader (importable: from score_leak import grade)
  run_agent.py        headless runner + results logging
  task/               what the agent sees
  hidden/             grader-only files
  runs/               one folder per run, plus results.csv
notes/                dated lab notes
```

## License

TBD
