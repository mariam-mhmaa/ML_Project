# Milestone 1 report assembly and submission checklist

This checklist is the Member 5 handoff for assembling the supplied team sections
and preparing the Milestone 1 repository ZIP. It does not invent missing member
details or silently mark incomplete work as finished.

## Report assembly order

1. Project objective and H2 dataset scope.
2. Team member names, university IDs, GitHub usernames, and contribution summary.
3. Data understanding, cleaning, EDA, and shared split strategy.
4. Regression Milestone 1 section from `docs/regression_milestone1.md`.
5. Classification Milestone 1 section from `docs/classification_milestone1.md`.
6. Clustering Milestone 1 section from
   `docs/clustering_milestone1_explanation.md`.
7. Combined preliminary leakage audit covering all three tasks.
8. Initial results and limitations.
9. Milestone 2 plan: remaining models, tuning, PCA/LDA, final evaluation,
   application, and saved artifacts.

Each member remains responsible for the accuracy of their supplied section.
Assembly means combining and formatting those sections, not writing unsupported
claims on another member's behalf.

## Fresh-setup commands

Run from the repository root using 64-bit Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe src/preprocessing/classification/prepare_classification_data.py
.\.venv\Scripts\python.exe src/classification/audit_training_data.py
.\.venv\Scripts\python.exe src/classification/run_milestone1_baseline.py
.\.venv\Scripts\python.exe src/preprocessing/regression/prepare_regression_data.py
.\.venv\Scripts\python.exe src/regression/audit_training_data.py
.\.venv\Scripts\python.exe src/regression/train_baseline.py
.\.venv\Scripts\python.exe src/clustering/run_milestone1_baseline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The shared preprocessing and split outputs are already tracked. If the team wants
to regenerate them too, run the shared commands in README steps 1-7 before the
task-specific commands above.

## Required pre-ZIP checks

- [ ] Confirm the repository URL in `README.md` is correct.
- [ ] Insert all member names, IDs, GitHub usernames, and contribution summaries
  into the submitted progress report.
- [ ] Combine the supplied member sections in the report order above.
- [ ] Resolve any conflicting prediction-point or leakage statements across
  sections instead of preserving contradictions.
- [ ] Run the fresh-setup commands and confirm every command exits successfully.
- [ ] Confirm regression, classification, and clustering results in the report
  match the tracked output files.
- [ ] Confirm the clustering run states that `test.csv` was not used.
- [ ] Confirm `git status --short` contains only intended submission changes.
- [ ] Commit the submitted version, then place the actual output of
  `git rev-parse HEAD` in the submission copy of the README/report.
- [ ] Create the ZIP from that committed version.

## ZIP include/exclude rules

Include source code, tests, documentation/report, README, requirements, tracked
input/reference data, and result tables/figures required to verify Milestone 1.

Exclude:

- `.git/`;
- `.venv/`, `.python-runtime/`, and other local environments;
- `__pycache__/`, `.pytest_cache/`, and tool caches;
- fitted `.joblib`, `.pkl`, or other serialized model/pipeline artifacts;
- temporary files, editor state, and unrelated downloads.

Do not create the final ZIP before the member information, report assembly, final
commit, and submitted commit SHA are complete.
