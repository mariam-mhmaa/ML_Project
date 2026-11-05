# Regression artifacts

This folder holds locally generated fitted regression pipelines. To recreate the
Milestone 1 baseline, run from the repository root using the environment described
in the [main README](../../README.md):

```powershell
python src/regression/train_baseline.py --save-artifact
```

The resulting `linear_regression_baseline.joblib` contains input cleanup, fitted
imputation/scaling/encoding, and the untuned regressor. It is fitted on eligible
shared training rows after cross-validation and is a preliminary baseline.

The [regression note](../../docs/regression_milestone1.md#reproduction) provides the
input contract and loading example. Run loading code from the repository root.

Keep this guide in Git. Fitted `.joblib` files are ignored by Git and must also be
excluded from the submission ZIP; regenerate them locally using the command above.
