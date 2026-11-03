"""Split H2.csv into a training set and a test set. Requires pandas and scikit-learn.

Run from the project folder:
    python src/preprocessing/split_data.py

Strategy: train/test split followed by cross-validation during modeling.
- Keep all rows: identical records may be different bookings.
- Keep each exact-duplicate group entirely in training or testing.
- Hold out one of five stratified group folds for testing (about 20%),
  using IsCanceled and random_state=42.
- The same split is used for regression (ADR) and classification (IsCanceled).
- Model selection and tuning should use grouped 5-fold cross-validation on
  train.csv so matching model inputs stay in one fold; this is not implemented here.
- test.csv is used once, for the final evaluation of the chosen models.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# To keep splits reproducible (number is arbitrary)
RANDOM_STATE = 42
N_SPLITS = 5

project_folder = Path(__file__).resolve().parents[2]
csv_path = project_folder / "data" / "raw" / "H2.csv"
output_folder = project_folder / "data" / "processed"

# Read everything as text so the saved files keep the raw values exactly.
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
groups = df.groupby(df.columns.tolist(), sort=False).ngroup()
splitter = StratifiedGroupKFold(
    n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
)
train_idx, test_idx = next(splitter.split(df, df["IsCanceled"], groups))
train, test = df.iloc[train_idx], df.iloc[test_idx]

output_folder.mkdir(parents=True, exist_ok=True)
train.to_csv(output_folder / "train.csv", index=False)
test.to_csv(output_folder / "test.csv", index=False)

for name, part in [("Train", train), ("Test", test)]:
    cancel_rate = (part["IsCanceled"] == "1").mean()
    print(f"{name}: {len(part):,} rows, {cancel_rate:.1%} canceled")
