"""Split H2.csv into a training set and a test set. Requires pandas and scikit-learn.

Run from the project folder:
    python src/preprocessing/split_data.py

Strategy: train/test split combined with cross-validation.
- Exact duplicate rows are dropped first, so no copy of a test row is in training.
- 80% train / 20% test, stratified on IsCanceled, random_state=42.
- The same split is used for regression (ADR) and classification (IsCanceled).
- Model selection and tuning use 5-fold cross-validation on train.csv only.
- test.csv is used once, for the final evaluation of the chosen models.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# To keep splits reproducible (number is arbitrary)
RANDOM_STATE = 42
TEST_SIZE = 0.2

project_folder = Path(__file__).resolve().parents[2]
csv_path = project_folder / "data" / "raw" / "H2.csv"
output_folder = project_folder / "data" / "processed"

# Read everything as text so the saved files keep the raw values exactly.
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
df = df.drop_duplicates()

train, test = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df["IsCanceled"],
    random_state=RANDOM_STATE,
)

output_folder.mkdir(parents=True, exist_ok=True)
train.to_csv(output_folder / "train.csv", index=False)
test.to_csv(output_folder / "test.csv", index=False)

for name, part in [("Train", train), ("Test", test)]:
    cancel_rate = (part["IsCanceled"] == "1").mean()
    print(f"{name}: {len(part):,} rows, {cancel_rate:.1%} canceled")
