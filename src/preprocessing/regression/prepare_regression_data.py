"""Create the deterministic regression training view; never fit preprocessing here."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.regression.preprocessing import (
    PREPARED_OUTPUT, load_training_frame, prepare_training_frame,
)


def main():
    raw = load_training_frame()
    prepared, exclusions = prepare_training_frame(raw)
    PREPARED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output = PROJECT_ROOT / "outputs/regression"
    output.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PREPARED_OUTPUT, index=False)
    exclusions.to_csv(output / "preparation_exclusions.csv", index=False)
    print(f"Regression: {len(prepared):,} rows retained; {len(exclusions):,} zero-night rows excluded.")
    print(f"Saved {PREPARED_OUTPUT.relative_to(PROJECT_ROOT)}")
    print("Missing Children and Country remain missing for fold-local imputation.")


if __name__ == "__main__":
    main()
