"""Play the agent's part yourself.

Edit `predict` below, then run:

    python3 try_it.py

It rebuilds the exact same task the agent gets (same seed, same split), scores your
predictions the same way `score` does, and keeps your best attempt as the submission -
all through the same machinery the notebook uses.

    python3 try_it.py --final     # also score the submission on test, once

Some rungs to try, roughly in order of how well they do:

    return train[TARGET].mean()                                     # 1.1188
    return pairs[COL].map(train.groupby(COL)[TARGET].mean())        # 0.9769
    ... then biases, then something low-rank.

The reference ladder is in ../Supporting Material/measure_ladder.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lab import movielens, Scorer  # noqa: E402

task = movielens(anonymize=True, seed=773)
ROW, COL, TARGET = task.keys[0], task.keys[1], task.target


# ---------------------------------------------------------------------------------
# EDIT ME. Return one prediction per row of `pairs`, in order.
# ---------------------------------------------------------------------------------
def predict(train: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    item_mean = train.groupby(COL)[TARGET].mean()
    return pairs[COL].map(item_mean).fillna(train[TARGET].mean()).to_numpy()


# ---------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true",
                    help="also score the submission on test (do this once)")
    args = ap.parse_args()

    scorer = Scorer(task)
    work = Path(task.workdir)
    train = pd.read_csv(work / "train.csv")

    # Exactly the contract the agent is held to: predict both, every attempt.
    for split, out in (("dev", "predictions.csv"), ("test", "test_predictions.csv")):
        pairs = pd.read_csv(work / f"{split}_pairs.csv")
        pairs = pairs.assign(prediction=predict(train, pairs))
        pairs[task.keys + ["prediction"]].to_csv(work / out, index=False)

    print(scorer.score())

    if args.final:
        print()
        scorer.final()


if __name__ == "__main__":
    main()
