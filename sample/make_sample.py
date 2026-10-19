"""Materialise a sample of what the agent sees, plus what it does not.

    python3 make_sample.py              # the anonymised task (the default)
    python3 make_sample.py --labelled   # the same data with its identity intact

Everything here is regenerable in seconds and safe to delete.
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from lab import movielens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--labelled", action="store_true",
                help="build the non-anonymised version instead")
args = ap.parse_args()

task = movielens(anonymize=not args.labelled, seed=773)

seen = HERE / "agent-sees"
hidden = HERE / "withheld"
for d in (seen, hidden):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)

# --- exactly what lands in the agent's working directory ----------------------------
for f in sorted(Path(task.workdir).iterdir()):
    shutil.copy(f, seen / f.name)

# --- and exactly what we keep back ---------------------------------------------------
for split in ("dev", "test"):
    df = getattr(task, split)[task.keys + [task.target]]
    df.to_csv(hidden / f"{split}_answers.csv", index=False)

print(f"task: {task.name}\n")
print("agent-sees/")
for f in sorted(seen.iterdir()):
    print(f"   {f.name:22s} {f.stat().st_size / 1e6:6.1f} MB")
print("\nwithheld/   (never written during a real run - these live only in memory)")
for f in sorted(hidden.iterdir()):
    print(f"   {f.name:22s} {f.stat().st_size / 1e6:6.1f} MB")

print(f"\nThe gap between the two is the task: train.csv has "
      f"{', '.join(task.train_columns)},\nwhile the pairs files have only "
      f"{', '.join(task.pair_columns)}.")
