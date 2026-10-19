"""The one place a number comes from — and the one place the deliverable is kept.

The held-out answers never touch disk. They live on the Task, in this process, and this is the
only thing that reads them. The agent writes predictions; this reads them, lines them up against
the answers, and returns a single number.

Two design decisions here, both of them lessons paid for the hard way:

**score captures the submission.** Every attempt writes predictions for *both* the dev pairs and
the test pairs. Whenever an attempt turns out to be the best so far, this class quietly keeps a
copy of that attempt's test predictions, outside the agent's working directory where it cannot
be overwritten. The agent never decides when to submit. Earlier versions asked it to write the
final answer "when you are done", which failed twice over: one run never judged itself done and
so produced nothing at all, and another lost track of which model had been its best and shipped a
worse one. Coupling the deliverable to the stopping decision was the mistake; this removes it.

**score is a tool call**, so the agent cannot invoke it from inside its own Python. One
configuration per turn. That rules out hyperparameter sweeps, and enforces - by accident - the
discipline of changing one thing at a time and measuring it.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DEV_FILE = "predictions.csv"
TEST_FILE = "test_predictions.csv"


class Scorer:
    def __init__(self, task):
        self.task = task
        self.trials: list[dict] = []
        self.submission: dict | None = None          # {attempt, score, path}
        self._vault = Path(tempfile.mkdtemp(prefix="hodl_submission_"))

    # ---------------------------------------------------------------- the tool

    def score(self) -> str:
        """Score predictions.csv, and keep this attempt's submission if it is the best."""
        result = self._evaluate(DEV_FILE, "dev")
        if isinstance(result, str):
            return result

        m = self.task.metric
        attempt = len(self.trials) + 1
        self.trials.append({"attempt": attempt, m.name.lower(): round(result, 4)})

        note = self._capture(attempt, result)
        # Record what actually happened, not what the score implies: an attempt can be the
        # best one and still submit nothing, if it forgot to write its test predictions.
        self.trials[-1]["submitted"] = (self.submission or {}).get("attempt") == attempt

        line = (f"dev {m.name} = {result:.4f}   "
                f"(attempt {attempt}, best so far {self.best:.4f})")
        return line + note

    def _capture(self, attempt: int, score: float) -> str:
        """Keep this attempt's test predictions if it is the best we have seen."""
        if self.submission is not None and not self.task.metric.better(
                score, self.submission["score"]):
            return ""

        src = Path(self.task.workdir) / TEST_FILE
        if not src.exists():
            return (f"\n  NOTE: your best attempt so far, but {TEST_FILE} is missing - nothing "
                    f"was saved. Write it alongside {DEV_FILE} on every attempt.")

        dest = self._vault / f"attempt_{attempt:03d}.csv"
        shutil.copy(src, dest)
        self.submission = {"attempt": attempt, "score": score, "path": dest}
        return f"\n  saved as your submission (attempt {attempt})"

    # ---------------------------------------------------------------- our side only

    @property
    def scores(self) -> list[float]:
        key = self.task.metric.name.lower()
        return [t[key] for t in self.trials]

    @property
    def best(self) -> float:
        return self.task.metric.best_of(self.scores) if self.trials else float("nan")

    def final(self) -> None:
        """Score the captured submission. Test is touched exactly once: here."""
        m = self.task.metric
        print(f"best dev {m.name:<12} {self.best:.4f}   (over {len(self.trials)} attempts)")

        if self.submission is None:
            print(f"\nNo submission was ever captured - no attempt wrote {TEST_FILE}.")
            return

        sub = self.submission
        out = self._evaluate(sub["path"], "test")
        print(f"submitted from      attempt {sub['attempt']}, which scored "
              f"{sub['score']:.4f} on dev")
        if isinstance(out, str):
            print(f"test {m.name:<12} {out}")
            return
        print(f"test {m.name:<12} {out:.4f}")
        # Optimism = how much WORSE test is than dev, whichever direction the metric runs.
        gap = (out - sub["score"]) if m.lower_is_better else (sub["score"] - out)
        print(f"optimism{'':<13}{gap:+.4f}  "
              f"({'test is worse than dev' if gap > 0 else 'test held up'})")
        print("\nThe submission is the best-scoring attempt by construction, so this gap is "
              "the real\nthing: a score you selected on is not an honest estimate of anything.")

    def leaderboard(self) -> pd.DataFrame:
        lb = pd.DataFrame(self.trials)
        if len(lb):
            m = self.task.metric
            col = m.name.lower()
            lb["best_so_far"] = lb[col].cummin() if m.lower_is_better else lb[col].cummax()
            lb["submitted"] = np.where(lb.get("submitted", False), "yes", "")
        return lb

    # ---------------------------------------------------------------- internals

    def _evaluate(self, filename, split: str):
        task = self.task
        path = Path(filename)
        if not path.is_absolute():
            path = Path(task.workdir) / filename
        if not path.exists():
            return f"ERROR: {path.name} does not exist. Write it first."
        try:
            sub = pd.read_csv(path)
        except Exception as e:
            return f"ERROR: could not read {path.name}: {e}"

        needed = set(task.keys) | {"prediction"}
        if not needed.issubset(sub.columns):
            return (f"ERROR: {path.name} needs columns {task.prediction_columns} - "
                    f"found {list(sub.columns)}")

        pairs = task.pairs(split)
        merged = pairs.merge(sub.drop_duplicates(task.keys), on=task.keys, how="left")
        missing = int(merged.prediction.isna().sum())
        if missing:
            return (f"ERROR: {missing:,} of {len(pairs):,} rows in the {split} pairs have no "
                    f"prediction. Predict every row.")

        preds = merged.prediction.to_numpy()
        try:
            preds = preds.astype(float)      # numeric targets, the common case
        except (TypeError, ValueError):
            pass                             # categorical labels - hand them over as-is
        return task.metric(task.y(split), preds)
