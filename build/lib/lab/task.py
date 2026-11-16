"""What the agent is actually asked to do, described abstractly enough to swap.

A Task is a held-out prediction problem:

    keys        identifier columns, present in train and in the pairs files
    features    columns usable at prediction time - in train and in the pairs files
    train_only  columns present in train but NOT in the pairs files
    target      the column to predict
    metric      how a prediction is scored

Everything the prompt needs to say about the data is derived from those five things, so
pointing this at a different dataset is a new Task, not a rewritten prompt.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- metric

@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    lower_is_better: bool = True
    clip: tuple[float, float] | None = None

    def __call__(self, y, yhat) -> float:
        y, yhat = np.asarray(y), np.asarray(yhat)
        if self.clip is not None:
            yhat = np.clip(yhat.astype(float), *self.clip)
        return float(self.fn(y, yhat))

    @property
    def direction(self) -> str:
        return "Lower is better." if self.lower_is_better else "Higher is better."

    def better(self, a: float, b: float) -> bool:
        return a < b if self.lower_is_better else a > b

    def best_of(self, values: Sequence[float]) -> float:
        return min(values) if self.lower_is_better else max(values)


def _num(a):
    return np.asarray(a, dtype=float)


def rmse(clip: tuple[float, float] | None = None) -> Metric:
    return Metric("RMSE", lambda y, p: np.sqrt(np.mean((_num(y) - _num(p)) ** 2)), True, clip)


def mae(clip: tuple[float, float] | None = None) -> Metric:
    return Metric("MAE", lambda y, p: np.mean(np.abs(_num(y) - _num(p))), True, clip)


def accuracy() -> Metric:
    """Exact match. Works for numeric classes and for string labels alike."""
    def fn(y, p):
        if np.issubdtype(np.asarray(y).dtype, np.number):
            return np.mean(np.round(_num(p)) == _num(y))
        return np.mean(np.asarray(p).astype(str) == np.asarray(y).astype(str))
    return Metric("accuracy", fn, False, None)


# --------------------------------------------------------------------------- task

@dataclass
class Task:
    name: str
    blurb: str                       # one domain-neutral sentence for the prompt
    keys: list[str]
    target: str
    metric: Metric
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    workdir: Path
    features: list[str] = field(default_factory=list)
    train_only: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ construction

    @classmethod
    def build(cls, frame: pd.DataFrame, *, name: str, blurb: str, keys: Sequence[str],
              target: str, metric: Metric, features: Sequence[str] = (),
              train_only: Sequence[str] = (), notes: Sequence[str] = (),
              seed: int = 773, split: tuple[float, float] = (0.8, 0.1),
              side_info: pd.DataFrame | None = None) -> "Task":
        perm = np.random.default_rng(seed).permutation(len(frame))
        n_tr = int(split[0] * len(frame))
        n_dv = int(split[1] * len(frame))
        bounds = ((0, n_tr), (n_tr, n_tr + n_dv), (n_tr + n_dv, len(frame)))
        parts = [frame.iloc[perm[a:b]].reset_index(drop=True) for a, b in bounds]

        self = cls(name=name, blurb=blurb, keys=list(keys), target=target, metric=metric,
                   train=parts[0], dev=parts[1], test=parts[2],
                   workdir=Path(tempfile.mkdtemp(prefix=f"autoresearch_{name}_")),
                   features=list(features), train_only=list(train_only), notes=list(notes))
        self._write_agent_files(side_info)
        return self

    def _write_agent_files(self, side_info: pd.DataFrame | None) -> None:
        self.train[self.train_columns].to_csv(self.workdir / "train.csv", index=False)
        for split in ("dev", "test"):
            df = getattr(self, split)[self.pair_columns]
            df.to_csv(self.workdir / f"{split}_pairs.csv", index=False)
        if side_info is not None:
            side_info.to_csv(self.workdir / "side_info.csv", index=False)

    # ------------------------------------------------------------------ column sets

    @property
    def pair_columns(self) -> list[str]:
        """What the agent gets at prediction time."""
        return self.keys + self.features

    @property
    def train_columns(self) -> list[str]:
        return self.keys + self.features + self.train_only + [self.target]

    @property
    def prediction_columns(self) -> str:
        return ",".join(self.keys + ["prediction"])

    # ------------------------------------------------------------------ held-out answers

    def y(self, split: str) -> np.ndarray:
        """The held-out answers. Kept in whatever dtype the target actually is, so that
        categorical targets work as well as numeric ones."""
        return getattr(self, split)[self.target].to_numpy()

    def pairs(self, split: str) -> pd.DataFrame:
        return getattr(self, split)[self.keys]

    # ------------------------------------------------------------------ prompt pieces

    def file_table(self) -> str:
        train_cols = ",".join(self.train_columns)
        pair_cols = ",".join(self.pair_columns)
        pad = max(len(train_cols), len(pair_cols))
        counts = [f"({len(df):,} rows)" for df in (self.train, self.dev, self.test)]
        cpad = max(len(c) for c in counts)
        rows = [
            ("train.csv", train_cols, counts[0], "observed data"),
            ("dev_pairs.csv", pair_cols, counts[1], f"predict {self.target} for each"),
            ("test_pairs.csv", pair_cols, counts[2], "predict these at the very end"),
        ]
        out = [f"  {f:14s}  {c.ljust(pad)}  {n.ljust(cpad)}  - {d}" for f, c, n, d in rows]
        if (self.workdir / "side_info.csv").exists():
            out.append(f"  {'side_info.csv':14s}  extra attributes, if you want them")
        return "\n".join(out)

    def notes_block(self) -> str:
        lines = list(self.notes)
        if self.train_only:
            cols = ", ".join(f"`{c}`" for c in self.train_only)
            verb = "appears" if len(self.train_only) == 1 else "appear"
            lines.append(f"{cols} {verb} only in train.csv, not in the pairs files, so it "
                         f"cannot be used at prediction time.")
        return "\n".join(f"- {line}" for line in lines)

    # ------------------------------------------------------------------ display

    def summary(self) -> None:
        print(f"{self.name}:  predict `{self.target}` from {', '.join(self.keys)}"
              f"{'  (+ ' + ', '.join(self.features) + ')' if self.features else ''}")
        print(f"metric: {self.metric.name} ({'lower' if self.metric.lower_is_better else 'higher'}"
              f" is better)")
        print(f"train {len(self.train):,}   dev {len(self.dev):,}   test {len(self.test):,}")
        for k in self.keys:
            unseen = (~self.dev[k].isin(self.train[k])).sum()
            print(f"dev {k}s unseen in train: {unseen}")

    def show_agent_files(self) -> None:
        print(f"{self.workdir}\n")
        for f in sorted(self.workdir.iterdir()):
            print(f"  {f.name:16s} {f.stat().st_size / 1e6:6.1f} MB")
