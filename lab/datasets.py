"""Concrete Tasks.

Each function here returns a `Task`. Adding a dataset means adding one of these — the prompt,
the scorer and the loop all read from the Task and need no edits.
"""

from __future__ import annotations

import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .task import Task, rmse

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

ANON = {"user": "row_id", "movie": "col_id", "rating": "value", "timestamp": "t"}
PLAIN = {"user": "user", "movie": "movie", "rating": "rating", "timestamp": "t"}


def _movielens_files() -> Path:
    d = Path(tempfile.gettempdir()) / "ml-1m"
    if not (d / "ratings.dat").exists():
        print("downloading MovieLens 1M ...")
        zp = Path(tempfile.gettempdir()) / "ml-1m.zip"
        urllib.request.urlretrieve(ML_URL, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(Path(tempfile.gettempdir()))
    return d


def movielens(anonymize: bool = True, seed: int = 773,
              time_at_prediction: bool = False) -> Task:
    """MovieLens 1M as a held-out prediction task.

    anonymize           hide what the data *is*: neutral column names, reshuffled
                        identifiers, no titles or genres, and a row count that is not the
                        recognisable 1,000,209. The structure survives; the identity does
                        not. This is what separates reasoning from recall.
    time_at_prediction  put the time column in the pairs files. Without it, temporal
                        features cannot be used at prediction time - which the agent will
                        work out for itself, after wasting an attempt on it.
    """
    d = _movielens_files()
    raw = pd.read_csv(d / "ratings.dat", sep="::", engine="python",
                      names=["user", "movie", "rating", "timestamp"], encoding="latin-1")

    rng = np.random.default_rng(seed)

    if anonymize:
        frame = raw.sample(frac=0.997, random_state=seed).reset_index(drop=True)
        for src in ("user", "movie"):            # ids must not match the public file
            ids = frame[src].unique()
            frame[src] = frame[src].map(dict(zip(ids, rng.permutation(len(ids)) + 1)))
        frame["timestamp"] -= frame["timestamp"].min()
        names, side, blurb = ANON, None, (
            "You are given a sparse table of observations indexed by two identifiers.")
    else:
        frame = raw.copy()
        titles = pd.read_csv(d / "movies.dat", sep="::", engine="python",
                             names=["movie", "title", "genres"], encoding="latin-1")
        titles["year"] = titles.title.str.extract(r"\((\d{4})\)$").astype(float)
        names, blurb = PLAIN, "You are given a table of movie ratings by users."
        side = titles.rename(columns={"movie": PLAIN["movie"]})

    frame = frame.rename(columns=names)
    row, col, val, t = names["user"], names["movie"], names["rating"], "t"

    return Task.build(
        frame,
        name="movielens" + ("-anon" if anonymize else ""),
        blurb=blurb,
        keys=[row, col],
        target=val,
        metric=rmse(clip=(1, 5)),
        features=[t] if time_at_prediction else [],
        train_only=[] if time_at_prediction else [t],
        notes=[f"Each `{val}` is an integer from 1 to 5.",
               "`t` is a time index."],
        seed=seed,
        side_info=side,
    )
