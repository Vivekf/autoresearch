"""
Measure the classical MovieLens-1M ladder on the task's split.

These are the reference numbers in the notebook's closing table. They are what a
person who knows this problem would get on the *same* split the agent works against, so the
agent's score can be read directly against them.

The agent never sees any of this. Run it standalone:

    python3 measure_ladder.py

Must stay in sync with the notebook: same SEED, same 80/10/10 permutation split, same clip to
[1, 5] before scoring. If you change either, re-run this and update the table in the notebook.
"""

import tempfile, urllib.request, zipfile, time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 773
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def load_ratings():
    d = Path(tempfile.gettempdir()) / "ml-1m"
    if not (d / "ratings.dat").exists():
        print("downloading MovieLens 1M ...")
        zp = Path(tempfile.gettempdir()) / "ml-1m.zip"
        urllib.request.urlretrieve(ML_URL, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(Path(tempfile.gettempdir()))
    return pd.read_csv(d / "ratings.dat", sep="::", engine="python",
                       names=["user", "movie", "rating", "timestamp"], encoding="latin-1")


def split(ratings):
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ratings))
    n_tr, n_dv = int(0.8 * len(ratings)), int(0.1 * len(ratings))
    return (ratings.iloc[perm[:n_tr]].reset_index(drop=True),
            ratings.iloc[perm[n_tr:n_tr + n_dv]].reset_index(drop=True),
            ratings.iloc[perm[n_tr + n_dv:]].reset_index(drop=True))


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.clip(pred, 1, 5) - y) ** 2)))


def regularized_biases(train, mu, lam_i, lam_u):
    g = train.groupby("movie").rating.agg(["sum", "count"])
    bi = (g["sum"] - mu * g["count"]) / (g["count"] + lam_i)
    resid = train.rating - mu - train.movie.map(bi).fillna(0).to_numpy()
    g2 = pd.DataFrame({"u": train.user, "r": resid}).groupby("u").r.agg(["sum", "count"])
    bu = g2["sum"] / (g2["count"] + lam_u)
    return bi, bu


def matrix_factorization(train, dev, mu, k=32, epochs=25, lr=0.01, reg=0.05, batch=8192, seed=0):
    """Biased MF trained with mini-batch SGD. Pure numpy - no scipy, no sklearn."""
    uc = {u: i for i, u in enumerate(train.user.unique())}
    mc = {m: i for i, m in enumerate(train.movie.unique())}
    ui, mi = train.user.map(uc).to_numpy(), train.movie.map(mc).to_numpy()
    rt = train.rating.to_numpy(dtype=float)

    g = np.random.default_rng(seed)
    P = g.normal(0, 0.05, (len(uc), k))
    Q = g.normal(0, 0.05, (len(mc), k))
    bu, bi = np.zeros(len(uc)), np.zeros(len(mc))

    idx = np.arange(len(rt))
    for _ in range(epochs):
        g.shuffle(idx)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            u, m = ui[b], mi[b]
            err = rt[b] - (mu + bu[u] + bi[m] + np.sum(P[u] * Q[m], axis=1))
            np.add.at(bu, u, lr * (err - reg * bu[u]))
            np.add.at(bi, m, lr * (err - reg * bi[m]))
            np.add.at(P, u, lr * (err[:, None] * Q[m] - reg * P[u]))
            np.add.at(Q, m, lr * (err[:, None] * P[u] - reg * Q[m]))

    du = dev.user.map(uc).fillna(-1).astype(int).to_numpy()
    dm = dev.movie.map(mc).fillna(-1).astype(int).to_numpy()
    seen = (du >= 0) & (dm >= 0)
    pred = mu + np.where(du >= 0, bu[du], 0) + np.where(dm >= 0, bi[dm], 0)
    pred += np.where(seen, np.sum(P[np.clip(du, 0, None)] * Q[np.clip(dm, 0, None)], axis=1), 0)
    return pred


def main():
    ratings = load_ratings()
    train, dev, test = split(ratings)
    y = dev.rating.to_numpy(dtype=float)
    mu = train.rating.mean()

    print(f"{len(ratings):,} ratings | {ratings.user.nunique():,} users | "
          f"{ratings.movie.nunique():,} movies")
    print(f"train {len(train):,}  dev {len(dev):,}  test {len(test):,}")
    print(f"dev users unseen in train: {(~dev.user.isin(train.user)).sum()}  "
          f"| dev movies unseen: {(~dev.movie.isin(train.movie)).sum()}\n")

    rows = []

    rows.append(("predict the global mean", rmse(np.full(len(dev), mu), y)))

    um = train.groupby("user").rating.mean()
    rows.append(("predict each user's mean", rmse(dev.user.map(um).fillna(mu).to_numpy(), y)))

    im = train.groupby("movie").rating.mean()
    rows.append(("predict each movie's mean", rmse(dev.movie.map(im).fillna(mu).to_numpy(), y)))

    bi = train.groupby("movie").rating.mean() - mu
    resid = train.rating - mu - train.movie.map(bi).fillna(0).to_numpy()
    bu = pd.DataFrame({"u": train.user, "r": resid}).groupby("u").r.mean()
    rows.append(("global mean + user bias + movie bias",
                 rmse(mu + dev.movie.map(bi).fillna(0).to_numpy()
                      + dev.user.map(bu).fillna(0).to_numpy(), y)))

    best = None
    for lam_i in (2, 5, 10, 15, 25):
        for lam_u in (2, 5, 10, 15, 25):
            b_i, b_u = regularized_biases(train, mu, lam_i, lam_u)
            s = rmse(mu + dev.movie.map(b_i).fillna(0).to_numpy()
                     + dev.user.map(b_u).fillna(0).to_numpy(), y)
            if best is None or s < best[0]:
                best = (s, lam_i, lam_u)
    rows.append((f"the same, regularized (lam_i={best[1]}, lam_u={best[2]})", best[0]))

    for k, epochs, reg in ((32, 25, 0.05), (64, 30, 0.08)):
        t0 = time.time()
        s = rmse(matrix_factorization(train, dev, mu, k=k, epochs=epochs, reg=reg), y)
        rows.append((f"matrix factorization with biases (k={k}, reg={reg})  "
                     f"[{time.time() - t0:.0f}s]", s))

    width = max(len(r[0]) for r in rows)
    print(f"{'approach'.ljust(width)}   dev RMSE")
    print("-" * (width + 12))
    for name, s in rows:
        print(f"{name.ljust(width)}   {s:.4f}")


if __name__ == "__main__":
    main()
