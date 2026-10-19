# Sample working directory

A materialised copy of what the Lecture 4 agent sees — and, next to it, what it does not.
Everything here is regenerated in seconds by `make_sample.py` and is safe to delete.

```
agent-sees/     exactly what lands in the agent's scratch directory
withheld/       the answers. In a real run these are never written anywhere
make_sample.py  rebuild both
try_it.py       play the agent's part yourself
```

---

## What the agent gets

| File | Columns | Rows |
|---|---|---|
| `train.csv` | `row_id, col_id, t, value` | 797,766 |
| `dev_pairs.csv` | `row_id, col_id` | 99,720 |
| `test_pairs.csv` | `row_id, col_id` | 99,722 |

**The gap between row 1 and rows 2–3 is the task.** Training rows carry a `value` and a time
index; the pairs carry two bare identifiers. The agent has to reconstruct the first from the
second.

That gap is not hand-written anywhere. It falls out of how the columns are declared in
`lab/datasets.py`:

```python
keys       = [row_id, col_id]   # in train AND in the pairs files
features   = []                 # usable at prediction time - none here
train_only = [t]                # observed in training data, absent at prediction time
target     = value              # what to predict
```

`train.csv` is `keys + features + train_only + [target]`. The pairs files are `keys + features`.
Change a column's role and all three files change, and so does the prompt.

## What the agent does not get

`withheld/dev_answers.csv` and `withheld/test_answers.csv` are the held-out `value` column,
keyed so you can join them yourself.

**In a real run these files do not exist.** They live as numpy arrays on the `Task` object
inside the notebook process, and the only thing that ever reads them is `Scorer`. They are
written out here purely so you can see what is being kept back.

This is also why the agent cannot cheat. There is no answers file to find, and the original
public MovieLens data would not help either — which rows landed in dev is decided by a seeded
permutation the agent never sees.

## Why it says `row_id` and not `user`

Underneath this is MovieLens 1M, which is *the* standard benchmark for this task. That is a
problem: a model that has read papers about MovieLens can recall that matrix factorization works
well on it without discovering anything.

So the anonymised build strips the identity — neutral column names, reshuffled identifiers, no
titles or genres, and a row count that is not the recognisable 1,000,209. The *structure*
survives, deliberately: reasoning from a sparse table of integers 1–5 to matrix completion is
real generalisation, and we want to see whether it happens.

`python3 make_sample.py --labelled` builds the other version if you want to compare.

---

## Playing with it

`try_it.py` lets you be the agent. Edit one function:

```python
def predict(train, pairs):
    item_mean = train.groupby(COL)[TARGET].mean()
    return pairs[COL].map(item_mean).fillna(train[TARGET].mean()).to_numpy()
```

then

```
python3 try_it.py            # scores your predictions on dev
python3 try_it.py --final    # also scores the submission on test, once
```

It rebuilds the identical task (same seed, same split) and runs your predictions through the
same `Scorer` the agent uses — including the submission capture, so you will see
`saved as your submission (attempt 1)` just as the agent does.

Some rungs to climb, in roughly increasing order of difficulty:

| Try | dev RMSE on this split |
|---|---|
| the global mean | 1.1188 |
| each `col_id`'s mean | 0.9815 |
| global mean + row bias + column bias | ~0.91 |
| the same, regularized | ~0.91 |
| something low-rank | ~0.85 |

The last rung is where every interesting thing in the lecture happens. Three of the five
recorded agent runs never reached it.

> **A caveat on those numbers.** The reference ladder in
> `../Supporting Material/measure_ladder.py` is measured on the *full, non-anonymised* data,
> which is a different split from this one. Expect differences of a few thousandths — the item
> mean is 0.9769 there and 0.9815 here. Compare agent scores against numbers measured on the
> same split they ran on.
