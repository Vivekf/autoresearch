# Autoresearch

An agent that is given a number and told to make the number better.

It gets two tools — a Python session and a scorer — and a dataset whose held-out answers it
cannot see. It proposes a model, writes it, measures it, and uses the measurement to decide what
to try next. The loop is about twenty lines. Everything interesting happens inside it.

```python
for turn in range(max_turns):
    reply = client.chat.completions.create(
        model=MODEL, messages=messages, tools=TOOLS
    ).choices[0].message

    if not reply.tool_calls:           # asked for nothing -> it is done
        return reply.content

    messages.append(reply)

    for call in reply.tool_calls:
        args   = json.loads(call.function.arguments)
        result = TOOL_FUNCTIONS[call.function.name](**args)
        messages.append({"role": "tool",
                         "tool_call_id": call.id,
                         "content": str(result)})
```

There is no research in it. No planning, no hypothesis, no decision about what to try next —
every one of those happens inside the model call. The two `append` lines are the whole mechanism:
strip them out and the model starts from nothing every turn.

---

# 1 · Try it on the built-in problem

The repo ships with a rating-prediction task built from MovieLens 1M. Start here — it takes a
couple of minutes and the behaviour is much easier to reason about once you've watched it.

```bash
git clone https://github.com/Vivekf/autoresearch
cd autoresearch
pip install -r requirements.txt
```

### Without an API key — play the agent's part yourself

Worth doing first. It takes two minutes and makes the rest obvious.

```bash
python3 sample/make_sample.py     # writes out what the agent sees, and what it doesn't
python3 sample/try_it.py --final  # scores a prediction rule through the real Scorer
```

`sample/make_sample.py` drops two folders next to each other:

```
sample/agent-sees/     train.csv (with values), dev_pairs.csv, test_pairs.csv (no values)
sample/withheld/       the answers — which in a real run are never written anywhere
```

Open `train.csv` and `dev_pairs.csv` side by side. Training rows carry a value; the pairs carry
two bare identifiers. **That gap is the task.**

Then edit one function in `sample/try_it.py` and re-run it:

```python
def predict(train, pairs):
    item_mean = train.groupby(COL)[TARGET].mean()
    return pairs[COL].map(item_mean).fillna(train[TARGET].mean()).to_numpy()
```

You'll be scored by the same `Scorer` the agent uses, submission capture included. A ladder to
climb, roughly in order of difficulty:

| Try | dev RMSE |
|---|---|
| the global mean | 1.1188 |
| each `col_id`'s mean | 0.9815 |
| global mean + row bias + column bias | ~0.91 |
| something low-rank | ~0.85 |

That last rung is where everything interesting happens. Three of the five recorded agent runs
never reached it.

### With an API key — let the agent do it

```bash
cp .env.example .env          # add your OPENAI_API_KEY
jupyter lab research-agent.ipynb
```

Run the cells top to bottom. The first run downloads MovieLens (~6 MB). Expect a few dollars of
API spend and 20–40 minutes for a full run. You'll see the agent climb — global mean, then
per-item means, then biases, then if it's having a good day, matrix factorization — with the
leaderboard, the context curve, and an honest test score at the end.

`harness-walkthrough.ipynb` is the slower version: it builds each component, pokes it until its
behaviour is obvious, then assembles the prompt and shows where every line of it came from.

---

# 2 · Point it at your own task

This is the whole of it. **Write one function. Change one line. Nothing else.**

### Write one function

Add this to `lab/datasets.py`:

```python
import pandas as pd
from .task import Task, rmse

def sensors(seed=773):
    df = pd.read_csv("sensors.csv")       # whatever your data is

    return Task.build(
        df,
        name       = "sensors",
        blurb      = "You are given a table of readings from an industrial sensor network.",
        keys       = ["site_id", "sensor_id"],   # identify a row; present at prediction time
        features   = ["ambient_c", "shift_no"],  # usable at prediction time
        train_only = ["maintenance_flag"],       # observed historically, absent at prediction
        target     = "reading",                  # what to predict
        metric     = rmse(),
        notes      = ["`reading` is a float between 0 and 100."],
        seed       = seed,
    )
```

### Change one line

In either notebook's setup cell:

```python
from lab.datasets import sensors   # instead of: from lab import movielens
task = sensors()                   # instead of: movielens(anonymize=..., seed=SEED)
```

### That's it

Everything else follows from the five roles you declared:

| | |
|---|---|
| **The files** | `train.csv` is `keys + features + train_only + [target]`. The pairs files are `keys + features`. Written automatically to a scratch directory. |
| **The split** | 80/10/10 by seeded permutation. Train answers go in the file; dev and test answers stay in memory and are never written. |
| **The prompt** | Generated from the Task. Your `blurb`, the file table, the notes, the column list, the metric name and its direction all appear without you writing any prose. |
| **The scorer** | Merges on your `keys`, applies your `metric`, captures the best attempt's submission. |
| **The loop** | Unchanged. |

If you move a column from `train_only` to `features`, the problem changes, the files change, and
the prompt changes — including a generated line telling the agent which columns it *cannot* use
at prediction time. No prose to keep in sync.

### Choosing the five roles

The only real design work:

- **`keys`** — what identifies a row. These appear in the predictions file, so the scorer can
  line your predictions up against the answers. They must be unique per row.
- **`features`** — anything genuinely available when a prediction is made. Leave empty if the
  task is pure matrix completion.
- **`train_only`** — the subtle one. Columns you observed historically but *won't have* at
  prediction time. Putting a column here rather than in `features` is often the difference
  between an honest task and a leaky one.
- **`target`** — never appears in the pairs files, by construction.
- **`metric`** — `rmse()`, `mae()`, `accuracy()`, or your own. See below.

### A different metric

```python
from lab.task import Metric
import numpy as np

mape = Metric("MAPE",
              lambda y, p: np.mean(np.abs((y - p) / np.clip(np.abs(y), 1e-9, None))) * 100,
              lower_is_better=True)
```

The `lower_is_better` flag is load-bearing: it decides which attempt counts as an improvement
worth submitting, whether the leaderboard runs a running min or max, and which way the dev→test
gap points. `rmse(clip=(1, 5))` clamps predictions to a legal range before scoring.

### If the agent needs more time per step

```python
Sandbox(task.workdir, persistent=True, timeout=600)
```

`persistent=True` keeps variables, loaded data and trained models alive between calls, so
*"train five more epochs on the model I already have"* costs three lines instead of a full
rebuild. A step that runs past `timeout` is interrupted **inside** the session — the agent is
told, and its variables survive.

---

## Why the agent can't cheat

Worth understanding before you trust a number it produces.

The held-out answers exist only as arrays in the notebook process. The agent's code runs
elsewhere, in a scratch directory holding only the files above. And the split is a seeded
permutation the agent never sees — so even the original public dataset wouldn't tell it which
rows were held out. **The seed is the secret, not the data.**

The deliverable is also uncoupled from the stopping decision. Every attempt writes predictions
for both the dev pairs and the test pairs, and whenever an attempt is the best so far, its test
predictions are copied somewhere the agent cannot reach. Earlier versions asked the agent to
write its final answer "when you are done", which failed twice: one run never judged itself done
and produced nothing at all, another lost track of which model had been its best and shipped a
worse one.

## Anonymisation

MovieLens 1M is *the* standard benchmark for rating prediction, which means a capable model has
read papers about it and can recall that matrix factorization works well without discovering
anything.

So `movielens(anonymize=True)` strips the identity: neutral column names, reshuffled identifiers,
no titles or genres, and a row count that is not the recognisable 1,000,209. The *structure*
survives on purpose — reasoning from a sparse table of integers to matrix completion is
generalisation, reciting a leaderboard is not. In the recorded runs the anonymised agent reached
factorization anyway.

If you're building your own task from data the model has never seen, you don't need any of this.

## What's here

| | |
|---|---|
| `lab/` | `Task` and `Metric`, concrete `datasets`, the `Sandbox`, the `Scorer`, and `Trace`. |
| `research-agent.ipynb` | The lean notebook — tools, prompt, loop, run. |
| `harness-walkthrough.ipynb` | Builds each component and shows where every line of the prompt came from. |
| `runs/` | Four complete recorded runs, with the findings written up. |
| `tools/measure_ladder.py` | The reference ladder: what a person who knows the problem gets on the same split. |
| `tools/build_*.py` | The notebooks are generated. Edit these, not the `.ipynb`. |
| `sample/` | Materialise the agent's working directory; try the task by hand. |
| `docs/design-note.pdf` | The full write-up: how research gets done, the four components, the prompt dissected. |

## What actually happens

Agents satisfice. They climb part of the ladder and stop, usually well short of what their own
approach could have given them, and with large variance between runs of an identical
configuration. One recorded run never found matrix factorization at all. Another found it and
under-trained it. A third exhausted its budget and left nothing behind.

`runs/README.md` has the numbers.

## Licence

MIT.
