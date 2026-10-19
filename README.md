# Autoresearch

An agent that is given a number and told to make the number better.

It gets two tools — a Python session and a scorer — and a held-out dataset whose answers it
cannot see. Then it proposes a model, writes it, measures it, and uses the measurement to decide
what to try next. The loop is about twenty lines. Everything interesting happens inside it.

Built for **15.773 Hands-On Deep Learning** at MIT Sloan, Lecture 4.

---

## The loop

```python
def run_agent(goal, max_turns=60):
    messages = [{"role": "system", "content": PROMPT},
                {"role": "user",   "content": goal}]

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

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY
jupyter lab research-agent.ipynb
```

The first run downloads MovieLens 1M (~6 MB) and splits it. Expect a few dollars of API spend
and 20–40 minutes of wall clock for a full run.

**To play the agent's part yourself**, no API key needed:

```bash
python3 sample/make_sample.py     # materialise what the agent sees, and what it doesn't
python3 sample/try_it.py --final  # edit one function, get scored by the same Scorer
```

## What's here

| | |
|---|---|
| `lab/` | The harness. `Task` and `Metric`, concrete `datasets`, the `Sandbox` the agent's code runs in, the `Scorer`, and `Trace` for instrumentation. |
| `research-agent.ipynb` | The lean notebook — tools, prompt, loop, run. |
| `harness-walkthrough.ipynb` | The teaching notebook — builds each component and pokes it until its behaviour is obvious, then assembles the prompt and shows where every line of it came from. |
| `runs/` | Four complete recorded runs, with the findings written up. |
| `tools/measure_ladder.py` | The reference ladder: what someone who knows this problem gets on the same split. |
| `sample/` | Materialise the agent's working directory and try the task by hand. |
| `docs/design-note.pdf` | The full write-up: how research gets done, the four components, the prompt dissected. |

## The task

A held-out prediction problem, declared as five things:

```python
keys       = ["row_id", "col_id"]   # identify a row; present at prediction time
features   = []                     # usable at prediction time
train_only = ["t"]                  # observed historically, unavailable when predicting
target     = "value"                # what to predict
metric     = rmse(clip=(1, 5))
```

The files follow mechanically — `train.csv` is `keys + features + train_only + [target]`, the
pairs files are `keys + features`. **The gap between those two lines is the task.** Move a column
from `train_only` to `features` and the problem changes, the files change, and the prompt changes,
without editing any prose.

Pointing this at a different dataset is one function in `lab/datasets.py`. Nothing in the scorer,
the sandbox, the prompt or the loop needs to change — the prompt is generated from the `Task`, so
it rewrites itself.

## Two design decisions worth stealing

**The agent cannot cheat, structurally.** The held-out answers exist only as arrays in the
notebook process. The agent's code runs elsewhere, in a scratch directory holding only the files
above. And the split is a seeded permutation the agent never sees — so even the original public
dataset would not tell it which rows were held out. The seed is the secret, not the data.

**The deliverable is uncoupled from the stopping decision.** Every attempt writes predictions for
both the dev pairs and the test pairs, and whenever an attempt is the best so far, its test
predictions are copied somewhere the agent cannot reach. Earlier versions asked the agent to write
its final answer "when you are done", which failed twice: one run never judged itself done and
produced nothing at all, and another lost track of which model had been its best and shipped a
worse one. See `runs/README.md`.

## Anonymisation

Underneath this is MovieLens 1M — *the* standard benchmark for rating prediction, which means a
capable model has read papers about it and can recall that matrix factorization works well
without discovering anything.

So `movielens(anonymize=True)` strips the identity: neutral column names, reshuffled identifiers,
no titles or genres, and a row count that is not the recognisable 1,000,209. The *structure*
survives on purpose. Reasoning from a sparse table of integers to matrix completion is
generalisation; reciting a leaderboard is not. The switch separates them — and in the recorded
runs, the anonymised agent reached factorization anyway.

## What actually happens

Agents satisfice. They climb part of the ladder and stop, usually well short of what their own
approach could have given them, and with large variance between runs of an identical
configuration. One recorded run never found matrix factorization at all. Another found it and
under-trained it. A third exhausted its budget and left nothing behind.

`runs/README.md` has the numbers.
