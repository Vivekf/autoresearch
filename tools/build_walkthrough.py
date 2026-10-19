"""
Build harness-walkthrough.ipynb -- the walkthrough notebook.

    cd "Supporting Material" && python3 build_walkthrough.py

This is the teaching companion to docs/design-note.pdf and follows its structure exactly:
how research gets done, then the four things the loop needs (Task, Metric, Scorer, Sandbox),
then the prompt assembled and dissected, then the loop itself.

The other notebook -- research-agent.ipynb -- is the lean one that just runs the thing.
Keep both; they are for different purposes.
"""
import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})


def co(s):
    CELLS.append({"cell_type": "code", "execution_count": None,
                  "metadata": {}, "outputs": [], "source": s})


md("""
# The Harness

*Companion to `docs/design-note.pdf`, in the same order.*

The other notebook runs the research agent. This one takes it apart: we build each piece, poke
it until its behaviour is obvious, and only then assemble them into the thirty lines of text the
agent actually reads.

| | |
|---|---|
| **§1** | How research gets done — the loop, before we can run it |
| **§2** | What the loop needs — `Task`, `Metric`, `Scorer`, `Sandbox` |
| **§3** | The prompt, assembled — and where every line of it came from |
| **§4** | The loop, for real |
""")

md("## Setup")

co('''import json, os
import numpy as np, pandas as pd
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

from lab import movielens, Task, Sandbox, Scorer, Trace, rmse, accuracy

load_dotenv(find_dotenv(usecwd=True))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300)

MODEL      = "gpt-5.5"
SEED       = 773
ANONYMIZE  = True
PERSISTENT = True''')

# =====================================================================
md("""
---
# §1 · How research gets done

Strip research of its trappings and a loop is left. You have a question. You guess at what might
improve the answer. You build something concrete enough to try. You measure it. The measurement
changes your next guess.

```
state <- what I know so far

repeat:
    hypothesis <- propose(state)       # what might make it better?
    experiment <- build(hypothesis)    # make it concrete enough to run
    result     <- measure(experiment)  # a number I did not get to choose
    state      <- state + (hypothesis, result)
```

The third line is what separates research from guessing. **A result you chose is a decoration; a
result you did not choose can contradict you.** Most of what follows is machinery for making that
line real.

An agent is a prompt in a loop. Give that loop two tools — a
way to *build* and a way to *measure* — and it becomes the cycle above:

```
repeat:
    code    <- model(context)       # propose and build, in one step
    output  <- sandbox.run(code)    # stdout, or a traceback
    score   <- scorer.score()       # the measurement it does not control
    context <- context + [code, output, score]
```

Notice that *propose* and *build* have collapsed into one step. For a model that writes programs,
having the idea and expressing it are the same act.

We cannot run any of that yet — `sandbox` and `scorer` do not exist. Building them is §2.
""")

# =====================================================================
md("""
---
# §2 · What the loop needs

Four things, in the order the loop needs them: a **problem**, a definition of **better**, a
**measurement**, and somewhere to **run experiments**.
""")

# --------------------------------------------------------------- Task
md("""
## §2.1 · `Task` — what is being asked

A held-out prediction problem. One table of observations, split three ways, with some columns
visible at prediction time and some not. You declare five things:

| | |
|---|---|
| `keys` | columns that identify a row — in training data **and** at prediction time |
| `features` | columns usable at prediction time |
| `train_only` | observed historically, unavailable when predicting |
| `target` | the column to predict |
| `metric` | how a prediction is judged |
""")

co("""task = movielens(anonymize=ANONYMIZE, seed=SEED)
task.summary()""")

md("""
### The column algebra

Nothing about the files is hand-specified. They are *derived* from the five roles above.
""")

co('''print("keys       ", task.keys)
print("features   ", task.features)
print("train_only ", task.train_only)
print("target     ", task.target)
print()
print("train.csv        =", ",".join(task.train_columns))
print("dev/test_pairs   =", ",".join(task.pair_columns))
print("predictions.csv  =", task.prediction_columns)''')

md("""
**The gap between the first two lines is the task.** It is not written anywhere — it is the
arithmetic difference between two column lists. Here, that difference is `t` and `value`: the
agent is handed two bare identifiers and asked for a rating.
""")

co("""task.show_agent_files()""")

md("### The same three rows, from both sides")

co('''work = task.workdir
asked = pd.read_csv(work / "dev_pairs.csv").head(3)
held  = task.dev[task.keys + [task.target]].head(3)

print("what the agent is asked:");  print(asked.to_string(index=False))
print("\\nwhat we are holding back:"); print(held.to_string(index=False))''')

md("""
The answers on the right are **never written to disk** during a real run. They live as a numpy
array on the `Task` object, and `Scorer` is the only thing that ever reads them.

That is also why the agent cannot cheat. There is no answer file to find — and the public
MovieLens data would not help either, because which rows landed in dev was decided by a seeded
permutation the agent never sees. **The seed is the secret, not the data.**
""")

md("""
### Changing a role changes the problem

`t` currently sits in `train_only`, so the agent can see timestamps while training but has none
at prediction time. Move it to `features` and the task is a different one — with no prose edited
anywhere. Watch the third note **disappear**: it is generated from `train_only`, so when nothing
is held back there is nothing to warn about.
""")

co('''timed = movielens(anonymize=True, seed=SEED, time_at_prediction=True)

print("pairs files, default:            ", ",".join(task.pair_columns))
print("pairs files, time_at_prediction: ", ",".join(timed.pair_columns))

print("\\n--- notes when t is held back ---")
print(task.notes_block())
print("\\n--- notes when t is available ---")
print(timed.notes_block())''')

md("""
> **Why `row_id` and not `user`.** Underneath this is MovieLens 1M — *the* standard benchmark for
> rating prediction — so a capable model has read papers about it and can recall that matrix
> factorization works well without discovering anything. `ANONYMIZE` strips the identity: neutral
> column names, reshuffled identifiers, no titles or genres, and a row count that is not the
> recognisable 1,000,209. The *structure* survives on purpose. Reasoning from a sparse table to
> matrix completion is generalisation; reciting a leaderboard is not.
""")

# --------------------------------------------------------------- Metric
md("""
## §2.2 · `Metric` — what "better" means

A name, a function, a direction, and optionally a clamp. The function is the obvious part. The
**direction** is the one that is easy to underrate.
""")

co('''m = task.metric
print(f"name            {m.name}")
print(f"lower_is_better {m.lower_is_better}")
print(f"clip            {m.clip}")
print(f"direction       {m.direction}   <- this string goes straight into the prompt")''')

md("""
### The clamp

`clip=(1,5)` is applied to predictions before scoring. RMSE does not know that ratings stop at 5,
so the clamp belongs to the *task*, not to the metric family — hence a parameter rather than a
property.
""")

co('''y    = np.array([5.0, 5.0, 1.0])
wild = np.array([7.0, 5.0, -2.0])     # a model that overshoots in both directions

print("with    clip:", rmse(clip=(1, 5))(y, wild))
print("without clip:", rmse()(y, wild))''')

md("""
### Direction controls four different things

`lower_is_better` silently decides which attempt counts as an improvement worth keeping, whether
the leaderboard runs a running minimum or maximum, what "best so far" means, and which way the
gap between dev and test points. Get it wrong and the closing number prints
backwards — which it did, until the harness was pointed at an accuracy task.
""")

co('''acc = accuracy()
print(f"{'':12s} RMSE (lower better)   accuracy (higher better)")
print(f"{'better(0.8,0.9)':22s} {m.better(0.8, 0.9)!s:<10s} {acc.better(0.8, 0.9)}")
print(f"{'best_of([.8,.9,.85])':22s} {m.best_of([.8,.9,.85]):<10} {acc.best_of([.8,.9,.85])}")''')

# --------------------------------------------------------------- Scorer
md("""
## §2.3 · `Scorer` — the measurement, and the deliverable

Two jobs. The second one is a repair.

**The number.** Read the agent's predictions, line them up against the held-out answers by
`keys`, refuse if any row is unpredicted, apply the metric. One float — the agent's only sense
organ.

**The deliverable.** Every attempt writes predictions for the dev pairs *and* the test pairs.
Whenever an attempt is the best so far, its test predictions are copied somewhere the agent
cannot reach. The agent never decides when to submit.
""")

co("""demo = Scorer(task)      # a throwaway scorer, just to watch it behave""")

md("""
Let us play the agent's part by hand. A helper that writes predictions for both splits, exactly
as the contract requires:
""")

co('''train = pd.read_csv(work / "train.csv")

def submit(fn, write_test=True):
    """Write dev predictions, and optionally test predictions, the way an attempt must."""
    for split, out in (("dev", "predictions.csv"), ("test", "test_predictions.csv")):
        if split == "test" and not write_test:
            (work / out).unlink(missing_ok=True)
            continue
        pairs = pd.read_csv(work / f"{split}_pairs.csv")
        pairs["prediction"] = fn(train, pairs)
        pairs[task.keys + ["prediction"]].to_csv(work / out, index=False)

GLOBAL_MEAN = lambda tr, p: tr[task.target].mean()
ITEM_MEAN   = lambda tr, p: (p[task.keys[1]]
                             .map(tr.groupby(task.keys[1])[task.target].mean())
                             .fillna(tr[task.target].mean()).to_numpy())''')

md("**Attempt 1** — the global mean. It is the best so far, so it becomes the submission.")

co('''submit(GLOBAL_MEAN)
print(demo.score())''')

md("**Attempt 2** — deliberately worse. The submission must *not* move.")

co('''submit(lambda tr, p: 1.0)
print(demo.score())''')

md("**Attempt 3** — better, but it forgot to write test predictions. Nothing can be saved.")

co('''submit(ITEM_MEAN, write_test=False)
print(demo.score())''')

md("**Attempt 4** — the same model, this time writing both. Now it is captured.")

co('''submit(ITEM_MEAN)
print(demo.score())''')

co("""demo.leaderboard()""")

md("""
The `submitted` column records what *actually happened*, not what the score implies — attempt 3
was the best so far and still submitted nothing.

This matters because the obvious alternative failed twice in real runs. Asking the agent to write
its final answer *"when you are done"* meant one run never judged itself done, ran out of turns
and produced nothing at all; another lost track of which model had been its best and shipped a
worse one, reporting 0.8579 while submitting a 0.8616 model.

> **Coupling the deliverable to the stopping decision was the mistake.** Uncoupling them makes
> both failures impossible to express.
""")

md("""
### Why there is a dev set *and* a test set

Every time the agent reads a dev score and chooses what to try next, it spends a little
information about dev. Do that twenty-eight times and dev has been **fit** — not by gradient
descent, but through a sequence of keep-this, discard-that decisions. Each one is a parameter
update made of judgement instead of derivatives.

The agent is playing the role the optimiser usually plays, and it overfits the validation set
exactly the way an unconstrained hyperparameter search does. Test informed no decision, so test
is the honest number.
""")

co("""demo.final()""")

md("""
With four attempts the gap is tiny. Watch it open up after a real run of thirty.

One more property, easy to miss and consequential: **`score` is a tool call**, so the agent
cannot invoke it from inside its own Python. One configuration per turn. That rules out
hyperparameter sweeps — and enforces, by accident, the discipline of changing one thing at a time.
One recorded run, denied sweeps, built itself an internal validation split so it could tune
offline and spend its single scoring call on the winner. Nobody told it to.
""")

# --------------------------------------------------------------- Sandbox
md("""
## §2.4 · `Sandbox` — where experiments happen

Mostly plumbing. One decision in it is not: whether the process **persists between calls**, which
sets the price of a marginal experiment and therefore how much research gets done.
""")

co('''sandbox = Sandbox(task.workdir, persistent=PERSISTENT, timeout=180)
print("packages found:", sandbox.packages)''')

md("The package list is **probed, not declared** — a prompt that lies about the environment wastes turns.")

md("### State persists between calls")

co('''print(sandbox.run('import pandas as pd; tr = pd.read_csv("train.csv"); print(tr.shape)'))
print(sandbox.run('mu = tr.value.mean(); print(round(mu, 4))'))   # tr survived
print(sandbox.run('print("still here:", round(mu, 4))'))''')

md("### Errors come back as text, and do not kill the session")

co('''print(sandbox.run('print(1/0)')[-80:])
print()
print(sandbox.run('print("session is fine:", round(mu, 4))'))''')

md("""
### A long step is interrupted, but the session survives

The obvious implementation — kill the process when a step runs long — destroys every variable,
which punishes precisely the long training runs this task rewards. So the interrupt happens
*inside* the session.
""")

co('''slow = Sandbox(task.workdir, persistent=True, timeout=5)
slow.run("keepme = 42")
print(slow.run('print("starting"); import time; time.sleep(60)'))
print()
print("and the variable survived:", slow.run("print(keepme)"))
print("restarts:", slow.restarts, " soft timeouts:", slow.timeouts)''')

md("### What all of that becomes, in the prompt")

co("""print(sandbox.description())""")

# =====================================================================
md("""
---
# §3 · The prompt, assembled

Everything above exists to produce one artifact: the text the agent actually reads.
""")

md("## The two tools")

co('''scorer = Scorer(task)          # a fresh scorer for the real run

TOOLS = [
    {"type": "function", "function": {
        "name": "run_python",
        "description": "Execute Python in the working directory and return its stdout and "
                       "stderr. " + sandbox.description(),
        "parameters": {"type": "object",
                       "properties": {"code": {"type": "string"}},
                       "required": ["code"]}}},

    {"type": "function", "function": {
        "name": "score",
        "description": f"Score predictions.csv against the held-out dev data and return its "
                       f"{task.metric.name}. If this attempt is your best so far, its "
                       f"test_predictions.csv is kept as your submission.",
        "parameters": {"type": "object", "properties": {}}}},
]

TOOL_FUNCTIONS = {"run_python": sandbox.run, "score": scorer.score}


def call_tool(name, args):
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        return f"ERROR running {name}: {e}"   # the error goes back as an observation''')

md("## The prompt")

co('''PROMPT = f"""You are a researcher. {task.blurb}

Your job is to predict the missing `{task.target}` for each held-out row, as accurately as
possible. The metric is {task.metric.name}. {task.metric.direction}

Your working directory contains:

{task.file_table()}

{task.notes_block()}

{sandbox.description()}

For each attempt:
  1. Use run_python to build a model, then write BOTH of these, each with columns
     {task.prediction_columns}:
        predictions.csv       one row per row of dev_pairs.csv
        test_predictions.csv  one row per row of test_pairs.csv
  2. Call score. It scores your dev predictions, and whenever an attempt is your best so
     far it keeps that attempt's test predictions as your submission.
  3. Use the number to decide what to try next.

You cannot see the true dev values. score is the only way to learn how you are doing.

You never have to decide when to submit - your best attempt is saved automatically, so there
is no final step to remember and nothing is lost if you run out of turns.

Guidance:
- Start with the simplest thing that works, measure it, and improve from there.
- Change one thing at a time, so you know what caused a change in the score.

Stop when you are no longer making progress, and report what you tried and what worked.
"""

print(PROMPT)''')

md("""
## Where every line of it came from

Run the cell below to see the prompt split into blocks, each tagged with its source.
""")

co('''PROVENANCE = [
    ("You are a researcher",      "task.blurb -- the one sentence written per dataset. Where anonymisation lives or dies."),
    ("Your job is to predict",    "task.target, task.metric.name, task.metric.direction (2.2). The direction is printed, not assumed."),
    ("Your working directory",    "fixed wording"),
    ("  train.csv",               "task.file_table() -- computed from the column roles in 2.1."),
    ("- Each",                    "task.notes_block() -- your notes=[...], plus a line GENERATED from train_only."),
    ("You have a persistent",     "sandbox.description() (2.4) -- a fact about the object, not a claim about it."),
    ("Each run_python call",      "sandbox.description() (2.4) -- the non-persistent wording."),
    ("Available packages",        "sandbox.packages -- probed from the child process."),
    ("For each attempt",          "the cycle from 1, made operational. Column list is task.prediction_columns."),
    ("You cannot see",            "the Scorer design (2.3), said out loud."),
    ("You never have to decide",  "the Scorer repair (2.3) -- the line that makes 'ran out of turns, delivered nothing' impossible."),
    ("Guidance",                  "HAND-WRITTEN -- and note that none of it is about this problem."),
    ("Stop when you are",         "HAND-WRITTEN."),
]

def source_of(block):
    for prefix, src in PROVENANCE:
        if block.lstrip("\\n").startswith(prefix):
            return src
    return "(unmatched -- did the prompt change?)"

for block in PROMPT.strip().split("\\n\\n"):
    first = block.strip().splitlines()[0]
    print(f"  {first[:66]:<68s}")
    print(f"      <- {source_of(block)}\\n")''')

md("""
> **Every word that depends on the dataset is derived; every hand-written word is
> dataset-independent.** Swap the `Task` and the prompt rewrites itself. There is no prose to keep
> in sync, and no way for the instructions to end up describing the previous problem.

**And note what it does not say.** Nothing about *how* to predict a value. No baselines, no bias
terms, no factorization, no hint that a ladder of standard approaches exists. The prompt describes
the situation and the rules, then stops. Whether anything gets climbed is the agent's business —
which is the only way the question stays an open one.
""")

# =====================================================================
md("""
---
# §4 · The loop, for real

Now the loop from §1 can finally run. It has not grown: send the conversation, run whatever tools
the model asked for, append the results, send it back. Stop when it stops asking.
""")

co('''def run_agent(goal, max_turns=60, verbose=True):
    messages = [{"role": "system", "content": PROMPT},
                {"role": "user",   "content": goal}]
    trace = Trace(verbose)

    for turn in range(1, max_turns + 1):
        completion = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto")
        reply = completion.choices[0].message
        trace.turn(reply, completion.usage)

        if not reply.tool_calls:          # no tools asked for -> it is done
            return trace.finished(turn)

        messages.append(reply.model_dump(exclude_none=True))

        for call in reply.tool_calls:
            args   = json.loads(call.function.arguments or "{}")
            result = call_tool(call.function.name, args)
            trace.call(call.function.name, args, result)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

    return trace.exhausted(max_turns)''')

md("""
Everything that makes an agent interesting — the planning, the backtracking, the decision to try
factorization after biases stall — happens inside `client.chat.completions.create`. None of it is
in this function.
""")

co('''trace = run_agent("Build the best predictor you can. Report your progress as you go.")''')

md("## What it tried")

co("""scorer.leaderboard()""")

md("## What it cost")

co("""trace.summary(sandbox)
trace.plot(scorer)""")

md("""
## The honest number

The submission is, by construction, the attempt that scored best on dev. Test has been touched
exactly once: now.
""")

co("""scorer.final()""")

md("""
---
## For reference: the ladder, measured

What someone who knows this problem gets on the same split. The agent has seen none of it.
Produced by `tools/measure_ladder.py`.

| Approach | dev RMSE |
|---|---|
| predict the global mean | 1.1154 |
| predict each row's mean | 1.0341 |
| predict each column's mean | 0.9769 |
| global mean + row bias + column bias | 0.9062 |
| the same, regularized | 0.9056 |
| **matrix factorization with biases (k=32)** | **0.8476** |
| the same with k=64 and more regularization | 0.8551 |

Regularizing the biases buys 0.0006. The jump that matters is the *structural* one, from biases
to factorization — and across the recorded runs, that is the jump agents most often fail to make.

*(Measured on the full, non-anonymised split; the anonymised split used here differs by a few
thousandths. Compare agent scores against numbers measured on the same split they ran on.)*
""")


def lines(s):
    s = s.strip("\n")
    parts = s.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


for c in CELLS:
    c["source"] = lines(c["source"])

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent.parent / "harness-walkthrough.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(CELLS)} cells, {sum(c['cell_type']=='code' for c in CELLS)} code)")
