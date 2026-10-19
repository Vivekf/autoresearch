"""
Build research-agent.ipynb.

Run from this directory:  python3 build_notebook.py

The notebook is deliberately thin. Everything mechanical - loading and anonymising the data,
running the agent's Python, turning a predictions file into a number, printing and plotting -
lives in the `lab` package one level up. What stays in the notebook is what the lecture is
about: the tools, the prompt, and the loop.
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
# Lecture 4 — An Agent That Improves a Number

In Lecture 3 the coding agent stopped when **the code ran**. That is a real signal, but a weak
one: a program that runs is not a program that is right.

Today the agent stops on something much stronger — **a number that moved on data it cannot
see.** We give it a dataset, a held-out set whose answers it has no access to, and one
instruction: make the error smaller. Then we get out of the way and watch.
""")

md("""
## What we are building

| | |
|---|---|
| **The task** | Predict held-out values in a sparse table. Metric: RMSE, lower is better. |
| **The agent gets** | `train.csv` (with values), `dev_pairs.csv` and `test_pairs.csv` (**no** values). |
| **The agent does not get** | The held-out values. They stay in this notebook's memory. |
| **Tools** | `run_python` and `score`. That is the entire toolkit. |
| **The loop** | Build a model → predict → score → decide what to try next. Repeat. |

Everything mechanical lives in the `lab` package next to this notebook, so that the three
things worth reading — the **tools**, the **prompt** and the **loop** — fit on a screen each.
Open `lab/` if you are curious; nothing in there is the lecture.

**Why the agent cannot cheat.** The held-out values exist only as an array in *this* process.
The agent's code runs elsewhere, in a scratch directory holding only the files above. And the
split is a random permutation under a seed it never sees — so even the original public dataset
would not tell it which rows were held out.
""")

md("## Setup")

co('''import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os

from lab import movielens, Sandbox, Scorer, Trace

load_dotenv(find_dotenv(usecwd=True))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=300)

MODEL      = "gpt-5.5"
SEED       = 773
ANONYMIZE  = True     # hide what the data is, so the model cannot recall what works on it
PERSISTENT = True     # keep the Python session alive between calls

print(f"model={MODEL}  anonymize={ANONYMIZE}  persistent={PERSISTENT}")''')

md("""
## The data

Underneath it is MovieLens 1M — a million ratings, and the standard benchmark for this task.
Which is exactly the problem: the model has read papers about it, so a good score might be
recall rather than research.

So `ANONYMIZE` strips its identity. Neutral column names, reshuffled identifiers, no titles or
genres, and a row count that is not the recognisable 1,000,209. What survives is the *structure*
— a sparse table of integers 1–5 indexed by two entities. Reasoning from that to matrix
completion is real generalisation; recalling a leaderboard is not. The switch separates them.

*We* keep the identity, because the reference ladder at the bottom needs it. The agent does not.
""")

co("""task = movielens(anonymize=ANONYMIZE, seed=SEED)
task.summary()
task.train.head()""")

md("""
### What the agent can see

Everything it can reach is in one scratch directory. Note what goes in and what does not — the
pairs files carry two identifiers and nothing else.
""")

co("""task.show_agent_files()""")

md("""
## The two tools

`run_python` executes code where the data lives. `score` is the only channel through which a
number ever reaches the agent — and it does one more thing besides scoring.

**`score` captures the submission.** Every attempt writes predictions for the dev pairs *and* the
test pairs, and whenever an attempt turns out to be the best so far, its test predictions are
copied somewhere the agent cannot reach. The agent never decides when to submit.

That is not tidiness; it is a repair. Asking the agent to write the final answer "when you are
done" failed twice over. One run never judged itself done, ran out of turns, and produced nothing
at all. Another lost track of which model had been its best and shipped a worse one — reporting
0.8579 while submitting a 0.8616 model. Both failures come from coupling the deliverable to the
stopping decision, so we have uncoupled them.

Worth noticing too: `score` is a *tool call*, so the agent cannot invoke it from inside its own
Python. One configuration per turn. That rules out hyperparameter sweeps — and enforces, by
accident, the discipline of changing one thing at a time and measuring it.
""")

co('''sandbox = Sandbox(task.workdir, persistent=PERSISTENT)
scorer  = Scorer(task)

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
                       f"test_predictions.csv is kept as your submission. This is the only "
                       f"way to find out how well you are doing.",
        "parameters": {"type": "object", "properties": {}}}},
]

TOOL_FUNCTIONS = {"run_python": sandbox.run, "score": scorer.score}


def call_tool(name, args):
    try:
        return TOOL_FUNCTIONS[name](**args)
    except Exception as e:
        return f"ERROR running {name}: {e}"   # the error goes back to the model as an observation''')

md("""
## The prompt

Deliberately thin. It describes the files, the workflow and the metric — and says nothing at all
about *how* to predict a value. No baselines, no biases, no factorization. Whether the agent
finds a ladder is up to the agent.
""")

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
## The loop

This is the whole agent. It is the same loop as Lecture 3 and it has not grown: send the
conversation, run whatever tools the model asked for, append the results, send it back. Stop
when it stops asking.

Everything that makes an agent interesting — the planning, the backtracking, the decision to
try factorization after biases stall — happens inside `client.chat.completions.create`. None of
it is in this function.
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
## Run it

This takes a while, and the wall-clock cost is real — which is itself something the agent has
to manage.
""")

co('''trace = run_agent("Build the best predictor you can. Report your progress as you go.")''')

md("""
## What it tried

One row per call to `score`. The `submitted` column marks the attempts whose test predictions
were captured — the last one marked is what gets scored at the end.
""")

co("""scorer.leaderboard()""")

md("""
## What it cost

Every tool result — every traceback, every printed table, every score — stays in the window for
the rest of the run.
""")

co("""trace.summary(sandbox)
trace.plot(scorer)""")

md("""
## The honest number

The agent spent the whole run optimising against dev: choosing what to try next from the dev
score, keeping what helped, discarding what did not. Dev is no longer held out in any meaningful
sense — it has been fit, slowly, through the agent's choices.

The submission is, by construction, the attempt that scored best on dev. Test has been touched
exactly once: now.
""")

co("""scorer.final()""")

md("""
## For reference: the ladder, measured

What someone who knows this problem would get, on this same split. The agent has seen none of
it. Produced by `Supporting Material/measure_ladder.py`.

| Approach | dev RMSE |
|---|---|
| predict the global mean | 1.1154 |
| predict each row's mean | 1.0341 |
| predict each column's mean | 0.9769 |
| global mean + row bias + column bias | 0.9062 |
| the same, regularized | 0.9056 |
| **matrix factorization with biases (k=32)** | **0.8476** |
| the same with k=64 and more regularization | 0.8551 |

Three things worth noticing. The whole ladder is worth about 0.27 RMSE, and more than half of
that comes from the two simplest steps. Regularizing the biases buys almost nothing — 0.9062 to
0.9056 — so the jump that matters is the *structural* one, from biases to factorization. And the
last row is worse than the one above it: more capacity is not more accuracy.

(The factorization rows carry about ±0.002 of run-to-run noise. These are for the full dataset;
with `ANONYMIZE` on, 0.3% of rows are dropped, which moves them by less than that.)
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

out = Path(__file__).resolve().parent.parent / "research-agent.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(CELLS)} cells)")
