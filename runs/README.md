# Recorded runs

Full transcripts of the agent working the MovieLens task. `traces.json` holds the per-turn
token counts and trial logs; the `.log` files are the complete traces, agent code included.

| Run | Model | Execution | Data | Attempts | Turns | Best dev | Test |
|---|---|---|---|---|---|---|---|
| 1 | `gpt-5.4-mini` | fresh process | labelled | 5 | 14 | 0.9057 | — |
| 2 | `gpt-5.4` | fresh process | labelled | 8 | 19 | 0.8740 | 0.8817 |
| 3 | `gpt-5.4` | persistent | anonymised | 7 | 17 | 0.8579 | 0.8615 |
| 4 | `gpt-5.5` | persistent | anonymised | 28 | 60 (limit) | 0.8522 | — |

Reference ladder (`tools/measure_ladder.py`): biases 0.9056, **matrix factorization 0.8476**.

## What they show

1. **They satisfice.** Runs 1–3 stopped voluntarily, well inside their turn limits and well short
   of the reference. Run 4 was still improving when it hit the limit.
2. **Where they stop is highly variable.** A fifth run under run 3's exact configuration stalled
   at 0.9093 and never found factorization at all. Same model, same seed, same prompt. No single
   run supports a claim here.
3. **Run 1 never makes the structural jump.** It lands on the regularized-bias rung — matching
   the reference to four decimals — then tweaks it twice and quits.
4. **A persistent session bought depth, not length.** Run 3 ran *fewer* turns than run 2 and
   scored better. Cheaper experiments did not produce more experiments.
5. **Run 3 could not reproduce its own best result.** It scored 0.8579, re-predicted from the
   model it had stored, and got 0.8616 — so the model it shipped was not the model that scored
   best. This is why `Scorer` now captures the submission automatically.
6. **Run 4 produced no submission at all.** It never judged itself finished, so under the old
   contract it never wrote its test predictions. Same root cause as (5): the deliverable was
   coupled to the stopping decision.

Runs 1–4 predate the current submission-capture contract, so their `final()` numbers are not
comparable with anything produced now. A later run (not logged here) reached **0.8406 dev /
0.8452 test**, beating the reference ladder.
