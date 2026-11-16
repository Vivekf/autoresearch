"""Watching the run: printing, token accounting, and the two figures.

All of this is bookkeeping around the loop rather than part of it, which is why it lives here.
The loop calls four methods and otherwise ignores it.
"""

from __future__ import annotations

import json

import pandas as pd

CODE_SHOWN = 4000   # how much of the agent's code to print; keep this generous, the
                    # interesting failures hide in the parts people truncate away


class Trace:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.turns: list[dict] = []

    # ---------------------------------------------------------------- called by the loop

    def turn(self, reply, usage) -> None:
        self.turns.append({"turn": len(self.turns) + 1,
                           "prompt_tokens": usage.prompt_tokens,
                           "completion_tokens": usage.completion_tokens})
        if self.verbose and reply.content:
            print(reply.content)

    def call(self, name: str, args: dict, result) -> None:
        if not self.verbose:
            return
        if name == "score":
            print("\n--- the agent calls score() ---")
        else:
            print(f"\n--- the agent calls {name} ---")
            code = args.get("code", "")
            print(code[:CODE_SHOWN] + ("\n... [code truncated]" if len(code) > CODE_SHOWN else ""))
        print(f"--- {name} returned ---\n{str(result)[:1200]}\n")

    def finished(self, turn: int) -> "Trace":
        print(f"\n=== the agent stopped on its own, after {turn} turns ===")
        return self

    def exhausted(self, max_turns: int) -> "Trace":
        print(f"\n=== hit the turn limit ({max_turns}) ===")
        return self

    # ---------------------------------------------------------------- afterwards

    @property
    def tokens(self) -> pd.DataFrame:
        return pd.DataFrame(self.turns)

    def summary(self, sandbox=None) -> None:
        t = self.tokens
        print(f"turns                   {len(t)}")
        print(f"total prompt tokens     {t.prompt_tokens.sum():,}")
        print(f"total completion tokens {t.completion_tokens.sum():,}")
        print(f"context at the end      {t.prompt_tokens.iloc[-1]:,}")
        if sandbox is not None and sandbox.persistent:
            print(f"session restarts        {sandbox.restarts}")

    def plot(self, scorer=None):
        import matplotlib.pyplot as plt

        t = self.tokens
        n = 2 if scorer is not None and scorer.trials else 1
        fig, ax = plt.subplots(1, n, figsize=(5.5 * n, 3.4), squeeze=False)
        ax = ax[0]

        ax[0].plot(t.turn, t.prompt_tokens, marker="o", ms=3)
        ax[0].set(xlabel="turn", ylabel="prompt tokens", title="What the window costs")

        if n == 2:
            # The leaderboard names its score column after the metric, so ask rather than guess.
            metric = scorer.task.metric
            col = metric.name.lower()
            lb = scorer.leaderboard()
            ax[1].plot(lb.attempt, lb[col], marker="o", ms=4, lw=1, label="attempt")
            ax[1].plot(lb.attempt, lb.best_so_far, lw=2, label="best so far")
            ax[1].set(xlabel="attempt", ylabel=f"dev {metric.name}", title="Progress")
            ax[1].legend()

        plt.tight_layout()
        plt.show()

    def save(self, path, scorer=None, **meta) -> None:
        payload = {"meta": meta, "turns": self.turns,
                   "trials": scorer.trials if scorer is not None else []}
        with open(path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"saved {path}")
