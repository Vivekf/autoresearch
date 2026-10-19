"""Plumbing for Lecture 4, so the notebook can be about the loop and the prompt.

    from lab import movielens, Sandbox, Scorer, Trace

Task      a held-out prediction problem: keys, target, metric, splits (task.py)
datasets  concrete Tasks - movielens() and anything you add (datasets.py)
Sandbox   runs the agent's Python - fresh process or persistent session
Scorer    holds the held-out answers and turns a predictions file into one number
Trace     prints the run and records what it cost

Read these if you are curious. Nothing in them is the lecture.
"""

from .task import Task, Metric, rmse, mae, accuracy
from .datasets import movielens
from .sandbox import Sandbox
from .scoring import Scorer
from .trace import Trace

__all__ = ["Task", "Metric", "rmse", "mae", "accuracy",
           "movielens", "Sandbox", "Scorer", "Trace"]
