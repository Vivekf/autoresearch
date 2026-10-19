"""Where the agent's code runs.

Two flavours, same interface:

  Sandbox(workdir, persistent=False)   a fresh process every call - Lecture 3's tool.
  Sandbox(workdir, persistent=True)    one process that stays alive, like a notebook kernel.

The difference is what a marginal experiment costs. With a fresh process, "train five more
epochs on the model I already have" means reloading the data and retraining from scratch. With
a persistent session it is three lines.

Timeouts are two-tier, which matters more than it sounds. A long-running cell is interrupted
*inside* the session, so the agent gets a traceback and keeps every variable it had - losing a
trained model because one experiment ran long would punish exactly the behaviour persistence
exists to encourage. Only if that fails to land does the parent kill and restart the process,
and then it says so plainly.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

MAX_OUTPUT = 3000
HARD_GRACE = 25          # seconds past the soft limit before we give up on the session

# Packages worth telling the agent about, if they happen to be installed.
CANDIDATES = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
              "sklearn": "scikit-learn", "statsmodels": "statsmodels", "torch": "torch",
              "xgboost": "xgboost", "lightgbm": "lightgbm", "numba": "numba"}

# Runs inside the child process. Reads code until a marker, executes it in a namespace that
# survives between calls, and interrupts it with a timer rather than dying.
_DRIVER = r'''
import sys, io, signal, traceback, contextlib

TIMEOUT = {timeout}

class _Timeout(Exception):
    pass

signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))

ns, buf = {{}}, []
for line in iter(sys.stdin.readline, ""):
    if line.rstrip("\n") == "<<<RUN>>>":
        code, buf = "".join(buf), []
        out = io.StringIO()
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(compile(code, "<cell>", "exec"), ns)
        except _Timeout:
            out.write("\n\nTIMEOUT: this step ran for more than %ds and was stopped.\n"
                      "Your session is intact - every variable you had is still here.\n"
                      "Do less per step, or make it faster." % TIMEOUT)
        except BaseException:
            out.write(traceback.format_exc())
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        sys.stdout.write(out.getvalue() + "\n<<<DONE>>>\n")
        sys.stdout.flush()
    else:
        buf.append(line)
'''


def _truncate(out: str) -> str:
    out = out.strip()
    if not out:
        return "(ran, printed nothing)"
    return out[:MAX_OUTPUT] + ("\n... [output truncated]" if len(out) > MAX_OUTPUT else "")


class Sandbox:
    def __init__(self, workdir: Path, persistent: bool = True, timeout: int = 180):
        self.workdir, self.persistent, self.timeout = Path(workdir), persistent, timeout
        self.restarts = 0
        self.timeouts = 0
        if persistent:
            self._start()
        self.packages = self._detect_packages()

    # ---------------------------------------------------------------- the tool

    def run(self, code: str) -> str:
        return self._run_persistent(code) if self.persistent else self._run_fresh(code)

    def _run_fresh(self, code: str) -> str:
        try:
            p = subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, timeout=self.timeout, cwd=self.workdir)
        except subprocess.TimeoutExpired:
            self.timeouts += 1
            return f"ERROR: code timed out after {self.timeout}s. Make it faster."
        return _truncate(p.stdout + p.stderr)

    def _run_persistent(self, code: str) -> str:
        # The child interrupts itself at self.timeout. We wait a little longer than that,
        # and only intervene if the interrupt never arrived.
        deadline = time.time() + self.timeout + HARD_GRACE
        try:
            self.proc.stdin.write(textwrap.dedent(code).rstrip("\n") + "\n<<<RUN>>>\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self.restart()
            return "ERROR: the session had died and was restarted. All variables are gone."

        lines: list[str] = []
        while True:
            try:
                line = self.q.get(timeout=max(0.1, deadline - time.time()))
            except queue.Empty:
                self.restart()
                return (f"ERROR: the session stopped responding and had to be restarted, "
                        f"so all variables are gone. Something ran far past the {self.timeout}s "
                        f"limit without yielding. Work in smaller steps.")
            if line.rstrip("\n") == "<<<DONE>>>":
                out = _truncate("".join(lines))
                if "TIMEOUT: this step ran" in out:
                    self.timeouts += 1
                return out
            lines.append(line)

    # ---------------------------------------------------------------- process management

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _DRIVER.format(timeout=self.timeout)],
            cwd=self.workdir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.q: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        for line in iter(self.proc.stdout.readline, ""):
            self.q.put(line)

    def restart(self) -> None:
        try:
            self.proc.kill()
        except Exception:
            pass
        self.restarts += 1
        self._start()

    # ---------------------------------------------------------------- environment

    def _detect_packages(self) -> list[str]:
        """Ask the sandbox itself what it can import, rather than guessing."""
        probe = ("import importlib.util as u\n"
                 f"print(','.join(n for n in {list(CANDIDATES)!r} if u.find_spec(n)))")
        out = self.run(probe).strip().splitlines()
        found = out[-1].split(",") if out and out[-1] else []
        return [CANDIDATES[n] for n in found if n in CANDIDATES]

    # ---------------------------------------------------------------- display

    def description(self) -> str:
        """The execution-model paragraph that goes into the prompt."""
        pkgs = ", ".join(self.packages) if self.packages else "numpy, pandas"
        if self.persistent:
            how = (f"You have a persistent Python session: variables, loaded data and trained "
                   f"models stay alive between run_python calls, so load the data once and "
                   f"build on what you already have. Any single step that runs longer than "
                   f"{self.timeout}s is stopped and handed back to you as a timeout - your "
                   f"variables survive, so pick up where you left off and do less per step.")
        else:
            how = (f"Each run_python call is a fresh process; nothing persists except files you "
                   f"write. Any call running longer than {self.timeout}s is killed.")
        return f"{how}\n\nAvailable packages: {pkgs}."
