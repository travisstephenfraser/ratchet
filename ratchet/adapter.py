"""The seam contracts a project implements (satisfied by shape, not import)."""
from typing import Protocol, Callable, Tuple, Dict, Any


class Unparseable(Exception):
    """A Runner raises this — and ONLY this — when a model response can't be parsed
    into a valid prediction. The loop demotes it to a per-item MISS (counted against
    the candidate via missing-as-miss), so one bad frame doesn't abort a hill-climb.
    Any OTHER exception (transport, timeout, a harness bug) is NOT a property of the
    candidate: it propagates loudly and halts the run rather than being scored as a
    miss. Keeping this the single swallowed failure IS the fail-loud contract."""


class Runner(Protocol):
    def run(self, candidate: str, item: Any, policy: str = "") -> Any:
        """Apply ONE candidate to ONE item; return a prediction.

        - `policy` is the active constraints block (prepend it to the model prompt).
        - FAIL LOUD: if the model response can't be parsed into a valid prediction,
          raise `Unparseable`. The loop scores that as a miss against the candidate.
          Never resolve a malformed response to a silent 0 — that makes a good mutation
          look like a regression and corrupts the hill-climb.
        - Let any OTHER error (transport, timeout, a bug) propagate: those are not the
          candidate's fault and must halt the run, not be laundered into a miss.
        - Do arithmetic (sums, clamps) HERE in code; the model emits judgments only."""
        ...


Mutation = Tuple[str, Callable[[str], str]]
Ingest = Callable[[], Tuple[Dict[str, Any], Dict[str, str]]]
