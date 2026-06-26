"""The seam contracts a project implements (satisfied by shape, not import)."""
from typing import Protocol, Callable, Tuple, Dict, Any


class Runner(Protocol):
    def run(self, candidate: str, item: Any, policy: str = "") -> Any:
        """Apply ONE candidate to ONE item; return a prediction.

        - `policy` is the active constraints block (prepend it to the model prompt).
        - FAIL LOUD: if the model response can't be parsed into a valid prediction,
          RAISE. Never resolve a malformed response to a silent 0/miss — that makes a
          good mutation look like a regression and corrupts the hill-climb.
        - Do arithmetic (sums, clamps) HERE in code; the model emits judgments only."""
        ...


Mutation = Tuple[str, Callable[[str], str]]
Ingest = Callable[[], Tuple[Dict[str, Any], Dict[str, str]]]
