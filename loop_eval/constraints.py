"""The feedback channel: Claude/human verdicts re-enter as constraints prepended to
every candidate (as policy), NOT as patches into the searched prompt. Append-only, so
it rots into a contradictory ban-list if untended — hence dated+attributed entries, a
consolidation review(), and a ConstraintsLedger recording what each consolidation
changed and why (the second of the spec's two ledgers)."""
import hashlib
import json
from pathlib import Path

_ONE_SIDED = ("NEVER", "ALWAYS", "CRITICAL", "MUST NOT")


def _entries(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def add_constraint(path, text, author, timestamp):
    entry = {"text": text, "author": author, "timestamp": timestamp, "active": True}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def load_constraints(path):
    return "\n".join(f"- {e['text']}" for e in _entries(path) if e.get("active", True))


def current_version(path):
    blob = "\n".join(e["text"] for e in _entries(path) if e.get("active", True))
    return hashlib.sha256(blob.encode()).hexdigest()[:8] if blob else "none"


def review(path):
    report, seen = [], {}
    for e in [e for e in _entries(path) if e.get("active", True)]:
        body = e["text"].strip()
        if body in seen:
            report.append(f"duplicate: {body!r} (also at {seen[body]})")
        seen[body] = e["timestamp"]
        if any(tok in body.upper() for tok in _ONE_SIDED):
            report.append(f"one-sided language (prefer a two-sided criterion): {body!r}")
    return report


class ConstraintsLedger:
    def __init__(self, path):
        self.path = Path(path)

    def record(self, *, version, changed, why, author, timestamp):
        entry = {"version": version, "timestamp": timestamp, "author": author,
                 "why": why, "changed": list(changed)}
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def entries(self):
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
