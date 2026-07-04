"""Enforce versioning at the entry points. On a regime change, the run BLOCKS until a
ledger entry records what changed and why — turning 'don't silently change frozen
params' from a discipline rule into something the core won't let you skip."""
import json
import sys
from pathlib import Path

from .regime import regime_payload, regime_hash, diff_payload, RegimeLedger


def _covered(changes, entries):
    recorded = {}
    for e in entries:
        for c in e.get("changed", []):
            recorded[c["field"]] = c["new"]
    return all(recorded.get(f) == n for (f, _o, n) in changes)


def enforce_regime(project, constraints_version, ledger_path, truth):
    payload = regime_payload(project.config, constraints_version, truth)
    current = regime_hash(payload)
    state_path = Path(project.config.project_dir) / ".regime"
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if regime_hash(old) != current:
            changes = diff_payload(old, payload)
            if not _covered(changes, RegimeLedger(ledger_path).entries()):
                lines = "\n".join(f"  {f}: {o!r} -> {n!r}" for f, o, n in changes)
                msg = (f"regime changed without a ledger rationale:\n{lines}\n"
                       f"Record it: python -m ratchet.regime_cli --project "
                       f"{project.config.project_dir} --why '...' --impact '...'")
                print(msg, file=sys.stderr)
                sys.exit(2)
    state_path.write_text(json.dumps(payload, sort_keys=True))
    return current


def record_bump(project, constraints_version, why, impact, author, timestamp, ledger_path, truth):
    payload = regime_payload(project.config, constraints_version, truth)
    state_path = Path(project.config.project_dir) / ".regime"
    old = {}
    if state_path.exists():
        try:
            old = json.loads(state_path.read_text())
        except ValueError:
            # record_bump IS the recovery path for a corrupt baseline: warn, diff from empty.
            print(f"warning: existing {state_path} is corrupt; re-anchoring from scratch",
                  file=sys.stderr)
    changes = diff_payload(old, payload)
    RegimeLedger(ledger_path).record(version=project.config.version, changed=changes,
                                     why=why, impact=impact, author=author,
                                     timestamp=timestamp,
                                     regime=regime_hash(payload))
    # Rewrite the baseline in the same call, establishing the invariant that after any
    # sanctioned operation hash(.regime) == the ledger's newest anchor.
    state_path.write_text(json.dumps(payload, sort_keys=True))
    return changes
