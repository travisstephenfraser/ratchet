"""Enforce versioning at the entry points. On a regime change, the run BLOCKS until a
ledger entry records what changed and why — turning 'don't silently change frozen
params' from a discipline rule into something the core won't let you skip."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .regime import regime_payload, regime_hash, diff_payload, RegimeLedger


def _covered(changes, entries):
    recorded = {}
    for e in entries:
        for c in e.get("changed", []):
            recorded[c["field"]] = c["new"]
    return all(recorded.get(f) == n for (f, _o, n) in changes)


def _fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(2)


def _unblock(project):
    return (f"If this is deliberate, record it: python -m ratchet.regime_cli --project "
            f"{project.config.project_dir} --why '...' --impact '...'")


def _append_anchor(ledger, project, regime, why):
    ledger.record(version=project.config.version, changed=[], why=why, impact="",
                  author="auto", timestamp=datetime.now(timezone.utc).isoformat(),
                  regime=regime)


def enforce_regime(project, constraints_version, ledger_path, truth):
    payload = regime_payload(project.config, constraints_version, truth)
    current = regime_hash(payload)
    state_path = Path(project.config.project_dir) / ".regime"
    ledger = RegimeLedger(ledger_path)
    try:
        entries = ledger.entries()
    except ValueError as e:
        _fail(f"regime ledger is corrupt: {e}\n"
              f"Edit or restore the named line in {ledger_path} before running — the "
              f"ledger is the baseline's anchor and cannot be skipped. (regime_cli "
              f"cannot repair this: the ledger is append-only.)")
    # The anchor: the newest ledger entry that recorded a resulting regime hash.
    anchor = next((e["regime"] for e in reversed(entries) if "regime" in e), None)

    if not state_path.exists():
        if anchor is not None:
            _fail(f".regime is missing but the ledger anchors baseline {anchor} — an "
                  f"established project must not silently re-baseline.\n{_unblock(project)}")
        # True first run (or a legacy ledger predating anchors): baseline and anchor it.
        state_path.write_text(json.dumps(payload, sort_keys=True))
        _append_anchor(ledger, project, current,
                       "initial baseline" if not entries else "anchor existing baseline")
        return current

    try:
        old = json.loads(state_path.read_text())
        old_hash = regime_hash(old)
    except (ValueError, TypeError):
        _fail(f"{state_path} is corrupt — failing closed rather than re-baselining.\n"
              f"{_unblock(project)}")

    if anchor is None:
        # Legacy project with a real baseline but a pre-anchor ledger: adopt it.
        _append_anchor(ledger, project, old_hash, "anchor existing baseline")
        anchor = old_hash

    if old_hash != anchor:
        _fail(f".regime does not match the ledger anchor ({old_hash} != {anchor}) — the "
              f"baseline was hand-edited or drifted.\n{_unblock(project)}")

    # NOTE: `entries` is the pre-append snapshot; that is safe ONLY because anchor
    # entries always carry changed=[] (so they can never satisfy _covered) and the
    # legacy branch runs only when NO entry has a "regime" field (so the dedup check
    # below cannot be confused by it). If _append_anchor ever populates `changed`,
    # re-fetch entries here.
    if old_hash != current:
        changes = diff_payload(old, payload)
        if not _covered(changes, entries):
            lines = "\n".join(f"  {f}: {o!r} -> {n!r}" for f, o, n in changes)
            _fail(f"regime changed without a ledger rationale:\n{lines}\n"
                  f"Record it: python -m ratchet.regime_cli --project "
                  f"{project.config.project_dir} --why '...' --impact '...'")
        # Sanctioned change recorded by an older ratchet that didn't rewrite .regime:
        # advance the baseline and re-anchor so the next run's invariant holds.
        state_path.write_text(json.dumps(payload, sort_keys=True))
        if not any(e.get("regime") == current for e in entries):
            _append_anchor(ledger, project, current, "anchor sanctioned bump")
        return current

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
