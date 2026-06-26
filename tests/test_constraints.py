from ratchet.constraints import (
    add_constraint, load_constraints, current_version, review, ConstraintsLedger,
)


def test_add_and_load(tmp_path):
    p = tmp_path / "constraints.jsonl"
    add_constraint(p, "State both costs of escalation.", "reviewer", "2026-06-25T00:00:00Z")
    add_constraint(p, "Grade only what is written.", "claude", "2026-06-25T00:01:00Z")
    block = load_constraints(p)
    assert "State both costs" in block and "Grade only what is written" in block


def test_version_changes(tmp_path):
    p = tmp_path / "c.jsonl"
    add_constraint(p, "one", "t", "2026-06-25T00:00:00Z")
    v1 = current_version(p)
    add_constraint(p, "two", "t", "2026-06-25T00:01:00Z")
    assert current_version(p) != v1


def test_review_flags_oneside_and_dupes(tmp_path):
    p = tmp_path / "c.jsonl"
    add_constraint(p, "NEVER give benefit of the doubt.", "t", "2026-06-25T00:00:00Z")
    add_constraint(p, "NEVER give benefit of the doubt.", "t", "2026-06-25T00:01:00Z")
    report = review(p)
    assert any("duplicate" in r.lower() for r in report)
    assert any("one-sided" in r.lower() for r in report)


def test_empty_when_no_file(tmp_path):
    assert load_constraints(tmp_path / "missing.jsonl") == ""


def test_constraints_ledger_roundtrip(tmp_path):
    led = ConstraintsLedger(tmp_path / "constraints_log.jsonl")
    led.record(version="c2", changed=["removed dup 'NEVER...'", "merged lenient rules"],
               why="contradiction accumulation", author="reviewer", timestamp="2026-06-25T00:00:00Z")
    assert led.entries()[0]["changed"][0].startswith("removed")
