from pathlib import Path
from loop_eval.project import load_project
from loop_eval.verifier import split_ids
from loop_eval.loop import hill_climb, escalate
from loop_eval.constraints import load_constraints, current_version
from loop_eval.regime import regime_payload, regime_hash

TOY = Path(__file__).parent.parent / "projects" / "toy"


def test_toy_climbs_persists_then_survives_gate(tmp_path):
    proj = load_project(TOY)
    items, truth = proj.ingest()
    policy = load_constraints(TOY / "constraints.jsonl")
    cv = current_version(TOY / "constraints.jsonl")
    regime = regime_hash(regime_payload(proj.config, cv))
    train, holdout = split_ids(list(truth), proj.config.salt, proj.config.holdout_pct)

    best = hill_climb(proj, train, items, truth, rounds=proj.config.search["rounds"],
                      patience=proj.config.search["patience"], policy=policy,
                      regime=regime, out_dir=tmp_path)
    assert "lenient" in best["instructions"].lower()
    assert best["metrics"]["objective"] == 1.0
    # persistence happened, stamped with the regime
    assert (tmp_path / "LOOP_LOG.md").exists()
    assert (tmp_path / "candidates" / f"{best['cid']}.metrics.json").exists()

    gate = escalate(proj, best, train, holdout, items, truth,
                    log_path=tmp_path / "holdout_access.log", policy=policy)
    assert gate["overfit"] is False
    assert gate["holdout"]["objective"] == 1.0
