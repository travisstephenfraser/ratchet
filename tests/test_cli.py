import subprocess, sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOY = ROOT / "projects" / "toy"


def _run(args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)


def _copy_toy(tmp_path):
    proj = tmp_path / "toy"
    shutil.copytree(TOY, proj, ignore=shutil.ignore_patterns(
        ".regime", "regime_log.jsonl", "candidates", "*.log", "LOOP_LOG.md", "bench_*.json"))
    return proj


def test_constraints_review_runs():
    assert _run(["ratchet.constraints_cli", "--project", str(TOY), "--review"]).returncode == 0


def test_loop_cli_runs_and_reports_best(tmp_path):
    r = _run(["ratchet.loop_cli", "--project", str(_copy_toy(tmp_path))])
    assert r.returncode == 0 and "best" in r.stdout.lower()


def test_loop_cli_labels_anomaly_as_reject_not_pass(tmp_path):
    proj = _copy_toy(tmp_path)
    (proj / "config.yaml").write_text(
        (proj / "config.yaml").read_text().replace("anomaly_at: 1.0", "anomaly_at: 0.95"))
    result = _run(["ratchet.loop_cli", "--project", str(proj), "--escalate"])
    assert result.returncode == 0  # loop remains exploratory, per the public contract
    assert "ANOMALY — reject" in result.stdout
    assert "generalizes — pass" not in result.stdout


def test_loop_escalation_fails_closed_without_frozen_baseline(tmp_path):
    proj = _copy_toy(tmp_path)
    cfg = (proj / "config.yaml").read_text().replace(
        ", baseline: {train: 0.5161290322580645, holdout: 0.4444444444444444}", "")
    (proj / "config.yaml").write_text(cfg)
    result = _run(["ratchet.loop_cli", "--project", str(proj), "--escalate"])
    assert result.returncode == 2
    assert "missing frozen baseline" in result.stderr


def test_loop_cli_establishes_frozen_baseline_values(tmp_path):
    proj = _copy_toy(tmp_path)
    cfg = (proj / "config.yaml").read_text().replace(
        "baseline: {train: 0.5161290322580645, holdout: 0.4444444444444444}",
        "baseline: {train: CHANGEME, holdout: CHANGEME}")
    (proj / "config.yaml").write_text(cfg)

    result = _run(["ratchet.loop_cli", "--project", str(proj), "--establish-baseline"])

    assert result.returncode == 0
    assert "train: 0.5161290322580645" in result.stdout
    assert "holdout: 0.4444444444444444" in result.stdout
    assert (proj / "holdout_access.log").exists()


def test_bench_cli_runs(tmp_path):
    r = _run(["ratchet.bench_cli", "--project", str(_copy_toy(tmp_path))])
    assert r.returncode == 0 and "regime" in r.stdout.lower()


# The ledger rationale is the audit trail for a sanctioned regime bump; a blank
# --why/--impact defeats it, so the CLI must refuse before touching the project.

def test_regime_cli_rejects_blank_why():
    r = _run(["ratchet.regime_cli", "--project", str(TOY), "--why", "   ", "--impact", "x"])
    assert r.returncode == 2 and "blank" in r.stderr.lower()


def test_regime_cli_rejects_blank_impact():
    r = _run(["ratchet.regime_cli", "--project", str(TOY), "--why", "model swap", "--impact", ""])
    assert r.returncode == 2 and "blank" in r.stderr.lower()


def test_regime_cli_requires_impact():
    r = _run(["ratchet.regime_cli", "--project", str(TOY), "--why", "model swap"])
    assert r.returncode == 2


def test_verify_exits_2_on_low_coverage(tmp_path):
    # Toy has 40 items; a predictions file with one row must trip the configured
    # coverage floor. Toy's normal config has no min_coverage, so add it in a copy.
    proj = _copy_toy(tmp_path)
    cfg = (proj / "config.yaml").read_text().replace(
        "guards: {anomaly_at: 1.0, overfit_gap: 0.20, baseline: {train: 0.5161290322580645, holdout: 0.4444444444444444}}",
        "guards: {anomaly_at: 1.0, overfit_gap: 0.20, min_coverage: 0.5, baseline: {train: 0.5161290322580645, holdout: 0.4444444444444444}}")
    assert "min_coverage" in cfg, "toy guards line changed; update this test's replace()"
    (proj / "config.yaml").write_text(cfg)
    preds = tmp_path / "one.csv"
    preds.write_text("anon_id,score\ntoy001,10\n")
    r = _run(["ratchet.verify", "--project", str(proj), "--predictions", str(preds),
              "--split", "train"])
    assert r.returncode == 2
    assert '"low_coverage": true' in r.stdout


def test_verify_exits_2_when_candidate_regresses_below_frozen_baseline(tmp_path):
    proj = _copy_toy(tmp_path)
    cfg = (proj / "config.yaml").read_text().replace(
        "baseline: {train: 0.5161290322580645, holdout: 0.4444444444444444}",
        "baseline: {train: 0.5, holdout: 0.4}")
    assert "baseline:" in cfg, "toy guards line changed; update this test's replace()"
    (proj / "config.yaml").write_text(cfg)
    preds = tmp_path / "bad.csv"
    preds.write_text("anon_id,score\n" + "".join(f"toy{i:03d},0\n" for i in range(40)))

    result = _run(["ratchet.verify", "--project", str(proj), "--predictions", str(preds),
                   "--split", "gap"])

    assert result.returncode == 2
    assert '"regressed": true' in result.stdout


def test_verify_fails_closed_when_required_baseline_is_missing(tmp_path):
    proj = _copy_toy(tmp_path)
    cfg = (proj / "config.yaml").read_text().replace(
        ", baseline: {train: 0.5161290322580645, holdout: 0.4444444444444444}", "")
    assert "baseline:" not in cfg, "toy guards line changed; update this test's replace()"
    (proj / "config.yaml").write_text(cfg)
    preds = tmp_path / "good.csv"
    preds.write_text("anon_id,score\n" + "".join(f"toy{i:03d},10\n" for i in range(40)))

    result = _run(["ratchet.verify", "--project", str(proj), "--predictions", str(preds),
                   "--split", "train"])

    assert result.returncode == 2
    assert "missing frozen baseline" in result.stderr
