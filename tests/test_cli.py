import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOY = ROOT / "projects" / "toy"


def _run(args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, capture_output=True, text=True)


def test_constraints_review_runs():
    assert _run(["ratchet.constraints_cli", "--project", str(TOY), "--review"]).returncode == 0


def test_loop_cli_runs_and_reports_best():
    r = _run(["ratchet.loop_cli", "--project", str(TOY)])
    assert r.returncode == 0 and "best" in r.stdout.lower()


def test_bench_cli_runs():
    r = _run(["ratchet.bench_cli", "--project", str(TOY)])
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
    # toy truth has 8 items; a preds file with ONE perfect row passes anomaly/overfit
    # for mae but must trip a min_coverage gate. Toy's config has no min_coverage, so
    # build a copy with the knob set.
    import shutil
    proj = tmp_path / "toy"
    shutil.copytree(TOY, proj, ignore=shutil.ignore_patterns(
        ".regime", "regime_log.jsonl", "candidates", "*.log", "LOOP_LOG.md", "bench_*.json"))
    cfg = (proj / "config.yaml").read_text().replace(
        "guards: {anomaly_at: 0.95, overfit_gap: 0.20}",
        "guards: {anomaly_at: 0.95, overfit_gap: 0.20, min_coverage: 0.5}")
    assert "min_coverage" in cfg, "toy guards line changed; update this test's replace()"
    (proj / "config.yaml").write_text(cfg)
    preds = tmp_path / "one.csv"
    preds.write_text("anon_id,score\ntoy001,10\n")
    r = _run(["ratchet.verify", "--project", str(proj), "--predictions", str(preds),
              "--split", "train"])
    assert r.returncode == 2
    assert '"low_coverage": true' in r.stdout
