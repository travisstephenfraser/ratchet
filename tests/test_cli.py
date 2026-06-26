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
