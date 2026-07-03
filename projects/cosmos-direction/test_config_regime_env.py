"""Every env knob the RUNNER side reads must be declared in config.regime_env, so a
model, backend, mode, parse-path, or codebase swap bumps the regime instead of silently
pooling. Ingest-side knobs (COSMOS_RUNS, COSMOS_RESULTS_ROOT, COSMOS_MAX_PER_CLIP) are
deliberately NOT required here: frozen.truth fingerprints ingest's OUTPUT, so any change
they cause already moves the hash."""
from pathlib import Path

import yaml

RUNNER_KNOBS = {
    "COSMOS_PARSE",        # regex vs structured extraction
    "COSMOS_VLM_BACKEND",  # inference backend
    "COSMOS_VLM_MODEL",    # model override on the runner
    "COSMOS_MODE",         # pixel vs grounded prompt
    "COSMOS_TEMPERATURE",  # sampling temperature
    "COSMOS_DIR",          # which cosmos codebase the runner imports
}


def test_runner_env_knobs_are_regime_declared():
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    declared = set(cfg["regime_env"])
    missing = RUNNER_KNOBS - declared
    assert not missing, f"runner-side env knobs not folded into the regime: {sorted(missing)}"
