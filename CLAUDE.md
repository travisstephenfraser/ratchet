# ratchet

Portable eval scaffold for prompt/model changes scored against frozen ground truth. Two directions off one project instance: `verify` (regression gate — exit non-zero on regress/leak/overfit) and `loop` (hill-climb, winner must survive the held-out overfit gate). The README is thorough and current — read it for design rationale; this file is the working rules.

## The seam (HARD RULE)

`ratchet/` is the **never-edit core**; `projects/<name>/` is the only place project work happens. A project supplies five adapters (`config.yaml`, `ingest.py`, `runner.py`, `mutations.py`, `base.txt` — contracts in `ratchet/adapter.py`) and touches nothing else. If a project need seems to require a core change, that's a deliberate core PR with tests — never an inline tweak to make one project pass. External consumers (e.g. Rubrica's `calibration/grading_loop/`) **import** the core; never vendor or fork it.

## Comparability discipline

- Scores are comparable **only within a regime** (the fingerprint over frozen params + truth content + scoring source). Any change that affects comparability — model, tolerance, eval set, objective — must be declared in config and recorded in the ledger; the run blocks until it is. Never bypass the block; never compare numbers across regimes by hand.
- The baseline **fails closed**: a missing `.regime` exits 2 with the unblock command. Don't "fix" that by deleting state.
- **Never peek at or tune against the holdout.** The holdout exists to catch memorization; using it for iteration destroys the only unbiased signal.
- The runner **fails loud** — a missing judgment is ambiguous in a way a crashed run isn't. Don't add silent-skip or default-score paths.

## Review lens

The highest-value catch from this repo's own pre-build red-teams: features that are specified, tested green, and **never wired into the run path**. When reviewing or adding a feature, trace it from an entry point before trusting it.

## Commands

```bash
pytest                      # full suite from repo root (conftest.py here)
```

New project: copy `projects/_template/`, fill the five adapters. Architecture snapshot: `ratchet-architecture-2026-06-28.html`; deeper docs in `docs/`.
