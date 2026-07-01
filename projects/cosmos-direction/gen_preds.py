"""gen_preds.py -- run the base prompt over the ENTIRE bench and write a predictions
CSV for `verify`. This calls the model, so run it under the cosmos venv (httpx for the
lmstudio backend); score the CSV afterward with `verify` under ratchet's venv.

A frame the model can't answer is OMITTED -> it counts as a miss in verify (by design:
"missing predictions are misses"). CSV is flushed per row so a long run is resumable-ish
and partial progress survives an interruption.

Mode/back end come from the env the runner reads:
  COSMOS_VLM_BACKEND=lmstudio  COSMOS_VLM_MODEL=<...>  COSMOS_MODE=pixel|grounded

Usage (cosmos venv, LM Studio up):
  COSMOS_VLM_BACKEND=lmstudio COSMOS_MODE=pixel \
  <cosmos>/.venv/bin/python projects/cosmos-direction/gen_preds.py \
      --out projects/cosmos-direction/data/preds_pixel_gemma.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import ingest   # noqa: E402  (project adapter)
import runner   # noqa: E402  (project adapter)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(HERE / "base.txt"))
    args = ap.parse_args(argv)

    items, _ = ingest.ingest()
    base = Path(args.base).read_text()
    r = runner.Runner()
    ids = sorted(items)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n, ok, miss, t0 = len(ids), 0, 0, time.time()
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "direction"])
        for k, anon in enumerate(ids, 1):
            try:
                w.writerow([anon, r.run(base, items[anon])])
                ok += 1
            except Exception as e:                       # parse/transport failure -> miss
                miss += 1
                print(f"  miss {anon}: {e}", file=sys.stderr)
            f.flush()
            if k % 25 == 0 or k == n:
                el = time.time() - t0
                print(f"{k}/{n}  ok={ok} miss={miss}  {el:.0f}s ({el/k:.1f}s/frame)", flush=True)

    print(f"done: {ok} ok, {miss} miss -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
