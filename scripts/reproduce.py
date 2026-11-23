"""Rebuild every table and figure.

    uv run python scripts/reproduce.py           full run, writes results/
    uv run python scripts/reproduce.py --fast    smoke run, writes build/results-fast/

The full run trains 20 small networks on one CPU thread and takes about a
quarter of an hour. The fast run uses the same code with small sizes and
finishes in well under two minutes; its numbers are not the committed ones.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from hedgerow.deep import set_determinism
from hedgerow.experiments import RunConfig, with_device
from hedgerow.report import run_all

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fast", action="store_true", help="small sizes, for CI")
    parser.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args()

    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(
            f"device {args.device!r} requested but this PyTorch build has no CUDA support"
        )

    set_determinism(threads=1)
    cfg = with_device(RunConfig.fast() if args.fast else RunConfig.full(), args.device)
    out = args.out or (ROOT / "build" / "results-fast" if args.fast else ROOT / "results")
    start = time.perf_counter()
    run_all(cfg, out, log=True)
    print(f"elapsed: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
