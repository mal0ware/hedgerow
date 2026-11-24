"""Rebuild every table and figure.

    uv run python scripts/reproduce.py           full run, writes results/
    uv run python scripts/reproduce.py --fast    smoke run, writes build/results-fast/

    uv run python scripts/reproduce.py --workers 8    the same files, sooner

The full run trains 104 small networks, each on one CPU thread. That is about
three hours of CPU time; --workers N spreads it over N processes and does not
change any number. The fast run uses the same code with small sizes and
finishes in a few minutes; its numbers are not the committed ones.
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
    parser.add_argument("--workers", type=int, default=1, help="processes used for training")
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
    run_all(cfg, out, log=True, workers=args.workers)
    print(f"elapsed: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
