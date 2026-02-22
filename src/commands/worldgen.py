"""World generation command wrapper around the internal worldgen shell script."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional, Sequence

_MAX_PARALLEL = os.cpu_count() or 1


def _default_parallel() -> int:
    """Return the number of available CPU cores."""
    return _MAX_PARALLEL


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Terraria worlds via tModLoader automation")
    parser.add_argument("--num-worlds", type=int, default=20, help="Number of worlds to generate")
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help=f"Parallel generation jobs (default: all {_MAX_PARALLEL} CPU cores)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.num_worlds < 1:
        print("--num-worlds must be >= 1")
        return 1

    parallel = args.parallel if args.parallel is not None else _default_parallel()
    if parallel < 1:
        print("--parallel must be >= 1")
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "src" / "scripts" / "worldgen.sh"

    if not script_path.exists():
        print(f"Worldgen script not found: {script_path}")
        return 1

    print(f"Generating {args.num_worlds} world(s) with {parallel} parallel job(s)...")
    result = subprocess.run([str(script_path), str(args.num_worlds), str(parallel)], cwd=str(repo_root), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
