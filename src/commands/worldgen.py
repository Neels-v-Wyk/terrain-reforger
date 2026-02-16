"""World generation command wrapper around the internal worldgen shell script."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Terraria worlds via tModLoader automation")
    parser.add_argument("--num-worlds", type=int, default=20, help="Number of worlds to generate")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.num_worlds < 1:
        print("--num-worlds must be >= 1")
        return 1

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "src" / "scripts" / "worldgen.sh"

    if not script_path.exists():
        print(f"Worldgen script not found: {script_path}")
        return 1

    result = subprocess.run([str(script_path), str(args.num_worlds)], cwd=str(repo_root), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
