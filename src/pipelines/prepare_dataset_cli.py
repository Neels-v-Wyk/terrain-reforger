"""CLI helpers for preparing Terraria datasets in consolidated or chunked modes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from src.pipelines.dataset_preparation import (
    find_world_files,
    process_world,
    save_chunk_file,
    save_consolidated_dataset,
)


def build_parser(fixed_mode: Optional[str] = None) -> argparse.ArgumentParser:
    if fixed_mode == "chunked":
        description = "Prepare chunked Terraria dataset"
    elif fixed_mode == "consolidated":
        description = "Prepare consolidated Terraria dataset"
    else:
        description = "Prepare optimized Terraria dataset"

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--source", type=str, default="worldgen", help="Directory containing .wld files")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size")
    parser.add_argument("--overlap", type=int, default=8, help="Overlap between chunks")
    parser.add_argument("--min-diversity", type=float, default=0.20, help="Minimum diversity score")

    if fixed_mode is None:
        parser.add_argument(
            "--mode",
            choices=("consolidated", "chunked"),
            default="consolidated",
            help="Preparation mode",
        )

    if fixed_mode != "chunked":
        parser.add_argument("--output", type=str, default="data/dataset_optimized.pt", help="Consolidated output .pt file")
        parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")

    if fixed_mode != "consolidated":
        parser.add_argument("--output-dir", type=str, default="data/cache", help="Chunked output directory for per-world .pt files")

    return parser


def _process_and_save_world(world_path: Path, output_dir: Path, config: dict) -> None:
    output_path = output_dir / f"{world_path.stem}.pt"

    if output_path.exists():
        print(f"Skipping {world_path.name} (already exists)")
        return

    print(f"Loading {world_path.name}...")
    try:
        chunks = process_world(world_path, config, skip_errors=True)
    except Exception as error:
        print(f"Error loading {world_path}: {error}")
        return

    if chunks:
        print(f"  Saving {len(chunks)} chunks to {output_path.name}...")
        save_chunk_file(output_path, chunks, config, str(world_path.name))
    else:
        print(f"  No interesting chunks found in {world_path.name}")


def _run_chunked(world_files: list[Path], config: dict, output_dir: Path) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for world_file in world_files:
        _process_and_save_world(world_file, output_dir, config)

    print("\nProcessing complete. You can now use CachedTileDataset pointing to:")
    print(f"  {output_dir}")


def _run_consolidated(world_files: list[Path], config: dict, output_path: Path, no_dedup: bool) -> None:
    output_parent = output_path.parent
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    all_chunks = []

    for world_file in world_files:
        print(f"Loading {world_file.name}...")
        try:
            world_chunks = process_world(world_file, config)
        except Exception as error:
            print(f"Error loading {world_file}: {error}")
            continue

        print(f"  Accepted {len(world_chunks)} chunks from {world_file.name}")
        all_chunks.extend(world_chunks)

    print(f"\nTotal chunks extracted: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks generated! lowering diversity threshold might help.")
        return

    print("Packing dataset...")
    before_count = len(all_chunks)
    final_count = save_consolidated_dataset(
        output_path=output_path,
        chunks=all_chunks,
        config=config,
        source_files=[str(path.name) for path in world_files],
        deduplicate=not no_dedup,
    )

    if not no_dedup:
        print(f"Deduplication: {before_count} -> {final_count} (removed {before_count - final_count})")
    print(f"Saving to {output_path}...")
    print("Done!")


def run_preparation(args: argparse.Namespace, fixed_mode: Optional[str] = None) -> None:
    mode = fixed_mode or args.mode
    config = {
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "min_diversity": args.min_diversity,
    }

    source_dir = Path(args.source)
    world_files = find_world_files(source_dir)

    if not world_files:
        print(f"No .wld files found in {source_dir}")
        return

    print(f"Found {len(world_files)} worlds.")

    if mode == "chunked":
        _run_chunked(world_files, config, Path(args.output_dir))
        return

    _run_consolidated(world_files, config, Path(args.output), args.no_dedup)


def main(argv: Optional[Sequence[str]] = None, fixed_mode: Optional[str] = None) -> None:
    parser = build_parser(fixed_mode=fixed_mode)
    args = parser.parse_args(argv)
    run_preparation(args, fixed_mode=fixed_mode)
