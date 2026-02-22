"""Prepare training datasets from generated Terraria worlds.

Merges the logic previously split between pipelines/dataset_preparation.py
and pipelines/prepare_dataset_cli.py into a single, focused command.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm import tqdm

from src.terraria.chunk_processor_optimized import extract_optimized_chunk
from src.terraria.sampling_strategies import DiversitySampler, analyze_chunk, deduplicate_chunks
from src.terraria.world_handler import load_world


# ---------------------------------------------------------------------------
# World processing helpers
# ---------------------------------------------------------------------------

def find_world_files(source_dir: Path) -> List[Path]:
    """Return sorted list of .wld files in *source_dir*."""
    return sorted(source_dir.glob("*.wld"))


def process_world(world_path: Path, config: Dict[str, Any], skip_errors: bool = True) -> List[torch.Tensor]:
    """Extract diversity-filtered chunks from a single world file."""
    world = load_world(world_path)

    x_start, y_start = 50, 50
    x_end, y_end = world.size.x - 50, world.size.y - 50
    step = config["chunk_size"] - config["overlap"]

    sampler = DiversitySampler(
        min_diversity=config["min_diversity"],
        max_rejection_rate=0.5,
        adaptive=True,
    )

    chunks: List[torch.Tensor] = []
    total_possible = ((y_end - y_start) // step) * ((x_end - x_start) // step)

    with tqdm(total=total_possible, desc=f"Scanning {world_path.name}", unit="chunk") as pbar:
        for y in range(y_start, y_end, step):
            for x in range(x_start, x_end, step):
                try:
                    chunk_np = extract_optimized_chunk(world, x, y, config["chunk_size"], config["chunk_size"])
                    tensor = torch.from_numpy(chunk_np).permute(2, 0, 1).float()
                    should_accept, _ = sampler.should_accept(tensor)
                    if should_accept:
                        chunks.append(tensor)
                except Exception:
                    if not skip_errors:
                        raise
                finally:
                    pbar.update(1)

    return chunks


def _save_chunk_file(output_path: Path, chunks: List[torch.Tensor], config: Dict[str, Any], source_world: str) -> None:
    if not chunks:
        return
    payload = {
        "chunks": torch.stack(chunks),
        "config": config,
        "source_world": source_world,
    }
    torch.save(payload, output_path)


def _save_consolidated_dataset(
    output_path: Path,
    chunks: List[torch.Tensor],
    config: Dict[str, Any],
    source_files: List[str],
    deduplicate: bool = True,
) -> int:
    if not chunks:
        return 0

    if deduplicate:
        stats_list = [analyze_chunk(c) for c in tqdm(chunks, desc="Analyzing")]
        dedup_result = deduplicate_chunks(chunks, stats_list=stats_list, similarity_threshold=0.95)
        if isinstance(dedup_result, tuple):
            final_chunks, _ = dedup_result
        else:
            final_chunks = dedup_result
    else:
        final_chunks = chunks

    payload = {
        "chunks": torch.stack(final_chunks),
        "stats": [analyze_chunk(c) for c in tqdm(final_chunks, desc="Stats")],
        "config": config,
        "source_files": source_files,
    }
    torch.save(payload, output_path)
    return len(final_chunks)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _run_chunked(world_files: List[Path], config: dict, output_dir: Path) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for world_path in world_files:
        output_path = output_dir / f"{world_path.stem}.pt"
        if output_path.exists():
            print(f"Skipping {world_path.name} (already exists)")
            continue
        print(f"Loading {world_path.name}...")
        try:
            chunks = process_world(world_path, config, skip_errors=True)
        except Exception as error:
            print(f"Error loading {world_path}: {error}")
            continue
        if chunks:
            print(f"  Saving {len(chunks)} chunks to {output_path.name}...")
            _save_chunk_file(output_path, chunks, config, str(world_path.name))
        else:
            print(f"  No interesting chunks found in {world_path.name}")

    print(f"\nDone. Use CachedTileDataset pointing to: {output_dir}")


def _run_consolidated(world_files: List[Path], config: dict, output_path: Path, no_dedup: bool) -> None:
    os.makedirs(output_path.parent, exist_ok=True)
    all_chunks: List[torch.Tensor] = []
    for world_path in world_files:
        print(f"Loading {world_path.name}...")
        try:
            chunks = process_world(world_path, config)
        except Exception as error:
            print(f"Error loading {world_path}: {error}")
            continue
        print(f"  Accepted {len(chunks)} chunks from {world_path.name}")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks extracted: {len(all_chunks)}")

    if not all_chunks:
        print("No chunks generated! Lowering --min-diversity might help.")
        return

    print("Packing dataset...")
    before_count = len(all_chunks)
    final_count = _save_consolidated_dataset(
        output_path=output_path,
        chunks=all_chunks,
        config=config,
        source_files=[p.name for p in world_files],
        deduplicate=not no_dedup,
    )

    if not no_dedup:
        print(f"Deduplication: {before_count} -> {final_count} (removed {before_count - final_count})")
    print(f"Saved to {output_path}")
    print("Done!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare optimized Terraria dataset for training")
    parser.add_argument("--source", type=str, default="worldgen", help="Directory containing .wld files")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size in tiles")
    parser.add_argument("--overlap", type=int, default=8, help="Overlap between adjacent chunks")
    parser.add_argument("--min-diversity", type=float, default=0.20, help="Minimum diversity score (0–1)")
    parser.add_argument(
        "--mode",
        choices=("consolidated", "chunked"),
        default="consolidated",
        help="consolidated: single .pt file; chunked: one .pt per world (for --disk-mode training)",
    )
    parser.add_argument("--output", type=str, default="data/dataset_optimized.pt", help="Output path (consolidated mode)")
    parser.add_argument("--output-dir", type=str, default="data/cache", help="Output directory (chunked mode)")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication (consolidated mode)")
    return parser


def run_preparation(args: argparse.Namespace) -> None:
    """Entry point callable from the Typer CLI (receives a parsed Namespace)."""
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

    if args.mode == "chunked":
        _run_chunked(world_files, config, Path(args.output_dir))
    else:
        _run_consolidated(world_files, config, Path(args.output), args.no_dedup)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    run_preparation(args)


if __name__ == "__main__":
    main()
