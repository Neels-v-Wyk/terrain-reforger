"""World feature analysis command for natural ID mapping generation."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
from typing import Optional, Sequence

import lihzahrd
from tqdm import tqdm


def analyze_world(world_path: Path) -> dict:
    """Analyze a single world file to extract unique block and wall IDs."""
    world = lihzahrd.World.create_from_file(str(world_path))
    
    block_types = set()
    wall_types = set()
    
    for y in range(world.size.y):
        for x in range(world.size.x):
            tile = world.tiles[x, y]
            
            if tile.block is not None:
                block_id = tile.block.type.value if hasattr(tile.block.type, "value") else tile.block.type
                block_types.add(block_id)
            
            if tile.wall is not None:
                wall_id = tile.wall.type.value if hasattr(tile.wall.type, "value") else tile.wall.type
                wall_types.add(wall_id)
    
    return {"block_types": block_types, "wall_types": wall_types}


def merge_stats(all_stats: list) -> dict:
    """Merge block and wall types from multiple worlds."""
    merged_blocks = set()
    merged_walls = set()
    
    for stats in all_stats:
        merged_blocks.update(stats["block_types"])
        merged_walls.update(stats["wall_types"])
    
    return {"block_types": merged_blocks, "wall_types": merged_walls}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze worlds and generate natural Terraria ID mappings")
    parser.add_argument("--source-dir", default="worldgen", help="Directory containing .wld files")
    parser.add_argument("--output", default="src/terraria/natural_ids.py", help="Output path for generated mapping module")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    
    worldgen_dir = Path(args.source_dir)
    if not worldgen_dir.exists():
        print(f"Error: {worldgen_dir} directory not found!")
        return 1
    
    world_files = list(worldgen_dir.glob("*.wld"))
    if not world_files:
        print(f"No .wld files found in {worldgen_dir}!")
        return 1
    
    print(f"Analyzing {len(world_files)} worlds...")
    
    num_workers = min(os.cpu_count() or 1, len(world_files))
    all_stats = []
    failed = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_world, wf): wf for wf in world_files}
        with tqdm(total=len(world_files), unit="world", dynamic_ncols=True) as pbar:
            for future in concurrent.futures.as_completed(futures):
                wf = futures[future]
                try:
                    all_stats.append(future.result())
                except Exception as error:
                    failed.append(wf.name)
                    tqdm.write(f"Error analyzing {wf.name}: {error}")
                finally:
                    pbar.update(1)
    
    if failed:
        print(f"Warning: {len(failed)} worlds failed: {failed}")
    
    if not all_stats:
        print("No worlds were successfully analyzed!")
        return 1
    
    merged = merge_stats(all_stats)
    
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as handle:
        handle.write('"""Natural-world block and wall ID mappings.\n\n')
        handle.write("Auto-generated from world analysis and used by the 9-channel pipeline.\n")
        handle.write('"""\n\n')
        handle.write(f"NATURAL_BLOCK_IDS = {sorted(merged['block_types'])}\n\n")
        handle.write(f"NATURAL_WALL_IDS = {sorted(merged['wall_types'])}\n\n")
        handle.write("BLOCK_ID_TO_INDEX = {game_id: idx for idx, game_id in enumerate(NATURAL_BLOCK_IDS)}\n")
        handle.write("WALL_ID_TO_INDEX = {game_id: idx for idx, game_id in enumerate(NATURAL_WALL_IDS)}\n\n")
        handle.write("BLOCK_INDEX_TO_ID = {idx: game_id for game_id, idx in BLOCK_ID_TO_INDEX.items()}\n")
        handle.write("WALL_INDEX_TO_ID = {idx: game_id for game_id, idx in WALL_ID_TO_INDEX.items()}\n")
    
    print(f"Found {len(merged['block_types'])} blocks, {len(merged['wall_types'])} walls → {args.output}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
