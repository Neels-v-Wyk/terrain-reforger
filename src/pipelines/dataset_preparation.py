"""Shared dataset preparation pipeline for optimized 9-channel training data."""

from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

from src.terraria.world_handler import load_world
from src.terraria.chunk_processor_optimized import extract_optimized_chunk
from src.terraria.sampling_strategies import DiversitySampler, analyze_chunk, deduplicate_chunks


def find_world_files(source_dir: Path) -> List[Path]:
    return sorted(source_dir.glob("*.wld"))


def process_world(world_path: Path, config: Dict[str, Any], skip_errors: bool = True) -> List[torch.Tensor]:
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


def save_chunk_file(output_path: Path, chunks: List[torch.Tensor], config: Dict[str, Any], source_world: str) -> None:
    if not chunks:
        return

    payload = {
        "chunks": torch.stack(chunks),
        "config": config,
        "source_world": source_world,
    }
    torch.save(payload, output_path)


def save_consolidated_dataset(
    output_path: Path,
    chunks: List[torch.Tensor],
    config: Dict[str, Any],
    source_files: List[str],
    deduplicate: bool = True,
) -> int:
    if not chunks:
        return 0

    final_chunks: List[torch.Tensor] = chunks
    filtered_stats: Optional[List[Any]] = None

    if deduplicate:
        stats_list = [analyze_chunk(c) for c in tqdm(chunks, desc="Analyzing")]
        dedup_result = deduplicate_chunks(
            chunks,
            stats_list=stats_list,
            similarity_threshold=0.95,
        )
        if isinstance(dedup_result, tuple):
            final_chunks, filtered_stats = dedup_result
        else:
            final_chunks = dedup_result
            filtered_stats = stats_list
    else:
        filtered_stats = [analyze_chunk(c) for c in tqdm(chunks, desc="Analyzing")]

    payload = {
        "chunks": torch.stack(final_chunks),
        "stats": filtered_stats,
        "config": config,
        "source_files": source_files,
    }
    torch.save(payload, output_path)
    return len(final_chunks)
