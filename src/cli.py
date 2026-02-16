"""Unified pretty CLI for Terrain Reforger."""

from __future__ import annotations

import argparse
from typing import List

import typer
from rich.console import Console

from src.commands.analyze import main as analyze_main
from src.commands.infer import main as infer_main
from src.commands.train import main as train_main
from src.commands.worldgen import main as worldgen_main
from src.pipelines.prepare_dataset_cli import run_preparation

app = typer.Typer(help="Terrain Reforger CLI", no_args_is_help=True, rich_markup_mode="rich")
console = Console()
data_app = typer.Typer(help="Data preparation and analysis commands", no_args_is_help=True)
model_app = typer.Typer(help="Model training and inference commands", no_args_is_help=True)


def _run_prepare(
    mode: str = typer.Option("consolidated", "--mode", help="Preparation mode: consolidated or chunked"),
    source: str = typer.Option("worldgen", "--source", help="Directory containing .wld files"),
    chunk_size: int = typer.Option(32, "--chunk-size", help="Chunk size"),
    overlap: int = typer.Option(8, "--overlap", help="Overlap between chunks"),
    min_diversity: float = typer.Option(0.20, "--min-diversity", help="Minimum diversity score"),
    output: str = typer.Option("data/dataset_optimized.pt", "--output", help="Consolidated output file"),
    output_dir: str = typer.Option("data/cache", "--output-dir", help="Chunked output directory"),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Disable deduplication in consolidated mode"),
) -> None:
    if mode not in {"consolidated", "chunked"}:
        raise typer.BadParameter("mode must be either 'consolidated' or 'chunked'")

    args = argparse.Namespace(
        mode=mode,
        source=source,
        chunk_size=chunk_size,
        overlap=overlap,
        min_diversity=min_diversity,
        output=output,
        output_dir=output_dir,
        no_dedup=no_dedup,
    )
    run_preparation(args)


def _run_train(
    data: str | None = typer.Option(None, "--data", help="Path to preprocessed dataset (.pt) or cache dir"),
    world: str | None = typer.Option(None, "--world", help="Path to a single .wld file"),
    epochs: int = typer.Option(50, "--epochs", help="Number of epochs"),
    one_pass: bool = typer.Option(False, "--one-pass", help="Train for exactly one pass"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
    resume: str | None = typer.Option(None, "--resume", help="Resume from checkpoint"),
    disk_mode: bool = typer.Option(False, "--disk-mode", help="Use LRU disk-mode dataset loading"),
    cache_size: int = typer.Option(5, "--cache-size", help="LRU cache size for disk mode"),
) -> None:
    """Train the optimized VQ-VAE model."""
    argv: List[str] = []
    if data:
        argv.extend(["--data", data])
    if world:
        argv.extend(["--world", world])
    argv.extend(["--epochs", str(epochs)])
    if one_pass:
        argv.append("--one-pass")
    argv.extend(["--batch-size", str(batch_size)])
    if resume:
        argv.extend(["--resume", resume])
    if disk_mode:
        argv.append("--disk-mode")
    argv.extend(["--cache-size", str(cache_size)])

    train_main(argv)


def _run_infer(
    model: str | None = typer.Argument(None, help="Checkpoint/model path"),
    world: str = typer.Option("worldgen/World_20260211_213447_mBvegqrN.wld", "--world", help="Path to .wld file"),
    region: List[int] = typer.Option([1500, 1000, 1600, 1100], "--region", help="Inference region as X0 Y0 X1 Y1"),
    chunk_size: int = typer.Option(32, "--chunk-size", help="Chunk size"),
    samples: int = typer.Option(3, "--samples", help="Number of chunks to inspect"),
) -> None:
    """Run model inference and reconstruction comparison."""
    if len(region) != 4:
        raise typer.BadParameter("region requires exactly 4 integers: X0 Y0 X1 Y1")

    argv: List[str] = []
    if model:
        argv.append(model)
    argv.extend(["--world", world])
    argv.extend(["--region", str(region[0]), str(region[1]), str(region[2]), str(region[3])])
    argv.extend(["--chunk-size", str(chunk_size)])
    argv.extend(["--samples", str(samples)])

    infer_main(argv)


def _run_analyze(
    source_dir: str = typer.Option("worldgen", "--source-dir", help="Directory containing .wld files"),
    output: str = typer.Option("src/terraria/natural_ids.py", "--output", help="Output module path"),
) -> None:
    analyze_main(["--source-dir", source_dir, "--output", output])


@data_app.command("prepare")
def data_prepare_command(
    mode: str = typer.Option("consolidated", "--mode", help="Preparation mode: consolidated or chunked"),
    source: str = typer.Option("worldgen", "--source", help="Directory containing .wld files"),
    chunk_size: int = typer.Option(32, "--chunk-size", help="Chunk size"),
    overlap: int = typer.Option(8, "--overlap", help="Overlap between chunks"),
    min_diversity: float = typer.Option(0.20, "--min-diversity", help="Minimum diversity score"),
    output: str = typer.Option("data/dataset_optimized.pt", "--output", help="Consolidated output file"),
    output_dir: str = typer.Option("data/cache", "--output-dir", help="Chunked output directory"),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Disable deduplication in consolidated mode"),
) -> None:
    """Prepare training datasets from Terraria worlds."""
    _run_prepare(mode, source, chunk_size, overlap, min_diversity, output, output_dir, no_dedup)


@data_app.command("analyze")
def data_analyze_command(
    source_dir: str = typer.Option("worldgen", "--source-dir", help="Directory containing .wld files"),
    output: str = typer.Option("src/terraria/natural_ids.py", "--output", help="Output module path"),
) -> None:
    """Analyze generated worlds and regenerate natural ID mappings."""
    _run_analyze(source_dir, output)


@data_app.command("worldgen")
def data_worldgen_command(
    num_worlds: int = typer.Option(20, "--num-worlds", help="Number of worlds to generate"),
) -> None:
    """Generate Terraria worlds through the internal worldgen script."""
    exit_code = worldgen_main(["--num-worlds", str(num_worlds)])
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@model_app.command("train")
def model_train_command(
    data: str | None = typer.Option(None, "--data", help="Path to preprocessed dataset (.pt) or cache dir"),
    world: str | None = typer.Option(None, "--world", help="Path to a single .wld file"),
    epochs: int = typer.Option(50, "--epochs", help="Number of epochs"),
    one_pass: bool = typer.Option(False, "--one-pass", help="Train for exactly one pass"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
    resume: str | None = typer.Option(None, "--resume", help="Resume from checkpoint"),
    disk_mode: bool = typer.Option(False, "--disk-mode", help="Use LRU disk-mode dataset loading"),
    cache_size: int = typer.Option(5, "--cache-size", help="LRU cache size for disk mode"),
) -> None:
    """Train the optimized VQ-VAE model."""
    _run_train(data, world, epochs, one_pass, batch_size, resume, disk_mode, cache_size)


@model_app.command("infer")
def model_infer_command(
    model: str | None = typer.Argument(None, help="Checkpoint/model path"),
    world: str = typer.Option("worldgen/World_20260211_213447_mBvegqrN.wld", "--world", help="Path to .wld file"),
    region: List[int] = typer.Option([1500, 1000, 1600, 1100], "--region", help="Inference region as X0 Y0 X1 Y1"),
    chunk_size: int = typer.Option(32, "--chunk-size", help="Chunk size"),
    samples: int = typer.Option(3, "--samples", help="Number of chunks to inspect"),
) -> None:
    """Run model inference and reconstruction comparison."""
    _run_infer(model, world, region, chunk_size, samples)


app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
