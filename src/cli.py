"""Unified pretty CLI for Terrain Reforger."""

from __future__ import annotations

import argparse
from typing import List

import typer
from rich.console import Console

from src.commands.analyze import main as analyze_main
from src.commands.export import main as export_main
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
    ema_decay: float = typer.Option(0.99, "--ema-decay", help="EMA decay rate"),
    ema_reset_multiplier: float = typer.Option(0.5, "--ema-reset-multiplier", help="Dead-code threshold multiplier relative to uniform usage"),
    ema_reset_interval: int = typer.Option(500, "--ema-reset-interval", help="EMA dead-code reset interval in updates"),
    beta: float = typer.Option(0.25, "--beta", help="Commitment loss weight"),
    block_loss_weighted: bool = typer.Option(False, "--block-loss-weighted", help="Use inverse-frequency block loss weighting"),
    block_weight_min: float = typer.Option(0.5, "--block-weight-min", help="Minimum block class weight clamp"),
    block_weight_max: float = typer.Option(5.0, "--block-weight-max", help="Maximum block class weight clamp"),
    no_ema: bool = typer.Option(False, "--no-ema", help="Disable EMA quantizer"),
    metrics_stride: int = typer.Option(50, "--metrics-stride", help="Store metrics every N updates"),
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
    argv.extend(["--ema-decay", str(ema_decay)])
    argv.extend(["--ema-reset-multiplier", str(ema_reset_multiplier)])
    argv.extend(["--ema-reset-interval", str(ema_reset_interval)])
    argv.extend(["--beta", str(beta)])
    if block_loss_weighted:
        argv.append("--block-loss-weighted")
    argv.extend(["--block-weight-min", str(block_weight_min)])
    argv.extend(["--block-weight-max", str(block_weight_max)])
    if no_ema:
        argv.append("--no-ema")
    argv.extend(["--metrics-stride", str(metrics_stride)])

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


def _run_export(
    model: str | None = typer.Argument(None, help="Checkpoint/model path"),
    world: str | None = typer.Option(None, "--world", help="Path to .wld file"),
    region: List[int] | None = typer.Option(None, "--region", help="Region as X0 Y0 X1 Y1 (overrides x/y/width/height)"),
    x: int = typer.Option(1500, "--x", help="Starting X coordinate in world"),
    y: int = typer.Option(500, "--y", help="Starting Y coordinate in world"),
    width: int = typer.Option(64, "--width", help="Width of exported region (tiles)"),
    height: int = typer.Option(64, "--height", help="Height of exported region (tiles)"),
    output_dir: str = typer.Option("exports", "--output-dir", help="Output directory"),
    name: str | None = typer.Option(None, "--name", help="Schematic name (shown in TEdit)"),
    mode: str = typer.Option("compare", "--mode", help="Mode: compare (before/after), reconstruct (output only), generate"),
    samples: int = typer.Option(1, "--samples", help="Number of regions to export"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for generation"),
) -> None:
    """Export inference results as TEdit schematics."""
    argv: List[str] = []
    if model:
        argv.append(model)
    if world:
        argv.extend(["--world", world])
    if region and len(region) == 4:
        argv.extend(["--region", str(region[0]), str(region[1]), str(region[2]), str(region[3])])
    argv.extend(["--x", str(x)])
    argv.extend(["--y", str(y)])
    argv.extend(["--width", str(width)])
    argv.extend(["--height", str(height)])
    argv.extend(["--output-dir", output_dir])
    if name:
        argv.extend(["--name", name])
    argv.extend(["--mode", mode])
    argv.extend(["--samples", str(samples)])
    if seed is not None:
        argv.extend(["--seed", str(seed)])

    exit_code = export_main(argv)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


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
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel generation jobs (max 4)"),
) -> None:
    """Generate Terraria worlds through the internal worldgen script."""
    exit_code = worldgen_main(["--num-worlds", str(num_worlds), "--parallel", str(parallel)])
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
    ema_decay: float = typer.Option(0.99, "--ema-decay", help="EMA decay rate"),
    ema_reset_multiplier: float = typer.Option(0.5, "--ema-reset-multiplier", help="Dead-code threshold multiplier relative to uniform usage"),
    ema_reset_interval: int = typer.Option(500, "--ema-reset-interval", help="EMA dead-code reset interval in updates"),
    beta: float = typer.Option(0.25, "--beta", help="Commitment loss weight"),
    block_loss_weighted: bool = typer.Option(False, "--block-loss-weighted", help="Use inverse-frequency block loss weighting"),
    block_weight_min: float = typer.Option(0.5, "--block-weight-min", help="Minimum block class weight clamp"),
    block_weight_max: float = typer.Option(5.0, "--block-weight-max", help="Maximum block class weight clamp"),
    no_ema: bool = typer.Option(False, "--no-ema", help="Disable EMA quantizer"),
    metrics_stride: int = typer.Option(50, "--metrics-stride", help="Store metrics every N updates"),
) -> None:
    """Train the optimized VQ-VAE model."""
    _run_train(
        data,
        world,
        epochs,
        one_pass,
        batch_size,
        resume,
        disk_mode,
        cache_size,
        ema_decay,
        ema_reset_multiplier,
        ema_reset_interval,
        beta,
        block_loss_weighted,
        block_weight_min,
        block_weight_max,
        no_ema,
        metrics_stride,
    )


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


@model_app.command("export")
def model_export_command(
    model: str | None = typer.Argument(None, help="Checkpoint/model path"),
    world: str | None = typer.Option(None, "--world", help="Path to .wld file"),
    region: List[int] | None = typer.Option(None, "--region", help="Region as X0 Y0 X1 Y1 (overrides x/y/width/height)"),
    x: int = typer.Option(1500, "--x", help="Starting X coordinate in world"),
    y: int = typer.Option(500, "--y", help="Starting Y coordinate in world"),
    width: int = typer.Option(64, "--width", help="Width of exported region (tiles)"),
    height: int = typer.Option(64, "--height", help="Height of exported region (tiles)"),
    output_dir: str = typer.Option("exports", "--output-dir", help="Output directory"),
    name: str | None = typer.Option(None, "--name", help="Schematic name (shown in TEdit)"),
    mode: str = typer.Option("compare", "--mode", help="Mode: compare (before/after), reconstruct (output only), generate"),
    samples: int = typer.Option(1, "--samples", help="Number of regions to export"),
    seed: int | None = typer.Option(None, "--seed", help="Random seed for generation"),
) -> None:
    """Export inference results as TEdit schematics (.TEditSch).
    
    The default 'compare' mode exports both the original chunk (before VAE) and 
    the reconstructed chunk (after VAE) so you can compare quality in TEdit.
    
    Examples:
        terrain model export --width 100 --height 50
        terrain model export --x 2000 --y 300 --width 64 --height 64 --samples 3
    """
    _run_export(model, world, region, x, y, width, height, output_dir, name, mode, samples, seed)


app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
