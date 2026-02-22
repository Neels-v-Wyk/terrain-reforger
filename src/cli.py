"""Unified CLI for Terrain Reforger."""

from __future__ import annotations

import argparse
from typing import List, Optional

import typer
from rich.console import Console

from src.commands.analyze import main as analyze_main
from src.commands.export import run as run_export
from src.commands.infer import run as run_infer
from src.commands.prepare import run_preparation
from src.commands.train import run as run_train
from src.commands.worldgen import main as worldgen_main

app = typer.Typer(help="Terrain Reforger CLI", no_args_is_help=True, rich_markup_mode="rich")
console = Console()
data_app = typer.Typer(help="Data preparation and analysis commands", no_args_is_help=True)
model_app = typer.Typer(help="Model training and inference commands", no_args_is_help=True)


# ---------------------------------------------------------------------------
# data commands
# ---------------------------------------------------------------------------

@data_app.command("prepare")
def data_prepare_command(
    mode: str = typer.Option("consolidated", "--mode", help="consolidated: single .pt file; chunked: one .pt per world"),
    source: str = typer.Option("worldgen", "--source", help="Directory containing .wld files"),
    chunk_size: int = typer.Option(32, "--chunk-size", help="Chunk size in tiles"),
    overlap: int = typer.Option(8, "--overlap", help="Overlap between adjacent chunks"),
    min_diversity: float = typer.Option(0.20, "--min-diversity", help="Minimum diversity score (0-1)"),
    output: str = typer.Option("data/dataset.pt", "--output", help="Output path (consolidated mode)"),
    output_dir: str = typer.Option("data/cache", "--output-dir", help="Output directory (chunked mode)"),
    no_dedup: bool = typer.Option(False, "--no-dedup", help="Disable deduplication (consolidated mode)"),
) -> None:
    """Prepare training datasets from generated Terraria worlds."""
    if mode not in {"consolidated", "chunked"}:
        raise typer.BadParameter("--mode must be 'consolidated' or 'chunked'")
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


@data_app.command("analyze")
def data_analyze_command(
    source_dir: str = typer.Option("worldgen", "--source-dir", help="Directory containing .wld files"),
    output: str = typer.Option("src/terraria/natural_ids.py", "--output", help="Output module path"),
) -> None:
    """Analyze generated worlds and regenerate natural ID mappings."""
    analyze_main(["--source-dir", source_dir, "--output", output])


@data_app.command("worldgen")
def data_worldgen_command(
    num_worlds: int = typer.Option(20, "--num-worlds", help="Number of worlds to generate"),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel generation jobs (max 4)"),
) -> None:
    """Generate Terraria worlds through the internal worldgen script."""
    exit_code = worldgen_main(["--num-worlds", str(num_worlds), "--parallel", str(parallel)])
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# ---------------------------------------------------------------------------
# model commands
# ---------------------------------------------------------------------------

@model_app.command("train")
def model_train_command(
    data: Optional[str] = typer.Option(None, "--data", help="Path to preprocessed dataset (.pt) or cache dir"),
    world: Optional[str] = typer.Option(None, "--world", help="Path to a single .wld file"),
    epochs: int = typer.Option(50, "--epochs", help="Number of epochs"),
    one_pass: bool = typer.Option(False, "--one-pass", help="Train for exactly one pass over the data"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from checkpoint path"),
    disk_mode: bool = typer.Option(False, "--disk-mode", help="Use LRU disk-mode dataset loading"),
    cache_size: int = typer.Option(5, "--cache-size", help="LRU cache size for disk mode"),
    ema_decay: float = typer.Option(0.99, "--ema-decay", help="EMA decay rate"),
    ema_reset_multiplier: float = typer.Option(0.5, "--ema-reset-multiplier", help="Dead-code reset threshold (multiplier of uniform usage)"),
    ema_reset_interval: int = typer.Option(500, "--ema-reset-interval", help="Dead-code reset check interval (updates)"),
    beta: float = typer.Option(0.25, "--beta", help="Commitment loss weight"),
    block_loss_weighted: bool = typer.Option(False, "--block-loss-weighted", help="Use inverse-frequency block loss weighting"),
    block_weight_min: float = typer.Option(0.5, "--block-weight-min", help="Minimum block class weight clamp"),
    block_weight_max: float = typer.Option(5.0, "--block-weight-max", help="Maximum block class weight clamp"),
    no_ema: bool = typer.Option(False, "--no-ema", help="Disable EMA quantizer (use vanilla VQ-VAE)"),
    metrics_stride: int = typer.Option(50, "--metrics-stride", help="Store metrics every N updates"),
) -> None:
    """Train the VQ-VAE model."""
    run_train(argparse.Namespace(
        data=data,
        world=world,
        epochs=epochs,
        one_pass=one_pass,
        batch_size=batch_size,
        resume=resume,
        disk_mode=disk_mode,
        cache_size=cache_size,
        no_ema=no_ema,
        ema_decay=ema_decay,
        ema_reset_multiplier=ema_reset_multiplier,
        ema_reset_interval=ema_reset_interval,
        beta=beta,
        block_loss_weighted=block_loss_weighted,
        block_weight_min=block_weight_min,
        block_weight_max=block_weight_max,
        metrics_stride=metrics_stride,
    ))


@model_app.command("infer")
def model_infer_command(
    model: Optional[str] = typer.Argument(None, help="Checkpoint/model path (defaults to latest)"),
    world: Optional[str] = typer.Option(None, "--world", help="Path to .wld file (auto-detects if omitted)"),
    region: List[int] = typer.Option([1500, 1000, 1600, 1100], "--region", help="Inference region as X0 Y0 X1 Y1"),
    chunk_size: int = typer.Option(32, "--chunk-size", help="Chunk size"),
    samples: int = typer.Option(3, "--samples", help="Number of chunks to inspect"),
) -> None:
    """Run model inference and print reconstruction comparison."""
    if len(region) != 4:
        raise typer.BadParameter("--region requires exactly 4 integers: X0 Y0 X1 Y1")
    run_infer(argparse.Namespace(
        model=model,
        world=world,
        region=list(region),
        chunk_size=chunk_size,
        samples=samples,
    ))


@model_app.command("export")
def model_export_command(
    model: Optional[str] = typer.Argument(None, help="Checkpoint/model path (defaults to latest)"),
    world: Optional[str] = typer.Option(None, "--world", help="Path to .wld file (auto-detects if omitted)"),
    region: Optional[List[int]] = typer.Option(None, "--region", help="Region as X0 Y0 X1 Y1 (overrides --x/--y/--width/--height)"),
    x: int = typer.Option(1500, "--x", help="Starting X coordinate"),
    y: int = typer.Option(500, "--y", help="Starting Y coordinate"),
    width: int = typer.Option(64, "--width", help="Width of exported region (tiles)"),
    height: int = typer.Option(64, "--height", help="Height of exported region (tiles)"),
    output_dir: str = typer.Option("exports", "--output-dir", help="Output directory"),
    name: Optional[str] = typer.Option(None, "--name", help="Schematic name (shown in TEdit)"),
    mode: str = typer.Option("compare", "--mode", help="compare (original + reconstructed), reconstruct, or generate"),
    samples: int = typer.Option(1, "--samples", help="Number of regions to export"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for generation"),
) -> None:
    """Export inference results as TEdit schematics (.TEditSch).

    Examples:
        terrain model export --width 100 --height 50
        terrain model export --x 2000 --y 300 --width 64 --height 64 --samples 3
    """
    if region is not None and len(region) != 4:
        raise typer.BadParameter("--region requires exactly 4 integers: X0 Y0 X1 Y1")
    exit_code = run_export(argparse.Namespace(
        model=model,
        world=world,
        region=list(region) if region else None,
        x=x,
        y=y,
        width=width,
        height=height,
        output_dir=output_dir,
        name=name,
        mode=mode,
        samples=samples,
        seed=seed,
    ))
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


app.add_typer(data_app, name="data")
app.add_typer(model_app, name="model")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
