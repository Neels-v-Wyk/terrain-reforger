"""Unified CLI for Terrain Reforger."""

from __future__ import annotations

import argparse
from typing import List, Optional

import typer
from rich.console import Console

from src.commands.analyze import main as analyze_main
from src.commands.diagnose import run as run_diagnose
from src.commands.export import run as run_export
from src.commands.infer import run as run_infer
from src.commands.lr_find import run as run_lr_find
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
    workers: Optional[int] = typer.Option(None, "--workers", help="Parallel worker processes (default: number of CPU cores)"),
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
        workers=workers,
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
    parallel: Optional[int] = typer.Option(None, "--parallel", help="Parallel generation jobs (default: all available CPU cores)"),
) -> None:
    """Generate Terraria worlds through the internal worldgen script."""
    argv = ["--num-worlds", str(num_worlds)]
    if parallel is not None:
        argv += ["--parallel", str(parallel)]
    exit_code = worldgen_main(argv)
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
    learning_rate: float = typer.Option(2e-4, "--learning-rate", "--lr", help="Learning rate"),
    optimizer: str = typer.Option("adam", "--optimizer", help="Optimizer: adam or adamw"),
    weight_decay: float = typer.Option(0.0, "--weight-decay", help="Weight decay for AdamW"),
    lr_schedule: str = typer.Option("none", "--lr-schedule", help="LR schedule: none, cosine, or onecycle"),
    warmup_fraction: float = typer.Option(0.05, "--warmup-fraction", help="Fraction of steps for warmup"),
    grad_clip: float = typer.Option(0.0, "--grad-clip", help="Gradient clipping max norm (0=disabled)"),
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
    use_focal_loss: bool = typer.Option(False, "--use-focal-loss", help="Use focal loss for block classification"),
    focal_alpha: float = typer.Option(1.0, "--focal-alpha", help="Focal loss alpha parameter"),
    focal_gamma: float = typer.Option(2.0, "--focal-gamma", help="Focal loss gamma parameter"),
    no_ema: bool = typer.Option(False, "--no-ema", help="Disable EMA quantizer (use vanilla VQ-VAE)"),
    metrics_stride: int = typer.Option(50, "--metrics-stride", help="Store metrics every N updates"),
    val_split: float = typer.Option(0.1, "--val-split", help="Fraction of data to use for validation"),
) -> None:
    """Train the VQ-VAE model."""
    run_train(argparse.Namespace(
        data=data,
        world=world,
        epochs=epochs,
        one_pass=one_pass,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
        warmup_fraction=warmup_fraction,
        grad_clip=grad_clip,
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
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        metrics_stride=metrics_stride,
        val_split=val_split,
    ))


@model_app.command("diagnose")
def model_diagnose_command(
    data: str = typer.Option(..., "--data", help="Path to preprocessed dataset (.pt) or cache dir"),
    subset_fraction: float = typer.Option(0.2, "--subset-fraction", help="Fraction of data to use"),
    epochs: int = typer.Option(1, "--epochs", help="Number of diagnostic epochs"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Optional checkpoint to load and evaluate"),
    output: str = typer.Option("checkpoints/diagnostic.json", "--output", help="Output path for report"),
    beta: float = typer.Option(0.25, "--beta", help="Commitment loss weight"),
    ema_decay: float = typer.Option(0.99, "--ema-decay", help="EMA decay rate"),
    no_ema: bool = typer.Option(False, "--no-ema", help="Disable EMA quantizer"),
    block_loss_weighted: bool = typer.Option(False, "--block-loss-weighted", help="Use class-weighted block loss"),
) -> None:
    """Run quick diagnostics on training setup (fast validation on data subset)."""
    run_diagnose(argparse.Namespace(
        data=data,
        subset_fraction=subset_fraction,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint=checkpoint,
        output=output,
        beta=beta,
        ema_decay=ema_decay,
        no_ema=no_ema,
        block_loss_weighted=block_loss_weighted,
    ))


@model_app.command("lr-find")
def model_lr_find_command(
    data: str = typer.Option(..., "--data", help="Path to preprocessed dataset"),
    min_lr: float = typer.Option(1e-6, "--min-lr", help="Minimum LR to test"),
    max_lr: float = typer.Option(1e-2, "--max-lr", help="Maximum LR to test"),
    steps: int = typer.Option(300, "--steps", help="Number of LR test steps"),
    subset_fraction: float = typer.Option(0.1, "--subset-fraction", help="Fraction of data to use"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
    smoothing: float = typer.Option(0.05, "--smoothing", help="Loss smoothing factor"),
    output: str = typer.Option("checkpoints/lr_finder.png", "--output", help="Output plot path"),
    beta: float = typer.Option(0.25, "--beta", help="Commitment loss weight"),
    no_plot: bool = typer.Option(False, "--no-plot", help="Don't display plot, just save"),
) -> None:
    """Find optimal learning rate using LR range test (Leslie Smith method)."""
    run_lr_find(argparse.Namespace(
        data=data,
        min_lr=min_lr,
        max_lr=max_lr,
        steps=steps,
        subset_fraction=subset_fraction,
        batch_size=batch_size,
        smoothing=smoothing,
        output=output,
        beta=beta,
        no_plot=no_plot,
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
