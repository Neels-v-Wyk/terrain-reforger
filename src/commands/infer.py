"""Inference command for optimized VQ-VAE."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch

from src.autoencoder.dataset_optimized import OptimizedTerrariaTileDataset
from src.autoencoder.vqvae_optimized import VQVAEOptimized, compute_optimized_loss, DEFAULT_MODEL_CONFIG
from src.utils.checkpoint import load_model_for_inference, read_checkpoint_config
from src.utils.device import get_device
from src.utils.visualization import compare_optimized_tiles


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with optimized VQ-VAE")
    parser.add_argument("model", nargs="?", help="Checkpoint/model path")
    parser.add_argument("--world", default=None, help="Path to .wld file (auto-detects from worldgen/ if omitted)")
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=[1500, 1000, 1600, 1100],
        help="Inference region",
    )
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size")
    parser.add_argument("--samples", type=int, default=3, help="Number of chunks to inspect")
    return parser


def run(args: argparse.Namespace) -> None:
    """Execute inference from a pre-populated Namespace. Called by main() and the CLI."""
    device = get_device()
    print(f"Using device: {device}")

    if args.model:
        model_path = args.model
    else:
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            latest = checkpoints_dir / "latest_model.pt"
            best = checkpoints_dir / "best_model.pt"
            if latest.exists():
                model_path = str(latest)
            elif best.exists():
                model_path = str(best)
            else:
                print("No optimized checkpoints found in checkpoints/")
                return
        else:
            print("checkpoints directory not found. Please train an optimized model first.")
            return

    print(f"\n{'=' * 80}")
    print("LOADING MODEL FOR INFERENCE")
    print(f"{'=' * 80}")

    ckpt_config = read_checkpoint_config(model_path, str(device))
    if not ckpt_config:
        print("  [warn] No config found in checkpoint; using default architecture.")
    model_config = {**DEFAULT_MODEL_CONFIG, **ckpt_config}

    model = VQVAEOptimized(**model_config)
    model = load_model_for_inference(model, model_path, str(device))

    print(f"{'=' * 80}\n")

    print("Loading test data...")
    world_path = args.world
    if not world_path:
        worldgen_dir = Path("worldgen")
        worlds = sorted(worldgen_dir.glob("*.wld")) if worldgen_dir.exists() else []
        if not worlds:
            print("No .wld files found. Specify --world or run 'terrain data worldgen' first.")
            return
        world_path = str(worlds[0])
        print(f"Auto-detected world: {world_path}")

    test_data = OptimizedTerrariaTileDataset(
        world_path=world_path,
        region=tuple(args.region),
        chunk_size=args.chunk_size,
        overlap=0,
        use_diversity_filter=False,
        deduplicate=False,
    )

    print(f"Test chunks: {len(test_data)}")

    print(f"\n{'=' * 80}")
    print("RUNNING INFERENCE")
    print(f"{'=' * 80}\n")

    model.eval()
    with torch.no_grad():
        for index in range(min(args.samples, len(test_data))):
            print(f"\nTest Chunk {index + 1}")
            print("-" * 80)

            sample = test_data[index].unsqueeze(0).to(device)
            embedding_loss, reconstructed, perplexity, logits = model(sample)
            total_loss, loss_dict = compute_optimized_loss(sample, reconstructed, logits, embedding_loss)

            print(f"Perplexity: {perplexity.item():.2f}")
            print(f"Total loss: {total_loss.item():.4f}")
            print(
                f"Component losses - block: {loss_dict['block']:.4f}, "
                f"wall: {loss_dict['wall']:.4f}, liquid: {loss_dict['liquid']:.4f}, "
                f"continuous: {loss_dict['continuous']:.4f}"
            )

            original_sample = sample[0].cpu().detach().numpy()
            reconstructed_sample = reconstructed[0].cpu().detach().numpy()
            mid_h, mid_w = args.chunk_size // 2, args.chunk_size // 2
            for tile_idx in range(3):
                original_tile = original_sample[:, mid_h, mid_w + tile_idx]
                reconstructed_tile = reconstructed_sample[:, mid_h, mid_w + tile_idx]
                print(compare_optimized_tiles(original_tile, reconstructed_tile, tile_idx + 1))
            print()

    print(f"{'=' * 80}")
    print("INFERENCE COMPLETE")
    print(f"{'=' * 80}\n")

    print("✓ Model loaded and tested successfully")
    print(f"✓ Model path: {model_path}")
    print(f"✓ Device: {device}")
    print(f"✓ Codebook size: {model_config['n_embeddings']}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
