"""Export command for generating TEdit schematics from model inference."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from src.autoencoder.vqvae import VQVAE, DEFAULT_MODEL_CONFIG
from src.terraria.world_handler import load_world
from src.terraria.chunk_processor import extract_chunk
from src.utils.checkpoint import load_model_for_inference, read_checkpoint_config
from src.utils.device import get_device
from src.utils.tedit_export import (
    export_tensor_to_schematic,
    export_inference_result,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export model inference results as TEdit schematics"
    )
    parser.add_argument("model", nargs="?", help="Checkpoint/model path")
    parser.add_argument(
        "--world",
        help="Path to .wld file to extract region from",
    )
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Region to extract (overrides --width/--height)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=64,
        help="Width of exported region (tiles)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=64,
        help="Height of exported region (tiles)",
    )
    parser.add_argument(
        "--x",
        type=int,
        default=1500,
        help="Starting X coordinate in world",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=500,
        help="Starting Y coordinate in world",
    )
    parser.add_argument(
        "--output-dir",
        default="exports",
        help="Output directory for schematics",
    )
    parser.add_argument(
        "--name",
        help="Name for the schematic (shown in TEdit)",
    )
    parser.add_argument(
        "--mode",
        choices=["reconstruct", "generate", "compare"],
        default="compare",
        help="Export mode: compare (original + reconstructed), reconstruct (only output), generate (random)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of regions to export (offset by width each time)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for generation",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """Execute export from a pre-populated Namespace. Called by main() and the CLI."""
    device = get_device()
    print(f"Using device: {device}")

    # Find model checkpoint
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
                print("No checkpoints found in checkpoints/")
                return 1
        else:
            print("checkpoints directory not found. Please train a model first.")
            return 1

    # Determine region to export
    if args.region:
        x_start, y_start = args.region[0], args.region[1]
        width = args.region[2] - args.region[0]
        height = args.region[3] - args.region[1]
    else:
        x_start, y_start = args.x, args.y
        width, height = args.width, args.height

    print(f"\n{'=' * 80}")
    print("TEDIT SCHEMATIC EXPORT")
    print(f"{'=' * 80}")
    print(f"Model: {model_path}")
    print(f"Mode: {args.mode}")
    print(f"Size: {width}x{height} tiles")
    print(f"Output: {args.output_dir}/")

    # Load model — read architecture config from the checkpoint so the right
    # number of embeddings / dimensions are used even if they differ from the default.
    ckpt_config = read_checkpoint_config(model_path, str(device))
    if not ckpt_config:
        print("  [warn] No config found in checkpoint; using default architecture.")
    model_config = {**DEFAULT_MODEL_CONFIG, **ckpt_config}

    model = VQVAE(**model_config)
    model = load_model_for_inference(model, model_path, str(device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_paths = []

    if args.mode in ("reconstruct", "compare"):
        # Reconstruct from world data
        if not args.world:
            # Find first available world
            worldgen_dir = Path("worldgen")
            if worldgen_dir.exists():
                worlds = list(worldgen_dir.glob("*.wld"))
                if worlds:
                    args.world = str(worlds[0])
                    print(f"Using world: {args.world}")
                else:
                    print("No .wld files found in worldgen/")
                    return 1
            else:
                print("Please specify --world or have worlds in worldgen/")
                return 1

        # Load the world file
        print(f"\nLoading world: {args.world}")
        world = load_world(args.world)
        print(f"World size: {world.size.x}x{world.size.y}")

        print(f"\nExporting {args.samples} region(s) of {width}x{height} tiles...")

        for idx in range(args.samples):
            # Calculate region for this sample (offset by width for each sample)
            sample_x = x_start + (idx * width)
            
            # Check bounds
            if sample_x + width > world.size.x:
                print(f"  Sample {idx + 1}: Skipped (out of bounds)")
                continue
            if y_start + height > world.size.y:
                print(f"  Sample {idx + 1}: Skipped (out of bounds)")
                continue

            # Extract chunk directly from world
            chunk = extract_chunk(world, sample_x, y_start, width, height)
            
            # Convert to tensor: (H, W, C) -> (C, H, W) -> (1, C, H, W)
            original = torch.from_numpy(chunk).permute(2, 0, 1).unsqueeze(0).float().to(device)

            with torch.no_grad():
                _, reconstructed, perplexity, _ = model(original)

            original_np = original[0].cpu().numpy()
            reconstructed_np = reconstructed[0].cpu().numpy()

            # Calculate reconstruction accuracy
            mse = np.mean((original_np - reconstructed_np) ** 2)

            # Determine output paths
            if args.samples > 1:
                base_name = args.name or f"terrain_{timestamp}_{idx:03d}"
            else:
                base_name = args.name or f"terrain_{timestamp}"

            if args.mode == "compare":
                # Export both original and reconstructed
                orig_path, recon_path = export_inference_result(
                    original_np,
                    reconstructed_np,
                    output_dir,
                    base_name,
                )
                saved_paths.extend([orig_path, recon_path])
                print(f"\n  Chunk {idx + 1}/{args.samples} at ({sample_x}, {y_start}):")
                print(f"    ✓ Original:      {orig_path.name}")
                print(f"    ✓ Reconstructed: {recon_path.name}")
                print(f"    Perplexity: {perplexity.item():.2f}, MSE: {mse:.6f}")
            else:
                # Export only reconstructed
                output_path = export_tensor_to_schematic(
                    reconstructed_np,
                    output_dir / f"{base_name}.TEditSch",
                    name=args.name or f"Terrain Reforger - {timestamp}",
                )
                saved_paths.append(output_path)
                print(f"\n  Chunk {idx + 1}/{args.samples}: {output_path.name}")
                print(f"    Perplexity: {perplexity.item():.2f}, MSE: {mse:.6f}")

    elif args.mode == "generate":
        # Generate from random latent (future feature)
        print("\nGeneration mode: Creating from random latent codes...")

        if args.seed is not None:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)

        # For now, we'll create random input and reconstruct
        # In the future, this could sample from the latent space
        random_input = torch.randn(1, 9, height, width).to(device)

        # Clamp to valid ranges for categorical channels
        random_input[:, 0] = torch.randint(0, 218, (height, width)).float()
        random_input[:, 2] = torch.randint(0, 77, (height, width)).float()
        random_input[:, 4] = torch.randint(0, 5, (height, width)).float()

        # Binary channels
        for ch in [1, 3, 5, 6, 7, 8]:
            random_input[:, ch] = (torch.rand(height, width) > 0.5).float()

        with torch.no_grad():
            _, generated, perplexity, _ = model(random_input)

        generated_np = generated[0].cpu().numpy()

        base_name = args.name or f"generated_{timestamp}"
        output_path = export_tensor_to_schematic(
            generated_np,
            output_dir / f"{base_name}.TEditSch",
            name=args.name or f"Generated - {timestamp}",
        )
        saved_paths.append(output_path)
        print(f"\n✓ Generated: {output_path}")
        print(f"Perplexity: {perplexity.item():.2f}")

    print(f"\n{'=' * 80}")
    print("EXPORT COMPLETE")
    print(f"{'=' * 80}")

    print(f"\n✓ Region size: {width}x{height} tiles")
    print(f"✓ Saved {len(saved_paths)} schematic(s) to {output_dir}/")
    
    if args.mode == "compare":
        print(f"\nFiles exported in pairs:")
        print(f"  *_original.TEditSch     - Original chunk from world (before VAE)")
        print(f"  *_reconstructed.TEditSch - After passing through the VAE")
    
    print(f"\nTo import in TEdit:")
    print(f"  File → Import Schematic → Select .TEditSch file")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
