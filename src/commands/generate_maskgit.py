"""Generate terrain using MaskGIT with streaming support."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Callable, Iterator, Dict, Any

import torch
import numpy as np

from src.autoencoder.vqvae import VQVAE, DEFAULT_MODEL_CONFIG
from src.generative.maskgit import TerrainMaskGIT
from src.utils.checkpoint import load_model_for_inference
from src.utils.device import get_device
from src.utils.tedit_export import export_tensor_to_schematic


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate terrain with MaskGIT + streaming visualization"
    )
    
    # Model paths
    parser.add_argument(
        "--maskgit",
        type=str,
        default="checkpoints/transformer/best_model.pt",
        help="Path to trained MaskGIT checkpoint",
    )
    parser.add_argument(
        "--vqvae",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to VQVAE checkpoint",
    )
    
    # Generation parameters
    parser.add_argument("--width", type=int, default=256, help="Width in tiles")
    parser.add_argument("--height", type=int, default=256, help="Height in tiles")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of regions to generate")
    
    # MaskGIT parameters
    parser.add_argument("--num-iterations", type=int, default=12, help="Number of decoding iterations")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 to disable)")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling (0 to disable)")
    parser.add_argument("--schedule", type=str, default="cosine", 
                        choices=["linear", "cosine", "exponential"],
                        help="Unmasking schedule")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="exports/generated", help="Output directory")
    parser.add_argument("--export-format", type=str, default="tedit",
                        choices=["tedit", "numpy", "both"], help="Export format")
    parser.add_argument("--visualize", action="store_true", help="Show streaming progress")
    parser.add_argument("--save-tokens", action="store_true", help="Save token sequences")
    
    # Performance
    parser.add_argument("--batch-chunks", type=int, default=1, 
                        help="Generate multiple chunks in parallel")
    parser.add_argument("--progressive-decode", action="store_true",
                        help="Show progressive refinement during generation")
    parser.add_argument("--progressive-steps", type=int, default=3,
                        help="Number of progressive update steps (if --progressive-decode)")
    
    parser.add_argument("--seed", type=int, help="Random seed")
    
    return parser


class MaskGITStreamingGenerator:
    """
    Generates terrain using MaskGIT with chunk-by-chunk streaming.
    
    Much faster than autoregressive due to parallel token generation.
    """
    
    def __init__(
        self,
        maskgit: TerrainMaskGIT,
        vqvae: VQVAE,
        device: torch.device,
    ):
        self.maskgit = maskgit
        self.vqvae = vqvae
        self.device = device
        
        self.maskgit.eval()
        self.vqvae.eval()
        
        self.chunk_size = 32  # Fixed by VQVAE architecture
    
    def generate_region(
        self,
        width: int,
        height: int,
        num_iterations: int = 12,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        schedule: str = 'cosine',
        progressive_decode: bool = False,
        progressive_steps: int = 3,
        batch_size: int = 1,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate a region with streaming updates.
        
        Args:
            width: Width in tiles
            height: Height in tiles
            num_iterations: Number of MaskGIT decoding iterations
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            schedule: Unmasking schedule
            progressive_decode: Show progressive refinement
            progressive_steps: Number of intermediate decoding steps to show
            batch_size: Generate multiple chunks in parallel
            
        Yields:
            Dict with keys:
                - progress: float 0-100
                - chunk_x, chunk_y: chunk position
                - terrain: (32, 32, 9) terrain data
                - tokens: (64,) token sequence
                - time_elapsed: seconds elapsed
                - eta: estimated time remaining
        """
        # Calculate grid dimensions
        chunks_x = (width + self.chunk_size - 1) // self.chunk_size
        chunks_y = (height + self.chunk_size - 1) // self.chunk_size
        total_chunks = chunks_x * chunks_y
        
        # Initialize full region
        full_width = chunks_x * self.chunk_size
        full_height = chunks_y * self.chunk_size
        full_region = np.zeros((9, full_height, full_width), dtype=np.float32)
        
        start_time = time.time()
        
        # Generate chunks
        for chunk_idx in range(0, total_chunks, batch_size):
            chunk_batch_start = time.time()
            
            # Determine how many chunks in this batch
            actual_batch = min(batch_size, total_chunks - chunk_idx)
            
            # Progressive decoding: show intermediate states
            if progressive_decode:
                iter_checkpoints = [
                    num_iterations * (i + 1) // (progressive_steps + 1)
                    for i in range(progressive_steps)
                ]
                iter_checkpoints.append(num_iterations)  # Always include full generation
            else:
                iter_checkpoints = [num_iterations]
            
            # Generate tokens for this batch
            for target_iterations in iter_checkpoints:
                # Generate token batch with MaskGIT
                tokens_batch = self.maskgit.generate(
                    num_samples=actual_batch,
                    num_iterations=target_iterations,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p > 0 else None,
                    schedule=schedule,
                    device=self.device,
                )  # (actual_batch, 64)
                
                # Decode each chunk
                for batch_offset in range(actual_batch):
                    global_chunk_idx = chunk_idx + batch_offset
                    chunk_y = global_chunk_idx // chunks_x
                    chunk_x = global_chunk_idx % chunks_x
                    
                    tokens = tokens_batch[batch_offset]  # (64,)
                    
                    # Reshape to 8×8 grid for VQVAE
                    token_grid = tokens.reshape(8, 8).unsqueeze(0)  # (1, 8, 8)
                    
                    # Decode with VQVAE
                    with torch.no_grad():
                        # Get embeddings from quantizer
                        if hasattr(self.vqvae.vq, 'embedding'):
                            # VectorQuantizer with nn.Embedding
                            z_q = self.vqvae.vq.embedding(token_grid)
                        else:
                            # VectorQuantizerEMA with buffer
                            z_q = torch.nn.functional.embedding(
                                token_grid,
                                self.vqvae.vq.embedding if hasattr(self.vqvae.vq, 'embedding') else self.vqvae.vq._embedding
                            )
                        
                        # z_q shape: (1, 8, 8, embedding_dim)
                        # Decoder expects (1, embedding_dim, 8, 8)
                        z_q = z_q.permute(0, 3, 1, 2)
                        
                        # Decode to terrain
                        terrain_tensor = self.vqvae.decoder(z_q)  # (1, 9, 32, 32)
                        terrain = terrain_tensor.squeeze(0).cpu().numpy()  # (9, 32, 32)
                    
                    # Store in full region
                    y_start = chunk_y * self.chunk_size
                    x_start = chunk_x * self.chunk_size
                    full_region[:, y_start:y_start+self.chunk_size, x_start:x_start+self.chunk_size] = terrain
                    
                    # Calculate progress only after full generation (not progressive steps)
                    if target_iterations == num_iterations:
                        elapsed = time.time() - start_time
                        progress = ((global_chunk_idx + 1) / total_chunks) * 100
                        avg_time_per_chunk = elapsed / (global_chunk_idx + 1) if global_chunk_idx > 0 else elapsed
                        remaining_chunks = total_chunks - (global_chunk_idx + 1)
                        eta = avg_time_per_chunk * remaining_chunks
                        
                        yield {
                            'progress': progress,
                            'chunk_x': chunk_x,
                            'chunk_y': chunk_y,
                            'terrain': terrain,
                            'tokens': tokens.cpu().numpy(),
                            'time_elapsed': elapsed,
                            'eta': eta,
                            'is_progressive': False,
                        }
                    else:
                        # Progressive update (don't count in overall progress)
                        yield {
                            'progress': ((global_chunk_idx) / total_chunks) * 100,
                            'chunk_x': chunk_x,
                            'chunk_y': chunk_y,
                            'terrain': terrain,
                            'tokens': tokens.cpu().numpy(),
                            'time_elapsed': time.time() - start_time,
                            'eta': 0,
                            'is_progressive': True,
                            'iterations_done': target_iterations,
                        }
        
        # Return final full region
        yield {
            'progress': 100.0,
            'full_region': full_region,
            'time_elapsed': time.time() - start_time,
        }


def run(args: argparse.Namespace) -> None:
    """Main generation function."""
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load models
    print(f"\nLoading MaskGIT from {args.maskgit}...")
    checkpoint = torch.load(args.maskgit, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    from src.generative.maskgit import TerrainMaskGIT
    maskgit = TerrainMaskGIT(
        vocab_size=config['vocab_size'],
        seq_len=64,
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        grid_size=config['grid_size'],
        mask_token_id=config['mask_token_id'],
    ).to(device)
    maskgit.load_state_dict(checkpoint['model_state_dict'])
    maskgit.eval()
    
    print(f"Loading VQVAE from {args.vqvae}...")
    vqvae = load_model_for_inference(args.vqvae, VQVAE, DEFAULT_MODEL_CONFIG, device)
    
    # Create generator
    generator = MaskGITStreamingGenerator(
        maskgit=maskgit,
        vqvae=vqvae,
        device=device,
    )
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} region(s) of {args.width}×{args.height} tiles...")
    print(f"MaskGIT iterations: {args.num_iterations}, Schedule: {args. schedule}")
    
    for sample_idx in range(args.num_samples):
        print(f"\n{'='*60}")
        print(f"Sample {sample_idx + 1}/{args.num_samples}")
        print(f"{'='*60}")
        
        # Generate with streaming
        full_region = None
        all_tokens = []
        
        for update in generator.generate_region(
            width=args.width,
            height=args.height,
            num_iterations=args.num_iterations,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            schedule=args.schedule,
            progressive_decode=args.progressive_decode,
            progressive_steps=args.progressive_steps,
            batch_size=args.batch_chunks,
        ):
            if 'full_region' in update:
                # Final result
                full_region = update['full_region']
                total_time = update['time_elapsed']
                print(f"\n✓ Generation complete in {total_time:.2f}s")
            elif args.visualize:
                # Progress update
                if update.get('is_progressive', False):
                    print(f"  Chunk ({update['chunk_x']}, {update['chunk_y']}): "
                          f"iteration {update['iterations_done']}/{args.num_iterations} "
                          f"(progressive refinement)")
                else:
                    print(f"  [{update['progress']:5.1f}%] "
                          f"Chunk ({update['chunk_x']}, {update['chunk_y']}) | "
                          f"Elapsed: {update['time_elapsed']:.1f}s | "
                          f"ETA: {update['eta']:.1f}s")
                    
                    if args.save_tokens:
                        all_tokens.append(update['tokens'])
        
        # Save outputs
        if full_region is not None:
            timestamp = int(time.time())
            base_name = f"maskgit_sample{sample_idx+1}_{timestamp}"
            
            if args.export_format in ["numpy", "both"]:
                numpy_path = output_dir / f"{base_name}.npy"
                np.save(numpy_path, full_region)
                print(f"  ✓ Saved numpy: {numpy_path}")
            
            if args.export_format in ["tedit", "both"]:
                tedit_path = output_dir / f"{base_name}.TEditSch"
                export_tensor_to_schematic(
                    full_region,
                    str(tedit_path),
                    name=f"MaskGIT Terrain {sample_idx+1}"
                )
                print(f"  ✓ Saved TEdit schematic: {tedit_path}")
            
            if args.save_tokens and all_tokens:
                tokens_path = output_dir / f"{base_name}_tokens.npy"
                tokens_array = np.array(all_tokens)
                np.save(tokens_path, tokens_array)
                print(f"  ✓ Saved tokens: {tokens_path}")
    
    print("\n✓ All samples generated successfully!")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
