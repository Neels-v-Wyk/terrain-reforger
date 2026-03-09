"""Generate terrain using trained transformer + VQVAE with streaming support."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence, Callable, Iterator, Dict, Any

import torch
import numpy as np

from src.autoencoder.vqvae import VQVAE, DEFAULT_MODEL_CONFIG
from src.generative.transformer import TerrainTransformer, MEDIUM_CONFIG
from src.utils.checkpoint import load_model_for_inference, read_checkpoint_config
from src.utils.device import get_device
from src.terraria.converters import tensor_to_tile_array
from src.utils.tedit_export import export_to_tedit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate terrain with streaming visualization"
    )
    
    # Model paths
    parser.add_argument(
        "--transformer",
        type=str,
        help="Path to trained transformer checkpoint",
    )
    parser.add_argument(
        "--vqvae",
        type=str,
        help="Path to trained VQVAE checkpoint (default: checkpoints/best_model.pt)",
    )
    
    # Generation parameters
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Width in tiles (will be rounded to nearest 32)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Height in tiles (will be rounded to nearest 32)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of different regions to generate",
    )
    
    # Sampling parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (lower = more conservative, higher = more creative)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (0 to disable)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold (0 to disable)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports/generated",
        help="Output directory for generated terrain",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="tedit",
        choices=["tedit", "numpy", "both"],
        help="Export format (TEdit schematic, numpy array, or both)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show streaming visualization (prints progress)",
    )
    parser.add_argument(
        "--save-tokens",
        action="store_true",
        help="Save generated token sequences",
    )
    
    # Performance
    parser.add_argument(
        "--batch-chunks",
        type=int,
        default=1,
        help="Generate multiple chunks in parallel (faster but less smooth streaming)",
    )
    
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    return parser


class StreamingTerrainGenerator:
    """
    Generates terrain with chunk-by-chunk streaming support.
    
    Each chunk (32×32 tiles) is generated and decoded independently,
    allowing for real-time visualization and progressive rendering.
    """
    
    def __init__(
        self,
        transformer: TerrainTransformer,
        vqvae: VQVAE,
        device: torch.device,
    ):
        self.transformer = transformer
        self.vqvae = vqvae
        self.device = device
        
        self.transformer.eval()
        self.vqvae.eval()
        
        self.chunk_size = 32  # Fixed by VQVAE architecture
    
    def generate_region(
        self,
        width: int,
        height: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        on_chunk_complete: Optional[Callable[[int, int, int, int, np.ndarray, float], None]] = None,
        batch_size: int = 1,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate a region with streaming updates.
        
        Args:
            width: Total width in tiles (rounded to nearest 32)
            height: Total height in tiles (rounded to nearest 32)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            on_chunk_complete: Callback(chunk_x, chunk_y, x, y, terrain, elapsed_time)
            batch_size: Number of chunks to generate in parallel
            
        Yields:
            dict: Progress updates with keys:
                - 'chunk_idx': Current chunk index
                - 'total_chunks': Total number of chunks
                - 'progress': Progress as fraction (0-1)
                - 'chunk_pos': (chunk_x, chunk_y) position in grid
                - 'tile_pos': (x, y) position in tiles
                - 'terrain': (8, 32, 32) numpy array of generated terrain
                - 'tokens': (8, 8) token grid
                - 'elapsed': Time taken for this chunk
                - 'estimated_remaining': Estimated time remaining
        """
        # Round to nearest chunk boundaries
        chunks_x = (width + self.chunk_size - 1) // self.chunk_size
        chunks_y = (height + self.chunk_size - 1) // self.chunk_size
        total_chunks = chunks_x * chunks_y
        
        actual_width = chunks_x * self.chunk_size
        actual_height = chunks_y * self.chunk_size
        
        print(f"\nGenerating {actual_width}×{actual_height} region ({chunks_x}×{chunks_y} chunks)")
        print(f"Total chunks: {total_chunks}")
        print(f"Streaming: {batch_size} chunk(s) at a time")
        print(f"Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}\n")
        
        chunk_idx = 0
        times = []
        
        with torch.no_grad():
            # Generate chunks row by row
            for cy in range(chunks_y):
                for cx_start in range(0, chunks_x, batch_size):
                    cx_end = min(cx_start + batch_size, chunks_x)
                    current_batch_size = cx_end - cx_start
                    
                    # Generate batch of chunks
                    start_time = time.time()
                    
                    # Generate tokens
                    tokens = self.transformer.generate(
                        num_samples=current_batch_size,
                        temperature=temperature,
                        top_k=top_k if top_k > 0 else None,
                        top_p=top_p if top_p > 0 else None,
                        device=self.device,
                    )  # (batch, 64)
                    
                    # Decode through VQVAE
                    tokens_2d = tokens.view(current_batch_size, 8, 8)
                    
                    # Process each chunk in the batch
                    for i, cx in enumerate(range(cx_start, cx_end)):
                        # Get tokens for this chunk
                        chunk_tokens = tokens_2d[i:i+1]  # (1, 8, 8)
                        
                        # Look up embeddings and decode
                        terrain_tensor = self._decode_tokens(chunk_tokens)  # (1, 8, 32, 32)
                        
                        # Convert to numpy
                        terrain = terrain_tensor.squeeze(0).cpu().numpy()  # (8, 32, 32)
                        
                        elapsed = time.time() - start_time
                        times.append(elapsed)
                        
                        # Calculate positions
                        tile_x = cx * self.chunk_size
                        tile_y = cy * self.chunk_size
                        
                        # Estimate remaining time
                        avg_time = np.mean(times[-10:]) if times else elapsed
                        remaining_chunks = total_chunks - (chunk_idx + 1)
                        estimated_remaining = avg_time * remaining_chunks
                        
                        # Callback
                        if on_chunk_complete:
                            on_chunk_complete(cx, cy, tile_x, tile_y, terrain, elapsed)
                        
                        # Yield progress
                        yield {
                            'chunk_idx': chunk_idx,
                            'total_chunks': total_chunks,
                            'progress': (chunk_idx + 1) / total_chunks,
                            'chunk_pos': (cx, cy),
                            'tile_pos': (tile_x, tile_y),
                            'terrain': terrain,
                            'tokens': chunk_tokens.squeeze(0).cpu().numpy(),
                            'elapsed': elapsed,
                            'estimated_remaining': estimated_remaining,
                        }
                        
                        chunk_idx += 1
    
    def _decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode token grid through VQVAE.
        
        Args:
            tokens: (batch, 8, 8) token indices
            
        Returns:
            terrain: (batch, 8, 32, 32) decoded tiles
        """
        batch_size = tokens.shape[0]
        
        # Flatten to (batch, 64) for embedding lookup
        token_indices = tokens.view(batch_size, -1)
        
        # Look up embeddings in VQVAE codebook
        # Handle both VectorQuantizer and VectorQuantizerEMA
        if hasattr(self.vqvae.vq, 'embedding'):
            # VectorQuantizerEMA - embedding is a buffer
            embeddings = torch.nn.functional.embedding(token_indices, self.vqvae.vq.embedding)
        else:
            # VectorQuantizer - embedding is an Embedding layer
            embeddings = self.vqvae.vq.embedding(token_indices)
        
        # Reshape to (batch, 64, h_dim) -> (batch, h_dim, 8, 8)
        h_dim = embeddings.shape[-1]
        embeddings = embeddings.view(batch_size, 8, 8, h_dim)
        embeddings = embeddings.permute(0, 3, 1, 2).contiguous()
        
        # Decode through VQVAE decoder
        decoded = self.vqvae.conv_decoder(embeddings)
        terrain, _ = self.vqvae.tile_decoder(decoded)
        
        return terrain


def run(args: argparse.Namespace) -> None:
    """Execute generation."""
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    device = get_device()
    print(f"Using device: {device}\n")
    
    # -------------------------------------------------------------------------
    # Load models
    # -------------------------------------------------------------------------
    print(f"{'=' * 80}")
    print("LOADING MODELS")
    print(f"{'=' * 80}\n")
    
    # Load VQVAE
    if args.vqvae:
        vqvae_path = args.vqvae
    else:
        vqvae_path = "checkpoints/best_model.pt"
    
    print(f"Loading VQVAE: {vqvae_path}")
    vqvae_config = read_checkpoint_config(vqvae_path, str(device))
    if not vqvae_config:
        vqvae_config = {}
    vqvae_config = {**DEFAULT_MODEL_CONFIG, **vqvae_config}
    
    vqvae = VQVAE(**vqvae_config)
    vqvae = load_model_for_inference(vqvae, vqvae_path, str(device))
    print(f"✓ VQVAE loaded (codebook size: {vqvae_config['n_embeddings']})\n")
    
    # Load Transformer
    if args.transformer:
        transformer_path = args.transformer
    else:
        transformer_dir = Path("checkpoints/transformer")
        if (transformer_dir / "best_model.pt").exists():
            transformer_path = str(transformer_dir / "best_model.pt")
        elif (transformer_dir / "final_model.pt").exists():
            transformer_path = str(transformer_dir / "final_model.pt")
        else:
            print("❌ No transformer checkpoint found!")
            print("   Expected: checkpoints/transformer/best_model.pt")
            print("   Train a model first: terrain gen train")
            return
    
    print(f"Loading Transformer: {transformer_path}")
    checkpoint = torch.load(transformer_path, map_location=device, weights_only=False)
    transformer_config = checkpoint.get('model_config', MEDIUM_CONFIG)
    
    transformer = TerrainTransformer(
        vocab_size=vqvae_config['n_embeddings'],
        seq_len=64,
        **transformer_config
    )
    transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer = transformer.to(device)
    transformer.eval()
    
    print(f"✓ Transformer loaded")
    print(f"  Layers: {transformer_config['n_layers']}")
    print(f"  Dim: {transformer_config['d_model']}")
    print(f"  Heads: {transformer_config['n_heads']}")
    
    if 'val_metrics' in checkpoint:
        val_loss = checkpoint['val_metrics'].get('loss', 'N/A')
        val_ppl = checkpoint['val_metrics'].get('perplexity', 'N/A')
        print(f"  Val Loss: {val_loss}")
        print(f"  Val Perplexity: {val_ppl}")
    
    print()
    
    # -------------------------------------------------------------------------
    # Generate terrain
    # -------------------------------------------------------------------------
    print(f"{'=' * 80}")
    print("GENERATING TERRAIN (STREAMING)")
    print(f"{'=' * 80}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = StreamingTerrainGenerator(transformer, vqvae, device)
    
    # Progress callback
    def on_chunk_complete(cx, cy, x, y, terrain, elapsed):
        if args.visualize:
            print(f"  ✓ Chunk [{cx:2d},{cy:2d}] @ ({x:4d},{y:4d}) - {elapsed*1000:.1f}ms")
    
    for sample_idx in range(args.num_samples):
        print(f"\n{'─' * 80}")
        print(f"Sample {sample_idx + 1}/{args.num_samples}")
        print(f"{'─' * 80}\n")
        
        # Full region tensor
        chunks_x = (args.width + 31) // 32
        chunks_y = (args.height + 31) // 32
        full_width = chunks_x * 32
        full_height = chunks_y * 32
        full_region = np.zeros((8, full_height, full_width), dtype=np.float32)
        
        # Token grid
        if args.save_tokens:
            token_grid = np.zeros((chunks_y * 8, chunks_x * 8), dtype=np.int64)
        
        start_time = time.time()
        
        # Generate with streaming
        for update in generator.generate_region(
            width=args.width,
            height=args.height,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p > 0 else None,
            on_chunk_complete=on_chunk_complete if args.visualize else None,
            batch_size=args.batch_chunks,
        ):
            # Place chunk in full region
            cx, cy = update['chunk_pos']
            x, y = update['tile_pos']
            terrain = update['terrain']
            
            full_region[:, y:y+32, x:x+32] = terrain
            
            if args.save_tokens:
                tokens = update['tokens']
                token_grid[cy*8:(cy+1)*8, cx*8:(cx+1)*8] = tokens
        
        total_time = time.time() - start_time
        
        print(f"\n✓ Generation complete!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg per chunk: {total_time/(chunks_x*chunks_y)*1000:.1f}ms")
        print(f"  Throughput: {full_width*full_height/total_time:.0f} tiles/sec")
        
        # -------------------------------------------------------------------------
        # Export
        # -------------------------------------------------------------------------
        print(f"\nExporting...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"generated_{timestamp}_sample{sample_idx+1}"
        
        if args.export_format in ["numpy", "both"]:
            numpy_path = output_dir / f"{base_name}.npy"
            np.save(numpy_path, full_region)
            print(f"  ✓ Saved numpy: {numpy_path}")
        
        if args.export_format in ["tedit", "both"]:
            tedit_path = output_dir / f"{base_name}.TEditSch"
            tile_array = tensor_to_tile_array(torch.from_numpy(full_region))
            export_to_tedit(
                tile_array,
                str(tedit_path),
                name=f"Generated Terrain {sample_idx+1}",
                width=full_width,
                height=full_height,
            )
            print(f"  ✓ Saved TEdit schematic: {tedit_path}")
        
        if args.save_tokens:
            token_path = output_dir / f"{base_name}_tokens.npy"
            np.save(token_path, token_grid)
            print(f"  ✓ Saved tokens: {token_path}")
    
    print(f"\n{'=' * 80}")
    print("✓ ALL SAMPLES GENERATED")
    print(f"{'=' * 80}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
