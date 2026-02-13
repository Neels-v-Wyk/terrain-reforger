#!/usr/bin/env python3
"""
Terrain Reforger

Example usage demonstrating chunk extraction and tensor conversion
for machine learning applications.
"""

import sys
from pathlib import Path

from src.terraria import (
    load_world,
    extract_chunk,
    iter_chunks,
    print_chunk_summary,
    get_tensor_statistics,
    validate_chunk,
)


def example_single_chunk():
    """Extract and display information about a single chunk."""
    print("=" * 70)
    print("Example 1: Single Chunk Extraction")
    print("=" * 70)
    
    # Load a world file
    world_path = Path("worldgen/World_20260211_213447_mBvegqrN.wld")
    if not world_path.exists():
        print(f"World file not found: {world_path}")
        print("Please place a .wld file in the worldgen/ directory")
        return
    
    world = load_world(world_path)
    print(f"\nWorld Info:")
    print(f"  Name: {world.name}")
    print(f"  Size: {world.size.x} x {world.size.y} tiles")
    
    # Extract a chunk from the middle of the world
    center_x = world.size.x // 2
    center_y = world.size.y // 2
    chunk = extract_chunk(world, center_x, center_y, width=64, height=64)
    
    # Print formatted summary
    print()
    print_chunk_summary(chunk)
    
    # Validate the chunk
    validation = validate_chunk(chunk)
    if validation['valid']:
        print("✓ Chunk validation passed")
    else:
        print("⚠ Chunk validation issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    
    if validation['warnings']:
        print(f"⚠ {validation['warning_count']} warnings (showing first 10)")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Convert to tensor for ML
    tensor = chunk.to_tensor()
    print("\nTensor Representation:")
    print(f"  Shape: {tensor.shape} (height, width, channels)")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Memory: {tensor.nbytes / 1024:.2f} KB")
    
    # Get detailed statistics
    stats = get_tensor_statistics(tensor)
    print("\nTensor Statistics:")
    print(f"  Tiles with blocks: {stats['tiles']['with_blocks']}/{stats['tiles']['total']}")
    print(f"  Tiles with slopes: {stats['tiles']['with_slopes']}")
    print(f"  Tiles with walls: {stats['tiles']['with_walls']}")
    print(f"  Tiles with liquid: {stats['tiles']['with_liquid']}")
    print(f"  Unique block types: {stats['block_types']['unique']}")
    
    if stats['tiles']['with_liquid'] > 0:
        print(f"\nLiquid Details:")
        print(f"  Water tiles: {stats['liquids']['water_tiles']}")
        print(f"  Lava tiles: {stats['liquids']['lava_tiles']}")
        print(f"  Honey tiles: {stats['liquids']['honey_tiles']}")
        print(f"  Avg liquid amount: {stats['liquids']['avg_amount']:.2%}")


def example_chunk_iteration():
    """Demonstrate iterating over all chunks in a world."""
    print("\n" + "=" * 70)
    print("Example 2: Chunk Iteration")
    print("=" * 70)
    
    world_path = Path("worldgen/World_20260211_213447_mBvegqrN.wld")
    if not world_path.exists():
        print(f"World file not found: {world_path}")
        return
    
    world = load_world(world_path)
    
    # Iterate over chunks with overlap
    chunk_size = 64
    overlap = 8
    
    print(f"\nIterating over chunks (size={chunk_size}, overlap={overlap})...")
    
    chunk_count = 0
    total_blocks = 0
    total_slopes = 0
    total_liquids = 0
    
    # Limit to a smaller region for demonstration
    region = (0, 0, min(512, world.size.x), min(512, world.size.y))
    print(f"Region: ({region[0]}, {region[1]}) to ({region[2]}, {region[3]})")
    
    for chunk in iter_chunks(world, chunk_size=chunk_size, overlap=overlap, region=region):
        chunk_count += 1
        summary = chunk.get_summary()
        total_blocks += summary['active_blocks']
        total_slopes += summary['slopes']
        total_liquids += summary['liquids']
        
        if chunk_count <= 3:  # Show first few chunks
            print(f"  Chunk {chunk_count}: pos=({chunk.x_start:4d}, {chunk.y_start:4d}), "
                  f"blocks={summary['active_blocks']:4d}, "
                  f"slopes={summary['slopes']:3d}, "
                  f"liquids={summary['liquids']:3d}")
    
    print(f"\nProcessed {chunk_count} chunks")
    print(f"  Total blocks: {total_blocks:,}")
    print(f"  Total slopes: {total_slopes:,}")
    print(f"  Total liquids: {total_liquids:,}")
    print(f"  Avg blocks per chunk: {total_blocks / chunk_count:.1f}")


def main():
    """Run example demonstrations."""
    print("\n" + "=" * 70)
    print("TERRAIN REFORGER - Terraria ML Data Processing")
    print("=" * 70)
    
    try:
        example_single_chunk()
        example_chunk_iteration()
        
        print("\n" + "=" * 70)
        print("Examples completed successfully!")
        print("=" * 70)
 
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have a Terraria world file (.wld) in the worldgen/ directory")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
