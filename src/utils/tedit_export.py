"""
TEdit schematic export utilities using terraschem.

Converts model output tensors to TEdit-compatible .TEditSch files
that can be imported directly into TEdit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from terraschem import Tile, TileRegion, TileId, WallId, LiquidType, BrickStyle

from ..terraria.natural_ids import BLOCK_INDEX_TO_ID, WALL_INDEX_TO_ID


# Mapping from our block shape values (0-5) to BrickStyle enum
SHAPE_TO_BRICK_STYLE = {
    0: BrickStyle.FULL,
    1: BrickStyle.HALF_BRICK,
    2: BrickStyle.SLOPE_TOP_RIGHT,
    3: BrickStyle.SLOPE_TOP_LEFT,
    4: BrickStyle.SLOPE_BOTTOM_RIGHT,
    5: BrickStyle.SLOPE_BOTTOM_LEFT,
}

# Mapping from our liquid type values (0-4) to LiquidType enum
LIQUID_TYPE_MAP = {
    0: LiquidType.NONE,
    1: LiquidType.WATER,
    2: LiquidType.LAVA,
    3: LiquidType.HONEY,
    4: LiquidType.SHIMMER,
}


def tensor_to_tile(tensor_slice: np.ndarray) -> Tile:
    """
    Convert a single 8-channel tile tensor to a terraschem Tile.
    
    Args:
        tensor_slice: Array of shape (8,) representing one tile.
        
    Returns:
        terraschem Tile object.
    """
    # Channel indices (from vqvae.py)
    # 0: block_type, 1: block_shape (idx), 2: wall_type
    # 3: liquid_type (0=none)
    # 4: wire_red, 5: wire_blue, 6: wire_green, 7: actuator
    
    # Block type (index -> game ID)
    block_idx = int(round(tensor_slice[0]))
    block_idx = np.clip(block_idx, 0, len(BLOCK_INDEX_TO_ID) - 1)
    block_game_id = BLOCK_INDEX_TO_ID.get(block_idx, 0)
    
    # Block is active if it's not air (game ID 0)
    is_active = block_game_id > 0
    
    # Block shape (now direct index 0-5)
    shape_val = int(round(tensor_slice[1]))
    shape_val = np.clip(shape_val, 0, 5)
    brick_style = SHAPE_TO_BRICK_STYLE.get(shape_val, BrickStyle.FULL)
    
    # Wall type (index -> game ID)
    wall_idx = int(round(tensor_slice[2]))
    wall_idx = np.clip(wall_idx, 0, len(WALL_INDEX_TO_ID) - 1)
    wall_game_id = WALL_INDEX_TO_ID.get(wall_idx, 0)
    
    # Liquid
    liquid_type_idx = int(round(tensor_slice[3]))
    liquid_type_idx = np.clip(liquid_type_idx, 0, 4)
    
    if liquid_type_idx > 0:
        liquid_type = LIQUID_TYPE_MAP.get(liquid_type_idx, LiquidType.NONE)
        liquid_amount = 255  # Full liquid
    else:
        liquid_type = LiquidType.NONE
        liquid_amount = 0
    
    # Wiring
    wire_red = bool(tensor_slice[4] > 0.5)
    wire_blue = bool(tensor_slice[5] > 0.5)
    wire_green = bool(tensor_slice[6] > 0.5)
    actuator = bool(tensor_slice[7] > 0.5)
    
    return Tile(
        is_active=is_active,
        type_id=block_game_id,
        wall=wall_game_id,
        brick_style=brick_style,
        liquid_type=liquid_type,
        liquid_amount=liquid_amount,
        wire_red=wire_red,
        wire_blue=wire_blue,
        wire_green=wire_green,
        actuator=actuator,
    )


def tensor_to_region(
    tensor: torch.Tensor | np.ndarray,
    name: Optional[str] = None,
) -> TileRegion:
    """
    Convert a tensor of shape (C, H, W) or (H, W, C) to a TileRegion.
    
    Args:
        tensor: Tensor of shape (9, H, W) or (H, W, 9).
        name: Optional name for the schematic.
        
    Returns:
        TileRegion object that can be saved to .TEditSch.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach().numpy()
    
    # Handle both (C, H, W) and (H, W, C) formats
    if tensor.shape[0] == 9:
        # (C, H, W) -> (H, W, C)
        tensor = np.transpose(tensor, (1, 2, 0))
    
    height, width, channels = tensor.shape
    assert channels == 9, f"Expected 9 channels, got {channels}"
    
    # Create region - NOTE: TileRegion uses (width, height) order
    region = TileRegion(width, height)
    
    # Fill in tiles
    for y in range(height):
        for x in range(width):
            tile = tensor_to_tile(tensor[y, x])
            region.set_tile(x, y, tile)
    
    return region


def export_tensor_to_schematic(
    tensor: torch.Tensor | np.ndarray,
    output_path: str | Path,
    name: Optional[str] = None,
    version: int = 275,
) -> Path:
    """
    Export a tensor to a TEdit schematic file.
    
    Args:
        tensor: Tensor of shape (9, H, W) or (H, W, 9).
        output_path: Path to save the .TEditSch file.
        name: Optional name for the schematic (shown in TEdit).
        version: Terraria version number (default 275 for 1.4.4+).
        
    Returns:
        Path to the saved schematic file.
    """
    output_path = Path(output_path)
    
    # Ensure .TEditSch extension
    if output_path.suffix.lower() != ".teditsch":
        output_path = output_path.with_suffix(".TEditSch")
    
    # Convert tensor to region
    region = tensor_to_region(tensor, name)
    
    # Save to file
    region.save(str(output_path), name=name, version=version)
    
    return output_path


def export_batch_to_schematics(
    tensors: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    prefix: str = "generated",
    name_template: Optional[str] = None,
    version: int = 275,
) -> list[Path]:
    """
    Export a batch of tensors to TEdit schematic files.
    
    Args:
        tensors: Batch tensor of shape (B, 9, H, W).
        output_dir: Directory to save schematic files.
        prefix: Filename prefix for generated schematics.
        name_template: Template for schematic names (use {i} for index).
        version: Terraria version number.
        
    Returns:
        List of paths to saved schematic files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.cpu().detach().numpy()
    
    saved_paths = []
    for i, tensor in enumerate(tensors):
        filename = f"{prefix}_{i:04d}.TEditSch"
        output_path = output_dir / filename
        
        name = name_template.format(i=i) if name_template else f"{prefix} {i}"
        
        export_tensor_to_schematic(tensor, output_path, name=name, version=version)
        saved_paths.append(output_path)
    
    return saved_paths


def export_inference_result(
    original: torch.Tensor | np.ndarray,
    reconstructed: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    base_name: str = "inference",
) -> Tuple[Path, Path]:
    """
    Export both original and reconstructed tensors for comparison.
    
    Args:
        original: Original input tensor (9, H, W).
        reconstructed: Model reconstruction (9, H, W).
        output_dir: Directory to save files.
        base_name: Base name for the files.
        
    Returns:
        Tuple of (original_path, reconstructed_path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    original_path = export_tensor_to_schematic(
        original,
        output_dir / f"{base_name}_original.TEditSch",
        name=f"{base_name} (Original)",
    )
    
    reconstructed_path = export_tensor_to_schematic(
        reconstructed,
        output_dir / f"{base_name}_reconstructed.TEditSch",
        name=f"{base_name} (Reconstructed)",
    )
    
    return original_path, reconstructed_path
