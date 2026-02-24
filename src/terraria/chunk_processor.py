"""
Chunk processor for 8-channel format.
"""

import numpy as np
import lihzahrd
from .converters import tile_to_array


# ---------------------------------------------------------------------------
# Vectorised lookup tables, built once at import time from natural_ids
# ---------------------------------------------------------------------------

def _build_block_lut() -> np.ndarray:
    """Dense array: game_block_id -> compact index (0 for unknown IDs)."""
    from .natural_ids import BLOCK_ID_TO_INDEX
    max_id = max(BLOCK_ID_TO_INDEX) + 1
    lut = np.zeros(max_id, dtype=np.uint8)
    for gid, idx in BLOCK_ID_TO_INDEX.items():
        lut[gid] = idx
    return lut


def _build_wall_lut() -> np.ndarray:
    """Dense array: game_wall_id -> compact index (0 for unknown IDs)."""
    from .natural_ids import WALL_ID_TO_INDEX
    max_id = max(WALL_ID_TO_INDEX) + 1
    lut = np.zeros(max_id, dtype=np.uint8)
    for gid, idx in WALL_ID_TO_INDEX.items():
        lut[gid] = idx
    return lut


_BLOCK_LUT: np.ndarray = _build_block_lut()
_WALL_LUT: np.ndarray = _build_wall_lut()


def world_to_array(world: lihzahrd.World) -> np.ndarray:
    """
    Convert an entire lihzahrd World to a (world_H, world_W, 8) float32 array
    in a single pass over all tiles.

    **Vectorised implementation**: each tile attribute is written into a
    pre-allocated integer channel-array (no per-tile Python object creation,
    no dict lookup in the inner loop).  After the walk the compact block/wall
    indices are computed in one NumPy fancy-index operation per channel.

    Memory:  world_H * world_W * 8 * 4 bytes (~600 MB for a large world).
    The 9 temporary uint8/uint16/bool channel arrays are freed before return.

    Args:
        world: Loaded lihzahrd World object

    Returns:
        numpy array of shape (world_H, world_W, 8) with dtype float32
    """
    W = world.size.x
    H = world.size.y

    # Pre-allocate per-channel raw-value arrays, no per-tile heap allocation
    block_game_ids = np.zeros((H, W), dtype=np.uint16)  # BlockType.value
    block_active   = np.zeros((H, W), dtype=bool)
    block_shapes   = np.zeros((H, W), dtype=np.uint8)   # Shape.value  0-5
    wall_game_ids  = np.zeros((H, W), dtype=np.uint16)  # WallType.value
    liquid_types   = np.zeros((H, W), dtype=np.uint8)   # LiquidType.value 0-4
    wire_red       = np.zeros((H, W), dtype=bool)
    wire_blue      = np.zeros((H, W), dtype=bool)
    wire_green     = np.zeros((H, W), dtype=bool)
    actuator       = np.zeros((H, W), dtype=bool)

    # Single tile walk, raw attribute extraction only.
    # Avoids: np.zeros(8) per tile, dict lookups, hasattr checks, string ops.
    # LiquidType.value already equals 1-4 (WATER/LAVA/HONEY/SHIMMER), matching
    # the channel encoding directly.
    for y in range(H):
        for x in range(W):
            tile = world.tiles[x, y]

            b = tile.block
            if b is not None and b.is_active:
                block_active[y, x] = True
                block_game_ids[y, x] = b.type.value
                sv = b.shape.value
                block_shapes[y, x] = sv if sv <= 5 else 0

            w = tile.wall
            if w is not None:
                wall_game_ids[y, x] = w.type.value

            liq = tile.liquid
            if liq is not None and liq.volume > 0:
                liquid_types[y, x] = liq.type.value  # 1=WATER 2=LAVA 3=HONEY 4=SHIMMER

            wiring = tile.wiring
            if wiring:  # __bool__ short-circuits when all flags are False
                wire_red[y, x]   = wiring.red
                wire_blue[y, x]  = wiring.blue
                wire_green[y, x] = wiring.green
                actuator[y, x]   = wiring.actuator

    # Vectorised remapping and channel assembly.
    # Pre-allocate the final (H, W, 8) float32 array once, then write each
    # channel directly into a slice and immediately del the intermediate.
    # This avoids the np.stack peak where all 8 float32 channels + the output
    # coexist in memory simultaneously (saves ~250 MB for a large world).
    arr = np.empty((H, W, 8), dtype=np.float32)

    # Channel 0: block compact index (0 = air / inactive)
    safe_block_ids = np.minimum(block_game_ids, len(_BLOCK_LUT) - 1)
    del block_game_ids
    arr[:, :, 0] = _BLOCK_LUT[safe_block_ids]  # uint8 -> float32
    del safe_block_ids
    arr[:, :, 0][~block_active] = 0.0
    del block_active

    # Channel 1: block shape
    arr[:, :, 1] = block_shapes
    del block_shapes

    # Channel 2: wall compact index
    safe_wall_ids = np.minimum(wall_game_ids, len(_WALL_LUT) - 1)
    del wall_game_ids
    arr[:, :, 2] = _WALL_LUT[safe_wall_ids]
    del safe_wall_ids

    # Channels 3-7: liquid type, wiring
    arr[:, :, 3] = liquid_types;  del liquid_types
    arr[:, :, 4] = wire_red;      del wire_red
    arr[:, :, 5] = wire_blue;     del wire_blue
    arr[:, :, 6] = wire_green;    del wire_green
    arr[:, :, 7] = actuator;      del actuator

    return arr


def extract_chunk_from_array(
    world_array: np.ndarray,
    x_start: int,
    y_start: int,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Extract a chunk from a pre-built world array using a NumPy slice.

    O(chunk_size²) memory copy entirely in C, no Python-level tile loop.
    Handles out-of-bounds regions by zero-padding.

    Args:
        world_array: (world_H, world_W, 8) float32 array from world_to_array()
        x_start: Starting X coordinate (column)
        y_start: Starting Y coordinate (row)
        width: Chunk width in tiles
        height: Chunk height in tiles

    Returns:
        numpy array of shape (height, width, 8) with dtype float32
    """
    world_H, world_W = world_array.shape[:2]

    # Clamp to valid world bounds
    src_y0 = max(y_start, 0)
    src_y1 = min(y_start + height, world_H)
    src_x0 = max(x_start, 0)
    src_x1 = min(x_start + width, world_W)

    chunk = np.zeros((height, width, 8), dtype=np.float32)

    # Destination offsets inside the chunk (non-zero when x/y_start < 0)
    dst_y0 = src_y0 - y_start
    dst_x0 = src_x0 - x_start

    if src_y1 > src_y0 and src_x1 > src_x0:
        chunk[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0), :] = (
            world_array[src_y0:src_y1, src_x0:src_x1, :]
        )

    return chunk


def extract_chunk(
    world: lihzahrd.World,
    x_start: int,
    y_start: int,
    width: int,
    height: int
) -> np.ndarray:
    """
    Extract a chunk from the world in 8-channel format.

    Iterates tiles individually, kept for single-sample callers (export,
    dataset inference).  For bulk extraction over many windows, prefer
    ``world_to_array`` + ``extract_chunk_from_array``.

    Args:
        world: Loaded lihzahrd World object
        x_start: Starting X coordinate
        y_start: Starting Y coordinate  
        width: Chunk width in tiles
        height: Chunk height in tiles
        
    Returns:
        numpy array of shape (height, width, 8) with dtype float32
    """
    chunk = np.zeros((height, width, 8), dtype=np.float32)
    
    for y in range(height):
        world_y = y_start + y
        if world_y < 0 or world_y >= world.size.y:
            continue
            
        for x in range(width):
            world_x = x_start + x
            if world_x < 0 or world_x >= world.size.x:
                continue
            
            tile = world.tiles[world_x, world_y]
            chunk[y, x, :] = tile_to_array(tile)
    
    return chunk
