"""World feature analysis command for natural ID mapping generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import lihzahrd
import numpy as np


def analyze_world(world_path: Path) -> dict:
    """Analyze a single world file for feature usage."""
    print(f"\nAnalyzing {world_path.name}...")

    world = lihzahrd.World.create_from_file(str(world_path))

    stats = {
        "block_types": set(),
        "wall_types": set(),
        "liquid_types": set(),
        "liquid_amounts": [],
        "block_paints": set(),
        "wall_paints": set(),
        "block_illuminants": set(),
        "wall_illuminants": set(),
        "echo_coating": {"block": 0, "wall": 0},
        "wires": {"red": 0, "blue": 0, "green": 0, "yellow": 0},
        "actuator": 0,
        "block_active": {"active": 0, "inactive": 0},
        "block_shapes": set(),
        "total_tiles": 0,
    }

    print(f"  Scanning entire world ({world.size.x} x {world.size.y} tiles)...")

    total_tiles = world.size.x * world.size.y
    progress_interval = max(total_tiles // 20, 1)
    next_progress = progress_interval

    for y in range(world.size.y):
        for x in range(world.size.x):
            tile = world.tiles[x, y]
            stats["total_tiles"] += 1

            if stats["total_tiles"] >= next_progress:
                pct = (stats["total_tiles"] / total_tiles) * 100
                print(f"    Progress: {pct:.0f}% ({stats['total_tiles']:,} / {total_tiles:,} tiles)")
                next_progress += progress_interval

            if tile.block is not None:
                block_id = tile.block.type.value if hasattr(tile.block.type, "value") else tile.block.type
                stats["block_types"].add(block_id)
                if tile.block.is_active:
                    stats["block_active"]["active"] += 1
                else:
                    stats["block_active"]["inactive"] += 1

                shape_val = tile.block.shape.value if hasattr(tile.block.shape, "value") else tile.block.shape
                stats["block_shapes"].add(shape_val)

                if hasattr(tile.block, "paint") and tile.block.paint is not None and tile.block.paint != 0:
                    stats["block_paints"].add(tile.block.paint)

                if getattr(tile.block, "is_illuminant", False):
                    stats["block_illuminants"].add(True)

                if getattr(tile.block, "is_echo", False):
                    stats["echo_coating"]["block"] += 1

            if tile.wall is not None:
                wall_id = tile.wall.type.value if hasattr(tile.wall.type, "value") else tile.wall.type
                stats["wall_types"].add(wall_id)

                if hasattr(tile.wall, "paint") and tile.wall.paint is not None and tile.wall.paint != 0:
                    stats["wall_paints"].add(tile.wall.paint)

                if getattr(tile.wall, "is_illuminant", False):
                    stats["wall_illuminants"].add(True)

                if getattr(tile.wall, "is_echo", False):
                    stats["echo_coating"]["wall"] += 1

            if tile.liquid is not None:
                liquid_type = tile.liquid.type.name if hasattr(tile.liquid.type, "name") else str(tile.liquid.type)
                stats["liquid_types"].add(liquid_type)
                liquid_amt = getattr(tile.liquid, "volume", getattr(tile.liquid, "amount", 0))
                stats["liquid_amounts"].append(liquid_amt)

            if tile.wiring is not None:
                if getattr(tile.wiring, "red", False):
                    stats["wires"]["red"] += 1
                if getattr(tile.wiring, "blue", False):
                    stats["wires"]["blue"] += 1
                if getattr(tile.wiring, "green", False):
                    stats["wires"]["green"] += 1
                if getattr(tile.wiring, "yellow", False):
                    stats["wires"]["yellow"] += 1
                if getattr(tile.wiring, "actuator", False):
                    stats["actuator"] += 1

    return stats


def merge_stats(all_stats: list) -> dict:
    """Merge statistics from multiple worlds."""
    merged = {
        "block_types": set(),
        "wall_types": set(),
        "liquid_types": set(),
        "liquid_amounts": [],
        "block_paints": set(),
        "wall_paints": set(),
        "block_illuminants": set(),
        "wall_illuminants": set(),
        "echo_coating": {"block": 0, "wall": 0},
        "wires": {"red": 0, "blue": 0, "green": 0, "yellow": 0},
        "actuator": 0,
        "block_active": {"active": 0, "inactive": 0},
        "block_shapes": set(),
        "total_tiles": 0,
    }

    for stats in all_stats:
        merged["block_types"].update(stats["block_types"])
        merged["wall_types"].update(stats["wall_types"])
        merged["liquid_types"].update(stats["liquid_types"])
        merged["liquid_amounts"].extend(stats["liquid_amounts"])
        merged["block_paints"].update(stats["block_paints"])
        merged["wall_paints"].update(stats["wall_paints"])
        merged["block_illuminants"].update(stats["block_illuminants"])
        merged["wall_illuminants"].update(stats["wall_illuminants"])
        merged["echo_coating"]["block"] += stats["echo_coating"]["block"]
        merged["echo_coating"]["wall"] += stats["echo_coating"]["wall"]
        for wire_color in ["red", "blue", "green", "yellow"]:
            merged["wires"][wire_color] += stats["wires"][wire_color]
        merged["actuator"] += stats["actuator"]
        merged["block_active"]["active"] += stats["block_active"]["active"]
        merged["block_active"]["inactive"] += stats["block_active"]["inactive"]
        merged["block_shapes"].update(stats["block_shapes"])
        merged["total_tiles"] += stats["total_tiles"]

    return merged


def print_analysis(stats: dict) -> None:
    """Print comprehensive analysis of world features."""
    print("\n" + "=" * 80)
    print("WORLD FEATURE ANALYSIS - NATURAL GENERATION")
    print("=" * 80)

    print(f"\nTotal tiles analyzed: {stats['total_tiles']:,}")

    print("\n📦 BLOCK TYPES:")
    print(f"  Unique block types found: {len(stats['block_types'])}")
    print(f"  Block IDs: {sorted(stats['block_types'])[:20]}{'...' if len(stats['block_types']) > 20 else ''}")
    print(f"  (Full list: {sorted(stats['block_types'])})")

    print("\n🧱 WALL TYPES:")
    print(f"  Unique wall types found: {len(stats['wall_types'])}")
    print(f"  Wall IDs: {sorted(stats['wall_types'])[:20]}{'...' if len(stats['wall_types']) > 20 else ''}")
    print(f"  (Full list: {sorted(stats['wall_types'])})")

    print("\n💧 LIQUIDS:")
    print(f"  Liquid types found: {stats['liquid_types']}")
    if stats["liquid_amounts"]:
        amounts = np.array(stats["liquid_amounts"])
        print(f"  Liquid amount range: {amounts.min()} to {amounts.max()}")
        print(f"  Liquid amount mean: {amounts.mean():.1f}")
        print("  Liquid amount distribution:")
        hist, bins = np.histogram(amounts, bins=[0, 50, 100, 150, 200, 255])
        for i, count in enumerate(hist):
            print(f"    {bins[i]:.0f}-{bins[i + 1]:.0f}: {count} ({count / len(amounts) * 100:.1f}%)")

    print("\n🎨 PAINT:")
    print(f"  Block paints: {stats['block_paints'] if stats['block_paints'] else 'NONE (can remove!)'}")
    print(f"  Wall paints: {stats['wall_paints'] if stats['wall_paints'] else 'NONE (can remove!)'}")

    print("\n💡 ILLUMINANTS:")
    print(f"  Block illuminants: {stats['block_illuminants'] if stats['block_illuminants'] else 'NONE (can remove!)'}")
    print(f"  Wall illuminants: {stats['wall_illuminants'] if stats['wall_illuminants'] else 'NONE (can remove!)'}")

    print("\n🔊 ECHO COATING:")
    echo_rate = (stats["echo_coating"]["block"] + stats["echo_coating"]["wall"]) / stats["total_tiles"] * 100
    print(f"  Block echo: {stats['echo_coating']['block']} tiles ({stats['echo_coating']['block'] / stats['total_tiles'] * 100:.4f}%)")
    print(f"  Wall echo: {stats['echo_coating']['wall']} tiles ({stats['echo_coating']['wall'] / stats['total_tiles'] * 100:.4f}%)")
    print(f"  Recommendation: {'KEEP' if echo_rate > 0.01 else 'REMOVE (too rare!)'}")

    print("\n⚡ WIRING:")
    total_wires = sum(stats["wires"].values())
    wire_rate = total_wires / stats["total_tiles"] * 100
    for color, count in stats["wires"].items():
        print(f"  {color.capitalize()} wire: {count} tiles ({count / stats['total_tiles'] * 100:.4f}%)")
    print(f"  Recommendation: {'KEEP' if wire_rate > 0.01 else 'REMOVE (too rare!)'}")

    print("\n🔧 ACTUATORS:")
    actuator_rate = stats["actuator"] / stats["total_tiles"] * 100
    print(f"  Actuators: {stats['actuator']} tiles ({actuator_rate:.4f}%)")
    print(f"  Recommendation: {'KEEP' if actuator_rate > 0.01 else 'REMOVE (too rare!)'}")

    print("\n🟢 BLOCK ACTIVE STATE:")
    total_blocks = stats["block_active"]["active"] + stats["block_active"]["inactive"]
    if total_blocks > 0:
        print(f"  Active: {stats['block_active']['active']} ({stats['block_active']['active'] / total_blocks * 100:.1f}%)")
        print(f"  Inactive: {stats['block_active']['inactive']} ({stats['block_active']['inactive'] / total_blocks * 100:.1f}%)")
        print(
            f"  Recommendation: {'KEEP' if stats['block_active']['inactive'] / total_blocks > 0.01 else 'Could potentially remove if >99% active'}"
        )

    print("\n📐 BLOCK SHAPES:")
    print(f"  Unique shapes: {stats['block_shapes']}")

    print("\n" + "=" * 80)
    print("RECOMMENDED OPTIMIZATIONS")
    print("=" * 80)

    print("\n1. Embeddings:")
    print(f"   - Block types: Reduce from 693 to {len(stats['block_types'])} ({len(stats['block_types']) / 693 * 100:.1f}% of total)")
    print(f"   - Wall types: Reduce from 347 to {len(stats['wall_types'])} ({len(stats['wall_types']) / 347 * 100:.1f}% of total)")

    print("\n2. Features to REMOVE (not in natural generation):")
    removable = []
    if not stats["block_paints"] and not stats["wall_paints"]:
        removable.append("   - Paint (both block & wall)")
    if not stats["block_illuminants"] and not stats["wall_illuminants"]:
        removable.append("   - Illuminants (both block & wall)")
    if echo_rate < 0.01:
        removable.append("   - Echo coating (too rare)")
    if wire_rate < 0.01:
        removable.append("   - All wiring (too rare)")
    if actuator_rate < 0.01:
        removable.append("   - Actuators (too rare)")

    if removable:
        for item in removable:
            print(item)
    else:
        print("   - (All features have some presence)")

    print("\n3. Liquid simplification:")
    amounts = np.array(stats["liquid_amounts"]) if stats["liquid_amounts"] else np.array([])
    if stats["liquid_amounts"]:
        if amounts.std() < 50:
            print(f"   - Could binarize: liquid present/absent only (std={amounts.std():.1f})")
        else:
            print(f"   - Keep amount, has variation (std={amounts.std():.1f})")

    original_dims = 17
    new_dims = 0

    new_dims += 1
    new_dims += 1
    new_dims += 1

    if stats["liquid_types"]:
        new_dims += 1
        if amounts.size > 0 and amounts.std() >= 50:
            new_dims += 1

    if wire_rate >= 0.01:
        new_dims += 4

    if actuator_rate >= 0.01:
        new_dims += 1

    print("\n4. Dimensionality reduction:")
    print(f"   - Original channels: {original_dims}")
    print(f"   - Recommended channels: {new_dims}")
    print(f"   - Reduction: {original_dims - new_dims} channels ({(original_dims - new_dims) / original_dims * 100:.1f}%)")

    print("\n5. Embedding table sizes:")
    print(f"   - Block type embeddings: {len(stats['block_types'])} instead of 693 (saves {693 - len(stats['block_types'])} embeddings)")
    print(f"   - Wall type embeddings: {len(stats['wall_types'])} instead of 347 (saves {347 - len(stats['wall_types'])} embeddings)")
    total_embedding_reduction = (693 - len(stats["block_types"])) + (347 - len(stats["wall_types"]))
    print(f"   - Total embedding reduction: {total_embedding_reduction} fewer embeddings")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze worlds and generate natural Terraria ID mappings")
    parser.add_argument("--source-dir", default="worldgen", help="Directory containing .wld files")
    parser.add_argument("--output", default="src/terraria/natural_ids.py", help="Output path for generated mapping module")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    worldgen_dir = Path(args.source_dir)
    if not worldgen_dir.exists():
        print(f"Error: {worldgen_dir} directory not found!")
        return 1

    world_files = list(worldgen_dir.glob("*.wld"))
    if not world_files:
        print(f"No .wld files found in {worldgen_dir}!")
        return 1

    print(f"Found {len(world_files)} world files to analyze")

    all_stats = []
    for world_file in world_files:
        try:
            stats = analyze_world(world_file)
            all_stats.append(stats)
        except Exception as error:
            print(f"Error analyzing {world_file}: {error}")
            continue

    if not all_stats:
        print("No worlds were successfully analyzed!")
        return 1

    merged = merge_stats(all_stats)
    print_analysis(merged)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as handle:
        handle.write('"""Natural-world block and wall ID mappings.\n\n')
        handle.write("Auto-generated from world analysis and used by the optimized 9-channel pipeline.\n")
        handle.write('"""\n\n')
        handle.write(f"NATURAL_BLOCK_IDS = {sorted(merged['block_types'])}\n\n")
        handle.write(f"NATURAL_WALL_IDS = {sorted(merged['wall_types'])}\n\n")
        handle.write("BLOCK_ID_TO_INDEX = {game_id: idx for idx, game_id in enumerate(NATURAL_BLOCK_IDS)}\n")
        handle.write("WALL_ID_TO_INDEX = {game_id: idx for idx, game_id in enumerate(NATURAL_WALL_IDS)}\n\n")
        handle.write("BLOCK_INDEX_TO_ID = {idx: game_id for game_id, idx in BLOCK_ID_TO_INDEX.items()}\n")
        handle.write("WALL_INDEX_TO_ID = {idx: game_id for game_id, idx in WALL_ID_TO_INDEX.items()}\n")

    print(f"\n✅ ID mappings saved to {output_file}")
    print("   Import with: from src.terraria.natural_ids import BLOCK_ID_TO_INDEX, WALL_ID_TO_INDEX")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
