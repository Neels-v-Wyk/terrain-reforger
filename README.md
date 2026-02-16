# terrain-reforger (WIP)
A Terraria terrain generation tool using Discrete Diffusion and Content Aware Inpainting with PyTorch

## Notes

- Checkpoints, latest model, and training plots are written under `checkpoints/`.
- Natural block/wall ID mappings are in `src/terraria/natural_ids.py`.
- Generated worlds, datasets, and checkpoints are intentionally ignored by git (see `.gitignore` entries for `worldgen/`, `data/`, and `checkpoints/`).

## Reproducibility

- Generate worlds: `terrain data worldgen --num-worlds 20`
- Refresh natural ID mapping: `terrain data analyze --source-dir worldgen --output src/terraria/natural_ids.py`
- Build per-world cache: `terrain data prepare --mode chunked --source worldgen --output-dir data/cache`
- Train from cache: `terrain model train --data data/cache --disk-mode --cache-size 5 --epochs 50`

## TEdit Export

Export inference results as TEdit schematic files (`.TEditSch`) that can be imported directly into [TEdit](https://github.com/TEdit/Terraria-Map-Editor).

```bash
# Export a reconstructed region from a world file
terrain model export --world worldgen/MyWorld.wld --region 1500 500 1564 564 --output-dir exports

# Export both original and reconstructed for comparison
terrain model export --mode compare --width 64 --height 64

# Generate from random input (experimental)
terrain model export --mode generate --width 32 --height 32 --seed 42
```

Export modes:
- `reconstruct` (default): Extract region from world, pass through model, export result
- `compare`: Export both original and reconstructed schematics side-by-side
- `generate`: Create terrain from random latent codes (experimental)

To import in TEdit: **File → Import Schematic → Select `.TEditSch` file**
