# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
conda activate farm-flood
jupyter lab  # to run notebooks
```

All notebooks must be run with the `farm-flood` kernel. To register it: `python -m ipykernel install --user --name farm-flood`

GEE authentication (one-time): `earthengine authenticate`

## Pipeline order

Run notebooks in sequence — each step produces files that the next step consumes:

1. `01_terrain.ipynb` → `data/processed/` (TWI, flow_acc, sinks, slope, pit_filled `.tif`)
2. `02_flood_history.ipynb` → flood frequency raster (exported via GEE to Google Drive, then placed in `data/processed/`)
3. `03_ml_model.ipynb` → `outputs/model.joblib`, `outputs/cv_results.csv`, flood probability raster
4. `04_canal_optimization.ipynb` → drainage paths + retention site recommendations

## Architecture

All `src/` modules are pure functions imported by notebooks — no CLI entrypoints.

- `config.py` — loads `config.yaml`, exposes `load_config()` and `get_bbox(cfg)` (returns `(xmin, ymin, xmax, ymax)`)
- `terrain.py` — DEM → pysheds flow routing → TWI, slope, sinks. Sinks are **not filled** (Pampas lagunas are real closed depressions)
- `gee.py` — GEE init + Sentinel-1, Sentinel-2, CHIRPS helpers. Call `init_gee()` before any `ee.*` usage
- `flood.py` — Builds per-event SAR flood masks (backscatter drop detection) and a multi-date flood frequency raster
- `water_topo.py` — Reconstructs relative elevation from flood frequency (workaround for GLO-30 ~2-4 m vertical error on flat terrain)
- `features.py` — Assembles the raster feature matrix (TWI, slope, flood_freq, NDVI, soil, etc.) as a pixel DataFrame
- `model.py` — XGBoost with spatial block CV. **Always use spatial block CV**, not random split, to avoid inflated AUC from spatial autocorrelation
- `canals.py` — Identifies optimal drainage paths and retention sites from terrain + flood outputs
- `viz.py` — matplotlib raster plots and folium interactive maps

## Key config values (`config.yaml`)

- `bbox`: Florentino Ameghino district, Buenos Aires province (WGS84)
- `dem.crs`: `EPSG:32720` (UTM 20S) for metric calculations
- `terrain.flow_direction_method`: `dinf` (better than D8 on near-flat Pampas terrain)
- `gee.project`: `farm-flood`

## Data flow

```
Planetary Computer STAC  →  data/raw/dem_glo30.tif  →  data/processed/twi.tif, flow_acc.tif, sinks.tif ...
GEE (Sentinel-1 SAR)     →  Google Drive export     →  data/processed/flood_frequency.tif
GEE (Sentinel-2, CHIRPS) →  used in-memory in notebooks
data/processed/*.tif     →  features.py             →  model.py → outputs/
```

`data/raw/` and `outputs/` are gitignored. `data/processed/` holds intermediate rasters (also gitignored except `.gitkeep`).
