# Farm Flood Model — Ameghino, Buenos Aires

Flood risk model for the Florentino Ameghino district (Buenos Aires province).
Predicts flood probability per cadastral parcel and recommends drainage/retention works.

**Design principle (Ameghino, 1884):** drain where water exceeds capacity; retain in
natural sinks for dry cycles — not only drainage canals but also retention lagoons.

## Pipeline

| Step | Notebook | Key output |
|------|----------|------------|
| 1. Terrain analysis | `01_terrain.ipynb` | TWI, flow accumulation, natural sinks |
| 2. Flood history + water topo | `02_flood_history.ipynb` | Flood inventory 2014-2024, relative elevation from SAR |
| 3. ML model | `03_ml_model.ipynb` | Flood probability raster + parcel risk |
| 4. Canal optimization | `04_canal_optimization.ipynb` | Drainage paths, retention sites, crop suitability |

## Data sources (all free)

| Data | Source |
|------|--------|
| DEM GLO-30 | Planetary Computer STAC |
| Sentinel-1 SAR | Google Earth Engine (2014–2024) |
| Sentinel-2 NDVI | Google Earth Engine |
| CHIRPS rainfall | Google Earth Engine |
| Soils | GeoINTA |
| Cadastral parcels | ARBA (Buenos Aires) |
| Canals/water | OpenStreetMap via osmnx |

## Setup

```bash
conda env create -f environment.yml
conda activate farm-flood

# Register kernel for notebooks
python -m ipykernel install --user --name farm-flood

# Authenticate GEE (one-time)
earthengine authenticate
```

## Key design decisions

- **Sinks are not filled** — closed depressions are real lagunas in the Pampas endorheic landscape.
- **Water-derived topography:** Sentinel-1 flood frequency and waterlines reconstruct relative elevation
  independently from the DEM, mitigating GLO-30's ~2-4 m vertical error on flat terrain.
- **Spatial block CV:** XGBoost is validated with spatial block cross-validation (not random split)
  to avoid inflated AUC from spatial autocorrelation.
- **D-infinity flow direction** (not D8) for better performance on near-flat terrain.

## Structure

```
Farm-Flood/
├── config.yaml          # bbox, thresholds, model params
├── environment.yml      # conda environment
├── notebooks/           # step-by-step analysis notebooks
├── src/
│   ├── config.py        # config loader
│   ├── terrain.py       # DEM → TWI, flow, sinks
│   ├── gee.py           # GEE auth + data downloads
│   ├── flood.py         # SAR flood masks + inventory
│   ├── water_topo.py    # topography from water dynamics
│   ├── features.py      # feature matrix builder
│   ├── model.py         # XGBoost + spatial CV
│   ├── canals.py        # drainage paths + retention sites
│   └── viz.py           # matplotlib + folium maps
└── outputs/             # final maps and model (gitignored)
```
