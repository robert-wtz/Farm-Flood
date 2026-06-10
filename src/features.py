"""
Build feature matrix per pixel (and aggregate to cadastral parcels).
Features: TWI, relative elevation (DEM + water-derived), flood frequency,
          upstream area, soil type, NDVI, CHIRPS accumulations, distance to canals.
"""
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterstats import zonal_stats
from pathlib import Path


def load_raster_as_array(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        arr[arr == src.nodata] = np.nan
        return arr, src.transform, src.crs, src.meta


def build_pixel_features(raster_paths: dict, flood_target_path, valid_mask=None):
    """
    Stack all feature rasters into a pixel-wise DataFrame.

    raster_paths: dict {feature_name: path}
    flood_target_path: binary flood frequency or event mask for target
    Returns: pd.DataFrame with columns = feature names + 'flooded' target
    """
    arrays = {}
    ref_meta = None

    for name, path in raster_paths.items():
        arr, transform, crs, meta = load_raster_as_array(path)
        arrays[name] = arr.ravel()
        if ref_meta is None:
            ref_meta = meta
            shape = arr.shape

    # Target
    target, _, _, _ = load_raster_as_array(flood_target_path)
    arrays["flooded"] = (target.ravel() > 0.5).astype(int)

    df = pd.DataFrame(arrays)

    if valid_mask is not None:
        df = df[valid_mask.ravel()]

    df = df.dropna()
    return df


def compute_distance_to_canals(bbox, crs, output_path):
    """
    Download canal/water OSM features via osmnx, rasterize distance raster.
    """
    import osmnx as ox
    from rasterio.transform import from_bounds
    from scipy.ndimage import distance_transform_edt

    xmin, ymin, xmax, ymax = bbox
    tags = {"waterway": True, "natural": "water"}
    try:
        gdf = ox.features_from_bbox(ymax, ymin, xmax, xmin, tags=tags)
    except Exception as e:
        print(f"OSM download failed: {e}. Returning empty distance raster.")
        gdf = gpd.GeoDataFrame()

    # Placeholder: rasterize and compute distance transform
    # (full impl requires reference raster dimensions)
    print(f"OSM water features: {len(gdf)} features")
    return gdf


def aggregate_to_parcels(feature_raster_paths, parcel_shapefile, output_path):
    """
    Zonal stats (mean) of each feature raster over cadastral parcels (ARBA).
    Returns GeoDataFrame with features per parcel.
    """
    parcels = gpd.read_file(parcel_shapefile)
    result = parcels.copy()

    for name, path in feature_raster_paths.items():
        stats = zonal_stats(
            parcels,
            str(path),
            stats=["mean", "std", "min", "max"],
            nodata=np.nan,
            geojson_out=False,
        )
        result[f"{name}_mean"] = [s["mean"] for s in stats]
        result[f"{name}_std"] = [s["std"] for s in stats]

    result.to_file(output_path)
    print(f"Parcel features saved: {output_path}")
    return result
