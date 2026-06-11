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


def _resample_to_ref(src_path, ref_path):
    """Read raster resampled to match ref_path grid."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    with rasterio.open(ref_path) as ref:
        ref_meta = ref.meta.copy()
        shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs
    with rasterio.open(src_path) as src:
        if src.shape == shape and src.crs == ref_crs:
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            return arr
        out = np.empty(shape, dtype=float)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
        if src.nodata is not None:
            out[out == src.nodata] = np.nan
        return out


def build_pixel_features(raster_paths: dict, flood_target_path, valid_mask=None, max_pixels=500_000, seed=42):
    """
    Stack all feature rasters into a pixel-wise DataFrame.

    raster_paths: dict {feature_name: path}
    flood_target_path: binary flood frequency or event mask for target
    Returns: pd.DataFrame with columns = feature names + x/y coords + 'flooded' target
    """
    ref_path = next(iter(raster_paths.values()))

    # Extract pixel center coordinates from reference raster
    with rasterio.open(ref_path) as ref:
        from rasterio.transform import xy
        rows, cols = np.meshgrid(np.arange(ref.height), np.arange(ref.width), indexing='ij')
        xs, ys = xy(ref.transform, rows.ravel(), cols.ravel())

    arrays = {"x": np.array(xs), "y": np.array(ys)}

    for name, path in raster_paths.items():
        arr = _resample_to_ref(path, ref_path)
        arrays[name] = arr.ravel()

    # Target — threshold at 0.1 (10% flood frequency = frequently at risk)
    target = _resample_to_ref(flood_target_path, ref_path)
    arrays["flooded"] = (target.ravel() > 0.1).astype(int)

    df = pd.DataFrame(arrays)

    if valid_mask is not None:
        df = df[valid_mask.ravel()]

    df = df.dropna()
    if max_pixels and len(df) > max_pixels:
        df = df.sample(n=max_pixels, random_state=seed)
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
