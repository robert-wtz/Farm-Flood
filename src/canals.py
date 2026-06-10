"""
Canal optimization: least-cost drainage paths + retention lagoon siting.
Based on flow accumulation raster from terrain.py and sinks from water_topo.py.
Logic: drain where water exceeds capacity; retain in natural sinks for dry cycles.
"""
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import shapes
from scipy.ndimage import label
from shapely.geometry import shape, LineString, Point
from pathlib import Path


def least_cost_drainage_paths(flow_acc_path, threshold_cells=500, output_path=None):
    """
    Extract candidate drainage channels: cells with upstream area > threshold.
    threshold_cells: minimum upstream cells to be considered a channel.
    Returns GeoDataFrame of candidate drainage paths.
    """
    with rasterio.open(flow_acc_path) as src:
        acc = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs

    # Channel network: high accumulation cells
    channels = (acc > threshold_cells).astype("uint8")

    geoms = [
        {"geometry": shape(geom), "accumulation": float(val)}
        for geom, val in shapes(channels * acc.astype("float32"), transform=transform)
        if shape(geom).geom_type in ("Polygon", "MultiPolygon")
    ]

    gdf = gpd.GeoDataFrame(geoms, crs=crs)
    if output_path:
        gdf.to_file(output_path)
        print(f"Drainage paths saved: {output_path} ({len(gdf)} features)")
    return gdf


def identify_retention_sites(sinks_path, twi_path, min_area_pixels=20,
                              twi_threshold=10.0, output_path=None):
    """
    Identify natural sink depressions suitable for retention lagoons.
    Criteria: (1) is a sink, (2) TWI > threshold, (3) area > min_area_pixels.
    Returns GeoDataFrame of candidate retention sites with estimated area.
    """
    with rasterio.open(sinks_path) as src:
        sinks = src.read(1).astype(bool)
        transform = src.transform
        crs = src.crs
        pixel_area_m2 = abs(transform.a * transform.e)

    with rasterio.open(twi_path) as src:
        twi = src.read(1).astype(float)

    # Combine criteria
    candidate = sinks & (twi > twi_threshold)

    labeled, n = label(candidate)
    sites = []
    for i in range(1, n + 1):
        component = labeled == i
        area_px = component.sum()
        if area_px >= min_area_pixels:
            # Centroid approx
            rows, cols = np.where(component)
            mean_twi = twi[component].mean()
            geoms_raw = list(shapes(component.astype("uint8"), transform=transform))
            for geom, val in geoms_raw:
                if val == 1:
                    sites.append({
                        "geometry": shape(geom),
                        "area_ha": area_px * pixel_area_m2 / 10000,
                        "mean_twi": float(mean_twi),
                    })

    gdf = gpd.GeoDataFrame(sites, crs=crs)
    if output_path:
        gdf.to_file(output_path)
        print(f"Retention sites saved: {output_path} ({len(gdf)} sites)")
    return gdf


def crop_suitability_map(flood_prob_path, drainage_paths_path=None, output_path=None):
    """
    Invert flood probability to get crop suitability:
    suitability = 1 - flood_probability (optionally boosted near drainage paths).
    """
    with rasterio.open(flood_prob_path) as src:
        prob = src.read(1).astype(float)
        meta = src.meta.copy()

    suitability = 1.0 - np.clip(prob, 0, 1)

    meta.update(dtype="float32")
    if output_path:
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(suitability.astype("float32"), 1)
        print(f"Crop suitability raster saved: {output_path}")

    return suitability
