"""
Topography from water dynamics (Sentinel-1 SAR temporal stack).

Three complementary approaches:
1. Flood frequency raster → proxy for relative elevation
2. Waterline method → shorelines as iso-elevation contours
3. Fill order / connectivity graph → real flow direction in flat terrain
"""
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from scipy.ndimage import label, binary_erosion
from shapely.geometry import shape
from pathlib import Path


def frequency_to_relative_elevation(flood_freq_path, output_path):
    """
    Invert flood frequency to get relative elevation proxy.
    Pixels flooded most often are lowest; least often are highest.
    """
    with rasterio.open(flood_freq_path) as src:
        freq = src.read(1).astype(float)
        meta = src.meta.copy()

    # Normalize and invert: rel_elev = 1 - flood_freq
    freq_norm = (freq - np.nanmin(freq)) / (np.nanmax(freq) - np.nanmin(freq) + 1e-9)
    rel_elev = 1.0 - freq_norm

    output_path = Path(output_path)
    meta.update(dtype="float32")
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(rel_elev.astype("float32"), 1)

    print(f"Relative elevation from flood frequency saved: {output_path}")
    return rel_elev


def extract_waterlines(flood_mask_path, output_path):
    """
    Extract shoreline (boundary) of a flood mask as vector.
    Each shoreline = approximate iso-elevation contour at the water level of that date.
    """
    with rasterio.open(flood_mask_path) as src:
        mask = src.read(1).astype("uint8")
        transform = src.transform
        crs = src.crs

    # Erode mask and subtract to get boundary ring
    eroded = binary_erosion(mask, iterations=1)
    boundary = mask.astype(int) - eroded.astype(int)
    boundary = np.where(boundary > 0, 1, 0).astype("uint8")

    geoms = [
        shape(geom)
        for geom, val in shapes(boundary, transform=transform)
        if val == 1
    ]

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    output_path = Path(output_path)
    gdf.to_file(output_path)
    print(f"Waterlines saved: {output_path} ({len(gdf)} features)")
    return gdf


def analyze_fill_order(flood_masks_paths):
    """
    Given a list of flood mask rasters in chronological order,
    compute for each pixel the first date it got flooded.
    Earlier = lower elevation. Also identify connected water bodies per date.

    Returns:
        fill_order (np.array): date index (0-based) of first flood, or -1 if never flooded
        connectivity (list of np.array): labeled connected components per date
    """
    masks = []
    for path in flood_masks_paths:
        with rasterio.open(path) as src:
            masks.append(src.read(1).astype(bool))

    if not masks:
        return None, None

    shape = masks[0].shape
    fill_order = np.full(shape, -1, dtype=int)

    for i, m in enumerate(masks):
        # Mark first occurrence
        first_time = (fill_order == -1) & m
        fill_order[first_time] = i

    # Connected components per date
    connectivity = []
    for m in masks:
        labeled, n = label(m)
        connectivity.append((labeled, n))

    return fill_order, connectivity


def compare_dem_vs_water_topo(dem_path, rel_elev_path, output_path=None):
    """
    Compute correlation and discrepancy map between DEM and water-derived
    relative elevation. Low correlation → DEM unreliable for hydrology.
    Returns Spearman correlation coefficient.
    """
    from scipy.stats import spearmanr

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(float)
        meta = src.meta.copy()
        nodata = src.nodata

    with rasterio.open(rel_elev_path) as src:
        rel_elev = src.read(1).astype(float)

    # Mask nans/nodata
    valid = ~np.isnan(dem) & ~np.isnan(rel_elev)
    if nodata is not None:
        valid &= dem != nodata

    corr, pval = spearmanr(dem[valid].ravel(), rel_elev[valid].ravel())
    print(f"DEM vs water-topo Spearman r = {corr:.3f} (p={pval:.2e})")
    if corr < 0.4:
        print("WARNING: Low correlation — water-derived topo preferred over DEM for flow routing.")

    if output_path:
        discrepancy = np.full_like(dem, np.nan)
        # Normalize both to [0,1] and compute absolute difference
        d_norm = (dem - np.nanmin(dem)) / (np.nanmax(dem) - np.nanmin(dem) + 1e-9)
        w_norm = rel_elev
        discrepancy[valid] = np.abs(d_norm[valid] - w_norm[valid])
        meta.update(dtype="float32", nodata=np.nan)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(discrepancy.astype("float32"), 1)

    return corr
