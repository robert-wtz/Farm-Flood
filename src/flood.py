"""
Flood detection from Sentinel-1 SAR time series.
Builds binary flood masks per event using change detection (pre/post backscatter).
"""
import numpy as np
import rasterio
import ee
from pathlib import Path
from .gee import get_s1_image, export_to_drive


def speckle_filter(image, kernel_size=3):
    """Lee-style speckle filter approximation via focal mean in GEE."""
    kernel = ee.Kernel.square(kernel_size, "pixels")
    return image.focal_mean(kernel=kernel)


def flood_mask_from_event(bbox_ee, event_date, pre_days=20, post_days=10,
                          threshold_db=-2.0, polarization="VV"):
    """
    Single flood mask: backscatter drop > threshold after rain event.
    Returns GEE Image (1=flooded, 0=dry).
    """
    pre_start = (
        __import__("pandas").Timestamp(event_date) -
        __import__("pandas").Timedelta(days=pre_days+6)
    ).strftime("%Y-%m-%d")
    pre_end = (
        __import__("pandas").Timestamp(event_date) -
        __import__("pandas").Timedelta(days=6)
    ).strftime("%Y-%m-%d")

    import pandas as pd

    def _get_col(date_str):
        start = (pd.Timestamp(date_str) - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        end = (pd.Timestamp(date_str) + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        return (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(bbox_ee)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", polarization))
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
            .select(polarization)
        )

    pre_col = _get_col(pre_end)
    post_col = _get_col(event_date)

    # Skip events with no S1 imagery in either window
    if pre_col.size().getInfo() == 0 or post_col.size().getInfo() == 0:
        return None

    pre_filtered = speckle_filter(pre_col.median())
    post_filtered = speckle_filter(post_col.median())

    diff = post_filtered.subtract(pre_filtered)
    flooded = diff.lt(threshold_db).rename("flooded")
    return flooded


def build_flood_inventory(bbox_ee, event_dates, output_dir=None, polarization="VV"):
    """
    Build multi-date flood inventory as a GEE ImageCollection.
    Returns ImageCollection with one binary mask per event.
    """
    masks = []
    skipped = 0
    for date in event_dates:
        mask = flood_mask_from_event(bbox_ee, date, polarization=polarization)
        if mask is None:
            skipped += 1
            continue
        mask = mask.set("event_date", date)
        masks.append(mask)

    inventory = ee.ImageCollection(masks)
    print(f"Flood inventory built: {len(masks)} events ({skipped} skipped — no S1 imagery)")

    if output_dir:
        # Export stack sum (flood frequency raster) to Drive for local use
        freq = inventory.sum().rename("flood_frequency")
        export_to_drive(freq, "flood_frequency", bbox_ee)

    return inventory


def compute_flood_frequency(inventory):
    """
    Fraction of dates each pixel was flooded → proxy for relative elevation.
    Low elevation = high flood frequency.
    """
    count = inventory.count()
    freq = inventory.sum().divide(count).rename("flood_freq_fraction")
    return freq


def combine_jrc_sar_frequency(jrc_images, sar_inventory, bbox_ee):
    """
    Combine JRC long-term water history (1984–present) with recent SAR flood masks.

    JRC occurrence (0–100 scale) is normalized to 0–1 and weighted 60%.
    SAR flood frequency (0–1) is weighted 40% — adds recent events and cloud-free detection.
    Returns a single combined flood frequency image (0–1).
    """
    jrc_norm = jrc_images["occurrence"].divide(100).rename("jrc_freq")
    sar_freq = compute_flood_frequency(sar_inventory).rename("sar_freq")

    combined = (
        jrc_norm.multiply(0.6)
        .add(sar_freq.multiply(0.4))
        .rename("flood_freq_combined")
        .clip(bbox_ee)
    )
    return combined
