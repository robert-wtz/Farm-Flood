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

    pre = get_s1_image(bbox_ee, pre_end, window_days=6, polarization=polarization)
    post = get_s1_image(bbox_ee, event_date, window_days=6, polarization=polarization)

    pre_filtered = speckle_filter(pre)
    post_filtered = speckle_filter(post)

    diff = post_filtered.subtract(pre_filtered)
    flooded = diff.lt(threshold_db).rename("flooded")
    return flooded


def build_flood_inventory(bbox_ee, event_dates, output_dir=None, polarization="VV"):
    """
    Build multi-date flood inventory as a GEE ImageCollection.
    Returns ImageCollection with one binary mask per event.
    """
    masks = []
    for date in event_dates:
        mask = flood_mask_from_event(bbox_ee, date, polarization=polarization)
        mask = mask.set("event_date", date)
        masks.append(mask)

    inventory = ee.ImageCollection(masks)
    print(f"Flood inventory built: {len(event_dates)} events")

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
