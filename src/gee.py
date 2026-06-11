"""
Google Earth Engine helpers: auth, Sentinel-1 SAR series, Sentinel-2 NDVI, CHIRPS.
Requires: earthengine-api authenticated via `earthengine authenticate`.
"""
import ee
import geemap
import pandas as pd
from pathlib import Path


def init_gee(project=None):
    if project is None:
        from .config import load_config
        project = load_config().get("gee", {}).get("project")
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    print(f"GEE initialized (project: {project}).")


def get_chirps_events(bbox_ee, start="2014-01-01", end="2024-12-31", threshold_mm=40):
    """
    Return list of dates where CHIRPS daily rainfall > threshold within bbox.
    """
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(start, end)
        .filterBounds(bbox_ee)
    )

    def flag_event(img):
        max_rain = img.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=bbox_ee,
            scale=5000,
            maxPixels=1e8,
        ).get("precipitation")
        return img.set("max_rain", max_rain)

    flagged = chirps.map(flag_event)
    events = flagged.filter(ee.Filter.gte("max_rain", threshold_mm))

    dates = events.aggregate_array("system:time_start").getInfo()
    event_dates = pd.to_datetime(dates, unit="ms").strftime("%Y-%m-%d").tolist()
    print(f"Found {len(event_dates)} rain events > {threshold_mm} mm")
    return event_dates


def get_s1_image(bbox_ee, date_str, window_days=6, polarization="VV", orbit="DESCENDING"):
    """Return single Sentinel-1 image (median composite) around a date."""
    start = (pd.Timestamp(date_str) - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(date_str) + pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(bbox_ee)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", polarization))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .select(polarization)
    )
    return col.median()


def get_s1_full_series(bbox_ee, start="2014-01-01", end="2024-12-31",
                       polarization="VV", orbit="DESCENDING"):
    """Return full Sentinel-1 ImageCollection for temporal analysis."""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(bbox_ee)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", polarization))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit))
        .select(polarization)
    )


def get_ndvi_median(bbox_ee, start="2019-01-01", end="2023-12-31"):
    """Sentinel-2 median NDVI mosaic (cloud-masked)."""
    def add_ndvi(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return img.addBands(ndvi)

    def mask_clouds(img):
        qa = img.select("QA60")
        cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(cloud_mask)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(bbox_ee)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_clouds)
        .map(add_ndvi)
    )
    return s2.select("NDVI").median()


def get_jrc_water(bbox_ee):
    """
    JRC Global Surface Water (Pekel et al.) — water occurrence 1984–present.
    Returns three GEE Images clipped to bbox:
      - occurrence:   % of time covered by water (0–100)
      - seasonality:  months/year with water (0–12)
      - recurrence:   % of years with water (0–100)
    """
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").clip(bbox_ee)
    return {
        "occurrence":  jrc.select("occurrence"),
        "seasonality": jrc.select("seasonality"),
        "recurrence":  jrc.select("recurrence"),
    }


def export_to_drive(image, description, bbox_ee, scale=30, folder="FarmFlood"):
    """Export GEE image to Google Drive."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        region=bbox_ee,
        scale=scale,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Export task started: {description}")
    return task
