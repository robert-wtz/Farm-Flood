"""
Terrain analysis: DEM download → fill → flow direction → accumulation → TWI.
Uses pysheds. Key design decision: sinks are NOT filled indiscriminately —
they represent real lagunas in the Pampas endorheic landscape.
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from pysheds.grid import Grid
from scipy.ndimage import gaussian_filter


def load_dem(dem_path):
    """Load DEM raster, return (data, transform, crs)."""
    with rasterio.open(dem_path) as src:
        data = src.read(1).astype(float)
        data[data == src.nodata] = np.nan
        return data, src.transform, src.crs, src.meta.copy()


def smooth_dem(data, sigma=1.0):
    """Gaussian smoothing to reduce GLO-30 noise before flow routing."""
    mask = np.isnan(data)
    smoothed = gaussian_filter(np.where(mask, 0, data), sigma=sigma)
    counts = gaussian_filter((~mask).astype(float), sigma=sigma)
    result = np.where(counts > 0, smoothed / counts, np.nan)
    return result


def clip_dem_to_bbox(dem_path, bbox, output_path):
    """Clip DEM raster to bbox (xmin, ymin, xmax, ymax) and save."""
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box, mapping

    xmin, ymin, xmax, ymax = bbox
    geom = [mapping(box(xmin, ymin, xmax, ymax))]
    with rasterio.open(dem_path) as src:
        out_image, out_transform = rio_mask(src, geom, crop=True)
        meta = src.meta.copy()
        meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(out_image)
    return output_path


def compute_flow(dem_path, output_dir, method="dinf", smooth=True, sigma=1.0, bbox=None):
    """
    Fill pits only (not depressions) → flow direction → accumulation → TWI.
    bbox: (xmin, ymin, xmax, ymax) to clip DEM before processing.
    Returns dict of output raster paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if bbox is not None:
        clipped_path = output_dir / "dem_clipped.tif"
        clip_dem_to_bbox(dem_path, bbox, clipped_path)
        dem_path = clipped_path

    grid = Grid.from_raster(str(dem_path))
    dem = grid.read_raster(str(dem_path))

    if smooth:
        smoothed = smooth_dem(np.array(dem), sigma=sigma)
        dem = grid.view(dem)
        dem[:] = smoothed

    # Fill pits only — preserve closed depressions (lagunas)
    pit_filled = grid.fill_pits(dem)

    # Flow direction
    if method == "dinf":
        fdir = grid.flowdir(pit_filled, routing="dinf")
        acc = grid.accumulation(fdir, routing="dinf")
    else:
        fdir = grid.flowdir(pit_filled, routing="d8")
        acc = grid.accumulation(fdir, routing="d8")

    # Topographic Wetness Index: ln(a / tan(slope))
    dem_arr = np.array(pit_filled).astype(float)
    dy, dx = np.gradient(dem_arr)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_rad = np.maximum(slope_rad, 0.001)
    slope = np.degrees(slope_rad)
    twi = np.log((np.array(acc) + 1) / np.tan(slope_rad))

    # Save outputs
    outputs = {}
    dem_arr_shape = np.array(pit_filled).shape
    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": dem_arr_shape[1],
        "height": dem_arr_shape[0],
        "count": 1,
        "crs": grid.crs,
        "transform": grid.affine,
    }
    for name, arr in [("pit_filled", pit_filled), ("flow_acc", acc), ("twi", twi), ("slope", slope)]:
        out_path = output_dir / f"{name}.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(np.array(arr).astype("float32"), 1)
        outputs[name] = out_path

    # Detect lagunas: cells significantly lower than surroundings + high TWI,
    # plus pixels encoded as 0 in GLO-30 (existing water bodies)
    from scipy.ndimage import uniform_filter, label as ndlabel, binary_closing
    dem_raw = np.array(dem).astype(float)
    water_bodies = (dem_raw == 0)           # GLO-30 encodes water as 0
    dem_raw[dem_raw == 0] = np.nan

    local_mean = uniform_filter(np.where(np.isnan(dem_raw), 0, dem_raw), size=101)
    rel_elev = dem_raw - local_mean
    twi_arr = twi
    twi_thresh = np.nanpercentile(twi_arr, 90)
    terrain_sinks = (rel_elev < -1.0) & (twi_arr > twi_thresh) & ~np.isnan(dem_raw)

    # Merge water bodies + terrain sinks, close small gaps (radius ~150m = 5px)
    combined = terrain_sinks | water_bodies
    closed = binary_closing(combined, iterations=5)

    labeled, _ = ndlabel(closed)
    sizes = np.bincount(labeled.ravel())
    valid_ids = np.where(sizes > 30)[0]
    valid_ids = valid_ids[valid_ids != 0]
    depressions = np.isin(labeled, valid_ids).astype("uint8")

    sink_path = output_dir / "sinks.tif"
    with rasterio.open(sink_path, "w", **{**meta, "dtype": "uint8"}) as dst:
        dst.write(depressions, 1)
    outputs["sinks"] = sink_path

    print(f"Terrain outputs saved to {output_dir}")
    return outputs


