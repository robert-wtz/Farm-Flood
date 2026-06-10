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
    counts = gaussian_filter(~mask.astype(float), sigma=sigma)
    result = np.where(counts > 0, smoothed / counts, np.nan)
    return result


def compute_flow(dem_path, output_dir, method="dinf", smooth=True, sigma=1.0):
    """
    Fill pits only (not depressions) → flow direction → accumulation → TWI.
    Returns dict of output raster paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    # Using a small epsilon to avoid log(0) in flat areas
    slope = grid.calculate_slope(pit_filled)
    slope_rad = np.deg2rad(np.maximum(slope, 0.001))
    twi = np.log((acc + 1) / np.tan(slope_rad))

    # Save outputs
    outputs = {}
    meta = grid.to_rasterio_meta()
    for name, arr in [("pit_filled", pit_filled), ("flow_acc", acc), ("twi", twi), ("slope", slope)]:
        out_path = output_dir / f"{name}.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(np.array(arr).astype("float32"), 1)
        outputs[name] = out_path

    # Identify sinks (closed depressions = real lagunas)
    depressions = _identify_sinks(np.array(pit_filled), np.array(dem))
    sink_path = output_dir / "sinks.tif"
    with rasterio.open(sink_path, "w", **{**meta, "dtype": "uint8"}) as dst:
        dst.write(depressions.astype("uint8"), 1)
    outputs["sinks"] = sink_path

    print(f"Terrain outputs saved to {output_dir}")
    return outputs


def _identify_sinks(pit_filled, original_dem):
    """Mask of cells that are depressions (filled > original = real sink)."""
    diff = pit_filled - original_dem
    return (diff > 0.1).astype("uint8")
