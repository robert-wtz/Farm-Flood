"""
Visualization: static matplotlib maps + interactive folium maps.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import rasterio
from rasterio.plot import show
import geopandas as gpd
import folium
from pathlib import Path


def plot_raster(path, title="", cmap="viridis", figsize=(10, 8), output_path=None, log_scale=False):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        fig, ax = plt.subplots(figsize=figsize)
        if log_scale:
            data = np.where(data > 0, np.log1p(data), np.nan)
            title = f"{title} (log scale)"
        import matplotlib.colors as mc
        norm = mc.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
        im = ax.imshow(data, cmap=cmap, norm=norm)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.7)
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
    return fig


def plot_twi_and_sinks(twi_path, sinks_path, output_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, path, title, cmap in [
        (axes[0], twi_path, "TWI (Topographic Wetness Index)", "Blues"),
        (axes[1], sinks_path, "Natural Sinks (Lagunas candidates)", "Reds"),
    ]:
        with rasterio.open(path) as src:
            show(src, ax=ax, cmap=cmap, title=title)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def folium_map(center_lat=-34.817, center_lon=-62.450, zoom=11):
    """Base folium map centered on Ameghino."""
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles="CartoDB positron")
    return m


def add_raster_layer(m, raster_path, name, colormap="YlOrRd", opacity=0.6):
    """Add a raster overlay to a folium map via image overlay."""
    import rasterio
    from rasterio.warp import transform_bounds
    import matplotlib.cm as cm

    with rasterio.open(raster_path) as src:
        data = src.read(1).astype(float)
        nodata = src.nodata
        bounds = src.bounds
        crs = src.crs

    if nodata is not None:
        data[data == nodata] = np.nan

    # Normalize
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    data_norm = (data - vmin) / (vmax - vmin + 1e-9)

    cmap_fn = plt.get_cmap(colormap)
    rgba = cmap_fn(data_norm)
    rgba[np.isnan(data)] = [0, 0, 0, 0]

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        plt.imsave(f.name, rgba)
        img_path = f.name

    folium.raster_layers.ImageOverlay(
        image=img_path,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        name=name,
        opacity=opacity,
    ).add_to(m)

    return m


def add_vector_layer(m, gdf, name, color="blue", fill_color="blue", fill_opacity=0.4):
    folium.GeoJson(
        gdf.__geo_interface__,
        name=name,
        style_function=lambda x: {
            "color": color,
            "fillColor": fill_color,
            "fillOpacity": fill_opacity,
            "weight": 1,
        },
    ).add_to(m)
    return m


def final_map(flood_prob_path, drainage_gdf, retention_gdf, suitability_path,
              output_html="outputs/final_map.html"):
    """Render the final interactive map with all layers."""
    m = folium_map()
    add_raster_layer(m, flood_prob_path, "Flood Probability", colormap="YlOrRd")
    add_raster_layer(m, suitability_path, "Crop Suitability", colormap="YlGn")
    if drainage_gdf is not None and len(drainage_gdf):
        add_vector_layer(m, drainage_gdf, "Drainage Channels", color="blue")
    if retention_gdf is not None and len(retention_gdf):
        add_vector_layer(m, retention_gdf, "Retention Lagoons", color="cyan", fill_color="cyan")
    folium.LayerControl().add_to(m)
    m.save(output_html)
    print(f"Final map saved: {output_html}")
    return m
