import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

def load_config(path=None):
    path = path or REPO_ROOT / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)

def get_bbox(cfg, as_ee=False):
    """Return bbox as (xmin, ymin, xmax, ymax) or ee.Geometry.Rectangle."""
    b = cfg["bbox"]
    coords = [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
    if as_ee:
        import ee
        return ee.Geometry.Rectangle(coords)
    return tuple(coords)
