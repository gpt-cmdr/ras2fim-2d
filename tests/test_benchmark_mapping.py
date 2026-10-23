"""The performance harness must reject products that are merely close."""
import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

pytest.importorskip("resource", reason="Linux-only benchmark process accounting")
spec = importlib.util.spec_from_file_location("benchmark_mapping", Path(__file__).parents[1] / "benchmarks/benchmark_mapping.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def products(root):
    root.mkdir()
    values = np.array([[np.nan, 0.0], [0.25, 1.0]], dtype="float32")
    profile = dict(driver="GTiff", width=2, height=2, count=1, dtype="float32", crs="EPSG:3857",
                   transform=from_origin(0, 6, 3, 3), nodata=np.nan)
    for name in ("09_wsel_max_1_depth_nudge.tif", "10_wsel_max_1_wsel_nudge.tif"):
        with rasterio.open(root / name, "w", **profile) as dataset:
            dataset.write(values, 1)
    gpd.GeoDataFrame(geometry=[box(0, 0, 6, 3)], crs="EPSG:3857").to_file(root / "11_wsel_max_1_floodplain_ar.gpkg")


def test_accepts_identical_products(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    products(left)
    products(right)
    assert benchmark.compare_products(left, right, ["wsel_max_1"]) == 3


@pytest.mark.parametrize("change, error", [("ulp", "float32"), ("nodata", "Nodata"), ("grid", "Grid")])
def test_rejects_numerical_or_grid_changes(tmp_path, change, error):
    left, right = tmp_path / "left", tmp_path / "right"
    products(left)
    products(right)
    path = right / "09_wsel_max_1_depth_nudge.tif"
    with rasterio.open(path, "r+") as dataset:
        values = dataset.read(1)
        if change == "ulp":
            values[1, 0] = np.nextafter(values[1, 0], np.float32(np.inf))
        elif change == "nodata":
            values[0, 0] = 0
        else:
            dataset.transform = from_origin(1, 6, 3, 3)
        dataset.write(values, 1)
    with pytest.raises(AssertionError, match=error):
        benchmark.compare_products(left, right, ["wsel_max_1"])


def test_rejects_changed_floodplain(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    products(left)
    products(right)
    gpd.GeoDataFrame(geometry=[box(0, 0, 3, 3)], crs="EPSG:3857").to_file(right / "11_wsel_max_1_floodplain_ar.gpkg")
    with pytest.raises(AssertionError, match="Floodplain"):
        benchmark.compare_products(left, right, ["wsel_max_1"])
