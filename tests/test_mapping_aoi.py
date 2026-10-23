"""Tests for the Step 05 area-of-interest (bbox OR polygon) rules."""
import os
import sys

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Polygon, box

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, 'src'))

import mapping_aoi as ma  # noqa: E402

# 10 x 10 north-up grid, 1-unit pixels, upper-left (0, 10); centres x = i + 0.5, y = 9.5 - j
GRID = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 10.0)
CRS_GRID = 'EPSG:3857'


def window(aoi, transform=GRID, width=10, height=10, crs=CRS_GRID):
    return ma.fn_aoi_window(aoi, transform, width, height, crs)


# ---------------------------------------------------------------- bbox parsing
@pytest.mark.parametrize('values', [
    [0, 0, 1], [0, 0, 1, 1, 2], [0, float('nan'), 1, 1], [0, 0, float('inf'), 1],
    [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1], ['a', 0, 1, 1],
])
def test_bbox_rejects_bad_values(values):
    with pytest.raises(ma.MappingAOIError):
        ma.fn_parse_bbox(values)


def test_bbox_accepts_numbers_and_strings():
    assert ma.fn_parse_bbox([0, 1, 2, 3]) == (0.0, 1.0, 2.0, 3.0)
    assert ma.fn_parse_bbox('0 1, 2 3') == (0.0, 1.0, 2.0, 3.0)


# ---------------------------------------------------------------- resolution
def test_resolve_none_without_aoi():
    assert ma.fn_resolve_aoi() is None


def test_bbox_and_polygon_are_mutually_exclusive():
    with pytest.raises(ma.MappingAOIError, match='mutually exclusive'):
        ma.fn_resolve_aoi(bbox=[0, 0, 1, 1], polygon=box(0, 0, 1, 1), aoi_crs=CRS_GRID)


def test_orphan_aoi_options_are_rejected():
    with pytest.raises(ma.MappingAOIError, match='aoi_crs'):
        ma.fn_resolve_aoi(aoi_crs='EPSG:4326')
    with pytest.raises(ma.MappingAOIError, match='polygon_layer'):
        ma.fn_resolve_aoi(bbox=[0, 0, 1, 1], polygon_layer='x', default_bbox_crs=CRS_GRID)


def test_bbox_defaults_to_output_crs_and_requires_some_crs():
    aoi = ma.fn_resolve_aoi(bbox=[0, 0, 1, 1], default_bbox_crs='EPSG:3857')
    assert aoi.crs == CRS.from_epsg(3857) and aoi.kind == 'bbox'
    aoi = ma.fn_resolve_aoi(bbox=[0, 0, 1, 1], aoi_crs='EPSG:2277', default_bbox_crs='EPSG:3857')
    assert aoi.crs == CRS.from_epsg(2277)
    with pytest.raises(ma.MappingAOIError, match='CRS'):
        ma.fn_resolve_aoi(bbox=[0, 0, 1, 1])


def test_shapely_polygon_requires_crs_and_polygonal_valid_geometry():
    with pytest.raises(ma.MappingAOIError, match='no CRS'):
        ma.fn_resolve_aoi(polygon=box(0, 0, 1, 1))
    with pytest.raises(ma.MappingAOIError, match='Polygon or MultiPolygon'):
        ma.fn_resolve_aoi(polygon=LineString([(0, 0), (1, 1)]), aoi_crs=CRS_GRID)
    with pytest.raises(ma.MappingAOIError, match='empty'):
        ma.fn_resolve_aoi(polygon=Polygon(), aoi_crs=CRS_GRID)
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2)])
    with pytest.raises(ma.MappingAOIError, match='invalid'):
        ma.fn_resolve_aoi(polygon=bowtie, aoi_crs=CRS_GRID)


# ---------------------------------------------------------------- polygon files
def test_polygon_geojson_file_uses_file_crs(tmp_path):
    str_path = str(tmp_path / 'aoi.geojson')
    gpd.GeoDataFrame(geometry=[box(1, 1, 3, 3), box(5, 5, 6, 6)], crs='EPSG:3857').to_file(str_path)
    aoi = ma.fn_resolve_aoi(polygon=str_path)
    assert aoi.kind == 'polygon' and aoi.crs == CRS.from_epsg(3857)
    assert isinstance(aoi.geometry, MultiPolygon)


def test_polygon_file_crs_rules(tmp_path):
    str_nocrs = str(tmp_path / 'nocrs.gpkg')
    gpd.GeoDataFrame(geometry=[box(1, 1, 3, 3)], crs=None).to_file(str_nocrs, layer='a')
    with pytest.raises(ma.MappingAOIError, match='no CRS'):
        ma.fn_resolve_aoi(polygon=str_nocrs)
    assert ma.fn_resolve_aoi(polygon=str_nocrs, aoi_crs=CRS_GRID).crs == CRS.from_epsg(3857)

    str_crs = str(tmp_path / 'crs.gpkg')
    gpd.GeoDataFrame(geometry=[box(1, 1, 3, 3)], crs='EPSG:3857').to_file(str_crs, layer='a')
    with pytest.raises(ma.MappingAOIError, match='conflicts'):
        ma.fn_resolve_aoi(polygon=str_crs, aoi_crs='EPSG:4326')


def test_polygon_gpkg_layers(tmp_path):
    str_path = str(tmp_path / 'multi.gpkg')
    gpd.GeoDataFrame(geometry=[box(1, 1, 3, 3)], crs=CRS_GRID).to_file(str_path, layer='keep')
    gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs=CRS_GRID).to_file(str_path, layer='line')
    with pytest.raises(ma.MappingAOIError, match='layers'):
        ma.fn_resolve_aoi(polygon=str_path)
    assert ma.fn_resolve_aoi(polygon=str_path, polygon_layer='keep').geometry.equals(box(1, 1, 3, 3))
    with pytest.raises(ma.MappingAOIError, match='Polygon or MultiPolygon'):
        ma.fn_resolve_aoi(polygon=str_path, polygon_layer='line')
    with pytest.raises(ma.MappingAOIError, match='not in'):
        ma.fn_resolve_aoi(polygon=str_path, polygon_layer='missing')


def test_polygon_file_missing(tmp_path):
    with pytest.raises(ma.MappingAOIError, match='not found'):
        ma.fn_resolve_aoi(polygon=str(tmp_path / 'none.geojson'))


# ---------------------------------------------------------------- windows
def test_bbox_window_selects_pixel_centres_half_open():
    aoi = ma.fn_bbox_aoi([2, 3, 5, 7], CRS_GRID)
    win = window(aoi)
    # x centres 2.5, 3.5, 4.5 ; y centres 6.5 .. 3.5 -> rows 3..6
    assert win.window == [2, 3, 3, 4] and win.mask is None and not win.is_empty
    assert win.transform == Affine(1.0, 0.0, 2.0, 0.0, -1.0, 7.0)
    # a centre exactly on xmin is inside, exactly on xmax is outside
    assert window(ma.fn_bbox_aoi([2.5, 3, 4.5, 7], CRS_GRID)).window == [2, 3, 2, 4]


def test_bbox_window_clips_to_grid_and_empty_outside():
    assert window(ma.fn_bbox_aoi([-5, -5, 50, 50], CRS_GRID)).window == [0, 0, 10, 10]
    win = window(ma.fn_bbox_aoi([20, 20, 30, 30], CRS_GRID))
    assert win.is_empty and win.status == ma.AOI_STATUS_EMPTY
    # strictly between two centres: no pixel selected
    assert window(ma.fn_bbox_aoi([2.6, 3, 3.4, 7], CRS_GRID)).is_empty


def test_rectangular_polygon_equals_bbox_crop():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    win_b = window(ma.fn_bbox_aoi([2, 3, 5, 7], CRS_GRID))
    win_p = window(ma.fn_polygon_aoi_from_geometry(box(2, 3, 5, 7), CRS_GRID))
    assert win_p.window == win_b.window and win_p.mask.all()
    np.testing.assert_array_equal(ma.fn_apply_aoi_window(arr, win_b), ma.fn_apply_aoi_window(arr, win_p))
    np.testing.assert_array_equal(ma.fn_apply_aoi_window(arr, win_b), arr[3:7, 2:5])


def test_polygon_hole_is_masked():
    poly = Polygon(box(1, 1, 9, 9).exterior.coords, [box(4, 4, 6, 6).exterior.coords])
    win = window(ma.fn_polygon_aoi_from_geometry(poly, CRS_GRID))
    assert win.window == [1, 1, 8, 8]
    out = ma.fn_apply_aoi_window(np.ones((10, 10), np.float32), win)
    # hole covers centres x 4.5, 5.5 and y 5.5, 4.5 -> window rows/cols 3..4
    assert np.isnan(out[3:5, 3:5]).all()
    assert np.isfinite(out).sum() == 64 - 4


def test_multipart_polygon_masks_the_gap():
    mp = MultiPolygon([box(0, 0, 2, 2), box(7, 7, 9, 9)])
    win = window(ma.fn_polygon_aoi_from_geometry(mp, CRS_GRID))
    assert win.window == [0, 1, 9, 9]
    out = ma.fn_apply_aoi_window(np.ones((10, 10), np.float32), win)
    assert np.isfinite(out).sum() == 8
    assert np.isnan(out[4, 4])


def test_polygon_without_pixel_centres_is_empty():
    sliver = Polygon([(2.6, 2.6), (2.9, 2.6), (2.9, 2.9)])
    assert window(ma.fn_polygon_aoi_from_geometry(sliver, CRS_GRID)).is_empty


def test_aoi_in_other_crs_is_reprojected_with_mask():
    # grid in EPSG:3857 near (0, 0); bbox given in EPSG:4326 degrees
    grid = Affine(1000.0, 0.0, -5000.0, 0.0, -1000.0, 5000.0)
    aoi = ma.fn_bbox_aoi([-0.02, -0.02, 0.02, 0.02], 'EPSG:4326')
    win = ma.fn_aoi_window(aoi, grid, 10, 10, 'EPSG:3857')
    # +/-0.02 deg ~ +/-2226 m -> centres -1500 .. 1500
    assert win.window == [3, 3, 4, 4]
    assert win.mask is not None and win.mask.all()


def test_south_up_grid_is_rejected():
    with pytest.raises(ma.MappingAOIError, match='north-up'):
        ma.fn_aoi_window(ma.fn_bbox_aoi([0, 0, 1, 1], CRS_GRID), Affine(1, 0, 0, 0, 1, 0), 10, 10, CRS_GRID)
