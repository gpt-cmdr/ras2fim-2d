"""
Tests for Step 05 engine selection, shared sample preparation, Rust-engine
input derivation and the adapter I/O.

Synthetic tests build a small jittered mesh, its HDF cell centres, a
hydraulic-results GeoPackage and a terrain GeoTIFF in a temporary folder.
Tests marked "real" need external public fixtures given by environment
variables (nothing is committed):
  RAS2FIM_PUBLIC_SAMPLE       folder with results.gpkg, sample.p01.hdf, terrain.tif
  RAS2FIM_STEP05_ORACLE_FLOWS upstream Step 05 oracle flows/<col>/{prepared.npz, capture.json, 09_*}
  RAS2FIM_RUST_BINARY         built ras2fim-batch supporting the `step05` command
They are reported as skipped (not passed) when the variables are unset.
"""
import argparse
import configparser
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
from functools import partial

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from scipy.interpolate import LinearNDInterpolator, griddata
from shapely.geometry import Polygon, box

STR_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'src'))
sys.path.insert(0, STR_SRC)

import generate_depth_wsel_rasters_mp_05 as step05  # noqa: E402
import mapping_aoi  # noqa: E402
import rust_mapping  # noqa: E402

CRS_MODEL = 'EPSG:2277'
X0, Y0, CELL = 3_000_000.0, 10_000_000.0, 50.0
NX, NY = 14, 12
LIST_COLS = ['wsel_max_1', 'wsel_max_2']


# ============================================================ synthetic fixture
def _write_config(str_path, dict_engine=None):
    config = configparser.ConfigParser()
    config.read(os.path.join(STR_SRC, 'config_global.ini'))
    for str_key, str_value in (dict_engine or {}).items():
        config['05_raster_engine'][str_key] = str_value
    with open(str_path, 'w') as f:
        config.write(f)
    return str_path


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    """A 14 x 12 square-cell mesh with jittered centres and two WSE profiles."""
    root = tmp_path_factory.mktemp('synthetic')
    rng = np.random.default_rng(20261023)
    list_poly, list_idx, list_centers = [], [], []
    for j in range(NY):
        for i in range(NX):
            x, y = X0 + i * CELL, Y0 + j * CELL
            list_poly.append(box(x, y, x + CELL, y + CELL))
            list_idx.append(j * NX + i)
            list_centers.append((x + CELL / 2 + rng.uniform(-6, 6), y + CELL / 2 + rng.uniform(-6, 6)))
    arr_centers = np.array(list_centers)
    # stored cells are indexed by cell_idx; trailing rows stand in for ghost cells
    arr_stored = np.vstack([arr_centers, [[0.0, 0.0], [0.0, 0.0]]])

    gdf = gpd.GeoDataFrame({'cell_idx': list_idx}, geometry=list_poly, crs=CRS_MODEL)
    cx, cy = arr_centers[:, 0] - X0, arr_centers[:, 1] - Y0
    wse1 = 104.0 + 0.002 * cx - 0.001 * cy + rng.uniform(0, 0.05, len(gdf))
    wse2 = wse1 + 0.8
    wse1[(cx > 500) & (cy > 400)] = np.nan          # dry corner
    wse2[(cx > 650) & (cy > 520)] = np.nan
    gdf['wsel_max_1'] = wse1
    gdf['wsel_max_2'] = wse2
    # never-wet cells are absent from the Step 04 layer
    gdf = gdf[~((cx < 60) & (cy < 60))].reset_index(drop=True)

    str_gpkg = str(root / 'results.gpkg')
    gdf.to_file(str_gpkg, layer='02_cells_wsel_ar')

    str_hdf = str(root / 'plan.p01.hdf')
    with h5py.File(str_hdf, 'w') as f:
        f.create_dataset('/Geometry/2D Flow Areas/Perimeter 1/Cells Center Coordinate', data=arr_stored)

    # terrain: 5 ft pixels with a margin around the mesh, sloping ground
    flt_res = 5.0
    int_w, int_h = int((NX * CELL + 200) / flt_res), int((NY * CELL + 200) / flt_res)
    transform = from_origin(X0 - 100, Y0 + NY * CELL + 100, flt_res, flt_res)
    xs = transform.c + (np.arange(int_w) + 0.5) * flt_res - X0
    ys = transform.f - (np.arange(int_h) + 0.5) * flt_res - Y0
    gx, gy = np.meshgrid(xs, ys)
    arr_dem = (103.0 + 0.003 * gx - 0.0005 * gy + 0.6 * np.sin(gx / 40.0)).astype(np.float32)
    str_dem = str(root / 'terrain.tif')
    with rasterio.open(str_dem, 'w', driver='GTiff', width=int_w, height=int_h, count=1, dtype='float32',
                       crs=CRS_MODEL, transform=transform, nodata=-9999.0, tiled=True,
                       blockxsize=64, blockysize=64) as dst:
        dst.write(arr_dem, 1)

    str_config = _write_config(str(root / 'config.ini'))
    gdf_cells = gpd.read_file(str_gpkg, layer='02_cells_wsel_ar')
    gdf_centers = step05.fn_gdf_cell_centerpoints(str_hdf, sorted(gdf_cells['cell_idx'].unique()), gdf_cells)
    return dict(root=root, gpkg=str_gpkg, hdf=str_hdf, dem=str_dem, config=str_config,
                gdf=gdf_cells, centers=gdf_centers, res=step05.fn_get_raster_resolution(str_dem))


@pytest.fixture(scope='module')
def python_full(synthetic, tmp_path_factory):
    """Full-map products of the unmodified python engine (serial, no pool)."""
    out = tmp_path_factory.mktemp('python_full')
    for str_col in LIST_COLS:
        step05.fn_process_wsel(str_col, synthetic['gdf'], synthetic['centers'], synthetic['res'],
                               'EPSG:3857', 3.0, str(out), synthetic['dem'])
    for str_col in LIST_COLS:
        assert os.path.exists(out / f'09_{str_col}_depth_nudge.tif'), 'legacy engine failed on the fixture'
    return str(out)


def _read(str_path):
    with rasterio.open(str_path) as src:
        return src.read(1), src.transform, src.crs, (src.height, src.width)


def _fake_binary(tmp_path):
    str_bin = tmp_path / 'ras2fim-batch'
    str_bin.write_text('')
    str_bin.chmod(0o755)
    return str(str_bin)


# ============================================================ options / config
def test_default_options_are_the_python_engine():
    opts = rust_mapping.fn_resolve_mapping_options(configparser.ConfigParser())
    assert opts.engine == 'python' and opts.is_legacy_default and not opts.has_aoi


def test_shipped_configs_keep_python_default():
    for str_name in ('config_global.ini', 'config_global_windows.ini'):
        config = configparser.ConfigParser()
        config.read(os.path.join(STR_SRC, str_name))
        assert rust_mapping.fn_resolve_mapping_options(config).is_legacy_default


def test_resolved_threads_is_at_least_one(monkeypatch):
    monkeypatch.setattr(rust_mapping.mp, 'cpu_count', lambda: 1)
    assert rust_mapping.Step05MappingOptions(engine='rust').resolved_threads() == 1
    monkeypatch.setattr(rust_mapping.mp, 'cpu_count', lambda: 64)
    assert rust_mapping.Step05MappingOptions(engine='rust').resolved_threads() == 16
    assert rust_mapping.Step05MappingOptions(engine='rust', threads=3).resolved_threads() == 3


def test_config_then_keyword_precedence(tmp_path):
    config = configparser.ConfigParser()
    config.read(_write_config(str(tmp_path / 'c.ini'), {
        'str_mapping_engine': 'rust', 'int_mapping_threads': '3', 'str_aoi_polygon': 'x.gpkg'}))
    opts = rust_mapping.fn_resolve_mapping_options(config)
    assert (opts.engine, opts.threads, opts.polygon) == ('rust', 3, 'x.gpkg')
    opts = rust_mapping.fn_resolve_mapping_options(config, engine='python', threads=5, bbox=[0, 0, 1, 1])
    assert (opts.engine, opts.threads, opts.polygon, opts.bbox) == ('python', 5, None, [0, 0, 1, 1])


@pytest.mark.parametrize('dict_bad, str_match', [
    ({'str_mapping_engine': 'gdal'}, 'engine'),
    ({'int_mapping_threads': '0'}, 'threads'),
    ({'str_unknown_key': '1'}, 'unknown key'),
    ({'str_rust_binary': '/x/ras2fim-batch'}, 'engine is python'),
    ({'str_rust_nudge_method': 'exact'}, 'nudge'),
])
def test_invalid_config_is_rejected(tmp_path, dict_bad, str_match):
    config = configparser.ConfigParser()
    config.read(_write_config(str(tmp_path / 'c.ini'), dict_bad))
    with pytest.raises(ValueError, match=str_match):
        rust_mapping.fn_resolve_mapping_options(config)


def test_missing_explicit_binary_fails(tmp_path):
    with pytest.raises(rust_mapping.RustBinaryNotFoundError, match='not found'):
        rust_mapping.fn_resolve_rust_binary(str(tmp_path / 'ras2fim-batch'))


def test_missing_binary_on_path_fails_without_fallback(monkeypatch):
    monkeypatch.setattr(rust_mapping.shutil, 'which', lambda name: None)
    with pytest.raises(rust_mapping.RustBinaryNotFoundError, match='no automatic fallback'):
        rust_mapping.fn_resolve_rust_binary(None)


def test_binary_resolution_explicit_and_path(tmp_path, monkeypatch):
    str_bin = _fake_binary(tmp_path)
    assert rust_mapping.fn_resolve_rust_binary(str_bin) == os.path.abspath(str_bin)
    monkeypatch.setattr(rust_mapping.shutil, 'which', lambda name: str_bin)
    assert rust_mapping.fn_resolve_rust_binary(None) == os.path.abspath(str_bin)


# ============================================================ CLI
def _parser():
    return rust_mapping.fn_add_mapping_arguments(argparse.ArgumentParser())


def test_cli_options_map_to_keywords():
    args = vars(_parser().parse_args(['--mapping-engine', 'rust', '--rust-binary', 'b', '--mapping-threads',
                                      '6', '--bbox', '1', '2', '3', '4', '--aoi-crs', 'EPSG:3857']))
    assert rust_mapping.fn_mapping_kwargs_from_args(args) == {
        'mapping_engine': 'rust', 'rust_binary': 'b', 'mapping_threads': 6,
        'bbox': [1.0, 2.0, 3.0, 4.0], 'aoi_crs': 'EPSG:3857'}
    assert rust_mapping.fn_mapping_kwargs_from_args(vars(_parser().parse_args([]))) == {}
    args = vars(_parser().parse_args(['--rust-max-inflight-profiles', '2', '--rust-profiles-per-request', '19']))
    assert rust_mapping.fn_mapping_kwargs_from_args(args) == {'rust_max_inflight_profiles': 2,
                                                             'rust_profiles_per_request': 19}
    with pytest.raises(ValueError, match='rust_max_inflight_profiles'):
        rust_mapping.fn_resolve_mapping_options(None, rust_max_inflight_profiles=0)


def test_cli_bbox_and_polygon_are_exclusive():
    with pytest.raises(SystemExit):
        _parser().parse_args(['--bbox', '1', '2', '3', '4', '--polygon', 'a.gpkg'])
    with pytest.raises(SystemExit):
        _parser().parse_args(['--mapping-engine', 'gdal'])


@pytest.mark.parametrize('str_script', ['generate_depth_wsel_rasters_mp_05.py', 'ras2fim-2d.py'])
def test_scripts_expose_mapping_options(str_script):
    proc = subprocess.run([sys.executable, os.path.join(STR_SRC, str_script), '--help'],
                          capture_output=True, text=True, cwd=STR_SRC)
    assert proc.returncode == 0, proc.stderr
    for str_opt in ('--mapping-engine', '--rust-binary', '--mapping-threads', '--bbox', '--polygon',
                    '--aoi-crs', '--polygon-layer', '--rust-max-inflight-profiles'):
        assert str_opt in proc.stdout


def test_step05_cli_exits_nonzero_for_missing_binary(synthetic, tmp_path):
    proc = subprocess.run([sys.executable, os.path.join(STR_SRC, 'generate_depth_wsel_rasters_mp_05.py'),
                           '-c', synthetic['config'], '-i', synthetic['hdf'], '-t', synthetic['dem'],
                           '-g', synthetic['gpkg'], '-o', str(tmp_path), '-r', 'f',
                           '--mapping-engine', 'rust', '--rust-binary', str(tmp_path / 'missing')],
                          capture_output=True, text=True, cwd=STR_SRC)
    assert proc.returncode != 0 and 'RustBinaryNotFoundError' in proc.stderr


# ============================================================ dispatch
class _RecordingPool:
    calls = []

    def __init__(self, processes):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, func, items):
        _RecordingPool.calls.append((self.processes, func, list(items)))
        return []


def _no_engine(*args, **kwargs):
    raise AssertionError('this engine must not be invoked')


def test_default_dispatch_is_unchanged(synthetic, tmp_path, monkeypatch):
    _RecordingPool.calls = []
    monkeypatch.setattr(step05.mp, 'Pool', _RecordingPool)
    monkeypatch.setattr(rust_mapping, 'fn_run_binary', _no_engine)
    result = step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'], synthetic['dem'],
                                                   synthetic['gpkg'], str(tmp_path), False)
    assert result is None
    assert os.listdir(tmp_path) == []
    (int_proc, func, list_cols), = _RecordingPool.calls
    assert int_proc == min(mp.cpu_count() - 1, 16)
    assert isinstance(func, partial) and func.func is step05.fn_process_wsel
    assert list_cols == LIST_COLS
    assert set(func.keywords) == {'gdf_cells_wsel', 'gdf_center_points', 'flt_pixel_width', 'str_output_crs',
                                  'flt_desired_res', 'str_output_dir', 'str_dem_path'}
    assert func.keywords['str_output_crs'] == 'EPSG:3857' and func.keywords['flt_desired_res'] == 3.0


def test_mapping_threads_sets_python_pool(synthetic, tmp_path, monkeypatch):
    _RecordingPool.calls = []
    monkeypatch.setattr(step05.mp, 'Pool', _RecordingPool)
    step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'], synthetic['dem'],
                                          synthetic['gpkg'], str(tmp_path), False, mapping_threads=2)
    assert _RecordingPool.calls[0][0] == 2


def test_rust_engine_missing_binary_fails_before_work(synthetic, tmp_path, monkeypatch):
    monkeypatch.setattr(step05.gpd, 'read_file', _no_engine)
    monkeypatch.setattr(step05.mp, 'Pool', _no_engine)
    with pytest.raises(rust_mapping.RustBinaryNotFoundError):
        step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'], synthetic['dem'],
                                              synthetic['gpkg'], str(tmp_path), False, mapping_engine='rust',
                                              rust_binary=str(tmp_path / 'missing-binary'))
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize('dict_kw, exc', [
    ({'bbox': [3, 0, 1, 1]}, mapping_aoi.MappingAOIError),
    ({'polygon': 'does-not-exist.gpkg'}, mapping_aoi.MappingAOIError),
    ({'bbox': [0, 0, 1, 1], 'polygon': 'x.gpkg'}, ValueError),
    ({'aoi_crs': 'EPSG:4326'}, mapping_aoi.MappingAOIError),
])
def test_invalid_aoi_fails_before_work(synthetic, tmp_path, monkeypatch, dict_kw, exc):
    monkeypatch.setattr(step05.mp, 'Pool', _no_engine)
    with pytest.raises(exc):
        step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'], synthetic['dem'],
                                              synthetic['gpkg'], str(tmp_path), False, **dict_kw)


# ============================================================ shared sample preparation
def _frozen_upstream_samples(gdf_pnt, gdf_ar, str_col, gdf_dissolved, flt_res):
    """Frozen copy of the original fn_create_raster_products sample recipe."""
    created, wsel = [], []
    hull = step05.fn_compute_convex_hull(gdf_pnt.geometry.apply(lambda p: (p.x, p.y)).tolist())
    for _, cell in gdf_ar.iterrows():
        for v in step05.fn_calculate_vertex_points(cell.geometry):
            if not step05.fn_is_point_inside_convex_hull(v, hull):
                created.append(v)
                wsel.append(cell[str_col])
    extra = gpd.GeoDataFrame(geometry=created)
    extra[str_col] = wsel
    merged = step05.fn_merge_points_within_threshold(extra, 0.01, str_col)
    pts = gdf_pnt[['geometry', str_col]].copy()
    if len(merged) > 0:
        pts = pd.concat([pts, merged[['geometry', str_col]].copy()], axis=0)
    xmin, ymin, xmax, ymax = gdf_dissolved.total_bounds
    return (pts.geometry.x.values, pts.geometry.y.values, pts[str_col].values,
            np.arange(xmin, xmax, flt_res), np.arange(ymin, ymax, flt_res))


@pytest.mark.parametrize('str_col', LIST_COLS)
def test_prepared_samples_match_frozen_upstream_recipe(synthetic, str_col):
    geom, gdf_dis, gdf_pnt, gdf_ar = step05.fn_wet_cells_per_run(str_col, synthetic['gdf'], synthetic['centers'])
    prep = step05.fn_prepare_interpolation_inputs(gdf_pnt, gdf_ar, str_col, gdf_dis, synthetic['res'])
    for a, b in zip((prep.x, prep.y, prep.z, prep.x_axis, prep.y_axis),
                    _frozen_upstream_samples(gdf_pnt, gdf_ar, str_col, gdf_dis, synthetic['res'])):
        assert np.array_equal(a, b)
    # wet centres first, in sorted cell_idx order, then merged extras
    n_wet = len(gdf_pnt)
    assert list(gdf_pnt['cell_idx']) == sorted(gdf_pnt['cell_idx'])
    assert np.array_equal(prep.x[:n_wet], gdf_pnt.geometry.x.values)
    assert len(prep.x) > n_wet
    # closing-ring duplicates and shared corners appear once (group mean WSE)
    keys = list(zip(np.round(prep.x[n_wet:], 2), np.round(prep.y[n_wet:], 2)))
    assert len(set(keys)) == len(keys)
    # query axes are np.arange from the wet-union origin; the raster affine keeps the pixel size
    assert prep.x_axis[0] == prep.xmin and prep.x_res == synthetic['res']


def test_merged_extras_use_group_mean_wse(synthetic):
    str_col = 'wsel_max_1'
    geom, gdf_dis, gdf_pnt, gdf_ar = step05.fn_wet_cells_per_run(str_col, synthetic['gdf'], synthetic['centers'])
    prep = step05.fn_prepare_interpolation_inputs(gdf_pnt, gdf_ar, str_col, gdf_dis, synthetic['res'])
    n_wet = len(gdf_pnt)
    # rebuild the raw extra occurrences (including ring-closing duplicates)
    dict_groups = {}
    hull = step05.fn_compute_convex_hull(gdf_pnt.geometry.apply(lambda p: (p.x, p.y)).tolist())
    for _, cell in gdf_ar.iterrows():
        for v in step05.fn_calculate_vertex_points(cell.geometry):
            if not step05.fn_is_point_inside_convex_hull(v, hull):
                dict_groups.setdefault((round(v.x, 2), round(v.y, 2)), []).append(cell[str_col])
    assert any(len(v) > 1 for v in dict_groups.values())
    for (x, y, z) in zip(prep.x[n_wet:], prep.y[n_wet:], prep.z[n_wet:]):
        list_vals = dict_groups[(round(x, 2), round(y, 2))]
        assert z == pd.Series(list_vals).mean()


def test_legacy_raster_01_uses_the_shared_preparation(synthetic, tmp_path):
    str_col = 'wsel_max_1'
    geom, gdf_dis, gdf_pnt, gdf_ar = step05.fn_wet_cells_per_run(str_col, synthetic['gdf'], synthetic['centers'])
    step05.fn_create_raster_products(gdf_pnt, gdf_ar, str_col, gdf_dis, synthetic['res'], str(tmp_path), geom,
                                     synthetic['dem'])
    arr01, transform01, _, _ = _read(str(tmp_path / f'01_{str_col}_interp.tif'))
    prep = step05.fn_prepare_interpolation_inputs(gdf_pnt, gdf_ar, str_col, gdf_dis, synthetic['res'])
    xi, yi = np.meshgrid(prep.x_axis, prep.y_axis)
    zi = griddata((prep.x, prep.y), prep.z, (xi, yi), method='linear')
    assert np.array_equal(arr01, zi, equal_nan=True)
    assert transform01 == Affine.translation(prep.xmin, prep.ymin) * Affine.scale(prep.x_res, prep.y_res)


def test_raster_resolution_is_the_terrain_pixel_width(synthetic):
    with rasterio.open(synthetic['dem']) as src:
        assert synthetic['res'] == src.transform.a == 5.0


def test_infinite_and_all_dry_profiles_are_rejected(synthetic):
    gdf = synthetic['gdf'].copy()
    gdf.loc[3, 'wsel_max_1'] = np.inf
    with pytest.raises(rust_mapping.UnsupportedForRustEngine, match='infinite'):
        step05.fn_prepare_global_samples('wsel_max_1', gdf, synthetic['centers'], synthetic['res'])
    gdf['wsel_max_1'] = np.nan
    with pytest.raises(ValueError, match='no wet cells'):
        step05.fn_prepare_global_samples('wsel_max_1', gdf, synthetic['centers'], synthetic['res'])


# ============================================================ engine inputs
def _grids(synthetic, str_col):
    prep, geom = step05.fn_prepare_global_samples(str_col, synthetic['gdf'], synthetic['centers'],
                                                  synthetic['res'])
    with rasterio.open(synthetic['dem']) as src:
        bounds = tuple(src.bounds)
    return prep, geom, rust_mapping.fn_derive_step05_grids(prep, geom, synthetic['gdf'].crs, bounds,
                                                           'EPSG:3857', 3.0)


@pytest.mark.parametrize('str_col', LIST_COLS)
def test_derived_grids_match_legacy_products(synthetic, python_full, str_col):
    prep, geom, grids = _grids(synthetic, str_col)
    _, transform09, _, shape09 = _read(os.path.join(python_full, f'09_{str_col}_depth_nudge.tif'))
    assert grids.dest_transform == transform09
    assert (grids.dest_height, grids.dest_width) == shape09
    from rasterio.features import geometry_mask
    expected = geometry_mask([geom], (grids.rows, grids.cols), grids.raster_affine, invert=True)
    assert np.array_equal(grids.wet_mask, expected)


def test_profile_inputs_follow_the_engine_contract(synthetic, tmp_path):
    prep, geom, grids = _grids(synthetic, 'wsel_max_1')
    str_dir = str(tmp_path / 'inputs' / 'wsel_max_1')
    rust_mapping.fn_write_profile_inputs(str_dir, 'wsel_max_1', prep, grids)
    with open(os.path.join(str_dir, 'inputs.json')) as f:
        meta = json.load(f)
    assert meta['schema'] == 'ras2fim-step05-inputs/1'
    assert (meta['rows'], meta['cols']) == (len(prep.y_axis), len(prep.x_axis))
    assert meta['raster_affine'] == {'x0': prep.xmin, 'y0': prep.ymin, 'res_x': prep.x_res, 'res_y': prep.y_res}
    for str_key, d in meta['files'].items():
        arr = np.fromfile(os.path.join(str_dir, d['file']), dtype=d['dtype'])
        assert arr.size == int(np.prod(d['shape'])), str_key
    samples = np.fromfile(os.path.join(str_dir, 'samples.f64'), '<f8').reshape(-1, 3)
    assert np.array_equal(samples, np.column_stack([prep.x, prep.y, prep.z]))
    simplices = np.fromfile(os.path.join(str_dir, 'simplices.u32'), '<u4').reshape(-1, 3)
    interp = LinearNDInterpolator(np.column_stack([prep.x, prep.y]), prep.z)
    assert np.array_equal(simplices, interp.tri.simplices)
    assert np.array_equal(np.fromfile(os.path.join(str_dir, 'transform.f64'), '<f8').reshape(-1, 3, 2),
                          interp.tri.transform, equal_nan=True)
    assert np.array_equal(np.fromfile(os.path.join(str_dir, 'x_axis.f64'), '<f8'), prep.x_axis)
    mask = np.fromfile(os.path.join(str_dir, 'wet_mask.u8'), 'u1').reshape(meta['rows'], meta['cols'])
    assert np.array_equal(mask.astype(bool), grids.wet_mask)


def test_duplicate_samples_are_unsupported(synthetic, tmp_path):
    prep, geom, grids = _grids(synthetic, 'wsel_max_1')
    prep = prep._replace(x=np.append(prep.x, prep.x[0]), y=np.append(prep.y, prep.y[0]),
                         z=np.append(prep.z, prep.z[0]))
    with pytest.raises(rust_mapping.UnsupportedForRustEngine, match='coplanar'):
        rust_mapping.fn_write_profile_inputs(str(tmp_path / 'd'), 'wsel_max_1', prep, grids)


def test_grid_changing_crops_are_unsupported(synthetic):
    prep, geom, _ = _grids(synthetic, 'wsel_max_1')
    with rasterio.open(synthetic['dem']) as src:
        bounds = tuple(src.bounds)
    small = Polygon([(X0 + 200, Y0 + 200), (X0 + 300, Y0 + 200), (X0 + 300, Y0 + 300)])
    with pytest.raises(rust_mapping.UnsupportedForRustEngine, match='wet-union'):
        rust_mapping.fn_derive_step05_grids(prep, small, CRS_MODEL, bounds, 'EPSG:3857', 3.0)
    with pytest.raises(rust_mapping.UnsupportedForRustEngine, match='terrain'):
        rust_mapping.fn_derive_step05_grids(prep, geom, CRS_MODEL, (X0 + 100, Y0, X0 + 900, Y0 + 900),
                                            'EPSG:3857', 3.0)


# ============================================================ python AOI baseline
def _copy_products(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f[:3] in ('09_', '10_', '11_'):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))


def _aoi_cases(python_full):
    _, t, _, (h, w) = _read(os.path.join(python_full, '09_wsel_max_1_depth_nudge.tif'))
    x0, y1 = t.c, t.f
    x1, y0 = x0 + w * t.a, y1 + h * t.e
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    ring = [(x0 + 30, y0 + 30), (xm, y0 + 20), (x1 - 25, ym), (xm, y1 - 30), (x0 + 40, ym)]
    hole = [(xm - 20, ym - 20), (xm + 20, ym - 20), (xm + 20, ym + 20), (xm - 20, ym + 20)]
    return {
        'bbox': dict(bbox=[x0 + 31.0, y0 + 47.5, xm + 12.25, ym + 60.0]),
        'polygon_hole': dict(polygon=Polygon(ring, [hole]), aoi_crs='EPSG:3857'),
        'multipart': dict(polygon=box(x0, y0, x0 + 60, y0 + 60).union(box(x1 - 50, y1 - 50, x1, y1)),
                          aoi_crs='EPSG:3857'),
    }


@pytest.mark.parametrize('str_case', ['bbox', 'polygon_hole', 'multipart'])
def test_python_aoi_postclip_equals_crop_of_full_map(synthetic, python_full, tmp_path, str_case):
    aoi = mapping_aoi.fn_resolve_aoi(default_bbox_crs='EPSG:3857', **_aoi_cases(python_full)[str_case])
    out = str(tmp_path / 'aoi')
    _copy_products(python_full, out)
    for str_col in LIST_COLS:
        info = step05.fn_postclip_python_products(str_col, aoi, 'EPSG:3857', out)
        assert info['status'] in ('complete', 'no_flooded_pixels')
        assert info['polygon_mask'] == (str_case != 'bbox')
        col0, row0, w, h = info['window']
        for str_name in (f'09_{str_col}_depth_nudge.tif', f'10_{str_col}_wsel_nudge.tif'):
            full, t_full, _, _ = _read(os.path.join(python_full, str_name))
            clip, t_clip, crs, _ = _read(os.path.join(out, str_name))
            expected = full[row0:row0 + h, col0:col0 + w].copy()
            if info['polygon_mask']:
                win = mapping_aoi.fn_aoi_window(aoi, t_full, full.shape[1], full.shape[0], 'EPSG:3857')
                assert (~win.mask).any()
                expected[~win.mask] = np.nan
            assert np.array_equal(clip, expected, equal_nan=True)
            assert t_clip == t_full @ Affine.translation(col0, row0) and crs.to_epsg() == 3857


def test_bbox_and_equivalent_polygon_give_identical_products(synthetic, python_full, tmp_path):
    bbox = _aoi_cases(python_full)['bbox']['bbox']
    out_b, out_p = str(tmp_path / 'b'), str(tmp_path / 'p')
    _copy_products(python_full, out_b)
    _copy_products(python_full, out_p)
    aoi_b = mapping_aoi.fn_resolve_aoi(bbox=bbox, default_bbox_crs='EPSG:3857')
    aoi_p = mapping_aoi.fn_resolve_aoi(polygon=box(*bbox), aoi_crs='EPSG:3857')
    step05.fn_postclip_python_products('wsel_max_1', aoi_b, 'EPSG:3857', out_b)
    step05.fn_postclip_python_products('wsel_max_1', aoi_p, 'EPSG:3857', out_p)
    for str_name in ('09_wsel_max_1_depth_nudge.tif', '10_wsel_max_1_wsel_nudge.tif'):
        a, ta, _, _ = _read(os.path.join(out_b, str_name))
        b, tb, _, _ = _read(os.path.join(out_p, str_name))
        assert ta == tb and np.array_equal(a, b, equal_nan=True)


def test_empty_intersection_writes_no_products(synthetic, python_full, tmp_path):
    out = str(tmp_path / 'empty')
    _copy_products(python_full, out)
    aoi = mapping_aoi.fn_resolve_aoi(bbox=[0, 0, 10, 10], default_bbox_crs='EPSG:3857')
    info = step05.fn_postclip_python_products('wsel_max_1', aoi, 'EPSG:3857', out)
    assert info['status'] == 'empty_intersection'
    assert not any(f.startswith(('09_wsel_max_1_', '10_wsel_max_1_', '11_wsel_max_1_')) for f in os.listdir(out))


# ============================================================ adapter I/O (emulated engine)
def _terrain_info(str_dem, b_epsg=True):
    with rasterio.open(str_dem) as src:
        return {'width': src.width, 'height': src.height, 'geo_transform': list(src.transform.to_gdal()),
                'epsg': src.crs.to_epsg() if b_epsg else None, 'crs_wkt_fnv1a64': '0123456789abcdef'}


def _emulated_engine(python_full, dict_seen, str_fault=None):
    """
    Stand-in for `ras2fim-batch`, used only to test the Python adapter. It
    validates the request/inputs layout and answers with crops of the python
    engine's full-map products; str_fault injects one failure into the second
    profile. It is not an engine qualification.
    """
    def run(list_args, str_what):
        if list_args[1] == 'terrain-info':
            return subprocess.CompletedProcess(list_args, 0, json.dumps(_terrain_info(list_args[2])), '')
        assert list_args[1] == 'step05' and not os.path.exists(list_args[3])
        str_req_dir = os.path.dirname(list_args[2])
        with open(list_args[2]) as f:
            req = json.load(f)
        dict_seen.setdefault('requests', []).append(req)
        assert req['schema'] == 'ras2fim-step05-request/1' and req['stages'] == []
        assert req['source_crs'] == {'kind': 'terrain'}
        assert req['nudge'] == {'dst_epsg': 3857, 'error_threshold': 0.125,
                                'memory_limit_bytes': 64 * 1024 * 1024, 'method': 'lookup'}
        int_exit = 0
        list_report_profiles = []
        for p in req['profiles']:
            with open(os.path.join(str_req_dir, p['inputs_dir'], 'inputs.json')) as f:
                assert json.load(f)['column'] == p['id']
            depth, t, _, (h, w) = _read(os.path.join(python_full, f'09_{p["id"]}_depth_nudge.tif'))
            wse, _, _, _ = _read(os.path.join(python_full, f'10_{p["id"]}_wsel_nudge.tif'))
            assert p['dest_grid'] == {'geotransform': list(t.to_gdal()), 'width': w, 'height': h}
            b_fault = str_fault is not None and p['id'] == 'wsel_max_2'
            b_failed = b_fault and str_fault in ('unit_failed', 'missing_unit', 'report_says_failed')
            list_report_profiles.append({'id': p['id'], 'status': 'failed' if b_failed else 'complete'})
            for aoi in p['aois']:
                col0, row0, ww, hh = aoi.get('window', [0, 0, w, h])
                d = depth[row0:row0 + hh, col0:col0 + ww].copy()
                e = wse[row0:row0 + hh, col0:col0 + ww].copy()
                if 'mask_u8' in aoi:
                    str_mask = os.path.join(str_req_dir, aoi['mask_u8'])
                    assert os.path.getsize(str_mask) == ww * hh
                    m = np.fromfile(str_mask, 'u1').reshape(hh, ww) == 0
                    d[m] = np.nan
                    e[m] = np.nan
                str_unit = os.path.join(list_args[3], 'units', p['id'], aoi['id'])
                if b_fault and str_fault == 'missing_unit':
                    int_exit = 2
                    continue
                os.makedirs(str_unit)
                unit = {'schema': 'ras2fim-step05-unit/1', 'profile': p['id'], 'aoi': aoi['id'],
                        'status': 'complete', 'window': [col0, row0, ww, hh],
                        'window_geotransform': list((t @ Affine.translation(col0, row0)).to_gdal()),
                        'crs_epsg': 3857, 'dtype': '<f4', 'nodata': 'nan',
                        'polygon_mask_applied': 'mask_u8' in aoi,
                        'finite_counts': {'depth.f32': int(np.isfinite(d).sum()),
                                          'wse.f32': int(np.isfinite(e).sum())}}
                if b_fault and str_fault == 'unit_failed':
                    unit.update(status='failed', error='synthetic profile failure')
                    int_exit = 2
                if b_fault and str_fault == 'bad_count':
                    unit['finite_counts']['depth.f32'] += 1
                if b_fault and str_fault == 'unit_no_schema':
                    del unit['schema']
                if b_fault and str_fault == 'unit_no_id':
                    del unit['profile']
                if b_fault and str_fault == 'unit_wrong_aoi':
                    unit['aoi'] = 'other'
                if b_fault and str_fault == 'nan_geotransform':
                    unit['window_geotransform'][0] = float('nan')
                if b_fault and str_fault == 'float_window':
                    unit['window'] = [float(v) for v in unit['window']]
                if b_fault and str_fault == 'report_says_failed':
                    int_exit = 2
                if b_fault and str_fault == 'mask_outside' and 'mask_u8' in aoi:
                    d = depth[row0:row0 + hh, col0:col0 + ww].copy()
                    unit['finite_counts']['depth.f32'] = int(np.isfinite(d).sum())
                d.astype('<f4').tofile(os.path.join(str_unit, 'depth.f32'))
                e.astype('<f4').tofile(os.path.join(str_unit, 'wse.f32'))
                if b_fault and str_fault == 'missing_f32':
                    os.remove(os.path.join(str_unit, 'wse.f32'))
                if b_fault and str_fault == 'short_f32':
                    e[:-1].astype('<f4').tofile(os.path.join(str_unit, 'wse.f32'))
                with open(os.path.join(str_unit, 'unit.json'), 'w') as f:
                    if b_fault and str_fault == 'malformed_unit':
                        f.write('{"status": "complete", ')
                    else:
                        json.dump(unit, f)
        if str_fault == 'exit2_all_complete':
            int_exit = 2
        dict_report = {'schema': 'ras2fim-step05-report/1', 'status': 'complete' if int_exit == 0 else 'failed',
                       'profiles': list_report_profiles, 'emulated': True}
        if str_fault == 'exit2_all_complete':
            dict_report['status'] = 'failed'
        if str_fault == 'report_status_mismatch':
            dict_report['status'] = 'failed'
        if str_fault == 'report_wrong_profiles':
            dict_report['profiles'] = dict_report['profiles'][:1]
        if str_fault == 'report_wrong_schema':
            dict_report['schema'] = 'other/1'
        if str_fault != 'report_missing':
            with open(os.path.join(list_args[3], 'report.json'), 'w') as f:
                if str_fault == 'report_malformed':
                    f.write('{"schema": ')
                else:
                    json.dump(dict_report, f)
        return subprocess.CompletedProcess(list_args, int_exit, '{}', 'emulated stderr')
    return run


@pytest.mark.parametrize('str_case', [None, 'bbox', 'polygon_hole'])
def test_rust_adapter_writes_upstream_products(synthetic, python_full, tmp_path, monkeypatch, str_case):
    str_bin = _fake_binary(tmp_path)
    dict_seen = {}
    monkeypatch.setattr(rust_mapping, 'fn_run_binary', _emulated_engine(python_full, dict_seen))
    dict_kw = _aoi_cases(python_full)[str_case] if str_case else {}
    out = str(tmp_path / 'rust')
    result = step05.fn_generate_depth_wsel_rasters(
        synthetic['config'], synthetic['hdf'], synthetic['dem'], synthetic['gpkg'], out, False,
        mapping_engine='rust', rust_binary=str_bin, mapping_threads=3, rust_profiles_per_request=1,
        **dict_kw)
    assert set(result) == set(LIST_COLS)
    assert all(v['status'] in ('complete', 'no_flooded_pixels') for v in result.values())
    # one bounded request per profile; threads forwarded; work directory removed
    assert len(dict_seen['requests']) == 2
    assert dict_seen['requests'][0]['budgets']['threads'] == 3
    assert not any(f.startswith('.rust_step05_') for f in os.listdir(out))
    with open(os.path.join(out, '05_mapping_result.json')) as f:
        manifest = json.load(f)
        assert manifest['engine'] == 'rust' and manifest['status'] == 'complete'
    with open(os.path.join(out, 'rust_mapping_report.json')) as f:
        report = json.load(f)
    assert len(report['batches']) == 2 and report['batches'][0]['report']['emulated']
    assert report['status'] == 'complete'

    ref = str(tmp_path / 'ref')
    _copy_products(python_full, ref)
    aoi = mapping_aoi.fn_resolve_aoi(default_bbox_crs='EPSG:3857', **dict_kw) if str_case else None
    for str_col in LIST_COLS:
        if aoi is not None:
            step05.fn_postclip_python_products(str_col, aoi, 'EPSG:3857', ref)
        for str_name in (f'09_{str_col}_depth_nudge.tif', f'10_{str_col}_wsel_nudge.tif'):
            a, ta, ca, _ = _read(os.path.join(out, str_name))
            b, tb, cb, _ = _read(os.path.join(ref, str_name))
            assert ta == tb and ca == cb and np.array_equal(a, b, equal_nan=True)
            with rasterio.open(os.path.join(out, str_name)) as src:
                assert src.dtypes[0] == 'float32' and np.isnan(src.nodata)
        if result[str_col]['status'] == 'complete':
            g11 = gpd.read_file(os.path.join(out, f'11_{str_col}_floodplain_ar.gpkg'))
            assert g11.crs.to_epsg() == 3857 and len(g11) == 1


@pytest.mark.parametrize('str_fault, dict_kw_case', [
    ('unit_failed', None), ('missing_unit', None), ('malformed_unit', None), ('missing_f32', None),
    ('short_f32', None), ('bad_count', None), ('mask_outside', 'polygon_hole'), ('unit_no_schema', None),
    ('unit_no_id', None), ('unit_wrong_aoi', 'bbox'), ('nan_geotransform', None), ('float_window', None),
    ('report_says_failed', None),
])
def test_rust_profile_failures_raise_after_writing_good_profiles(synthetic, python_full, tmp_path, monkeypatch,
                                                                 str_fault, dict_kw_case):
    str_bin = _fake_binary(tmp_path)
    monkeypatch.setattr(rust_mapping, 'fn_run_binary', _emulated_engine(python_full, {}, str_fault))
    dict_kw = _aoi_cases(python_full)[dict_kw_case] if dict_kw_case else {}
    out = str(tmp_path / 'rust')
    with pytest.raises(rust_mapping.RustMappingError, match=r"failed for 1 profile.*wsel_max_2"):
        step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'], synthetic['dem'],
                                              synthetic['gpkg'], out, False, mapping_engine='rust',
                                              rust_binary=str_bin, **dict_kw)
    with open(os.path.join(out, '05_mapping_result.json')) as f:
        columns = json.load(f)['columns']
    assert columns['wsel_max_1']['status'] in ('complete', 'no_flooded_pixels')
    assert columns['wsel_max_2']['status'] == 'failed' and columns['wsel_max_2']['error']
    listing = os.listdir(out)
    assert '09_wsel_max_1_depth_nudge.tif' in listing
    assert not any(f.startswith(('09_wsel_max_2', '10_wsel_max_2', '11_wsel_max_2')) for f in listing)


@pytest.mark.parametrize('str_fault', ['report_missing', 'report_malformed', 'report_wrong_schema',
                                       'report_status_mismatch', 'report_wrong_profiles'])
def test_rust_invalid_report_fails_the_call(synthetic, python_full, tmp_path, monkeypatch, str_fault):
    monkeypatch.setattr(rust_mapping, 'fn_run_binary', _emulated_engine(python_full, {}, str_fault))
    out = str(tmp_path / 'o')
    with pytest.raises(rust_mapping.RustMappingError, match='report'):
        step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'], synthetic['dem'],
                                              synthetic['gpkg'], out, False, mapping_engine='rust',
                                              rust_binary=_fake_binary(tmp_path))
    assert not any(f.startswith(('09_', '10_', '11_')) for f in os.listdir(out))


def test_rust_exit2_without_a_failed_unit_raises(synthetic, python_full, tmp_path, monkeypatch):
    monkeypatch.setattr(rust_mapping, 'fn_run_binary', _emulated_engine(python_full, {}, 'exit2_all_complete'))
    with pytest.raises(rust_mapping.RustMappingError, match='exited 2'):
        step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'], synthetic['dem'],
                                              synthetic['gpkg'], str(tmp_path / 'o'), False,
                                              mapping_engine='rust', rust_binary=_fake_binary(tmp_path))


def test_rust_invalid_request_raises(synthetic, tmp_path, monkeypatch):
    def run(list_args, str_what):
        if list_args[1] == 'terrain-info':
            return subprocess.CompletedProcess(list_args, 0, json.dumps(_terrain_info(list_args[2])), '')
        return subprocess.CompletedProcess(list_args, 1, '', 'invalid request: bad field')
    monkeypatch.setattr(rust_mapping, 'fn_run_binary', run)
    with pytest.raises(rust_mapping.RustMappingError, match='bad field'):
        step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'], synthetic['dem'],
                                              synthetic['gpkg'], str(tmp_path / 'o'), False,
                                              mapping_engine='rust', rust_binary=_fake_binary(tmp_path))


def test_terrain_identity_checks(synthetic, monkeypatch):
    dict_info = {}

    def run(list_args, str_what):
        return subprocess.CompletedProcess(list_args, 0, json.dumps(dict_info), '')
    monkeypatch.setattr(rust_mapping, 'fn_run_binary', run)

    dict_info.update(width=1, height=1, geo_transform=[0, 1, 0, 0, 0, -1], epsg=2277)
    with pytest.raises(rust_mapping.RustMappingError, match='size'):
        rust_mapping.fn_terrain_identity('bin', synthetic['dem'], CRS_MODEL)
    dict_info.update(_terrain_info(synthetic['dem']))
    with pytest.raises(rust_mapping.UnsupportedForRustEngine, match='terrain CRS'):
        rust_mapping.fn_terrain_identity('bin', synthetic['dem'], 'EPSG:3857')
    assert rust_mapping.fn_terrain_identity('bin', synthetic['dem'], CRS_MODEL)['crs_identity'] == {'kind': 'epsg'}
    # no engine EPSG: an explicit, matching WKT hash is required
    dict_info.update(_terrain_info(synthetic['dem'], b_epsg=False))
    with pytest.raises(rust_mapping.UnsupportedForRustEngine, match='str_rust_terrain_crs_fnv1a64'):
        rust_mapping.fn_terrain_identity('bin', synthetic['dem'], CRS_MODEL)
    with pytest.raises(rust_mapping.RustMappingError, match='does not match'):
        rust_mapping.fn_terrain_identity('bin', synthetic['dem'], CRS_MODEL, 'ffffffffffffffff')
    assert rust_mapping.fn_terrain_identity('bin', synthetic['dem'], CRS_MODEL, '0123456789abcdef')[
        'crs_identity'] == {'kind': 'caller_verified_wkt', 'fnv1a64': '0123456789abcdef'}


# ============================================================ real external fixtures
def _env_dir(str_var):
    str_dir = os.environ.get(str_var)
    if not str_dir or not os.path.isdir(str_dir):
        pytest.skip(f'{str_var} not set (real external fixture)')
    return str_dir


@pytest.fixture(scope='module')
def public_sample():
    str_dir = _env_dir('RAS2FIM_PUBLIC_SAMPLE')
    gdf = gpd.read_file(os.path.join(str_dir, 'results.gpkg'), layer='02_cells_wsel_ar')
    centers = step05.fn_gdf_cell_centerpoints(os.path.join(str_dir, 'sample.p01.hdf'),
                                              sorted(gdf['cell_idx'].unique()), gdf)
    str_dem = os.path.join(str_dir, 'terrain.tif')
    return dict(dir=str_dir, gdf=gdf, centers=centers, dem=str_dem, res=step05.fn_get_raster_resolution(str_dem))


def test_real_prep_matches_upstream_oracle(public_sample):
    str_flows = _env_dir('RAS2FIM_STEP05_ORACLE_FLOWS')
    from scipy.spatial import Delaunay
    with rasterio.open(public_sample['dem']) as src:
        bounds = tuple(src.bounds)
    list_cols = [c for c in public_sample['gdf'].columns if c.startswith('wsel')]
    assert len(list_cols) == 19
    for str_col in list_cols:
        oracle = np.load(os.path.join(str_flows, str_col, 'prepared.npz'))
        with open(os.path.join(str_flows, str_col, 'capture.json')) as f:
            capture = json.load(f)
        # rasterio terrain resolution equals the oracle's (GDAL) pixel width bitwise
        assert public_sample['res'] == capture['pixel_width']
        prep, geom = step05.fn_prepare_global_samples(str_col, public_sample['gdf'], public_sample['centers'],
                                                      public_sample['res'])
        samples = np.column_stack([prep.x, prep.y, prep.z])
        assert np.array_equal(samples, oracle['samples']), str_col
        assert np.array_equal(prep.x_axis, oracle['grid_x']) and np.array_equal(prep.y_axis, oracle['grid_y'])
        assert np.array_equal(Delaunay(samples[:, :2]).simplices, oracle['simplices']), str_col
        grids = rust_mapping.fn_derive_step05_grids(prep, geom, public_sample['gdf'].crs, bounds, 'EPSG:3857', 3.0)
        _, t09, _, shape09 = _read(os.path.join(str_flows, str_col, f'09_{str_col}_depth_nudge.tif'))
        assert grids.dest_transform == t09 and (grids.dest_height, grids.dest_width) == shape09


def test_real_rust_engine_matches_python_engine(public_sample, tmp_path):
    str_bin = os.environ.get('RAS2FIM_RUST_BINARY')
    if not str_bin or not os.path.isfile(str_bin):
        pytest.skip('RAS2FIM_RUST_BINARY not set (built ras2fim-batch with step05)')
    str_config = _write_config(str(tmp_path / 'c.ini'))
    str_dir = public_sample['dir']
    args = (str_config, os.path.join(str_dir, 'sample.p01.hdf'), public_sample['dem'],
            os.path.join(str_dir, 'results.gpkg'))
    out_py, out_rs = str(tmp_path / 'py'), str(tmp_path / 'rs')
    os.makedirs(out_py)
    step05.fn_generate_depth_wsel_rasters(*args, out_py, False, mapping_threads=4)
    step05.fn_generate_depth_wsel_rasters(*args, out_rs, False, mapping_engine='rust', rust_binary=str_bin,
                                          mapping_threads=4)
    list_mismatch = []
    for str_col in [c for c in public_sample['gdf'].columns if c.startswith('wsel')]:
        for str_name in (f'09_{str_col}_depth_nudge.tif', f'10_{str_col}_wsel_nudge.tif'):
            a, ta, _, _ = _read(os.path.join(out_py, str_name))
            b, tb, _, _ = _read(os.path.join(out_rs, str_name))
            if ta != tb or a.shape != b.shape:
                list_mismatch.append((str_name, 'grid'))
            elif not np.array_equal(a, b, equal_nan=True):
                list_mismatch.append((str_name, int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())))
    assert list_mismatch == []


@pytest.mark.parametrize('str_case', ['bbox', 'polygon_hole'])
def test_real_rust_aoi_matches_python_postclip(public_sample, tmp_path, str_case):
    str_bin = os.environ.get('RAS2FIM_RUST_BINARY')
    if not str_bin or not os.path.isfile(str_bin):
        pytest.skip('RAS2FIM_RUST_BINARY not set (built ras2fim-batch with step05)')
    xm, ym = -10842000.0, 3553750.0
    ring = [(xm - 1500, ym - 900), (xm + 1400, ym - 1100), (xm + 1600, ym + 800), (xm - 1200, ym + 1000)]
    hole = [(xm - 300, ym - 300), (xm + 300, ym - 300), (xm + 300, ym + 300), (xm - 300, ym + 300)]
    dict_kw = {'bbox': dict(bbox=[-10843000.0, 3553000.0, -10841000.0, 3554500.0]),
               'polygon_hole': dict(polygon=Polygon(ring, [hole]), aoi_crs='EPSG:3857')}[str_case]
    str_config = _write_config(str(tmp_path / 'c.ini'))
    str_dir = public_sample['dir']
    args = (str_config, os.path.join(str_dir, 'sample.p01.hdf'), public_sample['dem'],
            os.path.join(str_dir, 'results.gpkg'))
    out_py, out_rs = str(tmp_path / 'py'), str(tmp_path / 'rs')
    os.makedirs(out_py)
    res_py = step05.fn_generate_depth_wsel_rasters(*args, out_py, False, mapping_threads=4, **dict_kw)
    res_rs = step05.fn_generate_depth_wsel_rasters(*args, out_rs, False, mapping_engine='rust',
                                                   rust_binary=str_bin, mapping_threads=4, **dict_kw)
    assert {c: (v['status'], v.get('window')) for c, v in res_py.items()} == \
        {c: (v['status'], v.get('window')) for c, v in res_rs.items()}
    list_mismatch = []
    for str_col, info in res_py.items():
        if info['status'] == 'empty_intersection':
            continue
        for str_name in (f'09_{str_col}_depth_nudge.tif', f'10_{str_col}_wsel_nudge.tif'):
            a, ta, _, _ = _read(os.path.join(out_py, str_name))
            b, tb, _, _ = _read(os.path.join(out_rs, str_name))
            if ta != tb or not np.array_equal(a, b, equal_nan=True):
                list_mismatch.append(str_name)
    assert list_mismatch == []
