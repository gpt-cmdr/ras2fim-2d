# Step 05 mapping-engine options and the adapter for the optional Rust
# engine ("ras2fim-batch step05").
#
# The Python engine (the original Step 05 code) is the default. The Rust
# engine is opt-in: it is selected with mapping_engine='rust' (keyword),
# [05_raster_engine] str_mapping_engine = rust (config) or
# --mapping-engine rust (CLI). A missing Rust binary is an error; there is
# no silent fallback to the Python engine. Python never builds the binary.
#
# The Rust engine consumes the same global sample preparation as the Python
# engine (fn_prepare_interpolation_inputs in Step 05), plus SciPy's Delaunay
# topology, the wet-union mask and the final output grid, all derived here
# from the unmodified upstream recipe. See docs/rust-mapping.md.
#
# Created for the optional Rust mapping engine - 2026.10.23
# ************************************************************
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field

import numpy as np
# ************************************************************

ENGINE_PYTHON = 'python'
ENGINE_RUST = 'rust'
LIST_ENGINES = (ENGINE_PYTHON, ENGINE_RUST)

STR_RUST_BINARY_NAME = 'ras2fim-batch'
STR_REQUEST_SCHEMA = 'ras2fim-step05-request/1'
STR_INPUTS_SCHEMA = 'ras2fim-step05-inputs/1'
STR_UNIT_SCHEMA = 'ras2fim-step05-unit/1'
STR_REPORT_SCHEMA = 'ras2fim-step05-report/1'

# Upstream reprojects with rasterio.warp.reproject defaults: GDAL's
# approximate transformer (0.125 px) and the default 64 MiB warp memory,
# which sets the chunking (and therefore the approximation).
FLT_NUDGE_ERROR_THRESHOLD = 0.125
INT_NUDGE_MEMORY_LIMIT_BYTES = 64 * 1024 * 1024

# Legacy Step 05 pool size: min(cpu_count - 1, 16)
INT_LEGACY_MAX_PROCESSORS = 16

STR_CONFIG_SECTION = '05_raster_engine'
STR_RESULT_MANIFEST = '05_mapping_result.json'
STR_ENGINE_REPORT = 'rust_mapping_report.json'

_RE_SAFE_ID = re.compile(r'^[A-Za-z0-9_.-]+$')


class RustMappingError(RuntimeError):
    """The Rust mapping engine failed or was misconfigured."""


class RustBinaryNotFoundError(RustMappingError, FileNotFoundError):
    """The ras2fim-batch binary could not be found."""


class UnsupportedForRustEngine(RustMappingError):
    """Input that the current Rust engine contract does not support."""


# ------------------------------
@dataclass(frozen=True)
class Step05MappingOptions:
    """Resolved Step 05 mapping options (keyword > CLI > config > default)."""
    engine: str = ENGINE_PYTHON
    rust_binary: str = None
    threads: int = None                 # None = legacy default
    rust_max_cache_mib: int = 64
    rust_max_profile_mib: int = 1024
    rust_profiles_per_request: int = 4
    rust_max_inflight_profiles: int = 1
    rust_nudge_method: str = 'lookup'
    rust_terrain_crs_fnv1a64: str = None
    rust_keep_work: bool = False
    bbox: object = None
    polygon: object = None
    polygon_layer: str = None
    aoi_crs: object = None

    @property
    def has_aoi(self):
        return self.bbox is not None or self.polygon is not None

    @property
    def has_aoi_options(self):
        return self.has_aoi or self.polygon_layer is not None or self.aoi_crs is not None

    @property
    def is_legacy_default(self):
        return self.engine == ENGINE_PYTHON and not self.has_aoi_options and self.threads is None

    def resolved_threads(self):
        if self.threads is not None:
            return self.threads
        # legacy pool size, but never 0 on a single-CPU host
        return max(1, min(mp.cpu_count() - 1, INT_LEGACY_MAX_PROCESSORS))
# ------------------------------


# Config keys in [05_raster_engine] -> (option name, parser)
def _fn_opt_str(value):
    value = value.strip()
    return value or None


def _fn_opt_int(value):
    value = value.strip()
    return int(value) if value else None


def _fn_opt_bool(value):
    value = value.strip().lower()
    if not value:
        return None
    if value in {'true', 't', '1', 'yes'}:
        return True
    if value in {'false', 'f', '0', 'no'}:
        return False
    raise ValueError(f'boolean expected, got {value!r}')


DICT_CONFIG_KEYS = {
    'str_mapping_engine': ('engine', _fn_opt_str),
    'str_rust_binary': ('rust_binary', _fn_opt_str),
    'int_mapping_threads': ('threads', _fn_opt_int),
    'int_rust_max_cache_mib': ('rust_max_cache_mib', _fn_opt_int),
    'int_rust_max_profile_mib': ('rust_max_profile_mib', _fn_opt_int),
    'int_rust_profiles_per_request': ('rust_profiles_per_request', _fn_opt_int),
    'int_rust_max_inflight_profiles': ('rust_max_inflight_profiles', _fn_opt_int),
    'str_rust_nudge_method': ('rust_nudge_method', _fn_opt_str),
    'str_rust_terrain_crs_fnv1a64': ('rust_terrain_crs_fnv1a64', _fn_opt_str),
    'b_rust_keep_work': ('rust_keep_work', _fn_opt_bool),
    'str_aoi_bbox': ('bbox', _fn_opt_str),
    'str_aoi_polygon': ('polygon', _fn_opt_str),
    'str_aoi_polygon_layer': ('polygon_layer', _fn_opt_str),
    'str_aoi_crs': ('aoi_crs', _fn_opt_str),
}


# ..............................
def fn_options_from_config(config):
    """Read [05_raster_engine] from a ConfigParser; returns only set keys."""
    dict_out = {}
    if config is None or STR_CONFIG_SECTION not in config:
        return dict_out
    section = config[STR_CONFIG_SECTION]
    for str_key in section:
        if str_key not in DICT_CONFIG_KEYS:
            raise ValueError(f'[{STR_CONFIG_SECTION}] unknown key {str_key!r}; '
                             f'known keys: {sorted(DICT_CONFIG_KEYS)}')
        str_opt, fn_parse = DICT_CONFIG_KEYS[str_key]
        try:
            value = fn_parse(section.get(str_key, ''))
        except ValueError as e:
            raise ValueError(f'[{STR_CONFIG_SECTION}] {str_key}: {e}') from e
        if value is not None:
            dict_out[str_opt] = value
    return dict_out
# ..............................


# ------------------------------
def fn_resolve_mapping_options(config=None, **overrides):
    """
    Combine defaults, [05_raster_engine] config values and keyword/CLI
    overrides (None means "not given"), then validate.
    """
    dict_values = fn_options_from_config(config)
    for str_key, value in overrides.items():
        if str_key not in Step05MappingOptions.__dataclass_fields__:
            raise TypeError(f'unknown Step 05 mapping option {str_key!r}')
        if value is not None:
            dict_values[str_key] = value
    # an explicit AOI from keyword/CLI replaces the other AOI kind from config
    b_bbox, b_polygon = overrides.get('bbox') is not None, overrides.get('polygon') is not None
    if b_bbox and b_polygon:
        raise ValueError('bbox and polygon AOIs are mutually exclusive')
    if b_bbox:
        dict_values.pop('polygon', None)
        if overrides.get('polygon_layer') is None:
            dict_values.pop('polygon_layer', None)
    if b_polygon:
        dict_values.pop('bbox', None)
    opts = Step05MappingOptions(**dict_values)
    fn_validate_options(opts)
    return opts
# ------------------------------


# ..............................
def fn_validate_options(opts):
    if opts.engine not in LIST_ENGINES:
        raise ValueError(f'mapping engine must be one of {LIST_ENGINES}, got {opts.engine!r}')
    if opts.threads is not None and (not isinstance(opts.threads, int) or opts.threads < 1):
        raise ValueError(f'mapping threads must be a positive integer, got {opts.threads!r}')
    for str_name in ('rust_max_cache_mib', 'rust_max_profile_mib',
                     'rust_profiles_per_request', 'rust_max_inflight_profiles'):
        value = getattr(opts, str_name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{str_name} must be a positive integer, got {value!r}')
    if opts.rust_nudge_method not in ('lookup', 'full_warp'):
        raise ValueError(f"rust_nudge_method must be 'lookup' or 'full_warp', got {opts.rust_nudge_method!r}")
    if opts.engine == ENGINE_PYTHON and opts.rust_binary is not None:
        raise ValueError('rust_binary was given but the mapping engine is python; '
                         'select mapping_engine=rust to use it')
    if opts.bbox is not None and opts.polygon is not None:
        raise ValueError('bbox and polygon AOIs are mutually exclusive')
# ..............................


# ------------------------------
def fn_add_mapping_arguments(parser):
    """Add the Step 05 mapping-engine and AOI options to an argparse parser."""
    grp = parser.add_argument_group('Step 05 mapping engine and area of interest')
    grp.add_argument('--mapping-engine', dest='mapping_engine', choices=LIST_ENGINES, default=None,
                     help='OPTIONAL: Step 05 mapping engine (default: config [05_raster_engine] '
                          'or python)')
    grp.add_argument('--rust-binary', dest='rust_binary', default=None, metavar='FILE',
                     help=f'OPTIONAL: path to the {STR_RUST_BINARY_NAME} binary (default: PATH lookup)')
    grp.add_argument('--mapping-threads', dest='mapping_threads', type=int, default=None, metavar='N',
                     help='OPTIONAL: Step 05 worker processes (python) or threads (rust); '
                          'default min(cpu_count - 1, 16)')
    grp.add_argument('--rust-cache-mib', dest='rust_max_cache_mib', type=int, default=None, metavar='MIB',
                     help='OPTIONAL: Rust decoded-terrain cache budget in MiB (default 64)')
    grp.add_argument('--rust-profile-mib', dest='rust_max_profile_mib', type=int, default=None,
                     metavar='MIB', help='OPTIONAL: Rust per-profile working-set budget in MiB (default 1024)')
    grp.add_argument('--rust-profiles-per-request', dest='rust_profiles_per_request', type=int,
                     default=None, metavar='N',
                     help='OPTIONAL: profiles prepared per Rust engine call (default 4)')
    grp.add_argument('--rust-max-inflight-profiles', dest='rust_max_inflight_profiles', type=int,
                     default=None, metavar='N',
                     help='OPTIONAL: profiles the Rust engine evaluates concurrently (default 1)')
    aoi = grp.add_mutually_exclusive_group()
    aoi.add_argument('--bbox', dest='bbox', type=float, nargs=4, default=None,
                     metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                     help='OPTIONAL: bounding-box AOI (default CRS: [06_output_products] str_output_crs)')
    aoi.add_argument('--polygon', dest='polygon', default=None, metavar='FILE',
                     help='OPTIONAL: polygon AOI file (GeoJSON or GeoPackage)')
    grp.add_argument('--polygon-layer', dest='polygon_layer', default=None, metavar='LAYER',
                     help='OPTIONAL: layer name in the polygon AOI file')
    grp.add_argument('--aoi-crs', dest='aoi_crs', default=None, metavar='CRS',
                     help='OPTIONAL: CRS of the bbox/polygon AOI (e.g. EPSG:2277)')
    return parser
# ------------------------------


# ..............................
def fn_mapping_kwargs_from_args(args):
    """Map parsed CLI values (vars(args)) to fn_generate_depth_wsel_rasters keywords."""
    dict_map = {
        'mapping_engine': 'mapping_engine',
        'rust_binary': 'rust_binary',
        'mapping_threads': 'mapping_threads',
        'rust_max_cache_mib': 'rust_max_cache_mib',
        'rust_max_profile_mib': 'rust_max_profile_mib',
        'rust_profiles_per_request': 'rust_profiles_per_request',
        'rust_max_inflight_profiles': 'rust_max_inflight_profiles',
        'bbox': 'bbox',
        'polygon': 'polygon',
        'polygon_layer': 'polygon_layer',
        'aoi_crs': 'aoi_crs',
    }
    return {dst: args[src] for src, dst in dict_map.items() if args.get(src) is not None}
# ..............................


# ------------------------------
def fn_resolve_rust_binary(str_path=None):
    """
    Return the absolute path of the Rust engine binary: the explicit path, or
    ras2fim-batch on PATH. Raises RustBinaryNotFoundError; never builds it.
    """
    if str_path:
        str_path = os.fspath(str_path)
        if not os.path.isfile(str_path):
            raise RustBinaryNotFoundError(f'Rust mapping binary not found: {str_path}')
        if not os.access(str_path, os.X_OK):
            raise RustBinaryNotFoundError(f'Rust mapping binary is not executable: {str_path}')
        return os.path.abspath(str_path)
    str_found = shutil.which(STR_RUST_BINARY_NAME)
    if str_found is None:
        raise RustBinaryNotFoundError(
            f'mapping engine "rust" was selected but {STR_RUST_BINARY_NAME} is not on PATH; '
            'pass --rust-binary / rust_binary or [05_raster_engine] str_rust_binary. '
            'The binary is built separately (see docs/rust-mapping.md); '
            'there is no automatic fallback to the python engine.')
    return os.path.abspath(str_found)
# ------------------------------


# ..............................
def fn_run_binary(list_args, str_what):
    """Run the engine without a shell; return CompletedProcess."""
    try:
        return subprocess.run(list_args, capture_output=True, text=True, check=False)
    except OSError as e:
        raise RustMappingError(f'{str_what}: cannot execute {list_args[0]}: {e}') from e
# ..............................


# ------------------------------
def fn_terrain_identity(str_binary, str_dem_path, crs_expected, str_override_fnv=None):
    """
    Build the request "terrain" block. The engine's own terrain-info is
    compared with rasterio's view of the same file (size, geotransform, CRS).
    """
    import rasterio
    from pyproj import CRS

    proc = fn_run_binary([str_binary, 'terrain-info', os.path.abspath(str_dem_path)], 'terrain-info')
    if proc.returncode != 0:
        raise RustMappingError(f'terrain-info failed ({proc.returncode}): {proc.stderr.strip()[-2000:]}')
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RustMappingError(f'terrain-info returned invalid JSON: {e}') from e

    with rasterio.open(str_dem_path) as src:
        tup_gt = tuple(src.transform.to_gdal())
        int_w, int_h = src.width, src.height
        crs_rio = CRS.from_user_input(src.crs) if src.crs else None
    if crs_rio is None:
        raise UnsupportedForRustEngine(f'terrain {str_dem_path} has no CRS')
    if crs_expected is not None and CRS.from_user_input(crs_expected) != crs_rio:
        raise UnsupportedForRustEngine('the Rust engine requires the hydraulic results CRS to equal the '
                                       f'terrain CRS ({crs_rio.to_string()})')
    if (info.get('width'), info.get('height')) != (int_w, int_h):
        raise RustMappingError(f'terrain size differs between engine {info.get("width")}x'
                               f'{info.get("height")} and rasterio {int_w}x{int_h}')
    list_gt = info.get('geo_transform')
    if list_gt is None or len(list_gt) != 6 or any(float(a) != float(b) for a, b in zip(list_gt, tup_gt)):
        raise RustMappingError(f'terrain geotransform differs between engine {list_gt} and rasterio {tup_gt}')

    int_epsg = crs_rio.to_epsg()
    if int_epsg is None:
        raise UnsupportedForRustEngine(f'terrain CRS has no EPSG code: {crs_rio.to_string()}')
    str_fnv = info.get('crs_wkt_fnv1a64')
    if info.get('epsg') is not None:
        if int(info['epsg']) != int_epsg:
            raise RustMappingError(f'terrain EPSG differs: engine {info["epsg"]}, rasterio {int_epsg}')
        dict_identity = {'kind': 'epsg'}
    elif str_override_fnv:
        # The user has verified that the engine's WKT is the terrain CRS.
        if str_fnv != str_override_fnv:
            raise RustMappingError(f'str_rust_terrain_crs_fnv1a64={str_override_fnv} does not match the '
                                   f'engine terrain WKT hash {str_fnv}')
        dict_identity = {'kind': 'caller_verified_wkt', 'fnv1a64': str_fnv}
    else:
        str_wkt = fn_find_wkt(info.get('crs'))
        if str_wkt is None or str_fnv is None:
            raise UnsupportedForRustEngine(
                'the Rust engine did not identify an EPSG code for the terrain and exposed no WKT to '
                'verify; after checking `ras2fim-batch terrain-info`, set [05_raster_engine] '
                'str_rust_terrain_crs_fnv1a64 to its crs_wkt_fnv1a64')
        try:
            crs_engine = CRS.from_wkt(str_wkt)
        except Exception as e:
            raise RustMappingError(f'engine terrain WKT is not parseable: {e}') from e
        if crs_engine != crs_rio:
            raise UnsupportedForRustEngine(f'engine terrain CRS differs from rasterio CRS {crs_rio.to_string()}')
        dict_identity = {'kind': 'caller_verified_wkt', 'fnv1a64': str_fnv}
    return {'path': os.path.abspath(str_dem_path), 'epsg': int_epsg, 'crs_identity': dict_identity}
# ------------------------------


# ..............................
def fn_find_wkt(obj):
    """Find a WKT string in the engine's serialized CRS object, if present."""
    if isinstance(obj, str):
        return obj if obj.lstrip().upper().startswith(('PROJCS', 'GEOGCS', 'PROJCRS', 'GEOGCRS',
                                                       'COMPD', 'BOUNDCRS')) else None
    if isinstance(obj, dict):
        for str_key in ('wkt', 'wkt2', 'wkt1'):
            if isinstance(obj.get(str_key), str):
                return obj[str_key]
        for value in obj.values():
            str_found = fn_find_wkt(value)
            if str_found:
                return str_found
    return None
# ..............................


# ------------------------------
@dataclass
class Step05Grids:
    """Grids derived from the upstream recipe for one profile."""
    rows: int
    cols: int
    raster_affine: object          # raster 01 affine (positive e)
    wet_mask: np.ndarray           # bool (rows, cols), True = inside wet union
    dest_transform: object         # final nudged grid (north-up)
    dest_width: int
    dest_height: int
    dest_crs: str


def fn_derive_step05_grids(prep, geom_wet, crs_src, tup_terrain_bounds, str_output_crs, flt_desired_res):
    """
    Reproduce the upstream grid decisions without writing rasters:
    02 = rasterio.mask(01, wet union, crop=True), 03 = crop of 02 to the
    terrain bounds, 09/10 grid = calculate_default_transform of the 07 file
    followed by the origin nudge. The crops must not change the 01 grid
    (the Rust engine works on the 01 grid); otherwise this is unsupported.
    """
    import rasterio
    import rasterio.mask
    from affine import Affine
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    from rasterio.warp import calculate_default_transform
    from shapely.geometry import box

    int_rows, int_cols = len(prep.y_axis), len(prep.x_axis)
    affine_01 = Affine.translation(prep.xmin, prep.ymin) * Affine.scale(prep.x_res, prep.y_res)
    dict_profile = dict(driver='GTiff', width=int_cols, height=int_rows, count=1, dtype='float32',
                        crs=crs_src, transform=affine_01, nodata=np.nan, compress='lzw')
    with MemoryFile() as memfile:
        with memfile.open(**dict_profile):
            pass
        with memfile.open() as ds:
            arr_out02, transform02, window02 = rasterio.mask.raster_geometry_mask(ds, [geom_wet], crop=True)
            if (int(window02.col_off), int(window02.row_off), int(window02.width), int(window02.height)) != \
                    (0, 0, int_cols, int_rows) or tuple(transform02) != tuple(affine_01):
                raise UnsupportedForRustEngine(f'wet-union crop changes the raster 01 grid ({window02})')
            arr_out03, transform03, window03 = rasterio.mask.raster_geometry_mask(
                ds, [box(*tup_terrain_bounds)], crop=True)
            if (int(window03.col_off), int(window03.row_off), int(window03.width), int(window03.height)) != \
                    (0, 0, int_cols, int_rows) or arr_out03.any():
                raise UnsupportedForRustEngine('the raster 01 grid extends beyond the terrain bounds')
            # upstream: 07_temp has the 03 profile; nudge uses its CRS and bounds
            crs_07 = ds.crs
            tup_bounds_07 = tuple(ds.bounds)

    flt_res = float(flt_desired_res)
    transform, int_width, int_height = calculate_default_transform(
        crs_07, str_output_crs, int_cols, int_rows, *tup_bounds_07)
    transform_dest = from_origin(round(transform.c / flt_res) * flt_res,
                                 round(transform.f / flt_res) * flt_res,
                                 flt_res, flt_res)
    return Step05Grids(int_rows, int_cols, affine_01, ~arr_out02, transform_dest,
                       int(int_width), int(int_height), str_output_crs)
# ------------------------------


# ..............................
def _fn_write_array(str_path, arr, dtype):
    arr = np.ascontiguousarray(arr, dtype=dtype)
    arr.tofile(str_path)
    return {'file': os.path.basename(str_path), 'dtype': np.dtype(dtype).str, 'shape': list(arr.shape)}
# ..............................


# ------------------------------
def fn_write_profile_inputs(str_dir, str_column, prep, grids):
    """Write one profile's prepared inputs (ras2fim-step05-inputs/1)."""
    from scipy.spatial import Delaunay
    import scipy

    os.makedirs(str_dir, exist_ok=False)
    arr_samples = np.column_stack([np.asarray(prep.x, dtype=np.float64),
                                   np.asarray(prep.y, dtype=np.float64),
                                   np.asarray(prep.z, dtype=np.float64)])
    # identical to the triangulation griddata(..., method='linear') builds
    tri = Delaunay(np.ascontiguousarray(arr_samples[:, :2]))
    if len(tri.coplanar):
        raise UnsupportedForRustEngine(f'{str_column}: {len(tri.coplanar)} coplanar/duplicate samples '
                                       'are not supported by the Rust engine')
    dict_files = {
        'samples': _fn_write_array(os.path.join(str_dir, 'samples.f64'), arr_samples, '<f8'),
        'simplices': _fn_write_array(os.path.join(str_dir, 'simplices.u32'), tri.simplices, '<u4'),
        'transform': _fn_write_array(os.path.join(str_dir, 'transform.f64'), tri.transform, '<f8'),
        'neighbors': _fn_write_array(os.path.join(str_dir, 'neighbors.i32'), tri.neighbors, '<i4'),
        'equations': _fn_write_array(os.path.join(str_dir, 'equations.f64'), tri.equations, '<f8'),
        'x_axis': _fn_write_array(os.path.join(str_dir, 'x_axis.f64'), prep.x_axis, '<f8'),
        'y_axis': _fn_write_array(os.path.join(str_dir, 'y_axis.f64'), prep.y_axis, '<f8'),
        'wet_mask': _fn_write_array(os.path.join(str_dir, 'wet_mask.u8'), grids.wet_mask, 'u1'),
    }
    a = grids.raster_affine
    dict_meta = {
        'schema': STR_INPUTS_SCHEMA,
        'column': str_column,
        'rows': grids.rows,
        'cols': grids.cols,
        'raster_affine': {'x0': a.c, 'y0': a.f, 'res_x': a.a, 'res_y': a.e},
        'delaunay': {'paraboloid_scale': float(tri.paraboloid_scale),
                     'paraboloid_shift': float(tri.paraboloid_shift),
                     'min_bound': tri.min_bound.tolist(), 'max_bound': tri.max_bound.tolist(),
                     'nsimplex': int(tri.nsimplex)},
        'provenance': {'scipy': scipy.__version__, 'numpy': np.__version__,
                       'samples': 'fn_prepare_interpolation_inputs (upstream order)'},
        'files': dict_files,
    }
    with open(os.path.join(str_dir, 'inputs.json'), 'w') as f:
        json.dump(dict_meta, f, indent=1)
# ------------------------------


# ------------------------------
@dataclass
class ProfileRequest:
    column: str
    inputs_dir: str                # relative to the request directory
    grids: Step05Grids
    aoi_window: object = None      # mapping_aoi.AOIWindow or None for the full grid
    mask_file: str = None          # relative path of the AOI mask (polygon only)


@dataclass
class ProfileResult:
    column: str
    status: str                    # 'complete' | 'failed'
    depth: np.ndarray = None
    wse: np.ndarray = None
    transform: object = None
    error: str = None
    unit: dict = field(default_factory=dict)


def fn_dest_geotransform(grids):
    t = grids.dest_transform
    return [t.c, t.a, t.b, t.f, t.d, t.e]


def fn_build_request(terrain, opts, list_profiles, int_dst_epsg):
    list_json = []
    for p in list_profiles:
        if not _RE_SAFE_ID.match(p.column):
            raise UnsupportedForRustEngine(f'column name {p.column!r} is not a safe engine id')
        if p.aoi_window is None:
            dict_aoi = {'id': 'full'}
        else:
            dict_aoi = {'id': 'aoi', 'window': p.aoi_window.window}
            if p.mask_file is not None:
                dict_aoi['mask_u8'] = p.mask_file
        list_json.append({
            'id': p.column,
            'inputs_dir': p.inputs_dir,
            'dest_grid': {'geotransform': fn_dest_geotransform(p.grids),
                          'width': p.grids.dest_width, 'height': p.grids.dest_height},
            'aois': [dict_aoi],
        })
    return {
        'schema': STR_REQUEST_SCHEMA,
        'terrain': terrain,
        'source_crs': {'kind': 'terrain'},
        'budgets': {'threads': opts.resolved_threads(),
                    'max_cache_bytes': opts.rust_max_cache_mib * 1024 * 1024,
                    'max_profile_bytes': opts.rust_max_profile_mib * 1024 * 1024,
                    'max_inflight_profiles': opts.rust_max_inflight_profiles},
        'interpolation': {'selection': 'auto'},
        'nudge': {'dst_epsg': int_dst_epsg, 'error_threshold': FLT_NUDGE_ERROR_THRESHOLD,
                  'memory_limit_bytes': INT_NUDGE_MEMORY_LIMIT_BYTES, 'method': opts.rust_nudge_method},
        'stages': [],
        'profiles': list_json,
    }
# ------------------------------


# ------------------------------
def fn_run_step05_request(str_binary, str_work_dir, int_batch, terrain, opts, list_profiles, int_dst_epsg):
    """
    Write the request for one bounded batch, run the engine, and return a
    list of ProfileResult. Exit 0/2 are read per profile; exit 1 (invalid
    request) or any other status raises. The engine report is required and
    must agree with the exit status and the requested profiles; a profile
    that is not complete in both the report and a consistent unit is
    returned as 'failed'. Returns (results, report).
    """
    dict_request = fn_build_request(terrain, opts, list_profiles, int_dst_epsg)
    str_request = os.path.join(str_work_dir, f'request_{int_batch:04d}.json')
    with open(str_request, 'w') as f:
        json.dump(dict_request, f, indent=1)
    str_out = os.path.join(str_work_dir, f'out_{int_batch:04d}')

    proc = fn_run_binary([str_binary, 'step05', str_request, str_out], 'step05')
    if proc.returncode not in (0, 2):
        raise RustMappingError(f'ras2fim-batch step05 failed with exit {proc.returncode}: '
                               f'{proc.stderr.strip()[-4000:]}')

    dict_report = fn_read_report(str_out, proc.returncode, [p.column for p in list_profiles])
    dict_report_status = {d['id']: d for d in dict_report['profiles']}

    list_results = []
    for p in list_profiles:
        dict_prof = dict_report_status[p.column]
        if dict_prof.get('status') != 'complete':
            list_results.append(ProfileResult(p.column, 'failed',
                                              error=f'engine report status {dict_prof.get("status")!r}: '
                                                    f'{dict_prof.get("error_kind", "")} {dict_prof.get("error", "")}'))
            continue
        try:
            list_results.append(fn_read_unit(p, str_out))
        except (RustMappingError, OSError, ValueError, KeyError, TypeError) as e:
            list_results.append(ProfileResult(p.column, 'failed',
                                              error=f'{e} (engine exit {proc.returncode}; '
                                                    f'stderr: {proc.stderr.strip()[-1000:]})'))
    if proc.returncode == 2 and all(r.status == 'complete' for r in list_results):
        raise RustMappingError('ras2fim-batch step05 exited 2 (profile failure) but every requested unit '
                               f'reads as complete: {proc.stderr.strip()[-2000:]}')
    return list_results, dict_report
# ------------------------------




# ..............................
def _fn_is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _fn_finite_numbers(values, int_count):
    return (isinstance(values, list) and len(values) == int_count
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in values))
# ..............................


# ------------------------------
def fn_read_report(str_out, int_exit, list_columns):
    """Read the engine's report.json and check it against the exit status and request."""
    str_report = os.path.join(str_out, 'report.json')
    if not os.path.isfile(str_report):
        raise RustMappingError(f'ras2fim-batch step05 exited {int_exit} without report.json')
    try:
        with open(str_report) as f:
            dict_report = json.load(f)
    except json.JSONDecodeError as e:
        raise RustMappingError(f'malformed engine report.json: {e}') from e
    if not isinstance(dict_report, dict) or dict_report.get('schema') != STR_REPORT_SCHEMA:
        raise RustMappingError(f'engine report.json schema is not {STR_REPORT_SCHEMA}')
    str_expected = 'complete' if int_exit == 0 else 'failed'
    if dict_report.get('status') != str_expected:
        raise RustMappingError(f'engine report status {dict_report.get("status")!r} disagrees with exit {int_exit}')
    list_profiles = dict_report.get('profiles')
    if not isinstance(list_profiles, list) or not all(isinstance(d, dict) for d in list_profiles):
        raise RustMappingError('engine report.json has no profile list')
    list_ids = [d.get('id') for d in list_profiles]
    if list_ids != list_columns:
        raise RustMappingError(f'engine report profiles {list_ids} differ from the request {list_columns}')
    return dict_report
# ------------------------------

# ------------------------------
def fn_read_unit(p, str_out):
    """
    Read and validate one profile's unit (ras2fim-step05-unit/1). Raises
    RustMappingError for anything other than a complete, consistent unit.
    """
    from affine import Affine

    str_aoi_id = 'full' if p.aoi_window is None else 'aoi'
    str_unit_dir = os.path.join(str_out, 'units', p.column, str_aoi_id)
    str_unit = os.path.join(str_unit_dir, 'unit.json')
    if not os.path.isfile(str_unit):
        raise RustMappingError(f'{p.column}: engine wrote no unit.json')
    try:
        with open(str_unit) as f:
            dict_unit = json.load(f)
    except json.JSONDecodeError as e:
        raise RustMappingError(f'{p.column}: malformed unit.json: {e}') from e
    if not isinstance(dict_unit, dict) or dict_unit.get('schema') != STR_UNIT_SCHEMA:
        raise RustMappingError(f'{p.column}: unit.json schema is not {STR_UNIT_SCHEMA}')
    if dict_unit.get('profile') != p.column or dict_unit.get('aoi') != str_aoi_id:
        raise RustMappingError(f'{p.column}: unit.json identifies {dict_unit.get("profile")!r}/'
                               f'{dict_unit.get("aoi")!r}, expected {p.column!r}/{str_aoi_id!r}')
    str_status = dict_unit.get('status')
    if str_status != 'complete':
        raise RustMappingError(f'{p.column}: engine unit status {str_status!r}: {dict_unit.get("error", "")}')

    list_window = dict_unit.get('window')
    if not (isinstance(list_window, list) and len(list_window) == 4 and all(_fn_is_int(v) for v in list_window)
            and min(list_window) >= 0):
        raise RustMappingError(f'{p.column}: unit.json window {list_window!r} is not four non-negative integers')
    if p.aoi_window is None:
        list_expected = [0, 0, p.grids.dest_width, p.grids.dest_height]
        transform_expected = p.grids.dest_transform
    else:
        list_expected = p.aoi_window.window
        transform_expected = p.aoi_window.transform
    if list_window != list_expected:
        raise RustMappingError(f'{p.column}: engine window {list_window} != requested {list_expected}')
    gt = dict_unit.get('window_geotransform')
    if not _fn_finite_numbers(gt, 6):
        raise RustMappingError(f'{p.column}: unit.json window_geotransform {gt!r} is not six finite numbers')
    transform_engine = Affine.from_gdal(*gt)
    flt_tol = 1e-6 * abs(transform_expected.a)
    if any(abs(u - v) > flt_tol for u, v in zip(tuple(transform_engine)[:6], tuple(transform_expected)[:6])):
        raise RustMappingError(f'{p.column}: engine window geotransform {gt} differs from '
                               f'{tuple(transform_expected)[:6]}')
    if 'polygon_mask_applied' in dict_unit and bool(dict_unit['polygon_mask_applied']) != (p.mask_file is not None):
        raise RustMappingError(f'{p.column}: polygon_mask_applied={dict_unit["polygon_mask_applied"]} '
                               f'but a mask was {"" if p.mask_file else "not "}requested')

    int_w, int_h = list_window[2], list_window[3]
    for str_key, int_value in (('width', int_w), ('height', int_h)):
        if str_key in dict_unit and dict_unit[str_key] != int_value:
            raise RustMappingError(f'{p.column}: unit.json {str_key} {dict_unit[str_key]!r} != window')
    dict_counts = dict_unit.get('finite_counts')
    if not isinstance(dict_counts, dict):
        raise RustMappingError(f'{p.column}: unit.json has no finite_counts')
    dict_arrays = {}
    for str_name in ('depth', 'wse'):
        str_file = os.path.join(str_unit_dir, f'{str_name}.f32')
        if not os.path.isfile(str_file):
            raise RustMappingError(f'{p.column}: unit is complete but {str_name}.f32 is missing')
        int_bytes = os.path.getsize(str_file)
        if int_bytes != 4 * int_w * int_h:
            raise RustMappingError(f'{p.column}: {str_name}.f32 has {int_bytes} bytes, '
                                   f'expected {4 * int_w * int_h}')
        arr = np.fromfile(str_file, dtype='<f4').reshape(int_h, int_w)
        str_key = next((k for k in dict_counts if k.split('.')[0] == str_name), None)
        if str_key is None:
            raise RustMappingError(f'{p.column}: unit.json finite_counts lacks {str_name}')
        if not _fn_is_int(dict_counts[str_key]) or dict_counts[str_key] != int(np.isfinite(arr).sum()):
            raise RustMappingError(f'{p.column}: {str_name}.f32 finite count {int(np.isfinite(arr).sum())} '
                                   f'differs from unit.json {dict_counts[str_key]}')
        if p.aoi_window is not None and p.aoi_window.mask is not None and np.isfinite(arr[~p.aoi_window.mask]).any():
            raise RustMappingError(f'{p.column}: {str_name} has finite values outside the polygon mask')
        dict_arrays[str_name] = arr
    return ProfileResult(p.column, 'complete', dict_arrays['depth'], dict_arrays['wse'],
                         transform_expected, unit=dict_unit)
# ------------------------------

# ..............................
def fn_output_epsg(str_output_crs):
    from pyproj import CRS
    int_epsg = CRS.from_user_input(str_output_crs).to_epsg()
    if int_epsg is None:
        raise UnsupportedForRustEngine(f'output CRS {str_output_crs!r} has no EPSG code')
    return int_epsg
# ..............................


# ..............................
def fn_chunks(list_items, int_size):
    for i in range(0, len(list_items), int_size):
        yield list_items[i:i + int_size]
# ..............................
