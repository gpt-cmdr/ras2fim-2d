# Area-of-interest (AOI) constraints for the Step 05 mapping engines.
#
# An AOI selects output pixels on the final (nudged) Step 05 grid. It never
# changes sample preparation, triangulation, interpolation axes, the wet
# mask, terrain sampling, the output grid origin or the reprojection. Both
# mapping engines ("python" and "rust") use the window and mask computed
# here, so an AOI result is a crop of the full-map result.
#
# An AOI is either a bounding box or a polygon (mutually exclusive):
#   * bbox: four finite coordinates (xmin, ymin, xmax, ymax), xmin < xmax and
#     ymin < ymax. If no CRS is given, the bbox is in the Step 05 output CRS
#     ([06_output_products] str_output_crs).
#   * polygon: a GeoJSON or GeoPackage file (optional layer) or, from the
#     Python API, a Shapely Polygon/MultiPolygon. Holes and multiple parts are
#     supported. A CRS is required, either from the file or explicitly.
#
# Created for the optional Rust mapping engine - 2026.10.23
# ************************************************************
import math
import os
from dataclasses import dataclass

import numpy as np
import shapely
from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
# ************************************************************

AOI_STATUS_OK = 'ok'
AOI_STATUS_EMPTY = 'empty_intersection'

# Number of segments used to densify each edge's extent before an AOI is
# reprojected into the grid CRS.
INT_DENSIFY_SEGMENTS = 1000


class MappingAOIError(ValueError):
    """An AOI request is invalid or unsupported."""


# ------------------------------
@dataclass(frozen=True)
class MappingAOI:
    """A validated bbox or polygon AOI with its CRS."""
    kind: str                     # 'bbox' or 'polygon'
    geometry: BaseGeometry        # box or (Multi)Polygon in `crs`
    crs: CRS
    source: str                   # human-readable origin of the AOI

    def describe(self):
        return {
            'kind': self.kind,
            'source': self.source,
            'crs': self.crs.to_string(),
            'bounds': [float(v) for v in self.geometry.bounds],
        }
# ------------------------------


# ------------------------------
@dataclass(frozen=True)
class AOIWindow:
    """AOI selection on one north-up output grid."""
    status: str                   # AOI_STATUS_OK or AOI_STATUS_EMPTY
    col_off: int
    row_off: int
    width: int
    height: int
    transform: Affine             # affine of the window (same pixel size)
    mask: object = None           # bool array (height, width), True = keep; None = whole window

    @property
    def is_empty(self):
        return self.status == AOI_STATUS_EMPTY

    @property
    def window(self):
        return [self.col_off, self.row_off, self.width, self.height]

    def describe(self):
        return {
            'status': self.status,
            'window': self.window,
            'window_transform': list(self.transform)[:6],
            'polygon_mask': self.mask is not None,
        }
# ------------------------------


# ..............................
def fn_to_crs(value, str_what='aoi_crs'):
    if value is None:
        return None
    if isinstance(value, CRS):
        return value
    try:
        return CRS.from_user_input(value)
    except Exception as e:
        raise MappingAOIError(f'{str_what}: unrecognized CRS {value!r}: {e}') from e
# ..............................


# ..............................
def fn_parse_bbox(values):
    """Validate a bbox and return (xmin, ymin, xmax, ymax) as floats."""
    if isinstance(values, str):
        values = [v for v in values.replace(',', ' ').split() if v]
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError) as e:
        raise MappingAOIError(f'bbox must be four numbers (xmin ymin xmax ymax): {e}') from e
    if len(values) != 4:
        raise MappingAOIError(f'bbox must have exactly four values (xmin ymin xmax ymax), got {len(values)}')
    if not all(math.isfinite(v) for v in values):
        raise MappingAOIError(f'bbox values must be finite, got {values}')
    xmin, ymin, xmax, ymax = values
    if not (xmin < xmax and ymin < ymax):
        raise MappingAOIError(f'bbox must satisfy xmin < xmax and ymin < ymax, got {values}')
    return (xmin, ymin, xmax, ymax)
# ..............................


# ..............................
def fn_validate_polygon(geom, str_source):
    """Return a valid, non-empty Polygon/MultiPolygon or raise."""
    if geom is None:
        raise MappingAOIError(f'{str_source}: polygon AOI geometry is missing')
    if not isinstance(geom, (Polygon, MultiPolygon)):
        raise MappingAOIError(f'{str_source}: polygon AOI must be a Polygon or MultiPolygon, '
                              f'got {geom.geom_type}')
    if geom.is_empty:
        raise MappingAOIError(f'{str_source}: polygon AOI geometry is empty')
    if not geom.is_valid:
        raise MappingAOIError(f'{str_source}: polygon AOI geometry is invalid ({explain_validity(geom)})')
    coords = shapely.get_coordinates(geom)
    if not np.all(np.isfinite(coords)):
        raise MappingAOIError(f'{str_source}: polygon AOI has non-finite coordinates')
    return geom
# ..............................


# ..............................
def fn_bbox_aoi(values, crs):
    tup_bbox = fn_parse_bbox(values)
    crs = fn_to_crs(crs)
    if crs is None:
        raise MappingAOIError('bbox AOI requires a CRS')
    return MappingAOI('bbox', box(*tup_bbox), crs, f'bbox {list(tup_bbox)}')
# ..............................


# ..............................
def fn_polygon_aoi_from_geometry(geom, crs):
    crs = fn_to_crs(crs)
    if crs is None:
        raise MappingAOIError('a Shapely polygon AOI has no CRS: pass aoi_crs')
    geom = fn_validate_polygon(geom, 'polygon AOI')
    return MappingAOI('polygon', geom, crs, f'shapely {geom.geom_type}')
# ..............................


# ..............................
def fn_polygon_aoi_from_file(str_path, str_layer=None, crs=None):
    import geopandas as gpd

    str_path = os.fspath(str_path)
    if not os.path.isfile(str_path):
        raise MappingAOIError(f'polygon AOI file not found: {str_path}')
    try:
        df_layers = gpd.list_layers(str_path)
    except Exception as e:
        raise MappingAOIError(f'cannot read polygon AOI file {str_path}: {e}') from e
    list_layers = list(df_layers['name'])
    if str_layer is None:
        if len(list_layers) != 1:
            raise MappingAOIError(f'{str_path} has {len(list_layers)} layers {list_layers}: '
                                  'choose one with polygon_layer / --polygon-layer')
    elif str_layer not in list_layers:
        raise MappingAOIError(f'layer {str_layer!r} not in {str_path} (layers: {list_layers})')

    gdf = gpd.read_file(str_path, layer=str_layer)
    str_source = f'{os.path.basename(str_path)}' + (f' layer {str_layer}' if str_layer else '')
    if len(gdf) == 0:
        raise MappingAOIError(f'{str_source}: polygon AOI file has no features')

    crs_arg = fn_to_crs(crs)
    crs_file = fn_to_crs(gdf.crs, 'polygon file CRS') if gdf.crs is not None else None
    if crs_file is None and crs_arg is None:
        raise MappingAOIError(f'{str_source}: file has no CRS; pass aoi_crs / --aoi-crs')
    if crs_file is not None and crs_arg is not None and crs_file != crs_arg:
        raise MappingAOIError(f'{str_source}: file CRS {crs_file.to_string()} conflicts with '
                              f'aoi_crs {crs_arg.to_string()}')
    crs_use = crs_file if crs_file is not None else crs_arg

    for int_idx, geom in enumerate(gdf.geometry):
        fn_validate_polygon(geom, f'{str_source} feature {int_idx}')
    geom_union = shapely.union_all(np.asarray(gdf.geometry.values))
    geom_union = fn_validate_polygon(geom_union, f'{str_source} (union of features)')
    return MappingAOI('polygon', geom_union, crs_use, str_source)
# ..............................


# ------------------------------
def fn_resolve_aoi(bbox=None, polygon=None, polygon_layer=None, aoi_crs=None, default_bbox_crs=None):
    """
    Build a MappingAOI from user options, or None when no AOI is requested.

    bbox and polygon are mutually exclusive. A bbox without aoi_crs uses
    default_bbox_crs (the Step 05 output CRS). A polygon may be a file path
    or a Shapely Polygon/MultiPolygon.
    """
    b_bbox = bbox is not None and not (isinstance(bbox, str) and not bbox.strip())
    b_polygon = polygon is not None and not (isinstance(polygon, str) and not polygon.strip())
    if b_bbox and b_polygon:
        raise MappingAOIError('bbox and polygon AOIs are mutually exclusive')
    if polygon_layer and not b_polygon:
        raise MappingAOIError('polygon_layer was given without a polygon AOI')
    if not b_bbox and not b_polygon:
        if aoi_crs:
            raise MappingAOIError('aoi_crs was given without a bbox or polygon AOI')
        return None
    if b_bbox:
        crs = aoi_crs if aoi_crs else default_bbox_crs
        if crs is None:
            raise MappingAOIError('bbox AOI needs aoi_crs (no default output CRS is available)')
        return fn_bbox_aoi(bbox, crs)
    if isinstance(polygon, BaseGeometry):
        return fn_polygon_aoi_from_geometry(polygon, aoi_crs)
    return fn_polygon_aoi_from_file(polygon, polygon_layer or None, aoi_crs or None)
# ------------------------------


# ..............................
def fn_geometry_in_crs(aoi, crs_grid):
    """Return the AOI geometry in the grid CRS (densified before reprojection)."""
    crs_grid = fn_to_crs(crs_grid, 'grid CRS')
    if aoi.crs == crs_grid:
        return aoi.geometry
    xmin, ymin, xmax, ymax = aoi.geometry.bounds
    flt_step = max(xmax - xmin, ymax - ymin) / INT_DENSIFY_SEGMENTS
    geom = shapely.segmentize(aoi.geometry, flt_step) if flt_step > 0 else aoi.geometry
    transformer = Transformer.from_crs(aoi.crs, crs_grid, always_xy=True)

    def fn_xy(arr):
        x, y = transformer.transform(arr[:, 0], arr[:, 1])
        return np.column_stack([x, y])

    geom_out = shapely.transform(geom, fn_xy)
    if not np.all(np.isfinite(shapely.get_coordinates(geom_out))):
        raise MappingAOIError(f'AOI cannot be transformed from {aoi.crs.to_string()} '
                              f'to {crs_grid.to_string()} (non-finite coordinates)')
    return fn_validate_polygon(geom_out, f'AOI reprojected to {crs_grid.to_string()}')
# ..............................


# ..............................
def fn_center_index_range(flt_lo, flt_hi, flt_origin, flt_step, int_count):
    """
    Indices i in [0, int_count) whose pixel centre origin + (i + 0.5) * step
    lies in [lo, hi). step may be negative.
    Returns (start, stop) with stop exclusive (possibly start == stop).
    """
    def fn_center(i):
        return flt_origin + (i + 0.5) * flt_step

    def fn_inside(i):
        c = fn_center(i)
        return flt_lo <= c < flt_hi

    # Centres are monotonic in i, so the inside set is contiguous. Take the
    # continuous solution, then settle each end exactly on the predicate.
    flt_a = (flt_lo - flt_origin) / flt_step - 0.5
    flt_b = (flt_hi - flt_origin) / flt_step - 0.5
    int_lo = max(0, min(int_count, int(math.floor(min(flt_a, flt_b))) - 1))
    int_hi = max(0, min(int_count, int(math.ceil(max(flt_a, flt_b))) + 2))
    int_start = next((i for i in range(int_lo, min(int_lo + 4, int_hi)) if fn_inside(i)), None)
    if int_start is None:
        return (0, 0)
    int_last = next(i for i in range(int_hi - 1, int_start - 1, -1) if fn_inside(i))
    return (int_start, int_last + 1)
# ..............................


# ------------------------------
def fn_aoi_window(aoi, transform, int_width, int_height, crs_grid):
    """
    Select the AOI pixels on a north-up grid.

    The window holds the pixels whose centre lies in [xmin, xmax) x
    [ymin, ymax) of the AOI bounds in the grid CRS. A bbox in the grid CRS
    is that window. Any other AOI (a polygon, or a bbox in a different CRS,
    reprojected as a densified polygon) also masks the window pixels whose
    centre is outside the polygon (GDAL rasterization, all_touched=False);
    holes are excluded.
    """
    transform = Affine(*list(transform)[:6])
    if not (transform.b == 0 and transform.d == 0 and transform.a > 0 and transform.e < 0):
        raise MappingAOIError(f'AOI requires a north-up output grid, got {tuple(transform)[:6]}')
    crs_grid = fn_to_crs(crs_grid, 'grid CRS')
    b_rect = aoi.kind == 'bbox' and aoi.crs == crs_grid
    geom = aoi.geometry if b_rect else fn_geometry_in_crs(aoi, crs_grid)

    xmin, ymin, xmax, ymax = geom.bounds
    col0, col1 = fn_center_index_range(xmin, xmax, transform.c, transform.a, int_width)
    # rows: centre y = f + (j + 0.5) * e, e < 0
    row0, row1 = fn_center_index_range(ymin, ymax, transform.f, transform.e, int_height)
    int_w, int_h = col1 - col0, row1 - row0
    transform_win = transform @ Affine.translation(col0, row0)
    if int_w <= 0 or int_h <= 0:
        return AOIWindow(AOI_STATUS_EMPTY, col0, row0, max(int_w, 0), max(int_h, 0), transform_win)
    if b_rect:
        return AOIWindow(AOI_STATUS_OK, col0, row0, int_w, int_h, transform_win)

    from rasterio.features import geometry_mask
    arr_keep = geometry_mask([geom], out_shape=(int_h, int_w), transform=transform_win,
                             all_touched=False, invert=True)
    if not arr_keep.any():
        return AOIWindow(AOI_STATUS_EMPTY, col0, row0, int_w, int_h, transform_win)
    return AOIWindow(AOI_STATUS_OK, col0, row0, int_w, int_h, transform_win, arr_keep)
# ------------------------------


# ..............................
def fn_apply_aoi_window(arr, aoi_window):
    """Crop a full-grid array to the AOI window; NaN outside the polygon mask."""
    if aoi_window.is_empty:
        raise MappingAOIError('cannot apply an empty AOI window')
    arr_out = np.array(arr[aoi_window.row_off:aoi_window.row_off + aoi_window.height,
                           aoi_window.col_off:aoi_window.col_off + aoi_window.width], copy=True)
    if aoi_window.mask is not None:
        arr_out[~aoi_window.mask] = np.nan
    return arr_out
# ..............................
