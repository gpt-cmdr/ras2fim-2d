# Optional Rust mapping engine and area-of-interest constraints (Step 05)

Step 05 (`generate_depth_wsel_rasters_mp_05.py`) turns the Step 04 cell WSE profiles into the
`09_<col>_depth_nudge.tif`, `10_<col>_wsel_nudge.tif` and `11_<col>_floodplain_ar.gpkg` products.
Two engines are available:

| Engine | Selection | What runs |
|---|---|---|
| `python` (default) | nothing to configure | The original Step 05 code, unchanged |
| `rust` (opt-in) | `mapping_engine='rust'`, `[05_raster_engine] str_mapping_engine = rust` or `--mapping-engine rust` | Python prepares the same global inputs; the separately built `ras2fim-batch step05` command interpolates, samples terrain, applies the float32 depth/WSEL sequence and reprojects to the output grid |

Either engine can be limited to a bounding box **or** a polygon area of interest (AOI).
Rust and AOI runs require an output directory without existing 09/10/11 products
or mapping manifests. Existing maps are rejected and preserved. Each optional
run records `in_progress`, then `complete`, `failed` or `aborted`; a session
abort removes its uncommitted product set. Use a fresh directory to rerun.
Depth and WSE retain the source vertical units. Cell WSE and terrain elevations
must use compatible vertical units and reference; changing the horizontal output
CRS does not convert elevations. Output resolution uses the output CRS units
(metres for the default EPSG:3857 grid).

## Scope and status

- The Rust engine reproduces the pinned Step 05 recipe; it is not a new mapping method. Python
  still performs the upstream sample preparation (wet cell centres in sorted `cell_idx` order,
  then hull-outside cell vertices merged by rounded XY with their mean WSE), builds the SciPy
  Delaunay triangulation and the wet-union mask, and derives the final output grid. The engine
  receives these as binary inputs and does not re-derive them.
- Interpolation uses the exact `np.arange` query axes. Their effective step can differ from the
  terrain pixel size in the last digits; the raster affine keeps the pixel size.
- Product 11 is created by the original `fn_make_floodplain_gdf` from product 09.
- Speed comparisons are reported separately by the repository benchmark (`benchmarks/`). No
  speed ratio is implied here.
- [Method comparison and coverage limits](rust-mapping-qualification.md) describe the
  public-sample checks. Use the [benchmark protocol](../benchmarks/README.md) for
  complete-call throughput measurements.
- Not supported by the Rust engine (explicit errors, never a silent fallback): a hydraulic
  results CRS different from the terrain CRS, terrain without an EPSG code (unless its WKT is
  verified, below), non-TIFF terrain (VRT/HDF), an output CRS without an EPSG code, coplanar or
  duplicate samples, infinite WSE values, a wet union or terrain extent that would change the
  Step 05 grid crop, and the engine's own typed `unsupported` errors.

## Install a prebuilt engine wheel

The `Engine wheels` GitHub Actions workflow builds Linux x86-64 (manylinux
2.28, glibc 2.28 or newer) and Windows x86-64 wheels. Download the matching
`ras2fim-engine-<platform>` artifact from the test branch's successful workflow
run, extract it, and verify its `SHA256SUMS.txt` before installing:

```bash
python -m pip install /path/to/ras2fim_engine-0.1.0-py3-none-<platform>.whl
ras2fim-batch method
```

Use the actual wheel filename. Install it into the same environment used to
run Step05 so its scripts directory is on `PATH`. This optional wheel installs
only the compiled engine; it does not install the Python application or GIS
dependencies. No Rust compiler or system GDAL is needed by the engine. The
Python workflow still needs its existing raster libraries. Nothing is
published to PyPI by the workflow. Artifacts expire after 90 days; rebuild
from the retained source or retain downloaded wheels for longer testing.

Both wheels run an installed-binary mapping smoke test in CI. The retained
real-model equality evidence remains Linux-specific; a successful Windows
wheel smoke test does not establish real-model parity on Windows.

## Building and locating the binary

The engine source is in `rust/step05`. Python never builds it; build it yourself:

```bash
cd rust/step05
cargo build --release --locked
# binary: rust/step05/target/release/ras2fim-batch
```

The binary is found from, in order: `rust_binary=` / `--rust-binary`, `[05_raster_engine]
str_rust_binary`, then `ras2fim-batch` on `PATH`. If none exists, Step 05 raises
`RustBinaryNotFoundError` before reading any input. The engine is launched with an argument
list (no shell). Linux has retained real-model comparison evidence.

## Python API

```python
from generate_depth_wsel_rasters_mp_05 import fn_generate_depth_wsel_rasters

fn_generate_depth_wsel_rasters(
    str_config_file_path, str_hdf_file_path, str_dem_path,
    str_hydraulic_results_gpkg, str_output_dir, b_print_output,
    mapping_engine='rust',            # 'python' (default) or 'rust'
    rust_binary='rust/step05/target/release/ras2fim-batch',
    mapping_threads=8,                # python pool processes / rust threads
    bbox=(-10843000, 3553000, -10841000, 3554500),   # OR polygon=...
    aoi_crs=None,                     # bbox default: [06_output_products] str_output_crs
)
```

| Keyword | Meaning |
|---|---|
| `mapping_engine` | `'python'` or `'rust'` |
| `rust_binary` | engine path; only valid with the rust engine |
| `mapping_threads` | positive integer; default `min(cpu_count - 1, 16)` (the original pool size) |
| `bbox` | `(xmin, ymin, xmax, ymax)`, finite, `xmin < xmax`, `ymin < ymax` |
| `polygon` | GeoJSON/GeoPackage path or Shapely `Polygon`/`MultiPolygon` |
| `polygon_layer` | layer name; required when the file has more than one layer |
| `aoi_crs` | CRS of the AOI (any pyproj input such as `EPSG:2277`) |
| `rust_max_cache_mib` | decoded-terrain cache budget, default 64 |
| `rust_max_profile_mib` | per-profile working-set budget, default 1024 |
| `rust_profiles_per_request` | profiles per engine call, default 4 |
| `rust_max_inflight_profiles` | profiles the engine evaluates concurrently (1 or more), default 1 |
| `rust_nudge_method` | `lookup` (default) or `full_warp` |
| `rust_terrain_crs_fnv1a64` | verified terrain WKT hash, see below |
| `rust_keep_work` | keep the temporary request/inputs folder for inspection |

Keywords override `[05_raster_engine]` values (see `src/config_global.ini` for the key names);
unknown keys in that section are rejected. With no options and an empty section, the function
runs the original code and returns `None`. Otherwise it returns
`{column: {"status", "window", "window_transform", "polygon_mask", "error"?}}`.

## Command line

The Step 05 script and the full workflow (`ras2fim-2d.py`) accept the same options and pass
them to Step 05:

```bash
python src/ras2fim-2d.py -i SRC -o OUT -c src/config_global.ini -f "(1,1,1)" -s "(5,5)" \
    --mapping-engine rust --rust-binary rust/step05/target/release/ras2fim-batch \
    --mapping-threads 8 --polygon reach.gpkg --polygon-layer reach
```

`--mapping-engine python|rust`, `--rust-binary FILE`, `--mapping-threads N`, `--rust-cache-mib`,
`--rust-profile-mib`, `--rust-profiles-per-request`, `--rust-max-inflight-profiles N` (1 or more,
default 1), `--bbox XMIN YMIN XMAX YMAX` or
`--polygon FILE` (mutually exclusive), `--polygon-layer`, `--aoi-crs`.

## Area-of-interest rules

An AOI selects output pixels. It never changes sample preparation, triangulation, query axes,
the wet mask, terrain sampling, the output grid origin or the reprojection, so an AOI result
is a crop of the full-map result for both engines.

- **CRS.** A bbox without `aoi_crs` is in the Step 05 output CRS (`[06_output_products]
  str_output_crs`, EPSG:3857 by default). A polygon file must carry a CRS or be given
  `aoi_crs`; a conflicting file CRS and `aoi_crs` is an error. Shapely geometry requires
  `aoi_crs`.
- **Validity.** Polygon features must be non-empty, valid `Polygon`/`MultiPolygon` geometry with
  finite coordinates; features are unioned. Holes and multiple parts are supported.
- **Window.** On each profile's final 09/10 grid, the window holds the pixels whose centre lies
  in `[xmin, xmax) x [ymin, ymax)` of the AOI bounds (in the grid CRS). The window origin is a
  whole number of pixels from the full grid origin.
- **Mask.** For a polygon, window pixels whose centre is outside the polygon (or inside a hole)
  are set to NaN (rasterio `geometry_mask`, `all_touched=False`). A bbox in a CRS other than
  the output CRS is densified, reprojected and treated as a polygon.
- **Python engine.** Runs the original full map, then clips 09/10 to the window/mask and
  rebuilds 11 from the clipped 09. This is the AOI baseline.
- **Rust engine.** Receives the same window and mask; the engine evaluates only the source
  pixels that the AOI's output pixels use after a full-grid reprojection lookup.
- **Empty result.** If the window is empty or every pixel is masked, the profile's status is
  `empty_intersection` and no 09/10/11 files are written. A non-empty
  AOI with no depth in (0, 10000] writes 09/10, skips 11 and reports `no_flooded_pixels`.
- Downstream steps (06 onward) look for 09/10/11 by name and see only the AOI products.

## Outputs and failure policy

- 09/10: float32, LZW, NaN nodata, output CRS, window (or full-grid) transform. Pixel values,
  grid and nodata follow the original products; internal TIFF block layout may differ.
- `05_mapping_result.json`: engine, AOI, threads and per-column status (rust or AOI runs only).
- `rust_mapping_report.json`: the engine's `report.json` for every batch (cache, evaluated
  source pixels, stage timings), the terrain EPSG and CRS identity.
- **Rust failures are not silent.** The engine's `report.json` is required: a missing or
  malformed report, a wrong schema, a status that disagrees with the exit code (0 = complete,
  2 = failed) or a profile list that differs from the request fails the whole call. A profile is
  failed when the report does not mark it complete or its unit is missing, malformed, has the
  wrong schema or profile/AOI identity, is not `complete`, has a non-integer window, a
  non-finite or wrong window transform, missing or wrong-size `.f32` files, finite counts
  that disagree with `unit.json`, or finite values outside the polygon mask. Good profiles are
  still written, the failed ones get no products, and Step 05 then raises `RustMappingError`,
  so the script and `ras2fim-2d.py` exit non-zero. An invalid request (engine exit 1) raises
  immediately. Python AOI runs also raise for failed profiles. The unchanged
  Python path without an AOI keeps its original error behavior.

## Terrain CRS identity

The adapter compares the engine's `terrain-info` (size, geotransform, CRS) with rasterio. When
the engine reports an EPSG code it must equal rasterio's. When it reports only WKT, the WKT is
parsed with pyproj and must equal the terrain CRS; the engine is then given that WKT hash.
If neither check is possible, set `str_rust_terrain_crs_fnv1a64` to the `crs_wkt_fnv1a64` value
from `ras2fim-batch terrain-info terrain.tif` after verifying the CRS yourself.

## Memory and batching

Profiles are processed in batches of `rust_profiles_per_request`. Each batch's prepared inputs
are written to a temporary folder in the output directory and removed afterwards; the full-grid
wet mask is released after it is written, so Python keeps only grid metadata and AOI masks.
The engine's decoded-terrain cache is shared by the profiles of one call and starts empty for
the next call; raising `rust_profiles_per_request` increases reuse and the size of each request.
Python holds the returned depth/WSE arrays for one batch, so larger batches can
also increase Python memory use.
The decoded cache uses `rust_max_cache_mib`; `rust_max_profile_mib` controls
planned-allocation admission before profile reads. The plan includes estimates
and is not a complete accounting of live allocations or a process RSS cap.
Within a batch, `rust_max_inflight_profiles` permits concurrent profiles (default 1).
Profile and pixel work share the configured Rayon pool and terrain cache; raising
the profile limit does not create a separate thread pool for each profile. Memory
plans and observed concurrency are recorded in the native report. The profile
budget is per profile, so increasing concurrency can increase total memory use.
The cache must also accommodate the configured threads times one decoded terrain
block. Large strips can require a larger cache or fewer threads.

## Tests

```bash
uv run pytest tests/test_mapping_aoi.py tests/test_rust_mapping.py
```

Synthetic tests cover default dispatch, configuration, CLI, AOI rules, the shared preparation,
the derived grids against the original products, the engine input contract and adapter failure
handling. Real-data tests run only when the external public fixture and binary are supplied:
`RAS2FIM_PUBLIC_SAMPLE` (results.gpkg, sample.p01.hdf, terrain.tif),
`RAS2FIM_STEP05_ORACLE_FLOWS` (upstream Step 05 captures) and `RAS2FIM_RUST_BINARY`. Without
them those tests are reported as skipped.

## Upstream test branch

This contribution is based on `andycarter-pe/ras2fim-2d` main at `3146f647`.
The default Python engine retains upstream's GDAL subtraction and WSE-addition
operations. GDAL Python imports occur when those functions are used, allowing
the optional native engine to run without importing `osgeo`. The upstream
conda environment is retained. CI tests use a separate modern qualification
environment (Python 3.13 and SciPy 1.16.3); this does not qualify every older
SciPy version in existing installations. Install the normal project dependencies
in addition to the engine wheel. The native engine does not replace HEC-RAS
model execution or the other workflow steps.
