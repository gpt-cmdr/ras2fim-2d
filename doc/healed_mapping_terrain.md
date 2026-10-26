# Healed mapping terrain on Python 3.11

Steps 5–7 can use the **same** raise-only road restoration as FIM Commander.
Step 5 directly calls `fim_commander.inundation.healed_terrain.build_healed` in
this process. The companion uses Terrio 0.5.0 for raster reads, reprojection and
writes. No healing algorithm is copied into this repository, and no separate
Python 3.8 compatibility process is needed.

## Install

The active Dockerfile and conda environment now select Python 3.11. The old
`Dockerfile_old` is retained as historical material. The Python 3.8-only scientific
pins are replaced with Python 3.11-compatible ranges; native GDAL/Rasterio/PROJ
libraries still come from conda-forge. Terrio is an independent Rust backend.
Docker builds the checked-out source, so a feature-branch build includes its changes.

In the prepared Python 3.11 geo environment, install the optional companion:

```shell
python -m pip install --no-deps -r requirements-healed-terrain.txt
```

FIM Commander is a **separately licensed companion currently requiring authorized
repository access**. The pin names the Terrio/raise-only implementation; the core
change is tracked in [FIM Commander PR 408](https://github.com/CLB-Engineering-Corporation/fim-commander/pull/408).
Do not put GitHub credentials in Dockerfiles, build arguments or tracked files.
An authorized container build can instead install a prebuilt companion wheel with
`python -m pip install --no-deps /wheels/fim_commander.whl` (use its actual wheel filename).
No companion source or redistribution permission is granted by this repository's BSD license.
The default mapping mode does not import or require the companion.

## Configure

```ini
[05_raster_engine]
bln_heal_terrain = True
str_rem_tiles = /data/txdot-rem
str_terrain_vertical_unit = ftUS
flt_datum_tolerance_ft = 0.5
```

`str_rem_tiles` is a local folder, relative to the config file or absolute, containing
`<district>/dsm/<section>.tif`. Preserve this layout when mirroring the public
[TxDOT Road Elevation Model](https://web.corral.tacc.utexas.edu/nfiedata/road3d/).
The stage uses the 3 m **maintenance-section DSMs**, not the district `dsm_cog.tif`
mosaics, COPC point clouds, HAND products or bare-earth lidar DEMs. The June 2025
district COGs are resampled to EPSG:3857 with 4.777 m map pixels (about 4.1 m on
the ground near latitude 30 degrees); retain the native 3 m section grid for healing. A missing or
empty library is an error; a valid library with no road in this model's footprint
produces unchanged model terrain through the same builder.

The input model DEM must be a single-band north-up square projected raster with
NoData, and terrain modifications must already have been rasterized into it. An
HDF terrain definition or a raw lidar tile is not a substitute for that export.
The configured height unit must match the model and WSEL units. Only `ftUS` and
`ft` are accepted because this workflow's existing NetCDF packing is in feet.
The REM heights are metres; assumed NAVD88 is checked by the companion's **raw,
signed** road-core datum diagnostic. Failed datum QA stops mapping.

Blank/false `bln_heal_terrain` retains the model terrain and existing CLI signatures.
This is mapping-only: no terrain is supplied to HEC-RAS preprocessing or its solver.

## Results and resume behavior

The first Step 5 run builds once under
`05_large_terrain/_healed_terrain/<full input SHA-256>/`. Each folder contains:

- `healed.tif`: only raises where the road sample exceeds valid model terrain;
- `difference.tif` and `validity.tif`: applied changes and the raised-cell mask;
- `qa.json`: unchanged companion QA, including signed candidate differences;
- `mapping_terrain.json`: input and output hashes, units, method/backend identity,
  configuration and `approval_state: not_approved`.

Nearest footprint admission, bilinear sampling with nearest fallback, first-tile
precedence and model NoData preservation are owned by the companion. Work uses
its padded source windows and bounded output spill defaults. Scratch space must
also accommodate the uncompressed rasters. Inputs are read-only. Changed inputs
or parameters create a different cache folder; modified cached outputs fail validation.

Every Step 5 raster folder records the selected cache identity. Step 6 propagates
it to `06_simple_rasters/mapping_terrain.json`. Step 7 embeds the same healed terrain
in NetCDF, records `terrain_kind`, `mapping_terrain_sha256` and an explicit unapproved
state in global attributes, checks that identity against Step 6 and never computes a missing variant.
Changing terrain mode, REM files, datum tolerance, model terrain or units requires a
**new output folder and rerunning Steps 5–7**. Existing unrecorded rasters cannot be
silently relabeled as healed products. Rejected datum attempts retain `failed-*`
folders with diagnostic rasters and QA, without a reusable success marker.

Review original terrain, healed terrain, applied differences and raw signed datum
checks around bridges, narrow roads and corridor edges. Raise-only prevents
lowering; it does not prove every raise is correct or constitute engineering approval.

## Validation

```shell
python -m pip install pytest
python -m pytest
```

Tests use synthetic projected rasters and do not run HEC-RAS or download data.
The actual companion integration cases run when the authorized dependency is
installed; otherwise those cases are explicitly skipped. The remaining cases check
default behavior, entrypoint wiring, provenance handoffs, and Python 3.11 syntax.
Full hydraulic regression and a rebuilt container remain deployment qualification.
