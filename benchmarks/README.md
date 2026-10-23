# Mapping engine performance checks

See [development observations](performance.md) for the measured single-map and
catalog results, including the slower full-domain catalog case and its limits.

`benchmark_mapping.py` compares complete Step05 calls on the same public model inputs,
selected flow columns, CPU affinity and output storage. It includes Python preparation,
Rust process startup, mapping, GeoTIFF writes and floodplain generation. It never runs HEC-RAS.

The Python reference can be an unmodified copy of Step05 from the base revision, supplied
with `--reference-script`. For constrained cases, the reference generates the full products,
then applies the same pixel-center crop and polygon mask. The Rust backend receives the
constraint before mapping. Both must produce identical final grids, finite float32 values,
nodata masks and floodplain footprints before a performance result is accepted.
Raster values and masks use exact comparisons. Floodplain footprints are compared
by symmetric-difference area, at most 1e-6 squared output-CRS units; this is a
geometry comparison criterion, not a depth tolerance or GeoPackage byte comparison.

Run on Linux, with the pinned application dependencies and a release build of the
optional Rust engine. The reference uses rasterio's GDAL runtime; separate `osgeo`
bindings are unnecessary. Put the Python environment and
temporary work root on native Linux storage. In particular, WSL `/mnt/c` filesystem overhead
can dominate processing and must not be mistaken for an engine difference.
The harness rejects a Windows-mounted environment, base interpreter or harness
script. If the checkout is on `/mnt/c`, copy this script to native Linux storage
and run that copy, while supplying the original source paths as arguments.

```sh
python benchmarks/benchmark_mapping.py \
  --hdf /data/public-model.p01.hdf \
  --terrain /data/terrain.tif \
  --results /data/results.gpkg \
  --config src/config_global.ini \
  --script src/generate_depth_wsel_rasters_mp_05.py \
  --reference-script /data/base-step05.py \
  --rust-binary rust/step05/target/release/ras2fim-batch \
  --flow-counts 1 19 --cores 4 --repeats 5 \
  --bbox XMIN YMIN XMAX YMAX --aoi-crs EPSG:3857 \
  --polygon /data/reach.gpkg \
  --work-root /data/benchmark-scratch \
  --output benchmarks/runs/results.json
```

Omit `--bbox` and `--polygon` to measure only full-domain mapping. In this benchmark command
they select additional independent workloads; the mapping API itself accepts one constraint
at a time. A polygon file must carry a CRS. Models with fewer than 19 flows need appropriate
`--flow-counts` values.

The report retains interleaved raw measurements, median and interquartile range, flows/hour,
CPU time and process-tree memory samples. Each workload excludes one warm-up pair. RSS is
summed across the process tree and can count shared pages repeatedly; summed PSS is also
reported. Memory sampling occurs every 50 milliseconds and is not a kernel peak measurement.
The reference pool uses Linux `fork`, the default on the project's Python 3.13 runtime.
`--rust-inflight` (default 1) and `--rust-profiles-per-request` (default 4)
select explicit native concurrency and batch settings for separate experiments.
Increasing native concurrency does not parallelize Python preparation or output.

Inputs and outputs are staged in the temporary work root before timing. OS/storage caches
are not flushed, and host load is uncontrolled; disclose both when publishing results.
The report records package versions and source/binary/input hashes. Do not publish generated
reports without checking them for project-specific metadata. Kernel timings or terrain tile
byte reductions alone do not establish an end-to-end speedup.
