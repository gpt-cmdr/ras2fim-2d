# Development performance observations

One public 2D model was measured on 2026-10-23 with four logical CPUs on an
Intel Core i9-11900KF workstation, x86-64 WSL/Linux and native Linux **tmpfs**
working storage. The host was shared and load was uncontrolled. Five
interleaved repetitions followed one excluded warm-up pair per scenario.
Times include process startup, Python preparation, mapping, GeoTIFF writing
and floodplain generation. They are observations of this setup, not a general
throughput guarantee or physical-disk benchmark.

| Profiles | Output | Python median (s) | Python IQR (s) | Rust median (s) | Rust IQR (s) | Rust/Python throughput |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Full | 3.027 | 0.107 | 1.921 | 0.061 | 1.58× |
| 1 | Bbox | 3.322 | 0.198 | 1.757 | 0.051 | 1.89× |
| 1 | Polygon | 3.018 | 0.098 | 1.660 | 0.306 | 1.82× |
| 19 | Full | 15.955 | 1.231 | 23.787 | 2.280 | 0.67× |
| 19 | Bbox | 20.181 | 2.789 | 12.502 | 3.116 | 1.61× |
| 19 | Polygon | 16.588 | 0.836 | 11.156 | 0.740 | 1.49× |

A ratio above one means higher Rust throughput. The full 19-profile case was
slower in this run; it must not be omitted from a speed claim. Competing host
work was not measured separately, so the difference cannot be attributed to it.
The Rust path prepares and writes profiles serially in Python, while the
reference distributes its complete per-profile pipeline across four processes.
Native profile concurrency does not parallelize those Python stages.

Settings were the Python default engine with a four-process pool, versus Rust
with four Rayon threads, four profiles per request, one profile in flight and
a 64 MiB decoded-terrain cache. Optional native profile concurrency was tested
for output equality but was not benchmarked. The constrained Python reference
mapped the full domain before clipping; Rust received the constraint first.
The polygon included multiple parts and a hole.

Every timed pair passed exact finite float32, nodata-mask, grid and CRS
comparisons. Floodplain symmetric-difference area was at most 1e-6 squared
output-CRS units. The method's global samples and output lattice were retained.

The environment used Python 3.13.17, NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3,
rasterio 1.4.4 (GDAL 3.10.3), GeoPandas 1.1.1, Shapely 2.1.2 and pyproj 3.7.2.
The Rust executable SHA-256 was
`1f53abd2ff8c0d7ac0e8a400d6b09ad2eeaa8c4cb6a198d3216b51411e809ac9`.
These measurements preceded the final Python output-lifecycle hardening;
that wrapper was not retimed. The numerical Rust engine is unchanged.

Reproduce with the [benchmark harness](README.md). It records raw timings,
CPU time, sampled process-tree RSS/PSS, versions and input/source hashes.
Generated model outputs and workstation reports are retained outside Git.
