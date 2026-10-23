# Step05 mapping comparison

The optional engine implements the ras2fim-2D catalog mapping method. This
comparison does not establish equivalence to RAS Mapper stored maps or assess
the hydraulic accuracy of the source model.

## Method checks

The development oracle ran the original Step05 sequence on a public 2D example
with 19 water-surface profiles. It retained the global sample order, SciPy
triangulation and transforms, query axes, wet masks, intermediate rasters,
final depth and water-surface rasters, and floodplain products.

The Rust stage comparison used SciPy 1.18.1 prepared inputs and terrio 0.4.2.
For every checked raster, the criterion was identical nodata masks and identical
bits for each finite value. No numerical tolerance was used.

| Configuration | Raster comparisons | Result |
|---|---:|---|
| Full mapping, intermediate 03 and 05–08 plus final 09/10 | 133 | Exact |
| Full mapping with forced serial SciPy walk | 133 | Exact |
| Full, window, clipped edge, single pixel, bbox, polygon and outside AOIs | 304 | Exact |
| AOIs without requesting the full output | 266 | Exact |
| AOIs with bounded lookup retention | 266 | Exact |
| Full and AOIs with one worker thread | 304 | Exact |

AOI comparisons used windows of the original full destination grid, with a
pixel-center mask for polygons. They did not subset or rebuild the global
triangulation. One profile contained a degenerate simplex and used the serial
SciPy walk. The other profiles used guarded parallel interpolation.

Negative controls confirmed that the comparison detects changes to the method.
For profile 10, disabling reprojection approximation changed 702 mask pixels
and 9,536 finite values in the final depth raster. Reducing warp memory to
1 MiB changed 688 mask pixels and 9,369 finite values. Replacing the exact
NumPy query axes with `x0 + i * resolution` changed 233,087 finite values in
intermediate 03. These controls are not alternative engine settings qualified
for equivalence.

## Failure and resource checks

The shared terrain-cache and output code has regression tests for loader
panics, concurrent waiters, write failures, one-tile buffer limits, pre-read
budget rejection, and case-insensitive or reserved Windows output names.
Valid diagnostic-session outputs remained byte-identical after these fixes.
Partial data require a complete unit marker before a caller may accept them.

The integrated source passed 37 Rust unit tests and 12 tests that execute the
compiled engine, including the external public-fixture tests, with Rust 1.99.1
on Linux/WSL. Formatting and clippy checks passed with warnings treated as errors.
The concurrency test compared three valid profiles and one deliberately invalid
profile at in-flight limits of one and two: valid products and unit metadata
were byte-identical, output order was stable, the invalid profile wrote no units,
and observed active profiles stayed within the requested limit.

Decoded cache limits and planned allocation admission are enforced separately.
Memory plans include estimates for decoder and spatial-index working storage;
they are not a process RSS limit. Allocator behavior, thread stacks, codec
state and operating-system caches remain outside a hard total-memory bound.

## Coverage limits

The real-data comparison covers x86-64 Linux/WSL, one model, one single-band float32 GeoTIFF
terrain and the EPSG:2277 to EPSG:3857 mapping path. It does not establish
support for VRT terrain, multiple meshes, every codec or CRS pair, coplanar
triangulations, or real source grids exceeding 2²⁴ pixels. Multi-pass lookup
encoding above that size has unit tests. Wet cells over the terrain's nodata
sentinel were absent from this fixture. Windows installed-wheel smoke tests are separate; real-model equality on
Windows has not been qualified.

Sample preparation, SciPy triangulation and transforms, wet-union
rasterization, output-grid derivation, GeoTIFF writing and floodplain
polygonization remain Python operations. The Rust engine does not link GDAL;
the Python raster stack still uses its raster libraries.
The application-pinned SciPy 1.16.3 path was additionally compared with the
target-main Python method in complete-call benchmarks: all 19 profiles passed
exact raster comparisons in full, bbox and polygon scenarios.

The application submits one AOI per profile. Direct engine requests with several
AOIs can retain an earlier completed AOI if a later AOI fails; callers must check
the profile report. Ordinary typed profile errors are isolated, while an
unexpected native panic aborts the session and is rejected by the adapter.

Small AOIs reduce source-pixel interpolation and terrain evaluation. Each
profile still requires a geometry-only full-grid lookup warp, and the serial
interpolation fallback visits the full query grid. End-to-end throughput must
therefore be measured with the [complete-call benchmark](../benchmarks/README.md),
including preparation and output writing, rather than inferred from kernel
timings or the number of terrain tiles read.

## Platform wheel checks

The retained Linux x86-64 and Windows x86-64 wheels in `wheels/` were built
from identical engine source with Rust 1.99.1 and maturin 1.15.0. Both passed
installation and synthetic Step05 mapping tests (shared profiles, full maps,
bbox crop and polygon mask) in the public build recorded in `wheels/BUILD.json`.
Their checksums, bundled notices and source fingerprints are checked separately.
The Windows wheel also passed a local Commander public-model full/bbox/
multipart-polygon-with-hole crop test. That narrower check supplements the
retained Linux comparisons; it is not an all-model Windows equivalence claim.
