# Container validation

The delivered build includes the standalone Terrio healer and both public Rust
mapping methods. Commander is pinned to merged feature commit
`777c7746ceb35fc3b1c29d3718935c9013ae009c`, with fast-map 0.8.0 and fast-HDF 0.2.3.

The delivery package contains `delivery.json`, `SHA256SUMS`, exact source archive,
Docker-save archive, build/qualification logs and package inventories. These
post-build files identify the executable image and source snapshot independently
of the repository's version metadata. Large REM datasets remain mounted inputs.

Qualification procedure:

- Run all standalone tests on Windows and inside the actual Python 3.11 Linux image.
- Check Terrio/Rasterio exact synthetic sampling, raise-only/NoData behavior,
  datum failure, cache integrity, and Steps5-7 terrain/method provenance.
- Check real Rust Step05 arithmetic and native complete-profile/window preparation,
  stale input rejection, ghost/dry conventions, native grid and catalog delivery CRS.
- Import Steps5-7, run pipeline/healer CLI help and locate `nccopy` and Rust executable.
- Run native Step05 on a task-owned single-mesh Davis DS Channel derivative,
  preserving source model files. Its plan title is adapted to two discharge windows;
  it is an integration fixture, not an accepted hydraulic catalog.
- Inspect retained native rasters, output manifests, supplied profile hashes and
  delivered raster/vector CRS. Verify archive SHA-256 after transfer.

Full real-model catalog regression and H&H acceptance remain separate work.
Window aggregates and healed terrain do not inherit a universal native-parity or
speedup claim. `fast_step05` and `rasmapper_sloping` are explicitly different methods.
