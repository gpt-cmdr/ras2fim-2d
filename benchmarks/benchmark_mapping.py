"""Compare complete Step05 engines on equal inputs, CPU affinity and storage.

Run on Linux with the application's pinned dependencies.
External model inputs stay immutable. No HEC-RAS solver is executed.
The optional --reference-script can point at the original pre-change Step05.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def constrain_reference(module, output, columns, aoi):
    """Crop original full products on their global pixel-center lattice."""
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window
    from shapely.geometry import box

    if aoi["kind"] == "full":
        return
    geometry = (gpd.GeoSeries([box(*aoi["bounds"])], crs=aoi["crs"])
                if aoi["kind"] == "bbox" else gpd.read_file(aoi["path"]).geometry)
    for column in columns:
        for filename in (f"09_{column}_depth_nudge.tif", f"10_{column}_wsel_nudge.tif"):
            path = output / filename
            with rasterio.open(path) as source:
                projected = geometry.to_crs(source.crs)
                xmin, ymin, xmax, ymax = projected.total_bounds
                xs = source.transform.c + (np.arange(source.width) + 0.5) * source.transform.a
                ys = source.transform.f + (np.arange(source.height) + 0.5) * source.transform.e
                cols = np.flatnonzero((xs >= xmin) & (xs < xmax))
                rows = np.flatnonzero((ys >= ymin) & (ys < ymax))
                if not len(cols) or not len(rows):
                    raise ValueError("Benchmark AOI must intersect every selected full grid")
                window = Window(int(cols[0]), int(rows[0]), len(cols), len(rows))
                values = source.read(1, window=window)
                transform = source.window_transform(window)
                inside = geometry_mask(projected, values.shape, transform, invert=True, all_touched=False)
                values[~inside] = np.nan
                profile = source.profile
                profile.update(width=len(cols), height=len(rows), transform=transform)
            temporary = path.with_suffix(".benchmark-partial.tif")
            with rasterio.open(temporary, "w", **profile) as destination:
                destination.write(values, 1)
            os.replace(temporary, path)
        floodplain = module.fn_make_floodplain_gdf(str(output / f"09_{column}_depth_nudge.tif"), 0, 10000)
        floodplain.to_file(output / f"11_{column}_floodplain_ar.gpkg")


def worker(request):
    """Same interpreter/process boundary for each complete engine call."""
    import multiprocessing as mp

    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "GDAL_NUM_THREADS"):
        os.environ[key] = "1"
    os.sched_setaffinity(0, request["cpu_ids"])
    # Python3.13 Linux default; make it explicit on newer interpreters too.
    mp.set_start_method("fork", force=True)
    script = Path(request["reference"] if request["engine"] == "python" else request["script"])
    sys.path.insert(0, str(script.parent))
    module = load_module(script, "step05_benchmark_reference" if request["engine"] == "python" else "step05_benchmark_candidate")
    output = Path(request["output"])
    output.mkdir()
    if request["engine"] == "python":
        module.mp.cpu_count = lambda: len(request["cpu_ids"]) + 1
    start = time.perf_counter()
    common = [request[k] for k in ("config", "hdf", "terrain", "gpkg")] + [str(output), False]
    if request["engine"] == "python":
        module.fn_generate_depth_wsel_rasters(*common)
        constrain_reference(module, output, request["columns"], request["aoi"])
    else:
        options = dict(mapping_engine="rust", mapping_threads=len(request["cpu_ids"]),
                       rust_binary=request["rust_binary"],
                       rust_max_inflight_profiles=request.get("rust_inflight", 1),
                       rust_profiles_per_request=request.get("rust_profiles_per_request", 4))
        aoi = request["aoi"]
        if aoi["kind"] == "bbox":
            options.update(bbox=aoi["bounds"], aoi_crs=aoi["crs"])
        elif aoi["kind"] == "polygon":
            options["polygon"] = aoi["path"]
        module.fn_generate_depth_wsel_rasters(*common, **options)
    seconds = time.perf_counter() - start
    for column in request["columns"]:
        for name in (f"09_{column}_depth_nudge.tif", f"10_{column}_wsel_nudge.tif", f"11_{column}_floodplain_ar.gpkg"):
            if not (output / name).exists():
                raise RuntimeError(f"Required product missing: {name}")
    Path(request["worker_report"]).write_text(json.dumps({"mapping_seconds": seconds}))


def process_tree(pid):
    seen, pending = set(), [pid]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            pending.extend(int(value) for value in Path(f"/proc/{current}/task/{current}/children").read_text().split())
        except (OSError, ValueError):
            pass
    return seen


def measure(command, log, request):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    rss_peak = pss_peak = 0
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            rss = pss = 0
            for pid in process_tree(process.pid):
                try:
                    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                        if line.startswith("Rss:"):
                            rss += int(line.split()[1]) * 1024
                        elif line.startswith("Pss:"):
                            pss += int(line.split()[1]) * 1024
                except (OSError, ValueError):
                    pass
            rss_peak, pss_peak = max(rss_peak, rss), max(pss_peak, pss)
            time.sleep(0.05)
        if process.returncode:
            detail = log.read_text(errors="replace")[-4000:]
            raise RuntimeError(f"Engine failed with exit {process.returncode}:\n{detail}")
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(process_seconds=time.perf_counter() - start,
                process_tree_cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                sampled_process_tree_peak_rss_bytes=rss_peak, sampled_process_tree_peak_pss_bytes=pss_peak,
                **json.loads(Path(request["worker_report"]).read_text()))


def compare_products(reference, candidate, columns):
    import geopandas as gpd
    import numpy as np
    import rasterio

    checked = []
    for column in columns:
        for name in (f"09_{column}_depth_nudge.tif", f"10_{column}_wsel_nudge.tif"):
            with rasterio.open(reference / name) as left, rasterio.open(candidate / name) as right:
                if (left.shape, left.crs, left.transform) != (right.shape, right.crs, right.transform):
                    raise AssertionError(f"Grid differs: {name}")
                x, y = left.read(1), right.read(1)
                finite = np.isfinite(x)
                if not np.array_equal(finite, np.isfinite(y)):
                    raise AssertionError(f"Nodata mask differs: {name}")
                if x.dtype != np.dtype("f4") or y.dtype != np.dtype("f4") or not np.array_equal(x[finite].view("u4"), y[finite].view("u4")):
                    raise AssertionError(f"Finite float32 bits differ: {name}")
                checked.append(name)
        name = f"11_{column}_floodplain_ar.gpkg"
        x, y = gpd.read_file(reference / name), gpd.read_file(candidate / name)
        if x.crs != y.crs or x.geometry.union_all().symmetric_difference(y.geometry.union_all()).area > 1e-6:
            raise AssertionError(f"Floodplain footprint differs: {name}")
        checked.append(name)
    return len(checked)


def main(args):
    if sys.platform != "linux":
        raise SystemExit("Run on Linux so affinity and process-tree memory sampling are comparable")
    if any(str(Path(path)).startswith("/mnt/") for path in (sys.prefix, sys.base_prefix, __file__)):
        raise ValueError("Stage the Python environment, base interpreter and benchmark script on native Linux storage")
    import geopandas as gpd
    import numpy as np
    import rasterio

    if args.repeats < 3:
        raise ValueError("Use at least three interleaved repetitions")
    if args.rust_inflight < 1 or args.rust_profiles_per_request < 1:
        raise ValueError("Rust concurrency and batch size must be positive")
    if args.output.exists():
        raise FileExistsError("Use a fresh --output file; never reuse a stale successful report")
    cpu_ids = sorted(os.sched_getaffinity(0))[:args.cores]
    if len(cpu_ids) != args.cores:
        raise ValueError("Not enough CPUs in current affinity")
    os.sched_setaffinity(0, cpu_ids)
    hashes = {name: digest(getattr(args, name)) for name in ("hdf", "terrain", "results", "config", "rust_binary", "script", "reference_script")}
    harness_hash = digest(__file__)
    adapter_sources = {name: args.script.parent / name for name in ("rust_mapping.py", "mapping_aoi.py")}
    adapter_hashes = {name: digest(path) for name, path in adapter_sources.items()}
    results = []
    with tempfile.TemporaryDirectory(prefix="ras2fim-map-benchmark-", dir=args.work_root) as temporary:
        root = Path(temporary)
        # Stage code/binary/model data to the same storage before timing.
        for name in ("hdf", "terrain", "results", "config", "rust_binary"):
            destination = root / (name + Path(getattr(args, name)).suffix)
            shutil.copyfile(getattr(args, name), destination)
            if name == "rust_binary":
                destination.chmod(0o755)
        frame = gpd.read_file(root / ("results" + args.results.suffix), layer="02_cells_wsel_ar")
        shutil.copytree(args.script.parent, root / "candidate", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(args.reference_script, root / "reference.py")
        all_columns = sorted((c for c in frame.columns if c.startswith("wsel")), key=lambda c: int(c.rsplit("_", 1)[1]))
        scenarios = [dict(kind="full")]
        if args.bbox:
            scenarios.append(dict(kind="bbox", bounds=args.bbox, crs=args.aoi_crs))
        if args.polygon:
            scenarios.append(dict(kind="polygon", path=str(args.polygon.resolve())))
        for count in args.flow_counts:
            if count > len(all_columns) or count < 1:
                raise ValueError("Flow count outside the input table")
            columns = all_columns[:count]
            gpkg = root / f"flows-{count}.gpkg"
            frame.drop(columns=[c for c in all_columns if c not in columns]).to_file(gpkg, layer="02_cells_wsel_ar", driver="GPKG")
            for scenario in scenarios:
                timings = {"python": [], "rust": []}
                pair_checks = []
                for iteration in range(-1, args.repeats):
                    order = ("python", "rust") if iteration % 2 == 0 else ("rust", "python")
                    outputs = {}
                    for engine in order:
                        key = f"{count}-{scenario['kind']}-{iteration}-{engine}"
                        request = dict(engine=engine, cpu_ids=cpu_ids, script=str(root / "candidate" / args.script.name),
                                       rust_inflight=args.rust_inflight, rust_profiles_per_request=args.rust_profiles_per_request,
                                       reference=str(root / "reference.py"), columns=columns,
                                       gpkg=str(gpkg), config=str(root / ("config" + args.config.suffix)),
                                       hdf=str(root / ("hdf" + args.hdf.suffix)), terrain=str(root / ("terrain" + args.terrain.suffix)),
                                       rust_binary=str(root / ("rust_binary" + args.rust_binary.suffix)),
                                       output=str(root / (key + "-out")), worker_report=str(root / (key + "-report.json")), aoi=scenario)
                        path = root / (key + ".json")
                        path.write_text(json.dumps(request))
                        result = measure([sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(path)], root / (key + ".log"), request)
                        outputs[engine] = Path(request["output"])
                        if iteration >= 0:
                            timings[engine].append(result)
                    pair_checks.append(compare_products(outputs["python"], outputs["rust"], columns))
                    for output in outputs.values():
                        if not output.resolve().is_relative_to(root.resolve()):
                            raise AssertionError("Output cleanup escaped temporary benchmark workspace")
                        shutil.rmtree(output)
                summary = {}
                for engine, runs in timings.items():
                    values = [run["process_seconds"] for run in runs]
                    quartiles = np.quantile(values, [0.25, 0.75]).tolist()
                    summary[engine] = dict(median_seconds=statistics.median(values), quartiles_seconds=quartiles,
                                           iqr_seconds=quartiles[1] - quartiles[0],
                                           flows_per_hour=count * 3600 / statistics.median(values), runs=runs)
                summary["median_speed_ratio"] = summary["python"]["median_seconds"] / summary["rust"]["median_seconds"]
                results.append(dict(flows=count, scenario=scenario["kind"], exact_product_checks_per_pair=pair_checks, **summary))
                print(json.dumps({k: v for k, v in results[-1].items() if k not in ("python", "rust")}), flush=True)
    report = dict(schema="ras2fim-mapping-benchmark/1", source_hashes=hashes, adapter_source_hashes=adapter_hashes,
                  harness_sha256=harness_hash, rust_max_inflight_profiles=args.rust_inflight,
                  rust_profiles_per_request=args.rust_profiles_per_request,
                  python=sys.version.split()[0], rasterio_gdal=rasterio.__gdal_version__, packages={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "rasterio", "geopandas", "shapely", "pyproj")},
                  logical_cpus=args.cores, storage="caller-selected Linux work root; inputs and binary staged before timing",
                  python_environment_storage="Windows-mounted" if str(Path(sys.prefix)).startswith("/mnt/") else "Linux-native",
                  python_base_storage="Windows-mounted" if str(Path(sys.base_prefix)).startswith("/mnt/") else "Linux-native",
                  os_cache="warm-up excluded; OS/storage caches not flushed", multiprocessing="fork", rss_sampling_seconds=0.05,
                  host_load="shared, uncontrolled; interleaved engines", results=results)
    for name in ("hdf", "terrain", "results", "config", "rust_binary", "script", "reference_script"):
        if digest(getattr(args, name)) != hashes[name]:
            raise AssertionError(f"Input changed during benchmark: {name}")
    for name, path in adapter_sources.items():
        if digest(path) != adapter_hashes[name]:
            raise AssertionError(f"Adapter changed during benchmark: {name}")
    if digest(__file__) != harness_hash:
        raise AssertionError("Benchmark harness changed during the run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.output.with_suffix(args.output.suffix + ".partial")
    temporary_report.write_text(json.dumps(report, indent=2))
    os.replace(temporary_report, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    for name in ("hdf", "terrain", "results", "config", "rust-binary", "script", "reference-script", "polygon", "output", "work-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--flow-counts", type=int, nargs="+", default=[1, 19])
    parser.add_argument("--rust-inflight", type=int, default=1)
    parser.add_argument("--rust-profiles-per-request", type=int, default=4)
    parser.add_argument("--bbox", type=float, nargs=4)
    parser.add_argument("--aoi-crs", default="EPSG:3857")
    arguments = parser.parse_args()
    if arguments.worker:
        worker(json.loads(arguments.worker.read_text()))
    else:
        required = ("hdf", "terrain", "results", "config", "rust_binary", "script", "reference_script", "output")
        if any(getattr(arguments, name) is None for name in required):
            parser.error("Specify input paths, both scripts, --rust-binary and --output")
        if arguments.cores < 1 or arguments.cores > 16:
            parser.error("--cores must be 1..16 (legacy backend cap)")
        main(arguments)
