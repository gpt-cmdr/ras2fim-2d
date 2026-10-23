"""Exercise an installed platform wheel with a small real GeoTIFF and Step05 request.

Synthetic packaging smoke test; not real-model numerical qualification.
Run explicitly: python tests/smoke_engine_wheel.py.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rust_mapping import Step05Grids, fn_write_profile_inputs


def main():
    engine = shutil.which("ras2fim-batch")
    assert engine, "wheel did not install ras2fim-batch on PATH"
    method = json.loads(subprocess.check_output([engine, "method"], text=True))
    assert "step05_r2" in method
    with tempfile.TemporaryDirectory(prefix="ras2fim-wheel-") as tmp:
        root = Path(tmp)
        terrain = root / "terrain.tif"
        with rasterio.open(terrain, "w", driver="GTiff", width=8, height=8,
                           count=1, dtype="float32", crs="EPSG:3857",
                           transform=Affine(1, 0, 0, 0, -1, 8)) as ds:
            ds.write(np.full((8, 8), 100, dtype=np.float32), 1)
        prep = SimpleNamespace(x=np.array([0., 7., 0., 7.]),
                               y=np.array([0., 0., 7., 7.]), z=np.full(4, 105.),
                               x_axis=np.arange(1., 4.), y_axis=np.arange(1., 4.))
        grids = Step05Grids(3, 3, Affine(1, 0, 1, 0, 1, 1),
                            np.ones((3, 3), dtype=bool), Affine(1, 0, 1, 0, -1, 4),
                            3, 3, "EPSG:3857")
        inputs = root / "inputs"
        fn_write_profile_inputs(str(inputs), "wsel_a", prep, grids)
        mask = root / "mask.u8"
        mask.write_bytes(bytes([1, 0, 1, 1]))
        aois = [{"id": "full"}, {"id": "bbox", "window": [1, 1, 2, 2]},
                {"id": "polygon", "window": [1, 1, 2, 2], "mask_u8": str(mask)}]
        request = {"schema": "ras2fim-step05-request/1",
                   "terrain": {"path": str(terrain), "epsg": 3857,
                               "crs_identity": {"kind": "epsg"}},
                   "source_crs": {"kind": "terrain"},
                   "budgets": {"threads": 2, "max_cache_bytes": 4 << 20,
                               "max_profile_bytes": 32 << 20, "max_inflight_profiles": 2},
                   "interpolation": {"selection": "auto"},
                   "nudge": {"dst_epsg": 3857, "error_threshold": 0.125,
                             "memory_limit_bytes": 67108864., "method": "lookup"},
                   "profiles": [{"id": name, "inputs_dir": str(inputs),
                                 "dest_grid": {"geotransform": [1., 1., 0., 4., 0., -1.],
                                               "width": 3, "height": 3}, "aois": aois}
                                for name in ["wsel_a", "wsel_b"]]}
        rp = root / "request.json"
        rp.write_text(json.dumps(request))
        out = root / "out"
        subprocess.run([engine, "step05", str(rp), str(out)], check=True)
        report = json.loads((out / "report.json").read_text())
        assert report["status"] == "complete", report
        for name in ["wsel_a", "wsel_b"]:
            for product, expected in [("depth.f32", 5.), ("wse.f32", 105.)]:
                base = out / "units" / name
                full = np.fromfile(base / "full" / product, dtype="<f4").reshape(3, 3)
                np.testing.assert_array_equal(full, np.full((3, 3), expected))
                for aoi in ["bbox", "polygon"]:
                    got = np.fromfile(base / aoi / product, dtype="<f4").reshape(2, 2)
                    want = full[1:3, 1:3].copy()
                    if aoi == "polygon":
                        want[0, 1] = np.nan
                    np.testing.assert_array_equal(got, want)
        print("Installed wheel: Step05 shared-terrain batch, bbox and polygon mask PASS")


if __name__ == "__main__":
    main()
