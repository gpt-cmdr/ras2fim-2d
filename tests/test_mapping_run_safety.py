"""Regression checks for stale products and interrupted optional mapping runs."""
import json
import subprocess

import pytest

from test_rust_mapping import (
    _RecordingPool, _emulated_engine, _fake_binary, python_full, synthetic,
    rust_mapping, step05,
)


@pytest.mark.parametrize('engine', ['python', 'rust'])
def test_optional_run_rejects_stale_outputs_without_touching_them(synthetic, tmp_path, engine):
    output = tmp_path / 'out'
    output.mkdir()
    old = output / '09_wsel_max_1_depth_nudge.tif'
    old.write_bytes(b'previous map')
    options = dict(mapping_engine=engine, bbox=[0, 0, 10, 10])
    if engine == 'rust':
        options['rust_binary'] = _fake_binary(tmp_path)
    with pytest.raises(FileExistsError, match='fresh outputs'):
        step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'],
            synthetic['dem'], synthetic['gpkg'], str(output), False, **options)
    assert old.read_bytes() == b'previous map'
    assert not (output / '05_mapping_result.json').exists()


def test_python_aoi_worker_failure_is_not_success(synthetic, tmp_path, monkeypatch):
    monkeypatch.setattr(step05.mp, 'Pool', _RecordingPool)
    output = tmp_path / 'out'
    with pytest.raises(rust_mapping.RustMappingError, match='python AOI mapping failed'):
        step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'],
            synthetic['dem'], synthetic['gpkg'], str(output), False, bbox=[0, 0, 10, 10])
    manifest = json.loads((output / '05_mapping_result.json').read_text())
    assert manifest['status'] == 'failed'
    assert all(row['status'] == 'failed' for row in manifest['columns'].values())
    assert not step05._fn_mapping_products(str(output))


def test_later_rust_batch_abort_rolls_back_products_and_marks_reports(
        synthetic, python_full, tmp_path, monkeypatch):
    emulate = _emulated_engine(python_full, {})
    calls = 0

    def run(arguments, label):
        nonlocal calls
        if arguments[1] == 'step05':
            calls += 1
            if calls == 2:
                return subprocess.CompletedProcess(arguments, 1, '', 'injected later-batch abort')
        return emulate(arguments, label)

    monkeypatch.setattr(rust_mapping, 'fn_run_binary', run)
    output = tmp_path / 'out'
    with pytest.raises(rust_mapping.RustMappingError, match='later-batch abort'):
        step05.fn_generate_depth_wsel_rasters(synthetic['config'], synthetic['hdf'],
            synthetic['dem'], synthetic['gpkg'], str(output), False,
            mapping_engine='rust', rust_binary=_fake_binary(tmp_path), rust_profiles_per_request=1)
    assert calls == 2
    assert not step05._fn_mapping_products(str(output))
    for name in ('05_mapping_result.json', 'rust_mapping_report.json'):
        report = json.loads((output / name).read_text())
        assert report['status'] == 'aborted'
        assert 'later-batch abort' in report['error']
    assert not list(output.glob('*.partial'))
