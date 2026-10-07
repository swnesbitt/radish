"""SMN / SINARAME (Argentina) RMA BUFR backend.

Real-data tests run against the RELAMPAGO RMA1 volumes published in
open-radar-data (``data/ARBUFR/bufr_relampago/00/{0457,0620}``): set
``RADISH_ARGENTINA_BUFR_DIR`` to a directory holding those two volume
directories (the files can stay gzip-wrapped). Without it they skip.

* ``0457`` — task 0301 subtask 02: 3 sweeps (0.51–2.26°), 120 m gates
* ``0620`` — task 0301 subtask 01: 15 sweeps (0.51–29.99°), 450 m gates

Each volume has 12 moment files (CM, DBZH, DBZV, KDP, PHIDP, RHOHV, TDR,
TH, TV, VRAD, WRAD, ZDR).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import radish

MOMENTS = {"CM", "DBZH", "DBZV", "KDP", "PHIDP", "RHOHV", "TDR", "DBTH", "DBTV", "VRADH", "WRADH", "ZDR"}


def _root() -> Path:
    root = os.environ.get("RADISH_ARGENTINA_BUFR_DIR")
    if not root or not Path(root).is_dir():
        pytest.skip("RADISH_ARGENTINA_BUFR_DIR not set (RMA1 RELAMPAGO volumes 0457/0620)")
    return Path(root)


def _vol(name: str) -> Path:
    hits = [p for p in _root().rglob(name) if p.is_dir()]
    if not hits:
        pytest.skip(f"volume directory {name} not under RADISH_ARGENTINA_BUFR_DIR")
    return hits[0]


@pytest.fixture(scope="module")
def vol_0457() -> Path:
    return _vol("0457")


@pytest.fixture(scope="module")
def vol_0620() -> Path:
    return _vol("0620")


def test_directory_autodetects_and_merges_moments(vol_0457):
    assert radish.detect_backend(vol_0457) == "argentina_bufr"
    dt = radish.open_datatree(vol_0457)
    root = dt.ds
    assert root.attrs["instrument_name"] == "RMA1"
    assert root.attrs["scan_task"] == "0301" and root.attrs["scan_subtask"] == "02"
    assert root.attrs["institution"] == "SMN (SINARAME)"
    assert float(root.latitude) == pytest.approx(-31.44133, abs=1e-5)
    assert float(root.longitude) == pytest.approx(-64.19192, abs=1e-5)
    assert float(root.altitude) == 484.0
    sweeps = [k for k in dt.children if k.startswith("sweep_")]
    assert len(sweeps) == 3
    np.testing.assert_allclose(root.sweep_fixed_angle.values, [0.51, 1.3, 2.26], atol=1e-6)
    s0 = dt["sweep_0"].ds
    assert MOMENTS <= set(s0.data_vars)
    assert s0.sizes["azimuth"] == 360 and s0.sizes["range"] == 956
    assert float(s0.range[0]) == 2100.0 and float(s0.range[1] - s0.range[0]) == 120.0
    np.testing.assert_array_equal(s0.azimuth.values, np.arange(360, dtype=np.float32))
    assert str(s0.sweep_mode.values) == "azimuth_surveillance"
    dbz = s0.DBZH.values
    assert np.isfinite(dbz).mean() > 0.05
    assert -40 < np.nanmin(dbz) and np.nanmax(dbz) < 90
    rho = s0.RHOHV.values
    assert np.nanmax(rho) <= 1.2


def test_ray_times_follow_rotation_from_start_azimuth(vol_0457):
    s0 = radish.open_datatree(vol_0457)["sweep_0"].ds
    t = s0.time.values.astype("datetime64[ms]").astype(np.int64) / 1e3
    # sweep 0 starts at 13.00 deg, 00:04:57Z, ends 00:05:21Z
    assert t.argmin() == 13
    assert t.max() - t.min() == pytest.approx(24.0 * 359 / 360, abs=0.01)


def test_fifteen_tilt_volume_and_moment_filter(vol_0620):
    vol = radish.read_argentina_bufr([str(vol_0620)], moments=["DBZH", "VRADH"])
    assert vol.num_sweeps == 15
    angles = vol.metadata.sweep_fixed_angles
    assert angles[0] == pytest.approx(0.51) and angles[-1] == pytest.approx(29.99)
    assert all(a < b for a, b in zip(angles, angles[1:]))
    s = vol.get_sweep(0)
    assert sorted(s.moment_names()) == ["DBZH", "VRADH"]
    rng = np.asarray(s.range)
    assert rng.size == 521 and rng[1] - rng[0] == 450.0


def test_single_file_and_bytes_paths_agree(vol_0457):
    f = next(vol_0457.glob("*_DBZH_*.BUFR*"))
    a = radish.open_datatree(f)
    b = radish.open_datatree(f.read_bytes())
    for k in ("sweep_0", "sweep_2"):
        assert set(a[k].ds.data_vars) >= {"DBZH"}
        np.testing.assert_array_equal(a[k].ds.DBZH.values, b[k].ds.DBZH.values)
    # list of per-moment messages behaves like the directory
    files = sorted(vol_0457.glob("*.BUFR*"))
    c = radish.open_datatree([p.read_bytes() for p in files])
    d = radish.open_datatree(vol_0457)
    np.testing.assert_array_equal(c["sweep_1"].ds.VRADH.values, d["sweep_1"].ds.VRADH.values)


def test_uncompressed_bufr_reads_like_gzip(vol_0457, tmp_path):
    import gzip

    for p in vol_0457.glob("*_DBZH_*.BUFR.gz"):
        (tmp_path / p.name[:-3]).write_bytes(gzip.decompress(p.read_bytes()))
    plain = radish.open_datatree(tmp_path)
    gz = radish.open_datatree(next(vol_0457.glob("*_DBZH_*.BUFR.gz")))
    np.testing.assert_array_equal(plain["sweep_0"].ds.DBZH.values, gz["sweep_0"].ds.DBZH.values)


def test_directory_with_two_scans_is_grouped_not_merged(vol_0457, vol_0620, tmp_path):
    for v in (vol_0457, vol_0620):
        for p in v.glob("*_DBZH_*"):
            shutil.copy(p, tmp_path / p.name)
    with pytest.raises(RuntimeError, match="2 volumes"):
        radish.read_argentina_bufr([str(tmp_path)])
    groups = radish.argentina_bufr_groups([str(tmp_path)])
    assert [len(g) for g in groups] == [1, 1]
    assert "_02_" in Path(groups[0][0]).name and "_01_" in Path(groups[1][0]).name
