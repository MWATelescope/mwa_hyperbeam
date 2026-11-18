#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["numpy", "h5py", "everybeam", "python-casacore", "casadata", "mwa_hyperbeam"]
#
# [tool.uv.sources]
# mwa_hyperbeam = { path = "..", editable = false }
# ///
"""Benchmark hyperbeam's SKA-Low beams (CPU and GPU) against EveryBeam.

Run from the repository root with uv, which builds hyperbeam from this
checkout. Choose hyperbeam's features with MATURIN_PEP517_ARGS:

    # CPU only
    MATURIN_PEP517_ARGS="--features python,hdf5-static" \\
        uv run benches/ska_benchmark.py --feko-dir /path/to/EveryBeam/coeffs/skalow

    # with CUDA (nvcc on PATH; set HYPERBEAM_CUDA_COMPUTE, e.g. 80 for an A100)
    MATURIN_PEP517_ARGS="--features python,cuda,hdf5-static" \\
        uv run benches/ska_benchmark.py --feko-dir /path/to/EveryBeam/coeffs/skalow

uv caches the hyperbeam build; after changing the Rust code or the features,
add `--reinstall-package mwa_hyperbeam` to rebuild it.

What is compared, and how it is kept fair:

- Identical inputs: every implementation evaluates the same random directions
  above the horizon, at the same frequency, with the same data files. Outputs
  are cross-checked (max. relative difference) so that only equivalent results
  are timed.
- Element responses: EveryBeam's vectorised `ElementResponse.response` (its
  loop is in C++) against hyperbeam's `calc_jones_array` and the GPU.
- Station responses (a 256-element, SKA-Low-like, rotated station): the script
  writes a small throwaway SKA-Low Measurement Set (TELESCOPE_NAME "SKA-LOW",
  stations at the SKA-Low site in ITRF, each station's element offsets and
  rotation in its PHASED_ARRAY table, as in real data), which EveryBeam loads
  with `load_telescope` exactly as it would a real SKA-Low MS. EveryBeam's
  `station_response` (which can only be called one direction at a time from
  Python) is compared with hyperbeam's `calc_station_jones_array` and the GPU,
  for all three models; a second, differently rotated station is
  cross-checked too. Loading the MS (`load_telescope`) is not timed. EveryBeam's
  fixed per-call cost (the
  Python call plus building a new time-dependent point-response object, with
  casacore ITRF frame conversions, on every call; EveryBeam's C++ users reuse
  that object across directions) is measured with a 1-element station, and
  speed-ups are also given with it subtracted, i.e. comparing only the beam
  computation. That subtraction slightly favours EveryBeam, as the 1-element
  measurement also includes evaluating one element.
- Threads: EveryBeam's Python calls are single-threaded. hyperbeam CPU timings
  are given for 1 thread (`RAYON_NUM_THREADS=1`, in a subprocess) and for all
  threads.
- Timing: one untimed warm-up call (it also creates the GPU context), then the
  median of `--repeats` calls. hyperbeam caches the frequency-interpolated EEP
  grid and the FEKO coefficients in memory; EveryBeam reads FEKO coefficients
  from HDF5 on each call. The warm-up ("first call") times are reported too.
- Batched station responses: EveryBeam's `gridded_response` (its batched C++
  path, multi-threaded over image rows) on a sky grid around the MS's delay
  direction, against hyperbeam CPU and GPU on exactly the same directions (with
  hyperbeam's parallactic-angle correction, as EveryBeam applies its sky
  rotation). Both are timed on one core and on all cores; the MS is padded
  with 1-element stations so that EveryBeam can use every core.
- GPU: times include copying the directions to the device and the results back
  (and into a NumPy array). Preparing the GPU (`gpu_prepare`, done once) is
  reported separately, excluding the one-off creation of the GPU context. The
  GPU precision (single or double) is reported.

casacore may log warnings that its leap-second table is out of date; the
benchmark does not depend on absolute time, so they can be ignored.
"""

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version

import numpy as np

FLIP = np.array([1, -1, 1, -1])  # hyperbeam [X_theta, X_az, ...] <-> EveryBeam [X_theta, X_phi, ...]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feko-dir", help="Directory of SKA-Low FEKO coefficient files (e.g. EveryBeam's coeffs/skalow). The ska-low-eep-sph benchmarks are skipped without it.")
    p.add_argument("--grid-file", help="EEP grid HDF5 file. Default: a synthetic grid (1 deg in phi and theta, 31 frequencies 50-350 MHz, random values).")
    p.add_argument("--freq", type=float, default=100e6, help="Frequency [Hz] (default 100 MHz).")
    p.add_argument("--n-dirs", type=int, default=100_000, help="Directions for the analytic and grid element benchmarks.")
    p.add_argument("--n-sph-dirs", type=int, default=20_000, help="Directions for the spherical-wave element benchmark.")
    p.add_argument("--n-station-dirs", type=int, default=2_000, help="Directions for the station benchmarks.")
    p.add_argument("--grid-size", type=int, default=64, help="Pixels per side of the sky grid for the batched station benchmark (EveryBeam's gridded_response).")
    p.add_argument("--skip-batched", action="store_true", help="Skip the batched station benchmark.")
    p.add_argument("--repeats", type=int, default=5, help="Timed repeats per measurement (median reported).")
    p.add_argument("--json", help="Also write the results to this JSON file.")
    p.add_argument("--no-gpu", action="store_true", help="Skip the GPU even if hyperbeam was built with it.")
    p.add_argument("--_worker", help=argparse.SUPPRESS)
    p.add_argument("--_batched-worker", help=argparse.SUPPRESS)
    p.add_argument("--_ms", help=argparse.SUPPRESS)
    p.add_argument("--_one-core", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

def directions(n, seed=42):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 2 * np.pi, n)
    za = rng.uniform(0, np.deg2rad(89), n)
    return az, za


def station_layout():
    """A 256-element station, a Vogel spiral of radius ~19 m (like SKA-Low)."""
    n = 256
    i = np.arange(n)
    r = 19.0 * np.sqrt((i + 0.5) / n)
    a = i * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], axis=1)


STATION_ROTATION = 0.83  # radians, X dipoles from East towards North
STATION_POINTING = (0.6, 0.3)  # (az, za) radians


def make_synthetic_grid(path):
    import h5py

    phi = np.deg2rad(np.arange(0, 361, 1.0))
    theta = np.deg2rad(np.arange(0, 91, 1.0))
    freq = np.linspace(50e6, 350e6, 31)
    g = np.random.default_rng(0).standard_normal((phi.size, theta.size, freq.size, 4, 2))
    with h5py.File(path, "w") as f:
        grp = f.create_group("averageEEP")
        grp["Phi"], grp["Theta"], grp["Freq"], grp["averageEEP"] = phi, theta, freq, g


# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------

def timed(fn, repeats):
    """Return (first-call seconds, median seconds of `repeats` calls, last result)."""
    t = time.perf_counter()
    out = fn()
    first = time.perf_counter() - t
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t)
    return first, statistics.median(times), out


def rel_diff(a, b):
    a = np.asarray(a, dtype=np.complex128).reshape(-1, 4)
    b = np.asarray(b, dtype=np.complex128).reshape(-1, 4)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))


# -----------------------------------------------------------------------------
# hyperbeam
# -----------------------------------------------------------------------------

def hb_models(args, grid_file):
    """hyperbeam models, keyed by name. The grid model returns the raw grid
    values (no SKA-Low conjugation), like EveryBeam's element response."""
    import mwa_hyperbeam as hb

    models = {
        "ska-low-analytic": hb.SkaLowAnalytic(),
        "ska-low-eep-grid": hb.SkaLowEepGrid(grid_file, ska_low_convention=False),
    }
    if args.feko_dir:
        models["ska-low-eep-sph"] = hb.SkaLowEepSph(args.feko_dir)
    return models


def n_dirs_for(model, args):
    return args.n_sph_dirs if model == "ska-low-eep-sph" else args.n_dirs


def hb_cpu_benchmarks(args, grid_file):
    """hyperbeam CPU timings (run in the main process and in a 1-thread
    subprocess). Returns {key: {"first": s, "median": s}} and the outputs."""
    results, outputs = {}, {}
    pos = station_layout()
    for name, m in hb_models(args, grid_file).items():
        az, za = directions(n_dirs_for(name, args))
        first, med, out = timed(lambda: m.calc_jones_array(az, za, args.freq), args.repeats)
        results[f"element/{name}"] = {"first": first, "median": med, "n": len(az)}
        outputs[f"element/{name}"] = out

        az, za = directions(args.n_station_dirs, seed=7)
        first, med, out = timed(
            lambda: m.calc_station_jones_array(
                az, za, args.freq, *STATION_POINTING, pos, rotation_rad=STATION_ROTATION
            ),
            args.repeats,
        )
        results[f"station/{name}"] = {"first": first, "median": med, "n": len(az)}
        outputs[f"station/{name}"] = out
    return results, outputs


def hb_gpu_benchmarks(args, grid_file, cpu_outputs):
    import mwa_hyperbeam as hb

    if not hasattr(hb, "SkaLowGpu"):
        return None, "hyperbeam was built without a GPU feature (add cuda or hip to MATURIN_PEP517_ARGS)"
    results = {}
    pos = station_layout()
    precision = None
    # Create the GPU context first; it's a one-off per process, not part of
    # preparing any model.
    hb.SkaLowAnalytic().gpu_prepare([args.freq]).calc_jones([0.1], [0.1])
    for name, m in hb_models(args, grid_file).items():
        az, za = directions(n_dirs_for(name, args))
        t = time.perf_counter()
        gpu = m.gpu_prepare([args.freq])
        prep = time.perf_counter() - t
        first, med, out = timed(lambda: gpu.calc_jones(az, za), args.repeats)
        precision = "single" if out.dtype == np.complex64 else "double"
        results[f"element/{name}"] = {
            "prepare": prep, "first": first, "median": med, "n": len(az),
            "diff_vs_cpu": rel_diff(out[0, 0], cpu_outputs[f"element/{name}"]),
        }

        az, za = directions(args.n_station_dirs, seed=7)
        t = time.perf_counter()
        gpu = m.gpu_prepare([args.freq], positions_enu=pos[np.newaxis], rotations_rad=[STATION_ROTATION])
        prep = time.perf_counter() - t
        first, med, out = timed(lambda: gpu.calc_jones(az, za, *STATION_POINTING), args.repeats)
        results[f"station/{name}"] = {
            "prepare": prep, "first": first, "median": med, "n": len(az),
            "diff_vs_cpu": rel_diff(out[0, 0], cpu_outputs[f"station/{name}"]),
        }
    return {"precision": precision, "results": results}, None


# -----------------------------------------------------------------------------
# EveryBeam
# -----------------------------------------------------------------------------

EB_MODELS = {
    "ska-low-analytic": "oskar_dipole_cos",
    "ska-low-eep-grid": "gridded_eep",
    "ska-low-eep-sph": "skalow_feko",
}


def eb_version_tuple():
    v = version("everybeam")
    return tuple(int(x) for x in v.split(".")[:3] if x.isdigit())


def eb_options(eb, name, args, grid_file):
    o = eb.Options()
    o.element_response_model = getattr(eb.ElementResponseModel, EB_MODELS[name])
    if name == "ska-low-eep-grid":
        o.coeff_path = grid_file
    elif name == "ska-low-eep-sph":
        o.coeff_path = args.feko_dir
    o.use_channel_frequency = True
    return o


def eb_to_hb_order(name, j):
    """EveryBeam <= 0.9.1 stored gridded EEP correlations in a different order
    from master (and hyperbeam); reorder for comparison."""
    j = np.asarray(j).reshape(-1, 4)
    if name == "ska-low-eep-grid" and eb_version_tuple() <= (0, 9, 1):
        j = j[:, [1, 0, 3, 2]]
    return j


# The throwaway Measurement Set's stations: (name, centre offset East/North/Up
# [m] from the array centre, element positions, rotation [rad]). The first is
# the benchmarked station; the second (differently rotated) is only used to
# cross-check; the third (a single element) measures EveryBeam's per-call cost.
SKA_LOW_SITE = (np.deg2rad(-26.82472208), np.deg2rad(116.76444824), 377.8)  # lat, lon, height


def ms_stations():
    stations = [
        ("S1", [0.0, 0.0, 0.0], station_layout(), STATION_ROTATION),
        ("S2", [300.0, -150.0, 0.0], station_layout(), -1.2),
        ("S0", [0.0, 50.0, 0.0], np.zeros((1, 3)), 0.0),
    ]
    # EveryBeam's gridded_response uses min(CPUs, number of stations in the
    # telescope) threads, so pad with 1-element stations until there are at
    # least as many stations as CPUs (as in real SKA-Low data, which has 512).
    n_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    for i in range(max(0, n_cpus - len(stations))):
        stations.append((f"F{i}", [100.0 + 5.0 * i, 100.0, 0.0], np.zeros((1, 3)), 0.0))
    return stations


def enu_axes_itrf():
    """Rows: the local East, North and Up unit vectors (ITRF) at the SKA-Low
    site."""
    lat, lon, _ = SKA_LOW_SITE
    e = [-np.sin(lon), np.cos(lon), 0.0]
    n = [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    u = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    return np.array([e, n, u])


def enu_to_itrf(v):
    return np.asarray(v) @ enu_axes_itrf()


def site_itrf():
    lat, lon, h = SKA_LOW_SITE
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    n = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    return np.array([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - e2) + h) * np.sin(lat)])


def write_ska_low_ms(path, freq, delay_dir=(0.0, 0.0)):
    """Write a minimal SKA-Low Measurement Set (TELESCOPE_NAME "SKA-LOW"), with
    each station's element offsets and rotation (COORDINATE_AXES) in its
    PHASED_ARRAY table, as in real SKA-Low data. EveryBeam loads it through the
    same code as a real SKA-Low MS."""
    from casacore import tables as ct

    stations = ms_stations()
    n_st = len(stations)
    centre = site_itrf()
    ms = ct.default_ms(path, ct.maketabdesc(ct.makearrcoldesc("DATA", 0j, ndim=2, valuetype="complex")))

    ant = ct.table(path + "/ANTENNA", readonly=False, ack=False)
    ant.addrows(n_st)
    ant.putcol("NAME", [st[0] for st in stations])
    ant.putcol("STATION", [st[0] for st in stations])
    ant.putcol("TYPE", ["GROUND-BASED"] * n_st)
    ant.putcol("MOUNT", ["X-Y"] * n_st)
    ant.putcol("DISH_DIAMETER", np.full(n_st, 38.0))
    ant.putcol("POSITION", np.array([centre + enu_to_itrf(st[1]) for st in stations]))
    ant.close()

    obs = ct.table(path + "/OBSERVATION", readonly=False, ack=False)
    obs.addrows(1)
    obs.putcol("TELESCOPE_NAME", ["SKA-LOW"])
    obs.close()

    fld = ct.table(path + "/FIELD", readonly=False, ack=False)
    fld.addrows(1)
    for col in ["DELAY_DIR", "PHASE_DIR", "REFERENCE_DIR"]:
        fld.putcol(col, np.array(delay_dir, dtype=float).reshape(1, 1, 2))
    fld.close()

    spw = ct.table(path + "/SPECTRAL_WINDOW", readonly=False, ack=False)
    spw.addrows(1)
    spw.putcol("NUM_CHAN", np.array([1], dtype=np.int32))
    spw.putcol("CHAN_FREQ", np.array([[freq]]))
    for col in ["CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"]:
        spw.putcol(col, np.array([[1e3]]))
    spw.putcol("REF_FREQUENCY", np.array([freq]))
    spw.putcol("TOTAL_BANDWIDTH", np.array([1e3]))
    spw.close()

    m = {"QuantumUnits": ["m", "m", "m"]}
    cols = [
        ct.makearrcoldesc("POSITION", 0.0, ndim=1, shape=[3], keywords=m),
        ct.makearrcoldesc("COORDINATE_AXES", 0.0, ndim=2, shape=[3, 3], keywords=m),
        ct.makearrcoldesc("ELEMENT_OFFSET", 0.0, ndim=2, keywords=m),
        ct.makearrcoldesc("ELEMENT_FLAG", False, ndim=2),
    ]
    pa = ct.table(path + "/PHASED_ARRAY", ct.maketabdesc(cols), nrow=n_st, readonly=False, ack=False)
    for i, (_, c_enu, pos, rot) in enumerate(stations):
        pa.putcell("POSITION", i, centre + enu_to_itrf(c_enu))
        p = enu_to_itrf([np.cos(rot), np.sin(rot), 0.0])
        q = enu_to_itrf([-np.sin(rot), np.cos(rot), 0.0])
        r = enu_to_itrf([0.0, 0.0, 1.0])
        # In casacore the axes are the columns of a 3x3 matrix; python-casacore
        # transposes, so they are rows here. Likewise ELEMENT_OFFSET is 3 x N.
        pa.putcell("COORDINATE_AXES", i, np.array([p, q, r]))
        pa.putcell("ELEMENT_OFFSET", i, enu_to_itrf(pos))
        pa.putcell("ELEMENT_FLAG", i, np.zeros((len(pos), 2), dtype=bool))
    pa.close()
    ms.putkeyword("PHASED_ARRAY", "Table: " + os.path.abspath(path) + "/PHASED_ARRAY")
    ms.close()


def enu(az, za):
    return np.array([np.sin(za) * np.sin(az), np.sin(za) * np.cos(az), np.cos(za)])


EB_TIME = 5e9  # MJD seconds (2017); only used by EveryBeam for its (unused here) sky rotation.


def eb_station_reference(name, args, grid_file, station_index, az, za, pointing=STATION_POINTING):
    """hyperbeam's station response for the throwaway MS's station
    `station_index`, in EveryBeam's convention, to compare with EveryBeam."""
    import mwa_hyperbeam as hb

    if name == "ska-low-eep-grid":
        # EveryBeam master applies the SKA-Low conjugation/negation to gridded
        # EEPs in SKA-Low telescopes; 0.9.1 doesn't.
        model = hb.SkaLowEepGrid(grid_file, ska_low_convention=eb_version_tuple() > (0, 9, 1))
    else:
        model = hb_models(args, grid_file)[name]
    _, _, pos, rot = ms_stations()[station_index]
    out = model.calc_station_jones_array(az, za, args.freq, *pointing, pos, rotation_rad=rot)
    return out * FLIP


def eb_benchmarks(args, grid_file, hb_outputs, ms_path):
    import everybeam as eb

    results, notes = {}, []
    s0 = enu_to_itrf(enu(*STATION_POINTING))
    for name in hb_models(args, grid_file):
        # Element responses.
        az, za = directions(n_dirs_for(name, args))
        theta, phi = za, (np.pi / 2 - az) % (2 * np.pi)
        er = eb.ElementResponse.create(eb_options(eb, name, args, grid_file))
        f = np.full(len(az), args.freq)
        if name == "ska-low-eep-sph":
            ids = np.zeros(len(az), dtype=np.int32)
            fn = lambda: er.response(ids, f, theta, phi)
        else:
            fn = lambda: er.response(f, theta, phi)
        first, med, out = timed(fn, args.repeats)
        results[f"element/{name}"] = {
            "first": first, "median": med, "n": len(az),
            "diff_vs_hb": rel_diff(eb_to_hb_order(name, out), hb_outputs[f"element/{name}"] * FLIP),
        }

        # Station responses, from the throwaway SKA-Low MS.
        coeff = {"ska-low-eep-grid": grid_file, "ska-low-eep-sph": args.feko_dir}.get(name) or ""
        try:
            tel = eb.load_telescope(ms_path, element_response_model=EB_MODELS[name], coeff_path=coeff)
        except Exception as e:  # noqa: BLE001
            notes.append(f"EveryBeam couldn't load the SKA-Low MS with {EB_MODELS[name]}: {e}")
            continue
        az, za = directions(args.n_station_dirs, seed=7)
        dirs = [enu_to_itrf(enu(a, z)) for a, z in zip(az, za)]
        fn = lambda: np.array([tel.station_response(EB_TIME, 0, args.freq, d, s0, rotate=False).reshape(4)
                               for d in dirs])
        first, med, out = timed(fn, args.repeats)
        diff = rel_diff(eb_to_hb_order(name, out), eb_station_reference(name, args, grid_file, 0, az, za))
        # Also cross-check the second, differently rotated station (not timed).
        n_check = min(200, len(az))
        out2 = np.array([tel.station_response(EB_TIME, 1, args.freq, d, s0, rotate=False).reshape(4)
                         for d in dirs[:n_check]])
        diff2 = rel_diff(eb_to_hb_order(name, out2),
                         eb_station_reference(name, args, grid_file, 1, az[:n_check], za[:n_check]))
        results[f"station/{name}"] = {"first": first, "median": med, "n": len(az), "diff_vs_hb": max(diff, diff2)}

    # EveryBeam's fixed per-call cost for station responses, measured with the
    # MS's 1-element station.
    tel = eb.load_telescope(ms_path, element_response_model="oskar_dipole_cos")
    az, za = directions(args.n_station_dirs, seed=7)
    dirs = [enu_to_itrf(enu(a, z)) for a, z in zip(az, za)]
    _, med, _ = timed(lambda: [tel.station_response(EB_TIME, 2, args.freq, d, s0, rotate=False) for d in dirs],
                      args.repeats)
    results["station_call_overhead_per_dir"] = med / len(dirs)
    return results, notes


# -----------------------------------------------------------------------------
# Batched station responses: EveryBeam's gridded_response
# -----------------------------------------------------------------------------

BATCH_TIME = 5.2e9  # MJD seconds (2023)


def make_radec_to_itrf(t):
    """J2000 (RA, Dec) -> ITRF unit vector, exactly as EveryBeam's ItrfConverter
    does it (casacore, epoch `t`, and EveryBeam's fixed reference position,
    which EveryBeam 0.9.1 has without the leading '3' that master fixed)."""
    from casacore.measures import measures
    from casacore.quanta import quantity

    dm = measures()
    x = 826577.022720000 if eb_version_tuple() <= (0, 9, 1) else 3826577.022720000
    dm.do_frame(dm.position("itrf", f"{x}m", "461022.995082m", "5064892.814m"))
    dm.do_frame(dm.epoch("utc", quantity(t, "s")))

    def f(ra, dec):
        d = dm.measure(dm.direction("J2000", quantity(ra, "rad"), quantity(dec, "rad")), "ITRF")
        lon, lat = d["m0"]["value"], d["m1"]["value"]
        return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])

    return f


def itrf_to_azza(v):
    """(az, za) of ITRF direction vectors, which needn't be unit length
    (EveryBeam's gridded pixel directions aren't exactly: see
    batched_geometry)."""
    e, n, u = enu_axes_itrf() @ np.asarray(v).T
    return np.arctan2(e, n) % (2 * np.pi), np.arctan2(np.hypot(e, n), u)


def batched_geometry(args):
    """The sky grid used by EveryBeam's gridded_response, as (az, za) for
    hyperbeam, plus the pointing (the MS delay direction) and the vectors
    EveryBeam uses for its sky rotation."""
    from casacore.measures import measures
    from casacore.quanta import quantity

    # Point 0.15 rad from the zenith at BATCH_TIME.
    dm = measures()
    lat, lon, h = SKA_LOW_SITE
    dm.do_frame(dm.position("wgs84", quantity(lon, "rad"), quantity(lat, "rad"), quantity(h, "m")))
    dm.do_frame(dm.epoch("utc", quantity(BATCH_TIME, "s")))
    z = dm.measure(dm.direction("AZEL", "0rad", quantity(np.pi / 2, "rad")), "J2000")
    ra, dec = z["m0"]["value"], z["m1"]["value"] + 0.15

    to_itrf = make_radec_to_itrf(BATCH_TIME)
    lvec, mvec, nvec = to_itrf(ra + np.pi / 2, 0.0), to_itrf(ra, dec + np.pi / 2), to_itrf(ra, dec)
    ncp = to_itrf(0.0, np.pi / 2)

    # aocommon::ImageCoordinates::XYToLM, row-major [y][x]. +/- 40 deg in l, m.
    size = args.grid_size
    dl = 2 * np.sin(np.deg2rad(40)) / size
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    l = (size / 2 - x.ravel()) * dl
    m = (y.ravel() - size / 2) * dl
    n = np.sqrt(1 - l * l - m * m)
    # Exactly as EveryBeam does it. As the basis vectors are converted to ITRF
    # separately (with aberration), they aren't exactly orthonormal, so these
    # directions aren't exactly unit length (to ~1e-4).
    dirs = l[:, None] * lvec + m[:, None] * mvec + n[:, None] * nvec
    az, za = itrf_to_azza(dirs)
    paz, pza = itrf_to_azza(nvec[None, :])
    return {"ra": ra, "dec": dec, "dl": dl, "size": size, "dirs": dirs, "az": az, "za": za,
            "pointing": (float(paz[0]), float(pza[0])), "ncp": ncp}


def eb_sky_rotation(j, dirs, ncp):
    """Apply EveryBeam's `rotate` (its gridded_response always does): project
    the theta/phi columns onto sky north/east (from the celestial pole)."""
    enu = enu_axes_itrf()
    d = dirs @ enu.T
    east = np.cross(ncp, dirs)
    east /= np.linalg.norm(east, axis=1)[:, None]
    north = np.cross(dirs, east)
    east, north = east @ enu.T, north @ enu.T
    e_phi = np.stack([-d[:, 1], d[:, 0], np.zeros(len(d))], axis=1)
    e_phi /= np.linalg.norm(e_phi, axis=1)[:, None]
    e_theta = np.cross(e_phi, d)
    rot = np.stack([np.sum(e_theta * north, 1), np.sum(e_theta * east, 1),
                    np.sum(e_phi * north, 1), np.sum(e_phi * east, 1)], axis=1).reshape(-1, 2, 2)
    return np.einsum("nij,njk->nik", np.asarray(j).reshape(-1, 2, 2), rot).reshape(-1, 4)


def batched_benchmarks(args, grid_file, ms_path):
    """hyperbeam CPU vs EveryBeam's gridded_response on the same sky grid (run
    in a subprocess, with either all cores or one)."""
    import everybeam as eb

    geom = batched_geometry(args)
    az, za, pointing = geom["az"], geom["za"], geom["pointing"]
    lat = SKA_LOW_SITE[0]
    _, _, pos, rot = ms_stations()[0]
    gs = eb.GridSettings()
    gs.ra, gs.dec, gs.width, gs.height = geom["ra"], geom["dec"], geom["size"], geom["size"]
    gs.dl, gs.dm, gs.l_shift, gs.m_shift = geom["dl"], geom["dl"], 0.0, 0.0

    results = {}
    for name, m in hb_models(args, grid_file).items():
        first, med, _ = timed(lambda: m.calc_station_jones_array(
            az, za, args.freq, *pointing, pos, rotation_rad=rot, latitude_rad=lat, iau_order=True), args.repeats)
        r = {"n": len(az), "hb_first": first, "hb_median": med}
        coeff = {"ska-low-eep-grid": grid_file, "ska-low-eep-sph": args.feko_dir}.get(name) or ""
        try:
            tel = eb.load_telescope(ms_path, element_response_model=EB_MODELS[name], coeff_path=coeff)
            reps = args.repeats if name != "ska-low-eep-sph" else min(args.repeats, 2)
            first, med, out = timed(lambda: tel.gridded_response(gs, BATCH_TIME, args.freq, 0), reps)
            r.update({"eb_first": first, "eb_median": med})
            if not args._one_core:
                # hyperbeam's raw station response, put in this EveryBeam's
                # correlation order (eb_to_hb_order is its own inverse), then
                # EveryBeam's sky rotation applied.
                raw = eb_to_hb_order(name, eb_station_reference(name, args, grid_file, 0, az, za, pointing))
                ref = eb_sky_rotation(raw, geom["dirs"], geom["ncp"])
                r["diff"] = rel_diff(out, ref)
        except Exception as e:  # noqa: BLE001
            r["eb_error"] = str(e)
        results[name] = r
    return results


def batched_gpu(args, grid_file):
    import mwa_hyperbeam as hb

    if not hasattr(hb, "SkaLowGpu"):
        return None
    geom = batched_geometry(args)
    _, _, pos, rot = ms_stations()[0]
    out = {}
    for name, m in hb_models(args, grid_file).items():
        gpu = m.gpu_prepare([args.freq], positions_enu=pos[np.newaxis], rotations_rad=[rot])
        first, med, _ = timed(lambda: gpu.calc_jones(geom["az"], geom["za"], *geom["pointing"],
                                                     latitude_rad=SKA_LOW_SITE[0], iau_order=True), args.repeats)
        out[name] = {"first": first, "median": med}
    return out


def setup_casacore_data():
    """EveryBeam's station code uses casacore measures (leap seconds); point
    casacore at the measures data from the `casadata` package."""
    if os.environ.get("CASARCFILES"):
        return
    import casadata

    rc = tempfile.NamedTemporaryFile("w", suffix=".casarc", delete=False)
    rc.write(f"measures.directory: {casadata.datapath}\n")
    rc.close()
    os.environ["CASARCFILES"] = rc.name


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def system_info():
    info = {"python": platform.python_version(), "machine": platform.machine(), "cpus": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu"] = platform.processor()
    if shutil.which("nvidia-smi"):
        try:
            info["gpu"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    for pkg in ["mwa_hyperbeam", "everybeam", "numpy"]:
        try:
            info[pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            pass
    return info


def fmt_ms(s):
    return "—" if s is None else f"{s * 1e3:.2f}"


def main():
    args = parse_args()
    tmpdir = tempfile.mkdtemp(prefix="ska_benchmark_")
    grid_file = args.grid_file or os.path.join(tmpdir, "synthetic_eep_grid.h5")
    if not args.grid_file and args._worker is None and args._batched_worker is None:
        make_synthetic_grid(grid_file)

    if args._batched_worker:
        # Subprocess: the batched station benchmark, with all cores or (for
        # both hyperbeam and EveryBeam, which counts CPUs from the affinity
        # mask) just one.
        if args._one_core and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        res = batched_benchmarks(args, args._batched_worker, args._ms)
        print(json.dumps(res))
        return

    if args._worker:
        # Subprocess: hyperbeam CPU timings only (with RAYON_NUM_THREADS set by
        # the parent).
        res, _ = hb_cpu_benchmarks(args, args._worker)
        print(json.dumps(res))
        return

    import mwa_hyperbeam as hb

    grid_freqs = hb.SkaLowEepGrid(grid_file, ska_low_convention=False).get_freqs()
    if not grid_freqs[0] <= args.freq <= grid_freqs[-1]:
        sys.exit(f"--freq {args.freq:g} Hz is outside the EEP grid's frequency range "
                 f"({grid_freqs[0]:g} to {grid_freqs[-1]:g} Hz)")
    setup_casacore_data()

    info = system_info()
    print("# SKA-Low beam benchmark: hyperbeam vs EveryBeam\n")
    for k, v in info.items():
        print(f"- {k}: {v}")
    print(f"- frequency: {args.freq / 1e6:g} MHz; repeats: {args.repeats} (median reported)")
    print(f"- EEP grid: {args.grid_file or 'synthetic (1 deg, 31 freqs)'}")
    print(f"- FEKO coefficients: {args.feko_dir or 'not given (ska-low-eep-sph skipped)'}")
    print(f"- station: 256 elements (Vogel spiral, r = 19 m), rotated {STATION_ROTATION} rad, pointed at az/za "
          f"{STATION_POINTING} rad; EveryBeam loads it from a throwaway SKA-Low Measurement Set\n")

    print("Running hyperbeam (CPU, all threads)...", file=sys.stderr)
    hb_all, hb_outputs = hb_cpu_benchmarks(args, grid_file)

    print("Running hyperbeam (CPU, 1 thread)...", file=sys.stderr)
    cmd = [sys.executable, os.path.abspath(__file__), "--_worker", grid_file,
           "--freq", str(args.freq), "--n-dirs", str(args.n_dirs), "--n-sph-dirs", str(args.n_sph_dirs),
           "--n-station-dirs", str(args.n_station_dirs), "--repeats", str(args.repeats)]
    if args.feko_dir:
        cmd += ["--feko-dir", args.feko_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "RAYON_NUM_THREADS": "1"})
    if proc.returncode != 0:
        sys.exit(f"1-thread worker failed:\n{proc.stderr}")
    hb_one = json.loads(proc.stdout.strip().splitlines()[-1])

    gpu, gpu_note = (None, "--no-gpu given") if args.no_gpu else (None, None)
    if not args.no_gpu:
        print("Running hyperbeam (GPU)...", file=sys.stderr)
        gpu, gpu_note = hb_gpu_benchmarks(args, grid_file, hb_outputs)

    print("Running EveryBeam (casacore may warn about its leap-second table; that's harmless here)...",
          file=sys.stderr)
    ms_path = os.path.join(tmpdir, "ska_low.ms")
    geom = batched_geometry(args)
    write_ska_low_ms(ms_path, args.freq, delay_dir=(geom["ra"], geom["dec"]))
    eb_res, eb_notes = eb_benchmarks(args, grid_file, hb_outputs, ms_path)

    batched = None
    if not args.skip_batched:
        batched = {}
        base = [sys.executable, os.path.abspath(__file__), "--_batched-worker", grid_file, "--_ms", ms_path,
                "--freq", str(args.freq), "--grid-size", str(args.grid_size), "--repeats", str(args.repeats)]
        if args.feko_dir:
            base += ["--feko-dir", args.feko_dir]
        for label, extra, env in [("one", ["--_one-core"], {"RAYON_NUM_THREADS": "1"}), ("all", [], {})]:
            print(f"Running the batched station benchmark ({label} core{'s' if label == 'all' else ''})...",
                  file=sys.stderr)
            proc = subprocess.run(base + extra, capture_output=True, text=True, env={**os.environ, **env})
            if proc.returncode != 0:
                sys.exit(f"Batched worker failed:\n{proc.stderr}")
            batched[label] = json.loads(proc.stdout.strip().splitlines()[-1])
        if gpu:
            batched["gpu"] = batched_gpu(args, grid_file)

    # Tables.
    overhead = eb_res.get("station_call_overhead_per_dir")
    for kind, title in [("element", "Element responses"), ("station", "Station responses (256 elements, rotated)")]:
        print(f"\n## {title}\n")
        hdr = "| model | directions | EveryBeam [ms] | hyperbeam CPU 1 thread [ms] | hyperbeam CPU all threads [ms] |"
        sep = "|---|---|---|---|---|"
        if gpu:
            hdr += f" hyperbeam GPU ({gpu['precision']}) [ms] | GPU prepare [ms] |"
            sep += "---|---|"
        hdr += " speed-up vs EveryBeam (1 thread / all / GPU) | max rel. diff vs EveryBeam |"
        sep += "---|---|"
        print(hdr)
        print(sep)
        for name in hb_models(args, grid_file):
            key = f"{kind}/{name}"
            e = eb_res.get(key)
            a, o = hb_all[key], hb_one[key]
            g = gpu["results"][key] if gpu else None
            row = f"| {name} | {a['n']:,} | {fmt_ms(e['median'] if e else None)} | {fmt_ms(o['median'])} | {fmt_ms(a['median'])} |"
            if gpu:
                row += f" {fmt_ms(g['median'])} | {fmt_ms(g['prepare'])} |"
            if e:
                sp = [e["median"] / o["median"], e["median"] / a["median"]] + ([e["median"] / g["median"]] if g else [])
                row += " " + " / ".join(f"{x:.1f}x" for x in sp) + f" | {e['diff_vs_hb']:.1e} |"
            else:
                row += " — | — |"
            print(row)

        print("\nFirst-call (warm-up) times [ms]:")
        for name in hb_models(args, grid_file):
            key = f"{kind}/{name}"
            e = eb_res.get(key)
            parts = [f"EveryBeam {fmt_ms(e['first'] if e else None)}", f"hyperbeam CPU {fmt_ms(hb_all[key]['first'])}"]
            if gpu:
                parts.append(f"hyperbeam GPU {fmt_ms(gpu['results'][key]['first'])}")
            print(f"- {name}: " + ", ".join(parts))
        if gpu:
            print("\nGPU vs hyperbeam CPU, max rel. diff: " + ", ".join(
                f"{n} {gpu['results'][f'{kind}/{n}']['diff_vs_cpu']:.1e}" for n in hb_models(args, grid_file)))
        if kind == "station" and overhead is not None:
            n = args.n_station_dirs
            print(f"\nEveryBeam computes station responses one direction per Python call (loading the MS isn't "
                  f"timed). Each call also builds a new time-dependent point-response object, with casacore ITRF "
                  f"frame conversions, which EveryBeam's C++ users reuse across directions. This fixed per-call cost "
                  f"(measured with a 1-element station) is {overhead * 1e6:.1f} us per direction, i.e. "
                  f"{overhead * n * 1e3:.2f} ms of each EveryBeam station time above. Excluding it (comparing only "
                  f"the beam computation; this slightly favours EveryBeam), the speed-ups are:")
            for name in hb_models(args, grid_file):
                key = f"station/{name}"
                e = eb_res.get(key)
                if not e:
                    continue
                adj = e["median"] - overhead * n
                if adj <= 0:
                    print(f"- {name}: EveryBeam's time is dominated by its per-call cost")
                    continue
                sp = [adj / hb_one[key]["median"], adj / hb_all[key]["median"]]
                if gpu:
                    sp.append(adj / gpu["results"][key]["median"])
                print(f"- {name}: EveryBeam {adj * 1e3:.2f} ms -> " + " / ".join(f"{x:.1f}x" for x in sp)
                      + " (1 thread / all threads" + (" / GPU)" if gpu else ")"))

    if batched:
        n_cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        print(f"\n## Station responses, batched: EveryBeam's gridded_response ({args.grid_size}x{args.grid_size} sky grid, "
              f"256 elements, rotated)\n")
        hdr = ("| model | directions | EveryBeam 1 core [ms] | EveryBeam all cores [ms] | hyperbeam CPU 1 thread [ms] "
               "| hyperbeam CPU all threads [ms] |")
        sep = "|---|---|---|---|---|---|"
        if batched.get("gpu"):
            hdr += f" hyperbeam GPU ({gpu['precision']}) [ms] |"
            sep += "---|"
        hdr += " speed-up (1 core / all cores" + (" / GPU vs all cores" if batched.get("gpu") else "") + ") | max rel. diff vs EveryBeam |"
        sep += "---|---|"
        print(hdr)
        print(sep)
        for name in hb_models(args, grid_file):
            one, a = batched["one"][name], batched["all"][name]
            g = (batched.get("gpu") or {}).get(name)
            row = (f"| {name} | {a['n']:,} | {fmt_ms(one.get('eb_median'))} | {fmt_ms(a.get('eb_median'))} | "
                   f"{fmt_ms(one['hb_median'])} | {fmt_ms(a['hb_median'])} |")
            if batched.get("gpu"):
                row += f" {fmt_ms(g['median'])} |"
            if "eb_median" in a:
                sp = [one["eb_median"] / one["hb_median"], a["eb_median"] / a["hb_median"]]
                if g:
                    sp.append(a["eb_median"] / g["median"])
                row += " " + " / ".join(f"{x:.1f}x" for x in sp) + f" | {a.get('diff', float('nan')):.1e} |"
            else:
                row += f" — | — |"
            print(row)
        print(f"\nThis is EveryBeam's batched C++ path (no per-direction Python calls): the station beam on an image "
              f"grid of sky directions around the MS's delay direction, at one time, with EveryBeam's sky rotation "
              f"(it always applies it). hyperbeam computes the same directions (derived with the same J2000->ITRF "
              f"conversion as EveryBeam) with its parallactic-angle correction, i.e. equivalent work. EveryBeam "
              f"spreads the work over min(CPUs, stations in the MS) threads; the MS is padded so that is all "
              f"{n_cores} cores. For the 1-core columns both run pinned to a single CPU.")
        print("\nAgreement: EveryBeam builds each pixel direction as l*L + m*M + n*N from basis vectors that it "
              "converts to ITRF separately; with aberration they aren't exactly orthonormal, so its pixel directions "
              "aren't exactly unit length (by up to ~1e-4), which perturbs its array-factor phases by ~1e-5. "
              "hyperbeam uses exact unit directions. (Given the same, unit-length directions, EveryBeam and "
              "hyperbeam agree to ~1e-15; see the per-direction table.)")
        if "ska-low-eep-sph" in batched["all"] and "eb_median" in batched["all"]["ska-low-eep-sph"]:
            print("\nEveryBeam's ska-low-eep-sph reads each element's FEKO coefficients from HDF5 on every "
                  "evaluation, behind a lock unless its HDF5 is thread-safe, so it may not speed up (or may slow "
                  "down) with more cores.")
        for name in hb_models(args, grid_file):
            if "eb_error" in batched["all"][name]:
                print(f"- {name}: EveryBeam failed: {batched['all'][name]['eb_error']}")

    notes = []
    if gpu_note:
        notes.append(f"GPU skipped: {gpu_note}")
    notes += eb_notes
    if eb_version_tuple() <= (0, 9, 1):
        notes.append(f"EveryBeam {version('everybeam')} stores gridded-EEP correlations in its pre-0.9.2 order; "
                     "its results are reordered for comparison (hyperbeam follows EveryBeam master).")
    notes.append("EveryBeam element responses have no parallactic-angle correction, so none is applied anywhere; "
                 "station responses use rotate=False in EveryBeam for the same reason.")
    print("\n## Notes\n")
    for n in notes:
        print(f"- {n}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"system": info, "args": vars(args), "everybeam": eb_res, "batched": batched,
                       "hyperbeam_cpu_all": hb_all, "hyperbeam_cpu_1": hb_one, "hyperbeam_gpu": gpu,
                       "notes": notes}, f, indent=2)
        print(f"\nWrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
