#!/usr/bin/env python3
"""Make the small, fake SKA-Low EEP files in this directory, and EveryBeam's
responses for them, which hyperbeam's tests compare against.

    python3 test_files/ska_low/make_test_files.py

Needs numpy, h5py, everybeam and python-casacore (and casacore measures data;
set CASARCFILES if casacore can't find it). The outputs are:

- `eep_grid.h5`: a gridded EEP (EveryBeam `gridded_eep`) on a coarse grid (30
  deg in phi, 15 deg in theta) at 100, 150 and 200 MHz, with smooth, made-up
  values.
- `feko/`: spherical-wave coefficients (EveryBeam `skalow_feko`) for a
  5-element station at 100, 150 and 200 MHz, with different degrees (3, 5 and
  4), with made-up values.
- `everybeam_reference.h5`: EveryBeam's element responses for these files and
  its station responses for two rotated 5-element stations (for all three
  models), loaded from a throwaway SKA-Low Measurement Set as for real data.

The files aren't physical beams; they're only for testing that hyperbeam reads
and evaluates them as EveryBeam does.
"""

import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
FREQS_MHZ = [100.0, 150.0, 200.0]
FEKO_L_MAXES = [3, 5, 4]

# Two stations of the same 5 elements (East, North, Up [m]), with different
# rotations (X dipoles from East towards North [rad]).
POSITIONS = np.array([[0.0, 0.0, 0.0], [1.7, 0.4, 0.0], [-0.9, 1.5, 0.0], [-1.1, -1.3, 0.0], [0.6, -2.1, 0.0]])
ROTATIONS = [0.83, -1.2]
POINTING = (0.6, 0.3)  # (az, za) [rad]
SKA_LOW_SITE = (np.deg2rad(-26.82472208), np.deg2rad(116.76444824), 377.8)  # lat, lon, height
EB_TIME = 5e9  # MJD seconds

EB_MODELS = {"analytic": "oskar_dipole_cos", "grid": "gridded_eep", "sph": "skalow_feko"}
# The frequencies at which responses are compared: for the grid, on and
# between its frequencies; for FEKO, on and between files (no interpolation;
# the closest file is used).
TEST_FREQS = {"analytic": [110e6, 180e6], "grid": [100e6, 130e6, 200e6], "sph": [100e6, 140e6, 180e6]}


def directions():
    """Directions (az, za) above the horizon, including one near the zenith (at
    the zenith itself, phi is undefined and EveryBeam's depends on rounding)."""
    rng = np.random.default_rng(1)
    az = np.concatenate([[0.0, 1.0], rng.uniform(0, 2 * np.pi, 22)])
    za = np.concatenate([[0.02, 1.5], rng.uniform(0, np.deg2rad(88), 22)])
    return az, za


def write_grid(path):
    phi = np.deg2rad(np.arange(0, 361, 30.0))
    theta = np.deg2rad(np.arange(0, 91, 15.0))
    freq = np.array(FREQS_MHZ) * 1e6
    p, t, f, c = np.meshgrid(phi, theta, freq / 1e8, np.arange(4), indexing="ij")
    amp = (1 + 0.2 * c) * np.cos(t / 1.3) * np.cos(p - 0.7 * c) * (1 + 0.3 * (f - 1.5))
    phase = 0.5 * c + t * (1 + 0.5 * f) - 0.3 * p
    g = np.stack([amp * np.cos(phase), amp * np.sin(phase)], axis=-1)
    with h5py.File(path, "w") as h5:
        grp = h5.create_group("averageEEP")
        grp["Phi"], grp["Theta"], grp["Freq"], grp["averageEEP"] = phi, theta, freq, g


def write_feko(dir_):
    if dir_.exists():
        shutil.rmtree(dir_)
    dir_.mkdir()
    rng = np.random.default_rng(2)
    n_el = len(POSITIONS)
    for f, l_max in zip(FREQS_MHZ, FEKO_L_MAXES):
        modes = np.array([[1, m, n] for n in range(1, l_max + 1) for m in range(-n, n + 1)], dtype=np.int64).T
        nc = modes.shape[1]
        # Coefficients that fall with degree, so all degrees matter but
        # responses stay of order one.
        scale = 1.0 / modes[2][np.newaxis, :, np.newaxis] ** 1.5
        c = (rng.standard_normal((n_el, nc, 4)) + 1j * rng.standard_normal((n_el, nc, 4))) * scale
        with h5py.File(dir_ / f"SKA-Low_Test_Vogel_{n_el}_elem_FEKO_{f:.5f}_MHz.h5", "w") as h5:
            h5["coefficients"] = c
            h5["modes"] = modes


def enu_axes_itrf():
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


def enu(az, za):
    return np.array([np.sin(za) * np.sin(az), np.sin(za) * np.cos(az), np.cos(za)])


def write_ska_low_ms(path, freq):
    """A minimal SKA-Low Measurement Set with the two stations, each with its
    element offsets and rotation (COORDINATE_AXES) in its PHASED_ARRAY table,
    as in real SKA-Low data."""
    from casacore import tables as ct

    names = [f"S{i}" for i in range(len(ROTATIONS))]
    centre = site_itrf()
    offsets = [enu_to_itrf([30.0 * i, -20.0 * i, 0.0]) for i in range(len(ROTATIONS))]
    ms = ct.default_ms(path, ct.maketabdesc(ct.makearrcoldesc("DATA", 0j, ndim=2, valuetype="complex")))

    ant = ct.table(path + "/ANTENNA", readonly=False, ack=False)
    ant.addrows(len(names))
    ant.putcol("NAME", names)
    ant.putcol("STATION", names)
    ant.putcol("TYPE", ["GROUND-BASED"] * len(names))
    ant.putcol("MOUNT", ["X-Y"] * len(names))
    ant.putcol("DISH_DIAMETER", np.full(len(names), 38.0))
    ant.putcol("POSITION", np.array([centre + o for o in offsets]))
    ant.close()

    obs = ct.table(path + "/OBSERVATION", readonly=False, ack=False)
    obs.addrows(1)
    obs.putcol("TELESCOPE_NAME", ["SKA-LOW"])
    obs.close()

    fld = ct.table(path + "/FIELD", readonly=False, ack=False)
    fld.addrows(1)
    for col in ["DELAY_DIR", "PHASE_DIR", "REFERENCE_DIR"]:
        fld.putcol(col, np.zeros((1, 1, 2)))
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
    pa = ct.table(path + "/PHASED_ARRAY", ct.maketabdesc(cols), nrow=len(names), readonly=False, ack=False)
    for i, rot in enumerate(ROTATIONS):
        pa.putcell("POSITION", i, centre + offsets[i])
        p = enu_to_itrf([np.cos(rot), np.sin(rot), 0.0])
        q = enu_to_itrf([-np.sin(rot), np.cos(rot), 0.0])
        r = enu_to_itrf([0.0, 0.0, 1.0])
        # python-casacore transposes, so the axes are rows here, and
        # ELEMENT_OFFSET is 3 x N.
        pa.putcell("COORDINATE_AXES", i, np.array([p, q, r]))
        pa.putcell("ELEMENT_OFFSET", i, enu_to_itrf(POSITIONS))
        pa.putcell("ELEMENT_FLAG", i, np.zeros((len(POSITIONS), 2), dtype=bool))
    pa.close()
    ms.putkeyword("PHASED_ARRAY", "Table: " + os.path.abspath(path) + "/PHASED_ARRAY")
    ms.close()


def main():
    import everybeam as eb

    eb_version = version("everybeam")
    old_eb = tuple(int(x) for x in eb_version.split(".")[:3]) <= (0, 9, 1)

    grid_file = HERE / "eep_grid.h5"
    feko_dir = HERE / "feko"
    write_grid(grid_file)
    write_feko(feko_dir)
    coeff = {"analytic": "", "grid": str(grid_file), "sph": str(feko_dir)}

    def to_master_order(model, j):
        # EveryBeam <= 0.9.1 orders gridded EEP correlations differently from
        # later versions (and hyperbeam).
        j = np.asarray(j).reshape(-1, 4)
        return j[:, [1, 0, 3, 2]] if model == "grid" and old_eb else j

    az, za = directions()
    theta, phi = za, (np.pi / 2 - az) % (2 * np.pi)
    s0 = enu_to_itrf(enu(*POINTING))
    dirs = [enu_to_itrf(enu(a, z)) for a, z in zip(az, za)]

    with h5py.File(HERE / "everybeam_reference.h5", "w") as out, tempfile.TemporaryDirectory() as tmp:
        out.attrs["everybeam_version"] = eb_version
        # EveryBeam after 0.9.1 applies the SKA-Low conjugation and negation to
        # gridded EEPs in SKA-Low station responses (hyperbeam's
        # `ska_low_convention`); 0.9.1 doesn't. Element responses never do.
        out.attrs["grid_station_ska_low_convention"] = int(not old_eb)
        out["az"], out["za"] = az, za
        out["positions_enu"] = POSITIONS
        out["rotations"] = np.array(ROTATIONS)
        out["pointing_az_za"] = np.array(POINTING)

        for model, freqs in TEST_FREQS.items():
            grp = out.create_group(model)
            grp["freqs"] = np.array(freqs)
            if model != "analytic":
                o = eb.Options()
                o.element_response_model = getattr(eb.ElementResponseModel, EB_MODELS[model])
                o.coeff_path = coeff[model]
                o.use_channel_frequency = True
                er = eb.ElementResponse.create(o)
                # [freq, element (sph only), direction, 4]
                elements = [0, len(POSITIONS) - 1] if model == "sph" else [0]
                grp["element_ids"] = np.array(elements)
                resp = np.zeros((len(freqs), len(elements), len(az), 4), dtype=complex)
                for fi, freq in enumerate(freqs):
                    f = np.full(len(az), freq)
                    for ei, e in enumerate(elements):
                        if model == "sph":
                            r = er.response(np.full(len(az), e, dtype=np.int32), f, theta, phi)
                        else:
                            r = er.response(f, theta, phi)
                        resp[fi, ei] = to_master_order(model, r)
                grp["element_response"] = np.stack([resp.real, resp.imag], axis=-1)

            # [freq, station, direction, 4]
            resp = np.zeros((len(freqs), len(ROTATIONS), len(az), 4), dtype=complex)
            for fi, freq in enumerate(freqs):
                ms = os.path.join(tmp, f"{model}_{fi}.ms")
                write_ska_low_ms(ms, freq)
                tel = eb.load_telescope(ms, element_response_model=EB_MODELS[model], coeff_path=coeff[model])
                for si in range(len(ROTATIONS)):
                    r = [tel.station_response(EB_TIME, si, freq, d, s0, rotate=False).reshape(4) for d in dirs]
                    resp[fi, si] = to_master_order(model, r)
            grp["station_response"] = np.stack([resp.real, resp.imag], axis=-1)

    for p in sorted(HERE.rglob("*.h5")):
        print(f"{p.relative_to(HERE)}: {p.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
