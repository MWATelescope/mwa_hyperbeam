# mwa_hyperbeam

<div class="bg-gray-dark" align="center" style="background-color:#24292e">
<img src="hyperbeam.png" height="200px" alt="hyperbeam logo">
<br/>
<a href="https://crates.io/crates/mwa_hyperbeam">
  <img src="https://img.shields.io/crates/v/mwa_hyperbeam?logo=rust" alt="crates.io"></a>
<a href="https://docs.rs/crate/mwa_hyperbeam">
  <img src="https://img.shields.io/docsrs/mwa_hyperbeam?logo=rust" alt="docs.rs"></a>
<img src="https://img.shields.io/github/actions/workflow/status/MWATelescope/mwa_hyperbeam/run-tests.yml?label=Cross-platform%20tests&logo=github" alt="Cross-platform%20tests">
<a href="https://codecov.io/github/MWATelescope/mwa_hyperbeam">
  <img src="https://codecov.io/github/MWATelescope/mwa_hyperbeam/branch/main/graph/badge.svg?token=61JYU54DG2"/></a>
</div>

Primary beam code for the Murchison Widefield Array (MWA) radio telescope.

This code exists to provide a single correct, convenient implementation of
[Marcin
Sokolowski's](https://ui.adsabs.harvard.edu/abs/2017PASA...34...62S/abstract)
Full Embedded Element (FEE) primary beam model of the MWA, a.k.a. "the 2016
beam". This code should be used over all others. If there are soundness issues,
please raise them here so everyone can benefit.

See the
[changelog](https://github.com/MWATelescope/mwa_hyperbeam/blob/main/CHANGELOG.md)
for the latest changes to the code.

## Polarisation order

See [this
document](https://github.com/MWATelescope/mwa_hyperbeam/blob/main/fee_pols.pdf)
for details on the polarisation order of the beam-response Jones matrices. If
the parallactic-angle correction is applied, then it is possible for the code to
re-order the Jones matrices.

## Usage

`hyperbeam` requires the MWA FEE HDF5 file. This can be obtained with:

```bash
wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5
```

When making a new beam object, `hyperbeam` needs to know where this HDF5 file
is. The easiest thing to do is set the environment variable `MWA_BEAM_FILE`:

  `export MWA_BEAM_FILE=/path/to/mwa_full_embedded_element_pattern.h5`

(On Pawsey systems, this should be `export
MWA_BEAM_FILE=/pawsey/mwa/mwa_full_embedded_element_pattern.h5`)

`hyperbeam` can be used by any programming language providing FFI via C. In
other words, most languages. See Rust, C and Python examples of usage in the
`examples` directory. A simple Python example is:

```python
>>> import mwa_hyperbeam
>>> beam = mwa_hyperbeam.FEEBeam()
>>> help(beam.calc_jones)
Help on built-in function calc_jones:

calc_jones(az_rad, za_rad, freq_hz, delays, amps, norm_to_zenith, latitude_rad, iau_order) method of builtins.FEEBeam instance
    Calculate the beam-response Jones matrix for a given direction and
    pointing. If `latitude_rad` is *not* supplied, the result will match
    the original specification of the FEE beam code (possibly more useful
    for engineers).

    Astronomers are more likely to want to specify `latitude_rad` (which
    will apply the parallactic-angle correction using the Earth latitude
    provided for the telescope) and `iau_order`. If `latitude_rad` is not
    given, then `iau_reorder` does nothing. See this document for more
    information:
    <https://github.com/MWATelescope/mwa_hyperbeam/blob/main/fee_pols.pdf>

    `delays` and `amps` apply to each dipole in an MWA tile in the M&C
    order; see
    <https://wiki.mwatelescope.org/pages/viewpage.action?pageId=48005139>.
    `delays` *must* have 16 elements, whereas `amps` can have 16 or 32
    elements; if 16 are given, then these map 1:1 with dipoles, otherwise
    the first 16 are for X dipole elements, and the next 16 are for Y.

>>> print(beam.calc_jones(0, 0.7, 167e6, [0]*16, [1]*16, True, -0.4660608448386394, True))
[-1.51506097e-01-4.35034884e-02j -9.76099405e-06-1.21699926e-05j
  1.73003520e-05-1.53580286e-05j -2.23184781e-01-4.51051073e-02j]
```

### CUDA / HIP

`hyperbeam` also can also be run on NVIDIA GPUs, or AMD GPUs. To see an example
of usage, see any of the examples with "cuda" or "hip" in the name. GPU
functionality is provided with Cargo features; see installing from source
instructions below.

## Installation

### Python PyPI

If you're using Python version >=3.6:

```bash
pip install mwa_hyperbeam
```

### Pre-compiled

Have a look at the [GitHub
releases](https://github.com/MWATelescope/mwa_hyperbeam/releases) page. There is
a Python wheel for all versions of Python 3.6+, as well as shared and static
objects for C-style linking. To get an idea of how to link `hyperbeam`, see the
`fee.c` file in the [examples
directory](https://github.com/MWATelescope/mwa_hyperbeam/tree/main/examples).

Because these `hyperbeam` objects have the HDF5 and ERFA libraries compiled in,
their respective licenses are also distributed.

### From source

#### Prerequisites

<details>

- Cargo and a Rust compiler. `rustup` is recommended:

  `https://www.rust-lang.org/tools/install`

  The Rust compiler must be at least version 1.65.0:

  ```bash
  $ rustc -V
  rustc 1.65.0 (897e37553 2022-11-02)
  ```

- [hdf5](https://www.hdfgroup.org/hdf5)
  - Optional; use the `hdf5-static` or `all-static` features.
    - Requires `CMake` version 3.10 or higher.
  - Ubuntu: `libhdf5-dev`
  - Arch: `hdf5`
  - The C library dir can be specified manually with `HDF5_DIR`
    - If this is not specified, `pkg-config` is used to find the library.

</details>

Clone the repo, and run:

```bash
export RUSTFLAGS="-C target-cpu=native" # (optional) Use native CPU features (not portable!)
cargo build --release
```

For usage with other languages, an include file will be in the `include`
directory, along with C-compatible shared and static objects in the
`target/release` directory.

#### CUDA

Are you running `hyperbeam` on a desktop GPU? Then you probably want to compile
with single-precision floats:

```bash
cargo build --release --features=cuda,gpu-single
cargo build --release --features=hip,gpu-single
```

Otherwise, go ahead with double-precision floats:

```bash
cargo build --release --features=cuda
cargo build --release --features=hip
```

Desktop GPUs (e.g. NVIDIA GeForce RTX 2070) have significantly less
double-precision compute capability than "data center" GPUs (e.g. NVIDIA V100).
Allowing `hyperbeam` to switch on the float type allows the user to decide
between the performance and precision compromise.

`CUDA` can also be linked statically:

```bash
cargo build --release --features=cuda,cuda-static
```

#### HIP

The situation with `HIP` is similar to that of `CUDA`; use the `hip` feature and
use `gpu-single` if you want the code to use single-precision floats. `HIP` does
not appear to offer static libraries, so no static feature is provided.

#### Static dependencies

To make `hyperbeam` without a dependence on a system `HDF5` library, give the
`build` command a feature flag:

```bash
cargo build --release --features=hdf5-static
```

This will automatically compile the HDF5 source code and "bake" it into the
`hyperbeam` products, meaning that HDF5 is not needed as a system dependency.
`CMake` version 3.10 or higher is needed to build the HDF5 source.

To compile all C libraries statically:

```bash
cargo build --release --features=all-static
```

#### Python

To install `hyperbeam` to your currently-in-use virtualenv or conda environment,
you'll need the Python package `maturin` (can get it with `pip`), then run:

```bash
maturin develop --release -b pyo3 --features=python --strip
```

If you don't have or don't want to install HDF5 as a system dependency, include
the `hdf5-static` feature:

```bash
maturin develop --release -b pyo3 --features=python,hdf5-static --strip
```

## Comparing with other FEE beam codes

A high-level summary is below. Further details and info on how these results
were obtained can be found [here](./comparisons).

| Package | Config | Number of directions | Duration | Max. memory usage |
|:--------|:-------|---------------------:|---------:|------------------:|
| [mwa_pb](https://github.com/MWATelescope/mwa_pb) | serial           | 1       | 14.85 ms | 153 MiB  |
|                                                  | serial           | 1,000   | 130.1 ms | 201 MiB  |
|                                                  | serial           | 300,000 | 37.94 s  | 14.4 GiB |
| [pyuvdata](https://github.com/RadioAstronomySoftwareGroup/pyuvdata) | serial | 32,760 | 7.446 s | 653 MiB |
|                                                  | serial           | 130,320 | 11.85 s  | 1.92 GiB |
| [EveryBeam](https://git.astron.nl/RD/EveryBeam)  | serial           | 1       | 114 µs   | 61.7 MiB |
|                                                  | serial           | 1,000   | 103.9 ms | 61.9 MiB |
|                                                  | serial           | 300,000 | 31.16 s  | 71.1 MiB |
| mwa_hyperbeam                                    | serial           | 1       | 32.54 µs | 11.1 MiB |
|                                                  | serial           | 1,000   | 29.02 ms | 13.3 MiB |
|                                                  | parallel         | 1,000   | 4.598 ms | 13.5 MiB |
|                                                  | serial           | 300,000 | 8.610 s  | 33.9 MiB |
|                                                  | parallel         | 300,000 | 596.1 ms | 34.6 MiB |
|                                                  | CUDA             | 300,000 | 63.70 ms | 134 MiB  |
|                                                  | CUDA             | 999,999 | 164.4 ms | 195 MiB  |

|                                           | mwa_pb             | pyuvdata           | EveryBeam | mwa_hyperbeam      |
|-------------------------------------------|:------------------:|:------------------:|:---------:|:------------------:|
| Can be run in parallel?                   | :x:                | :x:                | :x:       | :white_check_mark: |
| Parallactic-angle correction?             | :x:                | :x:                | :x:       | :white_check_mark: |
| GPU (CUDA/HIP) support?                   | :x:                | :x:                | :x:       | :white_check_mark: |
| Supports MWA analytic beam?               | :white_check_mark: | :x:                | :x:       | :white_check_mark: |
| Supports per-dipole gains?                | :white_check_mark: | :white_check_mark: | :x:       | :white_check_mark: |
| Python interface?                         | :white_check_mark: | :white_check_mark: | :x:*      | :white_check_mark: |
| Can be called from other languages via C? | :x:                | :x:                | :x:       | :white_check_mark: |
| Supports MWA CRAM tile?                   | :x:                | :x:                | :x:       | :white_check_mark: |

*: `EveryBeam` has a Python interface, but it does not support the MWA beam.

## Troubleshooting

Run your code with `hyperbeam` again, but this time with the debug build. This
should be as simple as running:

```bash
cargo build
```

and then using the results in `./target/debug`.

If that doesn't help reveal the problem, report the version of the software
used, your usage and the program output in a new GitHub issue.

## hyperbeam?

AERODACTYL used HYPER BEAM!
