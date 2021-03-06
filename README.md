# mwa_hyperbeam

<div class="bg-gray-dark" align="center" style="background-color:#24292e">
<img src="hyperbeam.png" height="200px" alt="hyperbeam logo">
<br/>
<a href="https://docs.rs/crate/mwa_hyperbeam"><img src="https://docs.rs/mwa_hyperbeam/badge.svg" alt="docs"></a>
<img src="https://github.com/MWATelescope/mwa_hyperbeam/workflows/Tests/badge.svg" alt="Tests">
</div>

Primary beam code for the Murchison Widefield Array (MWA) radio telescope.

This code exists to provide a single correct, convenient implementation of
[Marcin
Sokolowski's](https://ui.adsabs.harvard.edu/abs/2017PASA...34...62S/abstract)
Full Embedded Element (FEE) primary beam model of the MWA, a.k.a. "the 2016
beam". This code should be used over all others. If there are soundness issues,
please raise them here so everyone can benefit.

## Usage
`hyperbeam` requires the MWA FEE HDF5 file. This can be obtained with:

  `wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5`

When making a new beam object, `hyperbeam` needs to know where this HDF5 file
is. The easiest thing to do is set the environment variable `MWA_BEAM_FILE`:

  `export MWA_BEAM_FILE=/path/to/mwa_full_embedded_element_pattern.h5`

(On Pawsey systems, this should be `export
MWA_BEAM_FILE=/pawsey/mwa/mwa_full_embedded_element_pattern.h5`)

`hyperbeam` can be used by any programming language providing FFI via C. In
other words, most languages. See Rust, C and Python examples of usage in the
`examples` directory. A simple Python example is:

    import mwa_hyperbeam
    beam = mwa_hyperbeam.FEEBeam()
    print(beam.calc_jones(0, 0.7, 167e6, [0]*16, [1]*16, True))

    [ 1.73003520e-05-1.53580286e-05j -2.23184781e-01-4.51051073e-02j
     -1.51506097e-01-4.35034884e-02j -9.76099405e-06-1.21699926e-05j]

## Installation
### Python PyPI
If you're using Python version >=3.6:

    pip install mwa_hyperbeam

### Pre-compiled
Have a look at the [GitHub
releases](https://github.com/MWATelescope/mwa_hyperbeam/releases) page. There is
a Python wheel for all versions of Python 3.6+, as well as shared and static
objects for C-style linking. To get an idea of how to link `hyperbeam`, see the
`beam_calcs.c` file in the examples directory.

Because these `hyperbeam` objects have the HDF5 library compiled in, the HDF5
license is also distributed.

### From source
#### Prerequisites
<details>

- Cargo and a Rust compiler. `rustup` is recommended:

  `https://www.rust-lang.org/tools/install`

  The Rust compiler must be at least version 1.47.0:
  ```bash
  $ rustc -V
  rustc 1.47.0 (18bf6b4f0 2020-10-07)
  ```

- [hdf5](https://www.hdfgroup.org/hdf5)
  - Optional; use the `hdf5-static` feature.
  - Ubuntu: `libhdf5-dev`
  - Arch: `hdf5`

</details>

Clone the repo, and run:

    cargo build --release

For usage with other languages, an include file will be in the `include`
directory, along with C-compatible shared and static objects in the
`target/release` directory.

To make `hyperbeam` without a dependence on a system HDF5 library, give the
`build` command a feature flag:

    cargo build --release --features=hdf5-static

This will automatically compile the HDF5 source code and "bake" it into the
`hyperbeam` products, meaning that HDF5 is not needed as a system dependency.
`CMake` version 3.10 or higher is needed to build the HDF5 source.

#### Python
To install `hyperbeam` to your currently-in-use virtualenv or conda environment,
you'll need the Python package `maturin` (can get it with `pip`), then run:

    maturin develop --release -b pyo3 --cargo-extra-args="--features python" --strip

If you don't have or don't want to install HDF5 as a system dependency, include
the `hdf5-static` feature:

    maturin develop --release -b pyo3 --cargo-extra-args="--features python,hdf5-static" --strip

## Comparing with other FEE beam codes
Below is a table comparing other implementations of the FEE beam code. All
benchmarks were done with unique azimuth and zenith angle directions, and all
on the same system. The CPU is a Ryzen 9 3900X, which has 12 cores and SMT (24
threads). All benchmarks were done in serial, unless indicated by "parallel".
Python times were taken by running `time.time()` before and after the
calculations. Memory usage is measured by running `time -v` on the command (not
the `time` associated with your shell; this is usually at `/usr/bin/time`).

| Code             | Number of directions | Duration | Max. memory usage |
|:-----------------|---------------------:|---------:|------------------:|
| [mwa_pb](https://github.com/MWATelescope/mwa_pb) | 500     | 98.8 ms  | 134.6 MiB |
|                                                  | 100000  | 13.4 s   | 5.29 GiB  |
|                                                  | 1000000 | 139.8 s  | 51.6 GiB  |
| mwa-reduce (C++)                                 | 500     | 115.2 ms | 48.9 MiB  |
|                                                  | 10000   | 2.417 s  | 6.02 GiB  |
| mwa_hyperbeam                                    | 500     | 30.8 ms  | 9.82 MiB  |
|                                                  | 100000  | 2.30 s   | 17.3 MiB  |
|                                                  | 1000000 | 22.5 s   | 85.6 MiB  |
| mwa_hyperbeam (parallel)                         | 1000000 | 1.73 s   | 86.1 MiB  |
| mwa_hyperbeam (via python)                       | 500     | 28.5 ms  | 35.0 MiB  |
|                                                  | 100000  | 4.25 s   | 51.5 MiB  |
|                                                  | 1000000 | 44.0 s   | 203.8 MiB |
| mwa_hyperbeam (via python, parallel)             | 1000000 | 3.40 s   | 203.2 MiB |

Not sure what's up with the C++ code. Maybe I'm calling `CalcJonesArray` wrong,
but it uses a huge amount of memory. In any case, `hyperbeam` seems to be
roughly 10x faster.

## Troubleshooting

Run your code with `hyperbeam` again, but this time with the debug build. This
should be as simple as running:

    cargo build
    
and then using the results in `./target/debug`.

If that doesn't help reveal the problem, report the version of the software
used, your usage and the program output in a new GitHub issue.

## hyperbeam?
AERODACTYL used HYPER BEAM!
