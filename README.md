# mwa_hyperbeam

Primary beam code for the Murchison Widefield Array (MWA) radio telescope.

This code exists to provide a single correct, convenient implementation of
[Marcin
Sokolowski's](https://ui.adsabs.harvard.edu/abs/2017PASA...34...62S/abstract)
Full Embedded Element (FEE) primary beam model of the MWA, a.k.a. "the 2016
beam". This code should be used over all others. If there are soundness issues,
please raise them here so everyone can benefit.

## Usage
`hyperbeam` can be used by any programming language providing FFI via C. In
other words, most languages. See Rust and C examples in the `examples`
directory.

### Python
`hyperbeam` can also be used via Python. This requires installation of the
Python package `maturin`. To install `hyperbeam` to your currently-in-use
virtualenv or conda environment, run:

  `maturin develop --release -b pyo3 --cargo-extra-args="--features python"`

See an example of usage in the `examples` directory.

Unfortunately it is not possible to provide `hyperbeam` via PyPI. This is
because `hyperbeam` depends on the C library of HDF5, and it will differ across
platforms. In other words, `hyperbeam` needs to be built from source.

## Comparing with other FEE beam codes
Below is a table comparing other implementations of the FEE beam code. All
benchmarks were done with unique azimuth and zenith angle pointings, and all on
the same system. The CPU is a Ryzen 9 3900X, which has 12 cores and SMT (24
threads). All benchmarks were done in serial, unless indicated by "parallel".
Python times were taken by running `time.time()` before and after the
calculations. Memory usage is measured by running `time -v` on the command (not
the `time` associated with your shell; this is usually at `/usr/bin/time`).

| Code             | Number of pointings | Duration | Max. memory usage |
|:-----------------|--------------------:|---------:|------------------:|
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

## Installation

### Prerequisites
<details>

- Cargo and a Rust compiler. `rustup` is recommended:

  `https://www.rust-lang.org/tools/install`

- [hdf5](https://www.hdfgroup.org/hdf5)
  - Ubuntu: `libhdf5-dev`
  - Arch: `hdf5`

</details>

### From source

Clone the repo, and run:

    cargo build --release

For usage with other languages, an include file will be in the `include`
directory, along with shared and static objects in the `target/release`
directory.

## Troubleshooting

Run your code with `hyperbeam` again, but this time with the debug build. This should be as simple as running:

    cargo build
    
And then linking against the object in `./target/debug`.

If that doesn't help reveal the problem, report the version of the software
used, your usage and the program output in a new GitHub issue.

## hyperbeam?
AERODACTYL used HYPER BEAM!
