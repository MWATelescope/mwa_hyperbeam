# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2021-10-14
### Added
- FEE beam code for CUDA
  - The original code is courtesy of Cristian Di Pietrantonio and Maciej
    Cytowski on behalf of the Pawsey Supercomputing Centre.
  - CHJ modified it to be easily called from Rust.
  - It is is possible to run the code in single- or double-precision (Cargo
    features "cuda-single" and "cuda", respectively). This is important because
    most NVIDIA desktop GPUs have significantly less double-precision compute
    capability.
  - There are examples of using the CUDA functionality from Rust, C and Python.
- Parallactic angle correction
  - Jack Line did a thorough investigation of what our beam responses should be;
    the write up is
    [here](https://github.com/JLBLine/polarisation_tests_for_FEE).
  - New Rust functions are provided (`*_eng*` for "engineering") to get the
    old-style beam responses. The existing functions do the corrections by
    default.
- A binary `verify-beam-file`
  - (In theory) verifies that an HDF5 FEE beam file has sensible contents.
  - The only way that standard beam calculations can fail is if the spherical
    harmonic coefficients are nonsensical, so this binary is an attempt to
    ensure that the files used are sensible.

### Changed
- Rust API
  - `calc_jones*_array` functions now return a `Vec`, not an `Array1`.
- Rust internals
  - Small optimisations.
  - Small documentation clean ups.
- C API
  - The caller must now specify if they want the parallactic angle correction.
  - All functions that can fail return an error code. If this is non-zero, the
    function failed.
  - The caller can also provide error strings to these fallible functions; in
    the event of failure, an error message is written to the string.
  - The example C files have been modified to conform with these changes.
- Python API
  - The caller must now specify if they want the parallactic angle correction.
