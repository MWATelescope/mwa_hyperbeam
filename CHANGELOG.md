# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- FFI function `calc_jones_cuda_device_inner`
  - This is the same as `calc_jones_cuda_device`, but allows the caller to pass
    in their own device buffers, so that `hyperbeam` doesn't need to allocate
    its own.

### Changed
- CUDA code now runs significantly faster.
- CUDA FFI functions now take `i32` instead of `u32` for the number of
  directions.
  - This isn't a downgrade; the internal code always used an `i32`, so it was
    dishonest for the code to look like it accepted more than `i32::MAX`
    directions.

### Fixed
- Calling CUDA FEE functions without any directions no longer causes a CUDA
  error.

## [0.5.1] - 2023-02-19
### Fixed
- A seemingly-rarely-occurring bug in CUDA FEE code.
  - Some Y dipole values were being used for X dipole values (:facepalm:), but
    despite this bug being present for many people over many thousands of
    observations, I only spotted this on a particular observation.
- Fix `get_num_unique_tiles` being unavailable for `FEEBeamCUDA`.
- Some function comments.
- Some clippy lints.

## [0.5.0] - 2022-08-23
### Added
- `calc_jones` functions have now been renamed to "_pair" functions, which take
  independent arguments of azimuths and zenith angles. The original functions
  (e.g. `FEEBeam::calc_jones`) now take `marlu::AzEl`, which may be more
  convenient for the caller by avoiding the need to allocate new arrays.

### Changed
- The minimum required Rust version is now 1.60.
- Python 3.6 support has been dropped, but 3.10 support is available.
- Rust function APIs have changed.
  - Previously, the MWA latitude was hard-coded when doing the parallactic-angle
    correction. Now, to get the correction, callers must supply a latitude.
  - The old "eng" functions have been removed, but their behaviour can be
    obtained by supplying `None` as the latitude.
  - See the note above about added "pair" functions.
- FFI function calls and error handling has changed. Please familiarise yourself
  with the new include file and/or examples.
- Function documentation is now more consistent and hopefully more readable.

### Fixed
- CUDA compilation on ozstar failed because of an arithmetic operation between
  two different types. Compilation has succeeded elsewhere, such as on Ubuntu,
  Arch, Pawsey's garrawarla and DUG. The code has changed to prevent the issue
  in the first place and no compilation issues have been spotted.
- CUDA function prototypes were being included in the C header, even if no CUDA
  feature was enabled.
- The CUDA library libcudart was always statically linked by mistake. It is now
  linked statically only if the cargo feature "cuda-static" is used, or one of
  the PKG_CONFIG environment variables is set.

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
