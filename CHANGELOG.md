# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.2] - 2025-04-11

- ⬆️ mwalib v1.8.7, marlu v0.16.1


## [0.10.1] - 2025-01-28

- dynamic python version in pyproject
- release CI builds linux-arm64

## [0.10.0] - 2024-11-12

- msrv1.80
- ⬆️ mwalib v1.8.2, marlu0.15.0 rubbl0.8 casacore3.5 ndarray0.16 pyo3 0.22 fitsio 0.21.6 lazy_static
- fix cuda keyring issue, update ubuntu2204 in CI
- use hdf5_metno

## [0.9.4] - 2024-08-20

Changes

- Native CPU compiler options off by default for better portability.

## [0.9.3] - 2024-07-31

Fixed

- fix a compile error when specifying env `HIP_FLAGS` with `--features=hip`

## [0.9.2] - 2024-06-21

Fixed

- reduce wasted cycles if the number of coefficients in the beam file is less than
  maximum N that the beam kernels are sized for. Since NMAX=31 and the default beam file
  is in the low 20s, this speeds things up.

## [0.9.1] - 2024-06-21

Fixed

- fix #11 : revert some of #9 , moving device memory allocations back onto the stack,
  this is fine because the kernels are smaller than before.

## [0.9.0] - 2024-05-24

Fixed

- Huge thanks to @robotopia for fixing <https://github.com/MWATelescope/mwa_hyperbeam/issues/9>

## [0.8.0] - 2024-05-24

Fixed

- update marlu 0.11.0 (no code changes)

## [0.7.2] - 2024-05-14

Fixed

- CI (no code changes).

## [0.7.1] - 2024-01-19

Fixed

- A (seemingly rare) race condition in FEE GPU code.
- examples/analytic_cuda_device.cu wasn't complete.

## [0.7.0] - 2023-10-31

Added

- Expose Rust function `fix_amps_ndarray`
- FFI functions

  See the changes below for an explanation.

  - `get_fee_freq_map`
  - `get_fee_tile_map`

Changed

- FFI functions

  Most of these changes are to disambiguate the FEE beam from the analytic beam
  and hopefully encourage a consistent naming scheme.

  - `calc_jones` -> `fee_calc_jones`
  - `calc_jones_array` -> `fee_calc_jones_array`
  - `closest_freq` -> `fee_closest_freq`
  - `calc_jones_gpu` -> `fee_calc_jones_gpu`
  - `calc_jones_gpu_device` -> `fee_calc_jones_gpu_device`
  - `calc_jones_gpu_device_inner` -> `fee_calc_jones_gpu_device_inner`
  - `get_freq_map` -> `get_fee_device_freq_map`
  - `get_tile_map` -> `get_fee_device_tile_map`
    - The old `get_*_map` functions returned device pointers. Callers should now
      use the `get_fee_device_*_map` functions for this purpose. New functions
      `get_fee_*_map` yield the host pointers for the maps.

## [0.6.1] - 2023-10-31

Added

- The "analytic" MWA beam
  - There are (unfortunately) two flavours -- "RTS" and "mwa_pb". Both are
    supported.

## [0.6.0] - 2023-09-14

Added

- Support for compiling GPU code with HIP
- FFI function `calc_jones_gpu_device_inner`
  - This is the same as `calc_jones_gpu_device`, but allows the caller to pass
    in their own device buffers, so that `hyperbeam` doesn't need to allocate
    its own.
- Set the $CXX variable to the C++ compiler in $CUDA_PATH, if $CXX is not
  already set and $CUDA_PATH/bin/g++ exists.

Changed

- The minimum required Rust version is now 1.64.
- Using single-precision floating point calculations on a GPU (CUDA or HIP) is
  now done with the `gpu-single` feature, not `cuda-single`.
- `hyperbeam` no longer depends on `ERFA`.
  - The [pure-Rust version](https://github.com/cjordan/rust-erfa) is now used
    instead, and this means that the C library is no longer required.
- CPU code now runs significantly faster.
- CUDA code now runs significantly faster.
- GPU FFI functions now take `i32` instead of `u32` for the number of
  directions.
  - This isn't a downgrade; the internal code always used an `i32`, so it was
    dishonest for the code to look like it accepted more than `i32::MAX`
    directions.
- `array_latitude_rad` arguments have been renamed to `latitude_rad`
  - This functionally changes nothing, but is maybe a little less confusing.
    "array" was used in the sense of the Murchison Widefield _Array_, not as a
    collection of numbers.

Fixed

- Calling GPU FEE functions without any directions no longer causes a GPU error.

## [0.5.1] - 2023-02-19

Fixed

- A seemingly-rarely-occurring bug in CUDA FEE code.
  - Some Y dipole values were being used for X dipole values (:facepalm:), but
    despite this bug being present for many people over many thousands of
    observations, I only spotted this on a particular observation.
- Fix `get_num_unique_tiles` being unavailable for `FEEBeamCUDA`.
- Some function comments.
- Some clippy lints.

## [0.5.0] - 2022-08-23

Added

- `calc_jones` functions have now been renamed to "_pair" functions, which take
  independent arguments of azimuths and zenith angles. The original functions
  (e.g. `FEEBeam::calc_jones`) now take `marlu::AzEl`, which may be more
  convenient for the caller by avoiding the need to allocate new arrays.

Changed

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

Fixed

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

Added

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

Changed

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
