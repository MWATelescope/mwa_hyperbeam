#!/bin/bash

# Update the Rust bindings to CUDA code (via a header). This script must be run
# whenever the CUDA code changes.

# This script requires bindgen. This can be provided by a package manager or
# installed with "cargo install bindgen".

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"

bindgen "${SCRIPTPATH}"/fee/cuda/fee.h \
    --allowlist-function "cuda_calc_jones.*" \
    --allowlist-type "FEECoeffs" \
    --size_t-is-usize \
    -- -D BINDGEN \
    > "${SCRIPTPATH}"/fee/cuda/double.rs

bindgen "${SCRIPTPATH}"/fee/cuda/fee.h \
    --allowlist-function "cuda_calc_jones.*" \
    --allowlist-type "FEECoeffs" \
    --size_t-is-usize \
    -- -D BINDGEN -D SINGLE \
    > "${SCRIPTPATH}"/fee/cuda/single.rs
