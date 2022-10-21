#!/bin/bash

# Update the Rust bindings to GPU code (via a header). This script must be run
# whenever the GPU code changes.

# This script requires bindgen. This can be provided by a package manager or
# installed with "cargo install bindgen-cli".

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"

bindgen "${SCRIPTPATH}"/fee/gpu/fee.h \
    --allowlist-function "gpu_calc_jones.*" \
    --allowlist-type "FEECoeffs" \
    -- -D BINDGEN \
    > "${SCRIPTPATH}"/fee/gpu/double.rs

bindgen "${SCRIPTPATH}"/fee/gpu/fee.h \
    --allowlist-function "gpu_calc_jones.*" \
    --allowlist-type "FEECoeffs" \
    -- -D BINDGEN -D SINGLE \
    > "${SCRIPTPATH}"/fee/gpu/single.rs
