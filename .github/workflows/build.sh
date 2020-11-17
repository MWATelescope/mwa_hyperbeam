#!/bin/bash

set -eux

# I don't know why, but I need to reinstall Rust. Probably something to do with
# GitHub overriding env variables.
curl https://sh.rustup.rs -sSf | sh -s -- -y

# Build python first
/usr/bin/maturin build --release --cargo-extra-args='--features=python,hdf5-static'

# Build C objects
cargo build --release --features hdf5-static
