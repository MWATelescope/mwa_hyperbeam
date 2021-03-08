#!/bin/bash

set -eux

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # I don't know why, but I need to reinstall Rust. Probably something to do with
    # GitHub overriding env variables.
    curl https://sh.rustup.rs -sSf | sh -s -- -y
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # As of 8 March 2020, the "macos-latest" GitHub actions runner is "macOS
    # 10.15", and the latest compiler for that (Homebrew GCC 10.2.0_4 ?) fails
    # to compile the HDF5 source C code, because it has an implicit declaration.
    # Use an older compiler to get around this issue.
    export CC=gcc-8
    export CXX=g++-8
    pip3 install maturin
fi

# Build python first
maturin build --release --cargo-extra-args='--features=python,hdf5-static' --strip

# Build C objects
cargo build --release --features hdf5-static

# Because we've compiled HDF5 into hyperbeam products, we legally must
# distribute the HDF5 license with the products.
curl https://raw.githubusercontent.com/HDFGroup/hdf5/develop/COPYING -o COPYING-hdf5

# Create new release asset tarballs
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    mv target/wheels/*.whl target/release/libmwa_hyperbeam.{a,so} include/mwa_hyperbeam.h .
    tar -acvf mwa_hyperbeam-$(git describe --tags)-Linux-C-library.tar.gz LICENSE COPYING-hdf5 libmwa_hyperbeam.{a,so} mwa_hyperbeam.h
    tar -acvf mwa_hyperbeam-$(git describe --tags)-Linux-Python.tar.gz LICENSE COPYING-hdf5 *.whl
elif [[ "$OSTYPE" == "darwin"* ]]; then
    mv target/wheels/*.whl target/release/libmwa_hyperbeam.{a,dylib} include/mwa_hyperbeam.h .
    tar -acvf mwa_hyperbeam-$(git describe --tags)-MacOSX-C-library.tar.gz LICENSE COPYING-hdf5 libmwa_hyperbeam.{a,dylib} mwa_hyperbeam.h
    tar -acvf mwa_hyperbeam-$(git describe --tags)-MacOSX-Python.tar.gz LICENSE COPYING-hdf5 *.whl
fi
