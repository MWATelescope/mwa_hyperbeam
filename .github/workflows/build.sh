#!/bin/bash

set -eux

# Copy the release readme to the project root so it can neatly be put in the
# release tarballs.
cp .github/workflows/releases-readme.md README.md

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # I don't know why, but I need to reinstall Rust. Probably something to do with
    # GitHub overriding env variables.
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal

    # Build a release for each x86_64 microarchitecture level. v4 can't be
    # compiled on GitHub for some reason.
    for level in "x86-64" "x86-64-v2" "x86-64-v3"; do
        export RUSTFLAGS="-C target-cpu=${level}"

        # Build python first
        maturin build --release --cargo-extra-args='--features=python,all-static' --strip

        # Build C objects
        cargo build --release --features all-static

        # Because we've compiled HDF5 and ERFA into hyperbeam products, we
        # legally must distribute their licenses with the products.
        curl https://raw.githubusercontent.com/HDFGroup/hdf5/develop/COPYING -o COPYING-hdf5
        curl https://raw.githubusercontent.com/liberfa/erfa/master/LICENSE -o LICENSE-erfa

        # Create new release asset tarballs
        mv target/wheels/*.whl target/release/libmwa_hyperbeam.{a,so} include/mwa_hyperbeam.h .
        tar -acvf mwa_hyperbeam-$(git describe --tags)-Linux-C-library-${level}.tar.gz \
            LICENSE COPYING-hdf5 LICENSE-erfa README.md \
            libmwa_hyperbeam.{a,so} mwa_hyperbeam.h
        tar -acvf mwa_hyperbeam-$(git describe --tags)-Linux-Python-${level}.tar.gz \
            LICENSE COPYING-hdf5 LICENSE-erfa README.md \
            ./*.whl
    done
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # As of 8 March 2020, the "macos-latest" GitHub actions runner is "macOS
    # 10.15", and the latest compiler for that (Homebrew GCC 10.2.0_4 ?) fails
    # to compile the HDF5 source C code, because it has an implicit declaration.
    # Use an older compiler to get around this issue.
    export CC=gcc-9
    export CXX=g++-9
    pip3 install maturin

    # Build python first
    maturin build --release --cargo-extra-args='--features=python,all-static' --strip

    # Build C objects
    cargo build --release --features all-static

    # Because we've compiled HDF5 and ERFA into hyperbeam products, we legally
    # must distribute their licenses with the products.
    curl https://raw.githubusercontent.com/HDFGroup/hdf5/develop/COPYING -o COPYING-hdf5
    curl https://raw.githubusercontent.com/liberfa/erfa/master/LICENSE -o LICENSE-erfa

    mv target/wheels/*.whl target/release/libmwa_hyperbeam.{a,dylib} include/mwa_hyperbeam.h .
    tar -acvf mwa_hyperbeam-$(git describe --tags)-MacOSX-C-library.tar.gz \
        LICENSE COPYING-hdf5 LICENSE-erfa README.md \
        libmwa_hyperbeam.{a,dylib} mwa_hyperbeam.h
    tar -acvf mwa_hyperbeam-$(git describe --tags)-MacOSX-Python.tar.gz \
        LICENSE COPYING-hdf5 LICENSE-erfa README.md \
        ./*.whl
fi
