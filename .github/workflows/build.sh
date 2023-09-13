#!/bin/bash

# This script is assumed to be using the docker image quay.io/pypa/manylinux_2_28_x86_64
# This should mean that glibc 2.28 is being used. More details at
# https://github.com/pypa/manylinux

set -eux

# Copy the release readme to the project root so it can neatly be put in the
# release tarballs.
cp .github/workflows/releases-readme.md README.md

release=v$(grep version Cargo.toml -m1 | cut -d' ' -f3 | tr -d '"')

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none
    source "$HOME/.cargo/env"
    rustup install $(grep rust-version Cargo.toml -m1 | cut -d' ' -f3 | tr -d '"')

    python3.11 -m venv venv
    source ./venv/bin/activate
    pip3 install maturin==0.14.13

    # Build a release for each x86_64 microarchitecture level. v4 can't be
    # compiled on GitHub for some reason.
    for level in "x86-64" "x86-64-v2" "x86-64-v3"; do
        export RUSTFLAGS="-C target-cpu=${level}"

        # Build python first
        maturin build --release --features=python,all-static --strip -f

        # We don't care about PyPy, sorry.
        rm target/wheels/*pypy*

        # Build C objects
        cargo build --release --features all-static

        # Because we've compiled HDF5 and ERFA into hyperbeam products, we
        # legally must distribute their licenses with the products.
        curl https://raw.githubusercontent.com/HDFGroup/hdf5/develop/COPYING -o COPYING-hdf5
        curl https://raw.githubusercontent.com/liberfa/erfa/master/LICENSE -o LICENSE-erfa

        # Create new release asset tarballs
        mv target/wheels/*.whl target/release/libmwa_hyperbeam.{a,so} include/mwa_hyperbeam.h .
        tar -acvf mwa_hyperbeam-"${release}"-Linux-C-library-${level}.tar.gz \
            LICENSE COPYING-hdf5 LICENSE-erfa README.md \
            libmwa_hyperbeam.{a,so} mwa_hyperbeam.h
        tar -acvf mwa_hyperbeam-"${release}"-Linux-Python-${level}.tar.gz \
            LICENSE COPYING-hdf5 LICENSE-erfa README.md \
            ./*.whl
    done
elif [[ "$OSTYPE" == "darwin"* ]]; then
    pip3 install maturin==0.14.13
    brew install automake

    # Build python first
    maturin build --release --features=python,all-static --strip

    # Build C objects
    cargo build --release --features all-static

    # Because we've compiled HDF5 and ERFA into hyperbeam products, we legally
    # must distribute their licenses with the products.
    curl https://raw.githubusercontent.com/HDFGroup/hdf5/develop/COPYING -o COPYING-hdf5
    curl https://raw.githubusercontent.com/liberfa/erfa/master/LICENSE -o LICENSE-erfa

    mv target/wheels/*.whl target/release/libmwa_hyperbeam.{a,dylib} include/mwa_hyperbeam.h .
    tar -acvf mwa_hyperbeam-"${release}"-MacOSX-C-library.tar.gz \
        LICENSE COPYING-hdf5 LICENSE-erfa README.md \
        libmwa_hyperbeam.{a,dylib} mwa_hyperbeam.h
    tar -acvf mwa_hyperbeam-"${release}"-MacOSX-Python.tar.gz \
        LICENSE COPYING-hdf5 LICENSE-erfa README.md \
        ./*.whl
fi
