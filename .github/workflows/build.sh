#!/bin/bash

set -eux

# Copy the release readme to the project root so it can neatly be put in the
# release tarballs.
cp .github/workflows/releases-readme.md README.md

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PATH=/root/.cargo/bin:$PATH
    # 1.63 is the newest rustc version that can use glibc >= 2.11, and we use it
    # because newer versions require glibc >= 2.17 (which this container
    # deliberately doesn't have; we want maximum compatibility, so we use an old
    # glibc).
    rustup install 1.63 --no-self-update
    rustup default 1.63
    pip3 install maturin==0.14.13

    # Build a release for each x86_64 microarchitecture level. v4 can't be
    # compiled on GitHub for some reason.
    for level in "x86-64" "x86-64-v2" "x86-64-v3"; do
        export RUSTFLAGS="-C target-cpu=${level}"

        # Build python first
        maturin build --release --features=python,all-static --strip -i 3.7 3.8 3.9 3.10

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
    tar -acvf mwa_hyperbeam-$(git describe --tags)-MacOSX-C-library.tar.gz \
        LICENSE COPYING-hdf5 LICENSE-erfa README.md \
        libmwa_hyperbeam.{a,dylib} mwa_hyperbeam.h
    tar -acvf mwa_hyperbeam-$(git describe --tags)-MacOSX-Python.tar.gz \
        LICENSE COPYING-hdf5 LICENSE-erfa README.md \
        ./*.whl
fi
