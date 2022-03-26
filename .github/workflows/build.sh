#!/bin/bash

set -eux

# Copy the release readme to the project root so it can neatly be put in the
# release tarballs.
cp .github/workflows/releases-readme.md README.md

# Save the Python version for later.
PYVER=$(python -c 'import sys; print(str(sys.version_info[0])+"."+str(sys.version_info[1]))')

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    python -m pip install --upgrade pip
    python -m pip install maturin

    # Build a release for each x86_64 microarchitecture level. v4 can't be
    # compiled on GitHub for some reason.
    for level in "x86-64" "x86-64-v2" "x86-64-v3"; do
        export RUSTFLAGS="-C target-cpu=${level}"

        # Build CPU-only code first. Python before C.
        maturin build --release --cargo-extra-args="--features=python,all-static" --strip -i $(command -v python)

        # C objects.
        cargo build --release --features all-static

        # Because we've compiled HDF5 and ERFA into hyperbeam products, we
        # legally must distribute their licenses with the products.
        curl https://raw.githubusercontent.com/HDFGroup/hdf5/develop/COPYING -o COPYING-hdf5
        curl https://raw.githubusercontent.com/liberfa/erfa/master/LICENSE -o LICENSE-erfa

        # Create new release asset tarballs.
        mv target/wheels/*.whl target/release/libmwa_hyperbeam.{a,so} include/mwa_hyperbeam.h .
        tar -acvf mwa_hyperbeam-$(git describe --tags)-Linux-C-library-${level}.tar.gz \
            LICENSE COPYING-hdf5 LICENSE-erfa README.md \
            libmwa_hyperbeam.{a,so} mwa_hyperbeam.h
        tar -acvf mwa_hyperbeam-$(git describe --tags)-Linux-Python-${PYVER}-${level}.tar.gz \
            LICENSE COPYING-hdf5 LICENSE-erfa README.md \
            ./*.whl

        # Now build CUDA-enabled assets. The empty string is double precision.
        for precision in "-single" ""; do
            maturin build --release --cargo-extra-args="--features=python,all-static,cuda${precision}" --strip -i $(command -v python)
            cargo build --release --features=all-static,cuda${precision}

            mv target/wheels/*.whl target/release/libmwa_hyperbeam.{a,so} include/mwa_hyperbeam.h .
            [[ -z $precision ]] && precision="-double"
            tar -acvf mwa_hyperbeam-$(git describe --tags)-Linux-C-library-${level}-CUDA${precision}.tar.gz \
                LICENSE COPYING-hdf5 LICENSE-erfa README.md \
                libmwa_hyperbeam.{a,so} mwa_hyperbeam.h
            tar -acvf mwa_hyperbeam-$(git describe --tags)-Linux-Python-${PYVER}-${level}-CUDA${precision}.tar.gz \
                LICENSE COPYING-hdf5 LICENSE-erfa README.md \
            ./*.whl
        done
    done
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # As of 8 March 2020, the "macos-latest" GitHub actions runner is "macOS
    # 10.15", and the latest compiler for that (Homebrew GCC 10.2.0_4 ?) fails
    # to compile the HDF5 source C code, because it has an implicit declaration.
    # Use an older compiler to get around this issue.
    export CC=gcc-9
    export CXX=g++-9
    python -m pip install maturin

    # Build python first
    maturin build --release --cargo-extra-args="--features=python,all-static" --strip -i $(command -v python)

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
    tar -acvf mwa_hyperbeam-$(git describe --tags)-MacOSX-Python-${PYVER}.tar.gz \
        LICENSE COPYING-hdf5 LICENSE-erfa README.md \
        ./*.whl
fi
