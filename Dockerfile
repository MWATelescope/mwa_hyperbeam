# syntax=docker/dockerfile:1
# cross-platform, cpu-only dockerfile for demoing MWA software stack
# on amd64, arm64
# ref: https://docs.docker.com/build/building/multi-platform/
# ARG BASE_IMG="ubuntu:20.04"
# HACK: newer python breaks on old ubuntu
ARG BASE_IMG="python:3.11-bookworm"
FROM ${BASE_IMG} as base

# Suppress perl locale errors
ENV LC_ALL=C
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    fontconfig \
    g++ \
    git \
    jq \
    lcov \
    libcfitsio-dev \
    liberfa-dev \
    libhdf5-dev \
    libpng-dev \
    pkg-config \
    procps \
    tzdata \
    unzip \
    wget \
    zip \
    && \
    apt-get clean all && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    apt-get -y autoremove

# Get Rust
ARG RUST_VERSION=stable
ENV RUSTUP_HOME=/opt/rust CARGO_HOME=/opt/cargo
ENV PATH="/opt/cargo/bin:${PATH}"
RUN mkdir -m755 $RUSTUP_HOME $CARGO_HOME && ( \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | env RUSTUP_HOME=$RUSTUP_HOME CARGO_HOME=$CARGO_HOME sh -s -- -y \
    --profile=minimal \
    --default-toolchain=${RUST_VERSION} \
    )

RUN python -m pip install --no-cache-dir \
    maturin[patchelf]==1.7.2 \
    ;

ADD . /app
WORKDIR /app
RUN maturin build --release --features=python && \
    python -m pip install $(ls -1 target/wheels/*.whl | tail -n 1) && \
    rm -rf ${CARGO_HOME}/registry