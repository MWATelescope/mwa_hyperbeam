#!/bin/bash
echo "ROCM Version: $(/opt/rocm/bin/hipconfig --version 2>&1)" | tee -a fee_hip.log
echo "start: $(date -Is)" | tee -a fee_hip.log
export RUSTUP_HOME=/tmp/rust CARGO_HOME=/tmp/cargo PATH=/tmp/cargo/bin:$PATH
mkdir -m755 $RUSTUP_HOME $CARGO_HOME
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet     --profile=minimal --default-toolchain=1.74
. $HOME/.cargo/env
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export LIBCLANG_PATH=$ROCM_PATH/llvm/lib RUST_BACKTRACE=1
for ndir in 1 10 100 1000 10000 1000000; do
    echo "ndir=$ndir" | tee -a fee_hip.log
    cargo run --example=fee_hip --features=hip 1 mwa_full_embedded_element_pattern.h5 | tee -a fee_hip.log
done
echo "end: $(date -Is)" | tee -a fee_hip.log
