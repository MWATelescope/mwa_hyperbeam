#!/bin/bash

# This script compares a bunch of different packages' implementations of the MWA
# FEE beam code. Please file an issue if this comparison is very misleading or
# incorrect.
#
# Requirements:
# - /usr/bin/time
# - Python (with a venv module inside it)
# - everybeam (https://git.astron.nl/RD/EveryBeam)
# - a Rust toolchain (see rustup)
#
# Optional:
# - CUDA

set -eu

# https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
SCRIPTPATH="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P)"

# Set up a Python env 
if [ ! -r venv ]; then
  python3 -m venv venv
  . ./venv/bin/activate
  pip install mwa_hyperbeam everybeam mwa_pb pyuvdata
else
  . ./venv/bin/activate
fi

# Adjust to suit your system
export MWA_BEAM_FILE="${SCRIPTPATH}/$(find venv -type f -name mwa_full_embedded_element_pattern.h5)" # Automatically provided by mwa_pb
# export MWA_BEAM_FILE="/usr/local/mwa_full_embedded_element_pattern.h5" # A hard-coded path
export MS="${SCRIPTPATH}/1090008640_2s_40kHz.trunc.ms"
USE_CUDA=1

if [ ! -r "${MWA_BEAM_FILE}" ]; then
  echo "The MWA HDF5 beam file doesn't exist; adjust the variable"
  exit 1
fi
if [ ! -r "${MS}" ]; then
  # Extract the MS out of the tarball
  if [ ! -r "${MS}.tar.gz" ]; then
    echo "The measurement set tarball needed for EveryBeam doesn't exist; adjust the variable"
    exit 1
  fi
  tar -xf "${MS}.tar.gz"
fi

#####

run_and_measure_memory () {
  local COMMAND_AND_ARGS=("$@")

  # https://stackoverflow.com/a/59592881
  {
      IFS=$'\n' read -r -d '' STDERR;
      IFS=$'\n' read -r -d '' STDOUT;
  } < <((printf '\0%s\0' "$(/usr/bin/time -v "${COMMAND_AND_ARGS[@]}")" 1>&2) 2>&1)

  echo "${STDOUT}"
  MEMORY=$(echo "${STDERR}" | grep "Maximum resident set size" | cut -d: -f2 | cut -d' ' -f2-)
  echo "Max memory use (kBytes): ${MEMORY}"
}

print_sys_info () {
  echo "*** System information ***"
  echo "uname -a:"
  echo "  $(uname -a)"

  echo "CPU:"
  NAME=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | cut -d' ' -f2-)
  CORES=$(grep -m1 "cpu cores" /proc/cpuinfo | cut -d: -f2 | cut -d' ' -f2-)
  THREADS=$(grep -m1 "siblings" /proc/cpuinfo | cut -d: -f2 | cut -d' ' -f2-)
  echo "  ${NAME} (${CORES} cores, ${THREADS} threads)"

  if [ $USE_CUDA == 1 ]; then
    echo "GPU:"
    echo "  $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
  fi

  echo "Total memory:"
  echo "  $(free -m | grep -m1 "Mem:" | sed -e 's|\(Mem:\s\+\)\([0-9]\+\)\(.*\)|\2|') MiB"

  local GLIBC=/usr/lib/libc.so.6
  if [ -r $GLIBC ]; then
    echo "glibc:"
    echo "  $($GLIBC | head -n1)"
  fi

  echo "Compilers:"
  echo "  GCC: $(g++ --version | head -n1)"
  echo "  Rust: $(rustc --version)"
  if [ $USE_CUDA == 1 ]; then
    echo "  nvcc: $(nvcc --version | grep release)"
  fi

  echo "Python:"
  echo "  $(python --version)"

  echo "***"
  echo ""
}

run_mwa_pb_python () {
  echo "*** mwa_pb Python results ***"
  run_and_measure_memory ./mwa_pb_example.py
  echo "***"
  echo ""
}

run_everybeam_python () {
  echo "*** EveryBeam Python results ***"
  run_and_measure_memory ./everybeam_example.py
  echo "***"
  echo ""
}

run_pyuvdata_python () {
  echo "*** pyuvdata Python results ***"
  run_and_measure_memory ./pyuvdata_example.py
  echo "***"
  echo ""
}

run_hyperbeam_python () {
  echo "*** hyperbeam Python results with 1 CPU core ***"
  export RAYON_NUM_THREADS=1
  run_and_measure_memory ./hyperbeam_example.py
  unset RAYON_NUM_THREADS

  echo ""
  echo "*** hyperbeam Python results with all CPU cores ***"
  if [ $USE_CUDA == 1 ]; then
    run_and_measure_memory ./hyperbeam_example.py cuda
  else
    run_and_measure_memory ./hyperbeam_example.py
  fi

  echo "***"
  echo ""
}

run_everybeam_cpp () {
  echo "*** Compiling EveryBeam C++ example ***"
  make
  echo ""
  echo "*** EveryBeam C++ results with 1 CPU core ***"
  run_and_measure_memory ./everybeam_example "${MS}"
  echo "***"
  echo ""
}

run_hyperbeam_rust () {
  echo "*** Compiling hyperbeam Rust code ***"
  if [ $USE_CUDA == 1 ]; then
    cargo build --release --features=cuda,gpu-single
  else
    cargo build --release
  fi
  echo ""

  echo "*** hyperbeam Rust results with 1 CPU core ***"
  export RAYON_NUM_THREADS=1
  run_and_measure_memory ./target/release/hyperbeam
  unset RAYON_NUM_THREADS
  echo ""

  echo "*** hyperbeam Rust results with all CPU cores ***"
  run_and_measure_memory ./target/release/hyperbeam

  if [ $USE_CUDA == 1 ]; then
    echo ""
    echo "*** hyperbeam Rust results with CUDA ***"
    run_and_measure_memory ./target/release/hyperbeam-cuda
  fi

  echo "***"
  echo ""
}

print_sys_info
run_mwa_pb_python
# run_everybeam_python # MWA not supported
run_pyuvdata_python
run_hyperbeam_python
cd everybeam
run_everybeam_cpp
cd ../hyperbeam
run_hyperbeam_rust
