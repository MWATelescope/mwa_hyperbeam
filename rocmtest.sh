LIBCLANG_PATH=/opt/rocm/llvm/lib RUST_BACKTRACE=1 cargo run --example fee_hip 1000 /opt/cal/mwa_full_embedded_element_pattern.h5



module load singularity/3.11.4-nohost
cd $MYSOFTWARE
[ -f mwa_full_embedded_element_pattern.h5 ] || wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5
git clone https://github.com/MWATelescope/mwa_hyperbeam.git --branch=setonix
cd mwa_hyperbeam

cat <<EOF > test.sh
#!/bin/bash
export ROCM_FULLVER=\$(/opt/rocm/bin/hipconfig --version 2>&1)
echo "ROCM Version: \$ROCM_FULLVER" | tee -a fee_hip.log
echo "start: \$(date -Is)" | tee -a fee_hip.log
export RUSTUP_HOME=/tmp/rust CARGO_HOME=/tmp/cargo PATH=/tmp/cargo/bin:\$PATH
mkdir -m755 \$RUSTUP_HOME \$CARGO_HOME
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --quiet \
    --profile=minimal --default-toolchain=1.74
. \$HOME/.cargo/env
export ROCM_PATH=\${ROCM_PATH:-/opt/rocm}
export LIBCLANG_PATH=\$ROCM_PATH/llvm/lib RUST_BACKTRACE=1
# seq 0 10 | sed 's/^/scale=99; a=e(/; s/$/ * l(10)); scale=0; a\/1/' | bc -l
for ndir in 1 9 99 999 9999 99999 999999; do
    echo "ndir=\$ndir" | tee -a fee_hip.log
    time cargo run --example=fee_hip --features=hip --quiet -- \$ndir mwa_full_embedded_element_pattern.h5 | tee -a fee_hip.log
done
echo "end: \$(date -Is)" | tee -a fee_hip.log
EOF
chmod +x test.sh
for ROCM_VER in 5.4.6 5.6.1 5.7.3; do # 6.0.2 6.1; do
    echo $ROCM_VER;
    export TAG="v0.3.0-setonix-rocm${ROCM_VER}"
    # export TAG="v0.3.0-setonix-rocm${ROCM_VER}"
    # singularity pull --force docker://d3vnull0/hyperdrive:$TAG
    singularity exec --rocm \
        --bind $PWD:/hyperbeam \
        --workdir /hyperbeam \
        --writable-tmpfs \
        --cleanenv \
        docker://rocm/dev-ubuntu-22.04:6.1-complete
        ./test.sh
done

        # docker://d3vnull0/hyperdrive:$TAG \
        # --bind $PWD/../hip-sys:/hip-sys
        # docker://quay.io/pawsey/rocm-mpich-base:rocm${ROCM_VER}-mpich3.4.3-ubuntu22