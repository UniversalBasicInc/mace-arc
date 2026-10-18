#!/usr/bin/env bash
# GROMACS 2026.3 SYCL (DPC++ 2026.1, Xe2 flags from Babel arc-b70.sh) + NNPot with libtorch XPU (pip torch 2.14.1+xpu),
# built INSIDE the arc-b70 container:  podman run --rm -v $ROOT:/work -v $ROOT:$ROOT localhost/arc-b70 bash /work/scripts/gmx/build-gmx-nnpot-xpu.sh
# No GPU is needed to build (JIT at first run), so this runs on the 6.1 kernel; running needs 6.18.
# setvars.sh is not clean under set -u (unbound variables kill the sourcing shell): source it first, then set the options.
ROOT=${MACE_ARC_ROOT:-/data/mace-arc}   # repository checkout (data disk on the original host)
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -euo pipefail
icpx --version | head -1
SRC=$ROOT/gmx/gromacs-2026.3; B=$ROOT/gmx/build-xpu; I=$ROOT/gmx/install-xpu
TORCH=/work/venv/lib/python3.12/site-packages/torch
# NNPot links libpython (torch extensions); the container has no python3-dev, so point cmake at uv's CPython (headers + libpython)
PYROOT=/work/.local/share/uv/python/cpython-3.12.15-linux-x86_64-gnu
export CC=icx CXX=icpx
cd "$SRC" && patch -p1 -N -r - < $ROOT/patches/gromacs-2026.3-nnpot-xpu.patch || true
rm -rf "$B"; cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$I" -DCMAKE_CXX_STANDARD=20 \
  -DGMX_GPU=SYCL -DGMX_SYCL=DPCPP -DGMX_GPU_NB_NUM_CLUSTER_PER_CELL_X=1 -DGMX_GPU_NB_CLUSTER_SIZE=8 \
  -DGMX_GPU_FFT_LIBRARY=mkl -DGMX_FFT_LIBRARY=mkl -DGMX_BUILD_OWN_FFTW=OFF \
  -DGMX_NNPOT=TORCH -DTorch_DIR="$TORCH/share/cmake/Torch" -DCMAKE_PREFIX_PATH="$TORCH" -DPython_ROOT_DIR="$PYROOT" \
  -DGMX_MPI=OFF -DGMX_OPENMP=ON -DGMX_HWLOC=OFF \
  -DREGRESSIONTEST_DOWNLOAD=OFF -DGMXAPI=OFF -DGMX_INSTALL_LEGACY_API=OFF -DGMX_EXTERNAL_BLAS=OFF -DGMX_EXTERNAL_LAPACK=OFF
grep -iE "torch|nnpot" "$B/CMakeCache.txt" | grep -v "^//" | head -20
cmake --build "$B" -j 48 && cmake --install "$B" && echo BUILD-XPU-DONE
