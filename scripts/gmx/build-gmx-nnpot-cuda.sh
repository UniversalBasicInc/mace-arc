#!/usr/bin/env bash
# GROMACS 2026.3 + NNPot (libtorch cu126 from the pip wheel) + CUDA 12.6 toolkit (torch cu126 cmake refuses older), on the JARVIS host (A5000 tier).
set -euo pipefail
SRC=/data/mace-arc/gmx/gromacs-2026.3; B=/data/mace-arc/gmx/build-cuda; I=/data/mace-arc/gmx/install-cuda
export PATH=/root/.local/cmake/bin:/data/opt/cuda-12.6/bin:$PATH CUDACXX=/data/opt/cuda-12.6/bin/nvcc
TORCH=/data/mace-cuda/venv/lib/python3.12/site-packages/torch
cd "$SRC" && patch -p1 -N -r - < /data/mace-arc/patches/gromacs-2026.3-nnpot-xpu.patch || true
rm -rf "$B"; cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$I" \
  -DGMX_GPU=CUDA -DCMAKE_CUDA_COMPILER=/data/opt/cuda-12.6/bin/nvcc -DCUDAToolkit_ROOT=/data/opt/cuda-12.6 \
  -DGMX_NNPOT=TORCH -DTorch_DIR="$TORCH/share/cmake/Torch" -DCMAKE_PREFIX_PATH="$TORCH" \
  -DGMX_MPI=OFF -DGMX_OPENMP=ON -DGMX_HWLOC=OFF -DGMX_FFT_LIBRARY=fftw3 -DGMX_BUILD_OWN_FFTW=ON \
  -DREGRESSIONTEST_DOWNLOAD=OFF -DGMXAPI=OFF -DGMX_INSTALL_LEGACY_API=OFF -DGMX_EXTERNAL_BLAS=OFF -DGMX_EXTERNAL_LAPACK=OFF
grep -iE "torch|nnpot|CUDA_ARCH" "$B/CMakeCache.txt" | grep -v "^//" | head -20
cmake --build "$B" -j 48 && cmake --install "$B" && echo BUILD-CUDA-DONE
