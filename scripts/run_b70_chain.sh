#!/usr/bin/env bash
# B70 lane, run on a host kernel >= 6.17 (xe drives 8086:e223). One command on the host:
#   podman run --rm --device /dev/dri --security-opt label=disable -v $ROOT:/work -v $ROOT:$ROOT \
#     -v /sys/class/drm:/sys/class/drm:ro -v /etc/localtime:/etc/localtime:ro localhost/arc-b70:latest bash /work/scripts/run_b70_chain.sh
# Same scripts, same geometries, same seeds as the A5000 lane (run_cuda_chain.sh) -> results/ab/b70_*.npz compare directly.
ROOT=${MACE_ARC_ROOT:-/data/mace-arc}   # repository checkout (data disk on the original host)
export HOME=/work XDG_CACHE_HOME=/work/.cache TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/work/venv/bin/python; cd /work/scripts; OUT=/work/results/ab
echo "host load $(cut -d' ' -f1 /proc/loadavg)"; set -x
$PY mace_ab.py --device xpu --tag b70 --out $OUT
$PY mace_power.py --device xpu --tag b70 --n 4 --seconds 60
$PY mace_power.py --device xpu --tag b70 --n 6 --seconds 60
$PY mace_nvt.py --device xpu --tag b70 --n 4 --ps 10
# GROMACS NNPot on the B70 (needs install-xpu from build-gmx-nnpot-xpu.sh)
if [ -f $ROOT/gmx/install-xpu/bin/GMXRC ]; then
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  source $ROOT/gmx/install-xpu/bin/GMXRC
  export LD_LIBRARY_PATH=/work/venv/lib/python3.12/site-packages/torch/lib:/work/venv/lib:${LD_LIBRARY_PATH:-}
  export ONEAPI_DEVICE_SELECTOR=level_zero:gpu GMX_NN_DEVICE=gpu
  rm -rf $ROOT/gmx/case-si512-b70 && cp -r $ROOT/gmx/case-si512 $ROOT/gmx/case-si512-b70
  bash /work/scripts/gmx/run_case.sh $ROOT/gmx/case-si512-b70 -ntomp 16
  $PY /work/scripts/gmx/compare_gmx.py $ROOT/gmx/case-si512-b70
fi
echo B70-CHAIN-DONE
