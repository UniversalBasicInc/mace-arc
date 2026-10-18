#!/usr/bin/env bash
# A5000 lane (kernel 6.1, driver 535), QUIET host (xmrig stopped with Dylan's go, 2026-10-15): A/B forces -> power -> NVT.
ROOT=${MACE_ARC_ROOT:-/data/mace-arc}   # repository checkout (data disk on the original host)
export HOME=/data/mace-cuda XDG_CACHE_HOME=/data/mace-cuda/.cache TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/data/mace-cuda/venv/bin/python; cd $ROOT/scripts; OUT=$ROOT/results/ab
echo "xmrig $(systemctl is-active xmrig); load $(cut -d" " -f1-3 /proc/loadavg)"; set -x
$PY mace_ab.py --device cuda --tag a5000 --out $OUT
$PY mace_ab.py --device cuda --tag a5000cueq --cueq --dtypes float32 --out $OUT
$PY mace_ab.py --device cpu  --tag cpu-epyc --iters 3 --out $OUT
$PY mace_compare.py $OUT a5000 a5000cueq cpu-epyc
$PY mace_power.py --device cuda --tag a5000 --n 4 --seconds 60
$PY mace_power.py --device cuda --tag a5000cueq --cueq --n 4 --seconds 60
$PY mace_power.py --device cuda --tag a5000 --n 6 --seconds 60
$PY mace_nvt.py --device cuda --tag a5000 --n 4 --ps 10
$PY mace_nvt.py --device cuda --tag a5000cueq --cueq --n 4 --ps 10
echo CHAIN-DONE
