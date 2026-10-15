#!/usr/bin/env bash
# after the benchmark chain (GPU exclusive) and the CUDA GROMACS build: NNPot Si512 case on the A5000, then a CPU-device force check
while ! grep -q CHAIN-DONE /data/mace-arc/results/cuda_chain.log || ! grep -q BUILD-CUDA-DONE /data/mace-arc/gmx/build-cuda.log; do
  grep -qE "error:|Error [0-9]" /data/mace-arc/gmx/build-cuda.log && { echo CUDA-BUILD-FAILED; exit 1; }; sleep 30; done
export HOME=/data/mace-cuda XDG_CACHE_HOME=/data/mace-cuda/.cache
source /data/mace-arc/gmx/install-cuda/bin/GMXRC
export LD_LIBRARY_PATH=/data/mace-cuda/venv/lib/python3.12/site-packages/torch/lib:/data/mace-cuda/.uvpython/cpython-3.12.15-linux-x86_64-gnu/lib:/data/opt/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}
export PATH=/data/opt/cuda-12.6/bin:$PATH
PY=/data/mace-cuda/venv/bin/python; cd /data/mace-arc/gmx
rm -rf case-si512-a5000 case-si512-cpu; cp -r case-si512 case-si512-a5000; cp -r case-si512 case-si512-cpu
echo "== A5000 (GMX_NN_DEVICE=gpu)"; export GMX_NN_DEVICE=gpu
bash /data/mace-arc/scripts/gmx/run_case.sh case-si512-a5000 -ntomp 16 -nb gpu; $PY /data/mace-arc/scripts/gmx/compare_gmx.py case-si512-a5000
grep -iE "nnpot|GMX_NN_DEVICE|Using .* device|torch" case-si512-a5000/force.log | head -5
echo "== CPU device (GMX_NN_DEVICE=cpu), force check only"; export GMX_NN_DEVICE=cpu; cd case-si512-cpu
gmx grompp -f force.mdp -c conf.gro -p topol.top -o force.tpr -maxwarn 2 > grompp-force.log 2>&1 && gmx mdrun -deffnm force -ntmpi 1 -ntomp 32 -nb cpu > mdrun-force.log 2>&1 \
 && echo 0 | gmx traj -s force.tpr -f force.trr -of forces.xvg -fp > traj.log 2>&1 && echo Potential | gmx energy -f force.edr -o energy.xvg > energy.log 2>&1 && cd .. && $PY /data/mace-arc/scripts/gmx/compare_gmx.py case-si512-cpu
echo GMX-CUDA-CASE-DONE
