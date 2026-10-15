#!/usr/bin/env bash
# Run one GROMACS NNPot case dir (from make_si_system.py): force check then NVT.  usage: run_case.sh <casedir> [mdrun flags...]
# Expects `gmx` on PATH (source the install's GMXRC) and GMX_NN_DEVICE set (gpu|cpu).
set -euo pipefail
cd "$1"; shift
gmx grompp -f force.mdp -c conf.gro -p topol.top -o force.tpr -maxwarn 2 > grompp-force.log 2>&1
gmx mdrun -deffnm force -ntmpi 1 "$@" > mdrun-force.log 2>&1
echo 0 | gmx traj -s force.tpr -f force.trr -of forces.xvg -fp > traj.log 2>&1
echo Potential | gmx energy -f force.edr -o energy.xvg > energy.log 2>&1
gmx grompp -f nvt.mdp -c conf.gro -p topol.top -o nvt.tpr -maxwarn 2 > grompp-nvt.log 2>&1
gmx mdrun -deffnm nvt -ntmpi 1 "$@" > mdrun-nvt.log 2>&1
grep -E "Performance|ns/day" nvt.log | tail -3
echo Temperature | gmx energy -f nvt.edr -o temp.xvg > temp.log 2>&1; grep -A1 "^Energy" temp.log | tail -2
echo CASE-DONE
