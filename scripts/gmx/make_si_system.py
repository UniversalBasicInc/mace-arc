# Build the GROMACS inputs for a pure-NNP MACE run on a rattled Si supercell, plus the ASE reference
# forces for the SAME coordinates (read back from the .gro so both codes see identical 0.001-nm positions).
#   python make_si_system.py --n 4 --dir case-si512 --model mace_mp_small_gmx.pt --device cpu
import argparse, os, numpy as np
from ase.build import bulk
from ase.io import read, write
from mace.calculators import mace_mp

p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=4); p.add_argument("--dir", required=True)
p.add_argument("--model", required=True); p.add_argument("--device", default="cpu"); p.add_argument("--cutoff", type=float, default=0.6)
p.add_argument("--nsteps", type=int, default=10000); a = p.parse_args()
os.makedirs(a.dir, exist_ok=True)
atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((a.n,) * 3); atoms.rattle(stdev=0.05, seed=7)
write(os.path.join(a.dir, "conf.gro"), atoms)
atoms = read(os.path.join(a.dir, "conf.gro")); N = len(atoms)
open(os.path.join(a.dir, "topol.top"), "w").write(f"""; Pure-NNP topology: one Si atom type with no classical interactions; MACE supplies everything.
[ defaults ]
; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
1 2 no 1.0 1.0
[ atomtypes ]
; name at.num mass charge ptype sigma epsilon
Si 14 28.0855 0.0 A 0.0 0.0
[ moleculetype ]
SI 3
[ atoms ]
; nr type resnr residue atom cgnr charge mass
1 Si 1 SI Si 1 0.0 28.0855
[ system ]
Si{N} rattled diamond, MACE-MP-0 small via NNPot
[ molecules ]
SI {N}
""")
common = f"""cutoff-scheme = Verlet
nstlist = 10
pbc = xyz
rlist = 0.7
coulombtype = cut-off
rcoulomb = 0.7
vdwtype = cut-off
rvdw = 0.7
nnpot-active = yes
nnpot-modelfile = {os.path.abspath(a.model)}
nnpot-input-group = System
nnpot-model-input1 = atom-positions
nnpot-model-input2 = atom-numbers
nnpot-model-input3 = atom-pairs
nnpot-model-input4 = pair-shifts
nnpot-model-input5 = box
nnpot-model-input6 = pbc
nnpot-pair-cutoff = {a.cutoff}
"""
open(os.path.join(a.dir, "force.mdp"), "w").write("integrator = md\nnsteps = 0\ndt = 0.001\nnstfout = 1\nnstenergy = 1\nnstcalcenergy = 1\nnstxout = 1\n" + common)
open(os.path.join(a.dir, "nvt.mdp"), "w").write(f"integrator = md\nnsteps = {a.nsteps}\ndt = 0.001\nnstenergy = 100\nnstcalcenergy = 100\nnstlog = 1000\nnstxout-compressed = 1000\n"
    "tcoupl = v-rescale\ntc-grps = System\ntau-t = 0.1\nref-t = 300\ngen-vel = yes\ngen-temp = 300\ngen-seed = 42\ncomm-mode = linear\nnstcomm = 100\n" + common)
atoms.calc = mace_mp(model="small", device=a.device, default_dtype="float32")
f = atoms.get_forces(); e = atoms.get_potential_energy()
np.savez(os.path.join(a.dir, "ase_ref.npz"), forces_eVA=f, energy_eV=e, forces_kJmolnm=f * 96.48533212 * 10, energy_kJmol=e * 96.48533212)
print(f"Si{N}: ASE/MACE fp32 {a.device} reference E = {e:.6f} eV = {e*96.48533212:.3f} kJ/mol, F_rms {np.sqrt((f**2).mean()):.4f} eV/A")
