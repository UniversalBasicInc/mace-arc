# Self-check of the exported GROMACS wrapper in pure Python, before any GROMACS build exists:
# build GROMACS-style inputs (nm, half pair list within r_max, shift vectors in nm) with ASE's neighbour
# list, autograd the energy w.r.t. positions, and compare to the MACE ASE calculator on the same cell.
#   python check_wrapper.py --model mace_mp_small_gmx.pt --n 4 [--device cpu]
import argparse, numpy as np, torch
from ase.build import bulk
from ase.neighborlist import primitive_neighbor_list
from mace.calculators import mace_mp
p = argparse.ArgumentParser(); p.add_argument("--model", required=True); p.add_argument("--n", type=int, default=4)
p.add_argument("--device", default="cpu"); a = p.parse_args()
atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((a.n,) * 3); atoms.rattle(stdev=0.05, seed=7)
m = torch.jit.load(a.model, map_location=a.device); r_max = float(m.r_max)
i, j, S = primitive_neighbor_list("ijS", atoms.pbc, atoms.cell, atoms.positions, r_max, self_interaction=False)
# half list: keep one direction per (i,j,S) pair, as GROMACS hands it over; shift vector = S @ cell, in nm
keep = (i < j) | ((i == j) & (S[:, 0] + S[:, 1] * 2 + S[:, 2] * 4 > 0))
pairs = torch.tensor(np.stack([i[keep], j[keep]], 1), dtype=torch.int32, device=a.device)
shifts = torch.tensor((S[keep] @ atoms.cell.array) / 10.0, dtype=torch.float32, device=a.device)
pos = torch.tensor(atoms.positions / 10.0, dtype=torch.float32, device=a.device, requires_grad=True)
Z = torch.tensor(atoms.numbers, dtype=torch.int32, device=a.device)
box = torch.tensor(atoms.cell.array / 10.0, dtype=torch.float32, device=a.device)
pbc = torch.tensor([True, True, True], device=a.device)
# sign convention check: the wrapper does r_ij = pos_j - pos_i + shift_ij with shifts negated for the half list;
# try both signs of the shift and report which reproduces the calculator (the one GROMACS must match).
atoms.calc = mace_mp(model="small", device=a.device, default_dtype="float32")
f_ref = atoms.get_forces() * 96.48533212 * 10; e_ref = atoms.get_potential_energy() * 96.48533212
for sign in (+1.0, -1.0):
    e = m(pos, Z, pairs, sign * shifts, box, pbc); e.sum().backward(); f = -pos.grad.detach().cpu().numpy(); pos.grad = None
    d = f - f_ref; rms = np.sqrt((d ** 2).mean()); frms = np.sqrt((f_ref ** 2).mean())
    print(f"shift sign {sign:+.0f}: E {float(e):.3f} vs ref {e_ref:.3f} kJ/mol (dE {float(e)-e_ref:+.4f}) | force RMS diff {rms:.3e} kJ/mol/nm (rel {rms/frms:.2e}) max {np.abs(d).max():.3e}")
print(f"pairs (half list): {len(pairs)} | r_max {r_max} A")
