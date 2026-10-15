# Compare GROMACS NNPot forces/energy (nsteps=0 run) with the ASE/MACE reference on identical coordinates.
#   python compare_gmx.py case-si512   (expects forces.xvg from `gmx traj -of` and energy.xvg from `gmx energy`)
import os, sys, numpy as np
d = sys.argv[1]
ref = np.load(os.path.join(d, "ase_ref.npz"))
rows = [l.split() for l in open(os.path.join(d, "forces.xvg")) if l and l[0] not in "#@"]
fg = np.array(rows[0][1:], dtype=float).reshape(-1, 3)               # kJ/mol/nm, frame 0
fr = ref["forces_kJmolnm"]
assert fg.shape == fr.shape, (fg.shape, fr.shape)
eg = [float(l.split()[1]) for l in open(os.path.join(d, "energy.xvg")) if l and l[0] not in "#@"][0]
er = float(ref["energy_kJmol"])
d_ = fg - fr; rms = np.sqrt((d_**2).mean()); frms = np.sqrt((fr**2).mean())
print(f"{d}: GROMACS vs ASE | E {eg:.3f} vs {er:.3f} kJ/mol (dE {eg-er:+.3f} kJ/mol = {(eg-er)/96.48533212*1000:+.2f} meV total, {(eg-er)/96.48533212*1000/len(fr):+.4f} meV/atom)"
      f" | force RMS diff {rms:.3e} kJ/mol/nm (rel {rms/frms:.2e}) max {np.abs(d_).max():.3e} | F_rms {frms:.2f}")
