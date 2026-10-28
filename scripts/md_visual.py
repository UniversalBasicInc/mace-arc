# Short MACE-MP-0 MD runs on the Arc, saving trajectories for the repository's figures.
# (a) Si512 at 600 K, (b) a 64-molecule water box at 300 K. Both Langevin, 1 fs, frames every 20 fs.
import os, sys, time, numpy as np, torch
from ase import Atoms, units
from ase.build import bulk, molecule
from ase.io import write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators import mace_mp
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "visual"); os.makedirs(out, exist_ok=True)
print("torch", torch.__version__, "| xpu:", torch.xpu.get_device_name(0), flush=True)
calc = mace_mp(model="small", device="xpu", default_dtype="float32")

def run(atoms, tag, T, ps, every=20):
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.RandomState(42))
    dyn = Langevin(atoms, 1 * units.fs, temperature_K=T, friction=0.02, rng=np.random.RandomState(7))
    steps = int(ps * 1000); fn = f"{out}/{tag}.extxyz"
    if os.path.exists(fn): os.remove(fn)
    t0 = time.time(); Ts = []
    for i in range(0, steps, every):
        dyn.run(every)
        Ts.append(atoms.get_temperature())
        write(fn, atoms, append=True, format="extxyz")
        if not np.isfinite(atoms.get_potential_energy()): print(tag, "non-finite energy"); return
    dt = time.time() - t0
    print(f"{tag}: {len(atoms)} atoms, {steps} steps ({ps} ps) at {T} K | T mean {np.mean(Ts[len(Ts)//2:]):.0f} K | "
          f"{steps/dt:.1f} steps/s = {steps*1e-6/dt*86400:.2f} ns/day | {len(Ts)} frames -> {fn}", flush=True)

# (a) silicon crystal, 600 K: visible thermal motion, stays crystalline
si = bulk("Si", "diamond", a=5.43, cubic=True).repeat((4, 4, 4))
run(si, "si512_600K", 600, 4.0)

# (b) liquid water: 64 molecules packed at random with a 2.6 A O-O exclusion, density ~1 g/cm3 (L = 12.42 A)
rng = np.random.RandomState(0); L = 12.42; mol = molecule("H2O"); mol.positions -= mol.get_center_of_mass()
pos = []
while len(pos) < 64:
    p = rng.uniform(0, L, 3)
    if all(np.linalg.norm(((p - q + L/2) % L) - L/2) > 2.6 for q in pos): pos.append(p)
w = Atoms(cell=[L, L, L], pbc=True)
for p in pos:
    m = mol.copy(); m.rotate(rng.uniform(0, 360), rng.normal(size=3)); m.translate(p); w += m
w.wrap()
# relax the random packing briefly at low T before the production segment
w.calc = calc
MaxwellBoltzmannDistribution(w, temperature_K=50, rng=np.random.RandomState(1))
Langevin(w, 0.5 * units.fs, temperature_K=50, friction=0.05, rng=np.random.RandomState(2)).run(1000)
run(w, "water64_300K", 300, 4.0)
print("VISUAL-DONE", flush=True)
