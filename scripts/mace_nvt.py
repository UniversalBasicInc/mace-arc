# Thermostatted production-style run: Langevin NVT at 300 K, 1 fs, MACE-MP-small, Si supercell.
#   python mace_nvt.py --device cuda --tag a5000 --n 4 --ps 10
# Reports T mean/std after equilibration, potential-energy stability, NaN/spike checks, ns/day.
import argparse, time, numpy as np, torch
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from mace.calculators import mace_mp

p = argparse.ArgumentParser()
p.add_argument("--device", required=True); p.add_argument("--tag", required=True)
p.add_argument("--n", type=int, default=4); p.add_argument("--dtype", default="float32")
p.add_argument("--ps", type=float, default=10); p.add_argument("--T", type=float, default=300)
p.add_argument("--cueq", action="store_true")
a = p.parse_args()

atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((a.n,) * 3)
MaxwellBoltzmannDistribution(atoms, temperature_K=a.T, rng=np.random.RandomState(42)); Stationary(atoms)
kw = dict(model="small", device=a.device, default_dtype=a.dtype)
if a.cueq: kw["enable_cueq"] = True
atoms.calc = mace_mp(**kw)
dyn = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=a.T, friction=0.01 / units.fs, rng=np.random.RandomState(7))
steps = int(a.ps * 1000); N = len(atoms); T, Ep = [], []
def log(): T.append(atoms.get_temperature()); Ep.append(atoms.get_potential_energy())
log(); t0 = time.perf_counter()
for c in range(steps // 100):
    dyn.run(100); log()
    if not np.isfinite(Ep[-1]): print(f"{a.tag}: non-finite energy at step {(c+1)*100}"); raise SystemExit(1)
wall = time.perf_counter() - t0
T = np.array(T); Ep = np.array(Ep); half = len(T) // 2
dEp = np.abs(np.diff(Ep)); spikes = int((dEp > 10 * np.median(dEp[dEp > 0])).sum()) if (dEp > 0).any() else 0
print(f"{a.tag} Si{N} {a.dtype}{' cueq' if a.cueq else ''} Langevin {a.T:.0f} K: {steps} steps ({a.ps:g} ps) | "
      f"T(2nd half) {T[half:].mean():.1f} ± {T[half:].std():.1f} K | Epot/atom 2nd half {Ep[half:].mean()/N:.4f} ± {Ep[half:].std()/N:.4f} eV "
      f"| max |dEp| between logs {dEp.max():.3f} eV | spikes>10xmed {spikes} | {steps/wall:.1f} steps/s = {steps*1e-6/(wall/86400):.3f} ns/day")
print("NVT-DONE")
