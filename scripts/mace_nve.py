# NVE stability: energy drift, spikes, temperature, ns/day. Chaos makes trajectories
# incomparable across devices; the comparable things are drift rates and spike counts.
import time, numpy as np, torch
from ase.build import bulk
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units
from mace.calculators import mace_mp
print("torch", torch.__version__, "| xpu:", torch.xpu.get_device_name(0), flush=True)

def nve(dev, dtype, n, steps, tag):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((n, n, n))
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.RandomState(42))
    Stationary(atoms)
    atoms.calc = mace_mp(model="small", device=dev, default_dtype=dtype)
    dyn = VelocityVerlet(atoms, timestep=1.0 * units.fs)
    N = len(atoms); E, T = [], []
    def log():
        e = atoms.get_potential_energy() + atoms.get_kinetic_energy()
        E.append(e); T.append(atoms.get_temperature())
    log()
    t0 = time.time()
    for chunk in range(steps // 50):
        dyn.run(50); log()
        if np.isnan(E[-1]): print(f"{tag}: NaN at step {(chunk+1)*50}"); return
    wall = time.time() - t0
    E = np.array(E); T = np.array(T)
    t_ps = np.arange(len(E)) * 50 * 1e-3  # ps
    drift = np.polyfit(t_ps, E, 1)[0]     # eV/ps
    dE = np.abs(np.diff(E))
    spikes = int((dE > 10 * np.median(dE[dE > 0] + 1e-30)).sum())
    nsday = (steps * 1e-6) / (wall / 86400.0)
    print(f"{tag}: {steps} steps ({steps/1000:.0f} ps) | drift {drift*1000/N:+.4f} meV/atom/ps "
          f"| E spread {E.max()-E.min():.4f} eV | spikes>10xmed {spikes} | T {T.mean():.0f}±{T.std():.0f} K "
          f"| {steps/wall:.1f} steps/s = {nsday:.3f} ns/day", flush=True)

nve("xpu", "float32", 4, 20000, "Si512 xpu32")
nve("xpu", "float32", 2, 20000, "Si64  xpu32")
nve("xpu", "float64", 2, 20000, "Si64  xpu64")
nve("cpu", "float32", 2, 20000, "Si64  cpu32")
print("NVE-DONE")
