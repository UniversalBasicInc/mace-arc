# MACE on Arc B70 — scaling + correctness with O(1) forces (rattle 0.05 A).
import time, numpy as np, torch
from ase.build import bulk
from mace.calculators import mace_mp
print("torch", torch.__version__, "| xpu:", torch.xpu.get_device_name(0))

def make(n):  # n = supercell repeat of the 8-atom cubic diamond cell
    a = bulk("Si", "diamond", a=5.43, cubic=True).repeat((n, n, n))
    a.rattle(stdev=0.05, seed=7)
    return a

def bench(dev, dtype, atoms, iters=5):
    calc = mace_mp(model="small", device=dev, default_dtype=dtype)
    a = atoms.copy(); a.calc = calc
    f = a.get_forces()                      # warm-up / compile
    ts = []
    for i in range(iters):
        a.rattle(stdev=1e-4, seed=i)
        t0 = time.time(); f = a.get_forces(); ts.append(time.time() - t0)
    return np.asarray(f, dtype=np.float64), np.median(ts)

for n, do64 in [(2, True), (4, True), (6, False)]:
    atoms = make(n); N = len(atoms)
    print(f"\n== Si {N} atoms")
    f32c, t32c = bench("cpu", "float32", atoms)
    f32x, t32x = bench("xpu", "float32", atoms)
    frms = np.sqrt((f32c ** 2).mean())
    d = f32x - f32c
    line = f"  cpu32 {t32c*1e3:8.1f} ms | xpu32 {t32x*1e3:8.1f} ms | xpu speedup {t32c/t32x:5.2f}x | F_rms {frms:.3f} eV/A | xpu-cpu32 RMS {np.sqrt((d**2).mean()):.2e} (rel {np.sqrt((d**2).mean())/frms:.2e}) max {np.abs(d).max():.2e}"
    if do64:
        f64, t64 = bench("cpu", "float64", atoms, iters=3)
        d32 = f32c - f64; dx = f32x - f64
        line += f"\n  vs cpu64: cpu32 rel {np.sqrt((d32**2).mean())/frms:.2e}, xpu32 rel {np.sqrt((dx**2).mean())/frms:.2e} (cpu64 {t64*1e3:.1f} ms)"
    print(line)
print("\nSCALE-DONE")
