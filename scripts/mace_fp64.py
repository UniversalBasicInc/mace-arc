# fp64 MACE on the B70 (Xe2 native FP64, ~1.4 TFLOPS per boltz2-intel-arc FINDINGS §7).
import time, numpy as np, torch
from ase.build import bulk
from mace.calculators import mace_mp
print("torch", torch.__version__, "| xpu:", torch.xpu.get_device_name(0))

def bench(dev, dtype, atoms, iters=3):
    calc = mace_mp(model="small", device=dev, default_dtype=dtype)
    a = atoms.copy(); a.calc = calc
    f = a.get_forces()
    ts = []
    for i in range(iters):
        a.rattle(stdev=1e-4, seed=i)
        t0 = time.time(); f = a.get_forces(); ts.append(time.time() - t0)
    return np.asarray(f, dtype=np.float64), np.median(ts)

for n in (2, 4):
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((n, n, n))
    atoms.rattle(stdev=0.05, seed=7)
    N = len(atoms)
    f64c, t64c = bench("cpu", "float64", atoms)
    f64x, t64x = bench("xpu", "float64", atoms)
    frms = np.sqrt((f64c ** 2).mean())
    d = f64x - f64c
    print(f"Si{N}: cpu64 {t64c*1e3:7.1f} ms | xpu64 {t64x*1e3:7.1f} ms | speedup {t64c/t64x:5.2f}x | "
          f"xpu64-cpu64 RMS {np.sqrt((d**2).mean()):.2e} (rel {np.sqrt((d**2).mean())/frms:.2e}) max {np.abs(d).max():.2e} eV/A")
print("FP64-DONE")
