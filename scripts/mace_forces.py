# MACE on Intel Arc (torch-xpu) — preliminary correctness + timing probe.
# Compares forces/energy across (device, dtype) lanes so the device delta is read
# against the dtype floor, Babel-style:
#   cpu/float64  = reference
#   cpu/float32  = dtype floor
#   xpu/float32  = the lane under test
import sys, time, traceback
import numpy as np
import torch

print("torch", torch.__version__, "| xpu available:", torch.xpu.is_available())
if not torch.xpu.is_available():
    sys.exit("no xpu device — stop")
print("xpu device:", torch.xpu.get_device_name(0))

from ase.build import molecule, bulk
from mace.calculators import mace_mp

systems = {
    "H2O": molecule("H2O"),
    "Si64": bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2)),
}

def run(dev, dtype, atoms):
    calc = mace_mp(model="small", device=dev, default_dtype=dtype)
    a = atoms.copy()
    a.calc = calc
    e = a.get_potential_energy()          # warm-up (includes model load/compile)
    f = a.get_forces()
    t0 = time.time()
    a.rattle(stdev=1e-6, seed=1)          # force a recompute
    f = a.get_forces()
    e = a.get_potential_energy()
    dt = time.time() - t0
    return e, np.asarray(f, dtype=np.float64), dt

lanes = [("cpu", "float64"), ("cpu", "float32"), ("xpu", "float32")]
for name, atoms in systems.items():
    print(f"\n== {name} ({len(atoms)} atoms)")
    res = {}
    for dev, dt_ in lanes:
        try:
            e, f, wall = run(dev, dt_, atoms)
            res[(dev, dt_)] = (e, f, wall)
            print(f"  {dev}/{dt_:8s}  E = {e:+.6f} eV   warm call {wall*1000:7.1f} ms")
        except Exception:
            print(f"  {dev}/{dt_:8s}  FAILED:")
            traceback.print_exc()
    ref = res.get(("cpu", "float64"))
    if ref is None:
        continue
    frms = np.sqrt((ref[1] ** 2).mean())
    for dev, dt_ in lanes[1:]:
        if (dev, dt_) in res:
            e, f, _ = res[(dev, dt_)]
            d = f - ref[1]
            print(f"  {dev}/{dt_} vs cpu/float64: dE = {e-ref[0]:+.3e} eV, "
                  f"force RMS diff {np.sqrt((d**2).mean()):.3e} (rel {np.sqrt((d**2).mean())/frms:.3e}), "
                  f"max {np.abs(d).max():.3e} eV/A")
    if ("cpu", "float32") in res and ("xpu", "float32") in res:
        d = res[("xpu", "float32")][1] - res[("cpu", "float32")][1]
        print(f"  xpu/float32 vs cpu/float32 (same dtype): force RMS diff {np.sqrt((d**2).mean()):.3e}, "
              f"max {np.abs(d).max():.3e} eV/A")
print("\nPROBE-DONE")
