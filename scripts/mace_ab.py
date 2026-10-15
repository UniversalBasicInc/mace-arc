# Device A/B for MACE-MP-small forces: one script, any torch device (xpu / cuda / cpu).
# Writes forces as .npz so lanes run on different days / kernels / machines compare directly.
#   python mace_ab.py --device cuda --tag a5000 [--cueq] [--dtypes float32,float64] [--sizes 2,4,6]
# Inputs are deterministic (rattle seeds fixed) so every lane sees identical geometries.
import argparse, json, os, platform, socket, time
import numpy as np, torch
from ase.build import bulk
from mace.calculators import mace_mp

p = argparse.ArgumentParser()
p.add_argument("--device", required=True)
p.add_argument("--tag", required=True)
p.add_argument("--cueq", action="store_true", help="cuEquivariance fast path (NVIDIA only)")
p.add_argument("--dtypes", default="float32,float64")
p.add_argument("--sizes", default="2,4,6", help="supercell repeats of the 8-atom Si cell")
p.add_argument("--iters", type=int, default=10)
p.add_argument("--out", default="results")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

def devname(d):
    if d.startswith("cuda"): return torch.cuda.get_device_name(0)
    if d.startswith("xpu"):  return torch.xpu.get_device_name(0)
    return platform.processor() or "cpu"

print(f"torch {torch.__version__} | {a.device} = {devname(a.device)} | host {socket.gethostname()} {platform.release()}", flush=True)

def make(n):
    s = bulk("Si", "diamond", a=5.43, cubic=True).repeat((n, n, n))
    s.rattle(stdev=0.05, seed=7)        # O(1) eV/A forces, identical in every lane
    return s

def bench(dtype, atoms, iters):
    kw = dict(model="small", device=a.device, default_dtype=dtype)
    if a.cueq: kw["enable_cueq"] = True
    calc = mace_mp(**kw)
    s = atoms.copy(); s.calc = calc
    s.get_forces()                      # warm-up: model load, JIT, allocator
    ts = []
    for i in range(iters):
        s.rattle(stdev=1e-4, seed=i)    # same perturbation sequence in every lane
        t0 = time.perf_counter(); f = s.get_forces()
        if a.device.startswith("cuda"): torch.cuda.synchronize()
        if a.device.startswith("xpu"):  torch.xpu.synchronize()
        ts.append(time.perf_counter() - t0)
    # comparison geometry = the base rattled cell, identical in every lane regardless of --iters
    s.positions = atoms.positions.copy()
    f = s.get_forces(); e = s.get_potential_energy()
    return np.asarray(f, dtype=np.float64), float(e), ts

meta = {"device": a.device, "device_name": devname(a.device), "torch": torch.__version__,
        "cueq": a.cueq, "host": socket.gethostname(), "kernel": platform.release(), "runs": {}}
for n in [int(x) for x in a.sizes.split(",")]:
    atoms = make(n); N = len(atoms)
    for dt in a.dtypes.split(","):
        if dt == "float64" and n >= 6: continue   # fp64 lane only up to Si512 (CPU reference cost)
        f, e, ts = bench(dt, atoms, a.iters)
        key = f"Si{N}_{dt}"
        np.savez(os.path.join(a.out, f"{a.tag}_{key}.npz"), forces=f, energy=e, times=np.array(ts))
        med = float(np.median(ts)); mn = float(np.min(ts))
        meta["runs"][key] = {"N": N, "median_ms": med * 1e3, "min_ms": mn * 1e3, "energy_eV": e}
        print(f"  {a.tag:12s} {key:16s} median {med*1e3:8.2f} ms  min {mn*1e3:8.2f} ms  E {e:+.6f} eV", flush=True)
json.dump(meta, open(os.path.join(a.out, f"{a.tag}_meta.json"), "w"), indent=1)
print("AB-DONE")
