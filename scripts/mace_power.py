# Energy per force call: sample the GPU power sensor while hammering the model.
#   python mace_power.py --device cuda --tag a5000 --n 4 --seconds 60
# NVIDIA: nvidia-smi power.draw (board power), sampled at 5 Hz. Intel xe: the hwmon exposes cumulative
# energy counters (energy*_input, uJ; the "card" label is the board), not instantaneous power, so the
# mean power is dE/dt over each window. Reports J per force call and the idle baseline.
import argparse, glob, os, subprocess, threading, time
import numpy as np, torch
from ase.build import bulk
from mace.calculators import mace_mp

p = argparse.ArgumentParser()
p.add_argument("--device", required=True); p.add_argument("--tag", required=True)
p.add_argument("--n", type=int, default=4); p.add_argument("--dtype", default="float32")
p.add_argument("--seconds", type=float, default=60); p.add_argument("--cueq", action="store_true")
a = p.parse_args()

def xe_hwmon():
    for h in glob.glob("/sys/class/drm/card*/device/hwmon/hwmon*"):
        name = open(os.path.join(h, "name")).read().strip() if os.path.exists(os.path.join(h, "name")) else ""
        if name == "xe": return h
    return None

def xe_energy_file():
    h = xe_hwmon()
    if h is None: raise SystemExit("no xe hwmon (B70 lane needs the 6.18 kernel)")
    cands = sorted(glob.glob(os.path.join(h, "energy*_input")))
    if not cands: raise SystemExit("xe hwmon has no energy*_input")
    for c in cands:                       # prefer the board-level counter
        lab = c.replace("_input", "_label")
        if os.path.exists(lab) and "card" in open(lab).read().lower(): return c
    return cands[0]

def read_power_w():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]).decode()
    return float(out.strip().splitlines[0]) if False else float(out.strip().splitlines()[0])

class Sampler(threading.Thread):
    """Mean power over a window: nvidia-smi samples at 5 Hz, or xe energy counter deltas."""
    def __init__(s):
        super().__init__(daemon=True); s.v = []; s.go = True
        s.xe = None if a.device.startswith("cuda") else xe_energy_file()
        if s.xe: s.e0 = (time.perf_counter(), int(open(s.xe).read()))
    def run(s):
        while s.go:
            try:
                if s.xe:
                    t, e = time.perf_counter(), int(open(s.xe).read())
                    if t - s.e0[0] > 0.5:
                        s.v.append((t, (e - s.e0[1]) / 1e6 / (t - s.e0[0]))); s.e0 = (t, e)   # W over the last window
                else:
                    s.v.append((time.perf_counter(), read_power_w()))
            except Exception: pass
            time.sleep(0.2)
    def mean(s): return float(np.mean([w for _, w in s.v])) if s.v else float("nan")
    def max(s): return float(np.max([w for _, w in s.v])) if s.v else float("nan")

idle = Sampler(); idle.start(); time.sleep(5); idle.go = False; idle.join(1)
idle_w = idle.mean()

atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((a.n,) * 3); atoms.rattle(stdev=0.05, seed=7)
kw = dict(model="small", device=a.device, default_dtype=a.dtype)
if a.cueq: kw["enable_cueq"] = True
atoms.calc = mace_mp(**kw); atoms.get_forces()
s = Sampler(); s.start()
t0 = time.perf_counter(); calls = 0
while time.perf_counter() - t0 < a.seconds:
    atoms.rattle(stdev=1e-4, seed=calls); atoms.get_forces(); calls += 1
if a.device.startswith("cuda"): torch.cuda.synchronize()
if a.device.startswith("xpu"):  torch.xpu.synchronize()
wall = time.perf_counter() - t0; s.go = False; s.join(1)
load_w = s.mean()
src = "nvidia-smi power.draw" if a.device.startswith("cuda") else f"xe {os.path.basename(s.xe)} ({open(s.xe.replace('_input','_label')).read().strip() if os.path.exists(s.xe.replace('_input','_label')) else '?'})"
print(f"{a.tag} Si{len(atoms)} {a.dtype}{' cueq' if a.cueq else ''}: {calls} force calls in {wall:.1f} s = {wall/calls*1e3:.1f} ms/call | "
      f"GPU power [{src}] idle {idle_w:.1f} W, under load {load_w:.1f} W (max window {s.max():.1f}) | "
      f"{load_w*wall/calls:.3f} J/call board, {(load_w-idle_w)*wall/calls:.3f} J/call above idle | "
      f"{calls/wall*86400*1e-6:.3f} ns/day at 1 fs")
print("POWER-DONE")
