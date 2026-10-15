# Energy per force call: sample the GPU power sensor while hammering the model.
#   python mace_power.py --device cuda --tag a5000 --n 4 --seconds 60
# NVIDIA: nvidia-smi power.draw (board power). Intel xe: hwmon energy1_input (uJ) if present,
# else power1_input (uW). Reports J per force call and idle baseline.
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

def read_power_w():
    if a.device.startswith("cuda"):
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"]).decode()
        return float(out.strip().splitlines()[0])
    h = xe_hwmon()
    if h is None: raise SystemExit("no xe hwmon (B70 lane needs the 6.18 kernel)")
    for f in ("power1_input", "power1_max"):
        pth = os.path.join(h, f)
        if os.path.exists(pth): return int(open(pth).read()) / 1e6
    raise SystemExit("xe hwmon has no power1_input")

class Sampler(threading.Thread):
    def __init__(s): super().__init__(daemon=True); s.v = []; s.go = True
    def run(s):
        while s.go:
            try: s.v.append((time.perf_counter(), read_power_w()))
            except Exception: pass
            time.sleep(0.2)

idle = Sampler(); idle.start(); time.sleep(5); idle.go = False
idle_w = float(np.mean([w for _, w in idle.v])) if idle.v else float("nan")

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
ws = np.array([w for _, w in s.v]); load_w = float(ws.mean())
print(f"{a.tag} Si{len(atoms)} {a.dtype}{' cueq' if a.cueq else ''}: {calls} force calls in {wall:.1f} s = {wall/calls*1e3:.1f} ms/call | "
      f"GPU power idle {idle_w:.1f} W, under load {load_w:.1f} W (max {ws.max():.1f}) | "
      f"{load_w*wall/calls:.3f} J/call board, {(load_w-idle_w)*wall/calls:.3f} J/call above idle | "
      f"{calls/wall*86400*1e-6:.3f} ns/day at 1 fs")
print("POWER-DONE")
