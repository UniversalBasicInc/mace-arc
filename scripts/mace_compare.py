# Compare force lanes written by mace_ab.py: python mace_compare.py results/ a5000 b70 cpu
# Prints RMS / max force deltas, relative to the force RMS, lane vs lane and vs the fp64 CPU reference.
import glob, os, sys, numpy as np
d = sys.argv[1]; tags = sys.argv[2:]
def load(tag, key):
    f = os.path.join(d, f"{tag}_{key}.npz")
    return np.load(f) if os.path.exists(f) else None
keys = sorted({os.path.basename(p).split("_", 1)[1][:-4] for p in glob.glob(os.path.join(d, "*_Si*.npz"))},
              key=lambda k: (int(k.split("_")[0][2:]), k))
for k in keys:
    lanes = {t: load(t, k) for t in tags}
    lanes = {t: v for t, v in lanes.items() if v is not None}
    if len(lanes) < 2: continue
    ref_t = next((t for t in tags if t in lanes and k.endswith("float64")), None)
    frms = np.sqrt((next(iter(lanes.values()))["forces"] ** 2).mean())
    print(f"== {k}  (F_rms {frms:.3f} eV/A)")
    ts = list(lanes)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            dF = lanes[ts[i]]["forces"] - lanes[ts[j]]["forces"]
            rms = np.sqrt((dF ** 2).mean())
            dE = float(lanes[ts[i]]["energy"]) - float(lanes[ts[j]]["energy"])
            print(f"   {ts[i]:>10s} vs {ts[j]:<10s} force RMS {rms:.2e} (rel {rms/frms:.2e}) max {np.abs(dF).max():.2e} eV/A | dE {dE:+.2e} eV")
    # dtype floor on the same lane
    for t in ts:
        if k.endswith("float32"):
            r64 = load(t, k.replace("float32", "float64"))
            if r64 is not None:
                dF = lanes[t]["forces"] - r64["forces"]; rms = np.sqrt((dF ** 2).mean())
                print(f"   {t:>10s} fp32 vs own fp64: force RMS {rms:.2e} (rel {rms/frms:.2e})")
