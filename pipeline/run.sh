#!/usr/bin/env bash
# End-to-end GROMACS + MACE on an Intel GPU, in the container from pipeline/Containerfile:
# export MACE-MP-0 for GROMACS, build a 512-atom silicon test system with its ASE reference forces,
# run a single-point force check and a 10 ps NVT run in GROMACS with the model on the GPU,
# and compare GROMACS's forces with ASE's.
#   pipeline/run.sh [outdir]          (ENGINE=docker to use docker instead of podman; IMAGE to override)
# Without a visible Intel GPU it stops; ALLOW_CPU=1 runs the same checks on the CPU with a 20-step NVT.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mkdir -p "${1:-$REPO/results/pipeline}" && cd "${1:-$REPO/results/pipeline}" && pwd)
ENGINE=${ENGINE:-podman}; IMAGE=${IMAGE:-gmx-mace-xpu}
"$ENGINE" image inspect "$IMAGE" >/dev/null 2>&1 || "$ENGINE" build -t "$IMAGE" "$REPO/pipeline"
"$ENGINE" run --rm --device /dev/dri -e ALLOW_CPU="${ALLOW_CPU:-0}" -v "$REPO:/repo:ro" -v "$OUT:/out" "$IMAGE" bash -c '
set -euo pipefail
cd /out
cat /opt/gromacs-commit
NSTEPS=10000
if ! python3 -c "import torch; ok = torch.xpu.is_available(); print(\"torch\", torch.__version__, \"| Intel GPU:\", torch.xpu.get_device_name(0) if ok else \"none\"); raise SystemExit(0 if ok else 1)"; then
  if [ "$ALLOW_CPU" != 1 ]; then echo "No Intel GPU visible in the container (check /dev/dri and the xe driver); ALLOW_CPU=1 to run on the CPU" >&2; exit 2; fi
  NSTEPS=20
fi
python3 /repo/scripts/gmx/export_gmx_mace.py --out /out/mace_mp_small_gmx.pt
python3 /repo/scripts/gmx/make_si_system.py --n 4 --dir /out/case-si512 --model /out/mace_mp_small_gmx.pt --nsteps $NSTEPS
bash /repo/scripts/gmx/run_case.sh /out/case-si512
grep -hE "GMX_NN_DEVICE|Using .*device|Using CPU" /out/case-si512/force.log | head -2
python3 /repo/scripts/gmx/compare_gmx.py /out/case-si512 | tee /out/compare.txt
python3 - <<PY
import re
rel = float(re.search(r"rel ([0-9.e+-]+)", open("/out/compare.txt").read()).group(1))
print("PIPELINE", "PASS" if rel < 1e-4 else "FAIL", f"(force rel RMS {rel:.2e}, threshold 1e-4)")
PY
'
