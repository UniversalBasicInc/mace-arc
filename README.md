# MACE on Intel Arc (PyTorch XPU)

MACE-MP-0 machine-learned interatomic potentials running on Intel Arc GPUs through PyTorch's
`xpu` backend, validated against the CPU reference and an NVIDIA RTX A5000 in the same machine,
and coupled to GROMACS 2026 through its NNPot interface (with a small patch that teaches NNPot
about Intel GPUs).

**Status: private working repository.** Nothing here has been proposed upstream yet.

| Lane | What runs | Where |
|---|---|---|
| `scripts/mace_ab.py` | force A/B: same geometries, any device, forces saved as `.npz` | B70 (`xpu`), A5000 (`cuda`), EPYC (`cpu`) |
| `scripts/mace_compare.py` | lane-vs-lane force deltas against the dtype floor | any |
| `scripts/mace_power.py` | joules per force call from the GPU power sensor | A5000 (`nvidia-smi`), B70 (`xe` hwmon) |
| `scripts/mace_nve.py`, `scripts/mace_nvt.py` | 20 ps NVE drift; 10 ps Langevin NVT production-style | both |
| `scripts/gmx/` | MACE → TorchScript for GROMACS NNPot, Si512 test case, force cross-check, builds | both |
| `patches/gromacs-2026.3-nnpot-xpu.patch` | GROMACS NNPot: Intel GPU device selection + `pair-shifts` typo fix | — |

Results and the engineering record: [`docs/FINDINGS.md`](docs/FINDINGS.md). Raw logs: `results/`.

## Reproduce

```bash
# Intel GPU (inside the Babel arc-b70 container, Ubuntu 26.04 user space; host kernel >= 6.17 for the B70)
uv venv --python 3.12 venv && uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/xpu
uv pip install mace-torch==0.3.16 ase
python scripts/mace_ab.py --device xpu --tag b70 --out results/ab
# NVIDIA: same with the cu126 wheel and --device cuda (--cueq for the cuEquivariance path)
python scripts/mace_compare.py results/ab b70 a5000 cpu-epyc
```

GROMACS: `scripts/gmx/build-gmx-nnpot-{cuda,xpu}.sh`, then `export_gmx_mace.py` → `make_si_system.py` → `run_case.sh` → `compare_gmx.py`.
