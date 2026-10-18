# MACE on Intel Arc

Run [MACE](https://github.com/ACEsuit/mace) machine-learned interatomic potentials on Intel Arc GPUs,
with the numbers to trust it: forces validated against the CPU reference and against an NVIDIA RTX A5000
in the same machine, molecular-dynamics stability checked, and GROMACS 2026 driving the model on the Arc
through its NNPot interface.

**MACE itself needs no change.** PyTorch's `xpu` backend is enough. What this repository adds is the
recipe, the validation record, and the GROMACS side: two contributions submitted upstream, and a patch
for the 2026.3 release for people who want to build today.

## Use it now

```bash
uv venv --python 3.12 venv && source venv/bin/activate
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/xpu
uv pip install mace-torch==0.3.16 ase
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # mace-torch 0.3.16 checkpoints are full pickles; torch 2.14 needs this
```

```python
from mace.calculators import mace_mp
calc = mace_mp(model="small", device="xpu")          # that is the whole port
```

Host requirements: an Intel GPU on the `xe` kernel driver and the Intel compute runtime (Level Zero).
The Arc Pro B70 (`8086:e223`) needs kernel 6.17 or later; the `torch` XPU wheel brings its own SYCL runtime.
The cuEquivariance fast path is NVIDIA-only, so Intel GPUs run the plain e3nn tensor-product path.

## What was measured

Arc Pro B70 (Battlemage, 32 GB) against an RTX A5000 and a 128-thread EPYC 7742 in the same chassis,
MACE-MP-0 small, diamond-Si supercells of 64, 512 and 1728 atoms. Full tables and caveats in
[`docs/FINDINGS.md`](docs/FINDINGS.md); raw logs and force arrays in `results/`.

| | result |
|---|---|
| Forces, fp32, B70 vs A5000 | 1.6–1.8e-06 relative RMS, below each card's own fp32-vs-fp64 floor |
| Forces, fp64, B70 vs A5000 and vs CPU | 4e-15: machine epsilon, the device is not in the answer |
| Speed, e3nn path, fp32 | B70 44 ms / A5000 43 ms per force call at Si512; 128 / 128 ms at Si1728; both ~10× the EPYC |
| Speed, fp64, Si512 | B70 110 ms, A5000 230 ms: **2.1× faster on the Arc** (Xe2 fp64 at 1/14 rate, GA102 at 1/32) |
| Energy per step, e3nn path | B70 7.3 J vs A5000 9.1 J at Si512: 20 % less |
| NVE, 20 ps at 1 fs | zero drift, no NaNs, statistically identical to the CPU lane |
| NVT, 10 ps Langevin | identical temperature and energy statistics on B70, A5000 and A5000+cuEquivariance |
| GROMACS NNPot, Si512 NVT | B70 2.93 ns/day, A5000 2.84 ns/day; forces match the ASE calculator to 1.2e-05 |

The honest caveat: this is the launch-bound e3nn path that Intel users get today. cuEquivariance, which
exists for NVIDIA only, buys the A5000 1.8× at Si1728 and halves its energy per step. An equivalent fused
tensor-product path for XPU is where the next 2× on Arc is.

## GROMACS

GROMACS 2026's NNPot interface runs a LibTorch model inside mdrun, but it only knew CUDA and HIP devices.
Two changes are submitted to GROMACS upstream:

| change | issue | merge request | branch |
|---|---|---|---|
| Run NNPot models on Intel GPUs through LibTorch XPU | [#5754](https://gitlab.com/gromacs/gromacs/-/issues/5754) | [!6372](https://gitlab.com/gromacs/gromacs/-/merge_requests/6372) (Draft) | `main` |
| Fix: models requesting only `pair-shifts` never got a pair list | [#5753](https://gitlab.com/gromacs/gromacs/-/issues/5753) | [!6370](https://gitlab.com/gromacs/gromacs/-/merge_requests/6370) | `release-2026` |

To build before they merge: the patches in [`upstream/gromacs/`](upstream/gromacs/) apply to the
`main` and `release-2026` branches (the verification record is in the same directory), and
[`patches/gromacs-2026.3-nnpot-xpu.patch`](patches/gromacs-2026.3-nnpot-xpu.patch) is the earlier
combined form for the 2026.3 release tarball. Build scripts: `scripts/gmx/build-gmx-nnpot-{xpu,cuda}.sh`.

Runtime rules learned the hard way, now in the upstream docs: PyTorch 2.14 headers need C++20; GROMACS and
LibTorch must share one SYCL runtime (`libsycl`), so put the LibTorch wheel's `lib/` first on
`LD_LIBRARY_PATH` or build GROMACS with the oneAPI release that LibTorch bundles; expose a single SYCL backend
(`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`); the mdp key is `nnpot-pair-cutoff`.

`scripts/gmx/export_gmx_mace.py` exports MACE-MP-0 to TorchScript with the NNPot calling convention
(positions nm→Å, GROMACS's half pair list mirrored with negated shifts, energy eV→kJ/mol);
`check_wrapper.py`, `make_si_system.py`, `run_case.sh` and `compare_gmx.py` are the test case and the
force cross-check against the ASE calculator.

## Repository map

| path | what |
|---|---|
| `scripts/mace_ab.py`, `mace_compare.py` | force A/B harness: same geometries on any device, forces saved as `.npz`, lane-vs-lane deltas |
| `scripts/mace_power.py` | joules per force call from the GPU energy counter (`xe` hwmon) or `nvidia-smi` |
| `scripts/mace_nve.py`, `mace_nvt.py` | 20 ps NVE drift; 10 ps Langevin NVT |
| `scripts/mace_fp64.py`, `mace_scale.py`, `mace_forces.py` | fp64 lane, size scaling, first probe |
| `scripts/run_b70_chain.sh`, `run_cuda_chain.sh` | the full lane on each card (paths via `MACE_ARC_ROOT`) |
| `scripts/gmx/` | GROMACS export, test case, cross-check, builds |
| `upstream/gromacs/` | the two upstream patches, verification record, end-to-end reproduction of the fix |
| `patches/` | combined patch for the GROMACS 2026.3 release |
| `docs/FINDINGS.md` | the engineering record, seven findings |
| `results/` | raw logs, force arrays, GROMACS cases; `results/miner-on/` is a busy-host calibration, see Finding 4 |

## Reproduce

```bash
export MACE_ARC_ROOT=$PWD                       # scripts default to /data/mace-arc
python scripts/mace_ab.py --device xpu --tag b70 --out results/ab      # Intel GPU
python scripts/mace_ab.py --device cuda --tag a5000 --out results/ab   # NVIDIA (add --cueq for cuEquivariance)
python scripts/mace_ab.py --device cpu --tag cpu --out results/ab
python scripts/mace_compare.py results/ab b70 a5000 cpu
```

GROMACS: build with the script for your tier, then `export_gmx_mace.py` → `make_si_system.py` → `run_case.sh` → `compare_gmx.py`.

## Related

[boltz2-intel-arc](https://github.com/UniversalBasicInc/boltz2-intel-arc): the same method applied to Boltz-2
protein structure prediction, merged into boltz-community.

## License

MIT. Copyright (c) 2026 Universal Basic Inc. (Dylan Holdren). MACE, e3nn, ASE, PyTorch and GROMACS are
their own projects under their own licenses; nothing of theirs is redistributed here except the GROMACS
patches, which are offered to GROMACS under its LGPL-2.1 terms.
