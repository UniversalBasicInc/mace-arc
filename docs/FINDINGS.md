# Findings — MACE-MP-0 on Intel Arc Pro B70 via PyTorch XPU

Engineering record for running MACE (ACEsuit, e3nn/PyTorch) on an Intel GPU, which upstream MACE
does not target (`--device` accepts `cuda`, `mps`, `cpu`; the cuEquivariance fast path is NVIDIA-only).
Same method as the Boltz-2 port: unmodified science code → device-agnostic → forces validated
against the CPU reference and an NVIDIA card in the same chassis → honest numbers.

## Environment

- **Intel GPU:** Arc Pro B70 (Battlemage G31, PCI `8086:e223`), `xe` driver, host kernel 6.18.53 (source-built LTS; `e223` enters the xe ID table at 6.17).
  User space in a podman container: Ubuntu 26.04, compute-runtime 26.35.39758.10, oneAPI 2026.1 (the Babel `arc-b70` image).
- **NVIDIA GPU:** RTX A5000 (24 GB, sm_86), driver 535.261.03, host kernel 6.1 (Debian 12). Same machine: 2× AMD EPYC 7742, 128 threads.
  The two GPU tiers need two kernels on this host (NVIDIA 535 does not build on ≥ 6.17), so the lanes were run on different boots.
- **PyTorch 2.14.1** on both: `+xpu` wheel (triton-xpu 3.8.0) and `+cu126` wheel (cuEquivariance 0.13.0). mace-torch 0.3.16, e3nn 0.4.4, ASE 3.29.
- **Model:** MACE-MP-0 `small` (`2023-12-10-mace-128-L0_energy_epoch-249`, r_max 6 Å, 89 species), the foundation model, unmodified.
- **Test systems:** diamond Si supercells (64, 512, 1728 atoms), rattled 0.05 Å, seed 7. Deterministic in every lane.

## Finding 1 — it runs unmodified

`mace_mp(model="small", device="xpu")` works as shipped. No source change in MACE or e3nn; the only
requirement is the `+xpu` PyTorch wheel. MACE's own device string check passes `xpu` through to
`torch.device`. cuEquivariance is absent on XPU, so the B70 runs the plain e3nn tensor-product path.

## Finding 2 — forces are correct to the dtype floor

Same-dtype device delta read against the fp32 dtype floor, Babel-style:

| System | xpu32 vs cpu32 (rel RMS) | cpu32 vs cpu64 (rel RMS) | xpu64 vs cpu64 (rel RMS) |
|---|---|---|---|
| Si64 | 2.2e-06 | 3.0e-03 | 4.0e-15 |
| Si512 | 2.0e-06 | 2.9e-03 | 4.2e-15 |
| Si1728 | 2.1e-06 | — | — |

The device contributes three orders of magnitude less than the precision choice. In fp64 the B70
matches the CPU at machine epsilon: the device is not in the answer. (B70 lane 2026-10-05, `results/b70-20261005/`.)

A5000 vs B70 on the same geometries: PENDING-A5000-COMPARE

## Finding 3 — the B70 does real FP64, and it is cheap here

Xe2 has IEEE fp64 at ~1/14 of fp32 rate (~1.4 TFLOPS on the B70, see the Boltz findings §7).
MACE in fp64 on the B70 costs only ~2.5× fp32 (Si512: 110 ms vs 44 ms), not 14×, because the e3nn
path is launch/memory-bound, not FLOP-bound. fp64 on the B70 is still 4.5–5.8× faster than fp64 on
the 128-thread EPYC. Geometry optimisation at full double precision is essentially free on this card.

## Finding 4 — speed (e3nn path, no cuEquivariance)

Median force call, Si512 / Si1728, fp32:

| Device | Si64 | Si512 | Si1728 | Notes |
|---|---|---|---|---|
| EPYC 7742, 128 thr, cpu32 | 83 ms | 435 ms | 1206 ms | quiet host, 2026-10-05 |
| Arc Pro B70, xpu32 | 30 ms | 44 ms | 128 ms | quiet host, 2026-10-05 |
| RTX A5000, cuda32 (e3nn) | PENDING | PENDING | PENDING | 2026-10-15, **RandomX miner resident, see caveat** |
| RTX A5000, cuda32 + cuEquivariance | PENDING | PENDING | PENDING | same |

B70 vs EPYC: 9.8× (Si512), 9.4× (Si1728).

**Caveat on the A5000 numbers.** On 2026-10-15 the host was running a 128-thread RandomX miner at
`SCHED_IDLE`. Our process pre-empts it for CPU time, but RandomX saturates L3 and memory bandwidth,
and the e3nn path is launch-bound (hundreds of small kernels per call, Python-driven). The CPU lane
measured 5× slower than on the quiet host two weeks earlier (Si512 cpu32: 2219 ms vs 435 ms), which
calibrates the contamination. The A5000 lane needs a quiet re-run before any cross-vendor speed claim;
forces are unaffected and the correctness comparison stands. `scripts/run_cuda_chain.sh` reruns it.

## Finding 5 — energy per force call (board power)

PENDING-POWER

## Finding 6 — MD stability: NVE drift and thermostatted NVT

NVE (Velocity Verlet, 1 fs, 20 000 steps = 20 ps, Si from a 300 K Maxwell–Boltzmann start):

| Lane | drift (meV/atom/ps) | E spread (eV) | spikes | steps/s | ns/day |
|---|---|---|---|---|---|
| Si512 xpu32 | +0.0000 | 0.010 | 2 | 21.6 | 1.87 |
| Si64 xpu32 | +0.0000 | 0.0013 | 0 | 32.0 | 2.76 |
| Si64 xpu64 | +0.0000 | 0.0014 | 0 | 31.9 | 2.76 |
| Si64 cpu32 | +0.0000 | 0.0014 | 0 | 13.7 | 1.18 |

No NaNs, no drift, lanes statistically identical. xpu64 == xpu32 speed at Si64: fp64 is free at small N.
(The ns/day column is recomputed; the script version that produced `nve.log` had a stray 1e-3 in
the formula, so the log's "0.002 ns/day" is wrong by 1000×.)

Langevin NVT, 300 K, 1 fs, 10 ps: PENDING-NVT

## Finding 7 — GROMACS 2026 NNPot coupling

GROMACS 2026 can drive an NNP through LibTorch (`nnpot-active`), and since 2026 passes its own pair
list and periodic shifts to the model, which is what MACE-style message-passing models need. The
interface knows CUDA and HIP devices only (`torchmodel.cpp: getMdrunActiveDevice`).

**What was built.**

- `scripts/gmx/export_gmx_mace.py`: MACE-MP-0 small → TorchScript with the NNPot calling convention
  (positions nm → Å, half pair list mirrored with negated shifts, energy eV → kJ/mol; forces by GROMACS's autograd).
  Lookup-table species encoding instead of the Python loop in the reference wrapper (TorchScript-friendly).
- `scripts/gmx/check_wrapper.py`: the exported model, fed an ASE-built half list, reproduces the MACE ASE
  calculator: force RMS diff 6.2e-03 kJ/mol/nm (rel 1.1e-05), energy within 0.005 kJ/mol of −263 596. The
  shift-sign convention is the one the NNPot authors use in `gmx-nnpot-tools`.
- `patches/gromacs-2026.3-nnpot-xpu.patch`: (a) with a DPC++ SYCL build, NNPot selects `torch::kXPU` when
  LibTorch sees an Intel GPU; device type/availability are resolved by string instead of hard-coded CUDA;
  (b) **bug found in upstream** `nnpot.cpp`: the pair-list subscription tests for `"pairs-shifts"` while the
  option is `"pair-shifts"`, so a model that requests only `pair-shifts` never gets a pair list.
- Si512 pure-NNP test case (`make_si_system.py`): a Si atom type with zero LJ/charge, `nnpot-input-group = System`,
  `pair-cutoff = 0.6` (= r_max). Force check at step 0 against the ASE calculator on the identical `.gro` coordinates.

**Results.** PENDING-GMX

## Operational notes

- The B70 lane lives in a container because Intel ships the compute runtime for Ubuntu 26.04 only; the
  Debian 12 host cannot load it. Only `/dev/dri` crosses the boundary.
- Keep `HOME`/`XDG_CACHE_HOME` on the data disk: the MACE model cache and uv's Python land in `~/.cache`,
  and this host's root disk is at 97 %.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is needed for mace-torch 0.3.16 on torch 2.14 (model checkpoints are
  full pickles). `torch.jit.script` is deprecated in 2.14 but still works for e3nn.

## What did **not** need changing

MACE, e3nn, ASE: nothing. The device string is already generic. The cuEquivariance import is optional and
simply absent on XPU.

## Open

- Quiet-host A5000 re-run (miner paused) for the cross-vendor speed table.
- B70 re-run of `mace_ab.py`/`mace_power.py`/`mace_nvt.py` with the new harness (`scripts/run_b70_chain.sh`) after the next 6.18 boot.
- GROMACS XPU lane run on the B70 (build is kernel-independent; the run is not).
- Larger systems / MACE-MP medium; torch.compile on XPU; memory ceiling on the 32 GB B70.
