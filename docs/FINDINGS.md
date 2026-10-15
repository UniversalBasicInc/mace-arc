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

A5000 lane (2026-10-15, quiet host), same geometries, `results/ab/`:

| System | cuda32 vs cpu32 (rel RMS) | cuda32 e3nn vs cuda32 cuEquivariance | cuda64 vs cpu64 (rel RMS) |
|---|---|---|---|
| Si64 | 1.6e-06 | 5.2e-07 | 3.4e-15 |
| Si512 | 1.6e-06 | 6.2e-07 | 3.0e-15 |
| Si1728 | 1.6e-06 | 6.5e-07 | — |

Direct B70 vs A5000, same `.npz` harness, B70 re-run 2026-10-15 after the 6.18 boot (`results/ab/b70_*`):

| System | B70 vs A5000, fp32 (rel RMS) | B70 vs A5000, fp64 (rel RMS) | B70 vs CPU, fp32 | B70 fp32 vs own fp64 |
|---|---|---|---|---|
| Si64 | 1.8e-06 | 3.8e-15 | 2.2e-06 | 4.5e-06 |
| Si512 | 1.6e-06 | 3.4e-15 | 2.0e-06 | 7.3e-06 |
| Si1728 | 1.7e-06 | — | 2.1e-06 | — |

The two vendors agree with each other as closely as either agrees with the CPU: ~2e-06 relative in
fp32 (below each card's own fp32-vs-fp64 floor), 4e-15 in fp64. cuEquivariance's kernels move the
A5000's forces by 6e-07 relative, a reordering of fp32 sums. The device is not in the answer.

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
| EPYC 7742, 128 thr, cpu32 | 148 ms | 454 ms | 1184 ms | quiet host, 2026-10-15 (repeat) |
| RTX A5000, cuda32 (e3nn) | 39 ms | 43 ms | 128 ms | quiet host, 2026-10-15 |
| RTX A5000, cuda32 + cuEquivariance | 42 ms | 43 ms | 71 ms | quiet host, 2026-10-15 |
| Arc Pro B70, xpu32 (repeat) | 32 ms | 45 ms | 126 ms | quiet host, 2026-10-15, kernel 6.18 |

fp64, Si512: **B70 107–110 ms, A5000 230 ms, EPYC 644–671 ms.** B70 Si64 fp64 = fp32 (32 ms): free.

Read: on the plain e3nn path **the Arc Pro B70 and the RTX A5000 are the same speed** (Si512 44 vs 43 ms,
Si1728 128 vs 128 ms), both ~10× the 128-thread EPYC. That path is launch-bound (Python-driven, hundreds
of small kernels per call), so neither card is near its compute roof; it is the path Intel users get
today. cuEquivariance, which exists for NVIDIA only, buys the A5000 1.8× at Si1728 and nothing at Si512.
In fp64 the B70 is **2.1× faster than the A5000**: Xe2 has fp64 at 1/14 rate, GA102 at 1/32.

**Measurement note.** A first A5000 pass on 2026-10-15 ran with a 128-thread RandomX miner resident at
`SCHED_IDLE`; the CPU lane came out 5× slower than on the quiet host (Si512 cpu32 2219 ms vs 435 ms) and the
A5000 e3nn lane 1.7–3.3× slower (it is launch-bound, and RandomX saturates L3/memory bandwidth even when
it yields the cores). Those numbers are kept in `results/miner-on/` as a calibration of how much a busy host
distorts a launch-bound benchmark; the table above is the quiet re-run (two reniced GROMACS builds were
running, and the CPU lane reproduced the 2026-10-05 numbers within 5 %).

## Finding 5 — energy per force call (board power)

GPU board power sampled at 5 Hz while hammering the force call for 60 s (`mace_power.py`), Si512 fp32:

| Device | ms/call | idle W | load W | J/call (board) | J/call above idle | ns/day at 1 fs |
|---|---|---|---|---|---|---|
| RTX A5000, e3nn | 54 | 18 | 168 | 9.1 | 8.1 | 1.60 |
| RTX A5000, cuEquivariance | 57 | 20 | 84 | 4.8 | 3.7 | 1.52 |
| Arc Pro B70, e3nn | 46 | 47 | 159 | 7.3 | 5.2 | 1.88 |
| Arc Pro B70, e3nn, **fp64** | 107 | 47 | 158 | 16.9 | 11.9 | 0.81 |

Si1728 fp32: A5000 e3nn 129 ms/call, 179 W, 23.2 J/call board (0.67 ns/day); B70 125 ms/call, 175 W, 22.0 J/call (0.69 ns/day).

On the path both vendors run (e3nn), the B70 does the same step in the same time for **20 % less energy**
than the A5000 (7.3 vs 9.1 J at Si512, 22.0 vs 23.2 J at Si1728). cuEquivariance, NVIDIA-only, halves the A5000's
energy at equal speed: its fused kernels leave the GPU half-idle instead of busy with launch overhead, which says
an equivalent fused tensor-product path for XPU is where the next 2× on Arc is. Full-double MD on the B70 costs
16.9 J/step at Si512, less than fp32 on the A5000's e3nn path costs per 1.9 steps.
The B70's idle board power reads 47 W on the `xe` counter (the A5000 reads 18 W on `nvidia-smi`); the two sensors
are not the same thing (card energy counter vs reported board draw), so compare the *above-idle* column across vendors
with that in mind.

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

Langevin NVT (ASE, friction 0.01/fs, 300 K, 1 fs, 10 000 steps = 10 ps, Si512 fp32), second half of the run:

| Lane | T (K) | E_pot/atom (eV) | spikes | steps/s | ns/day |
|---|---|---|---|---|---|
| A5000 e3nn | 299.7 ± 12.4 | −5.3305 ± 0.0016 | 0 after equilibration | 18.0 | 1.55 |
| A5000 cuEquivariance | 299.7 ± 12.4 | −5.3305 ± 0.0016 | 0 after equilibration | 19.5 | 1.69 |
| B70 e3nn | 299.7 ± 12.4 | −5.3305 ± 0.0016 | 0 after equilibration | 21.3 | 1.84 |

Same seeds, same thermostat: the three lanes produce statistically identical trajectories (the one
"spike" is the first 100-step window, the 16.6 eV equilibration from the cold lattice). Production-style
MD on Si512 with a foundation potential sits at 1.5–1.7 ns/day on one card.

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

**Results (A5000 tier, GROMACS 2026.3 + libtorch 2.14.1+cu126, patched).** Step-0 forces on the identical
`.gro` coordinates, Si512, `nnpot-input-group = System`, `nnpot-pair-cutoff = 0.6`:

| NNPot device | ΔE vs ASE/MACE | force RMS diff (rel) | max |
|---|---|---|---|
| A5000 (`GMX_NN_DEVICE=gpu`) | +0.05 kJ/mol of −263 604 (+0.001 meV/atom) | 6.7e-03 kJ/mol/nm (1.2e-05) | 2.7e-02 kJ/mol/nm |
| CPU (`GMX_NN_DEVICE=cpu`) | +0.02 kJ/mol | 6.7e-03 kJ/mol/nm (1.2e-05) | 2.7e-02 kJ/mol/nm |

GROMACS's own pair list + shift vectors, fed through the wrapper, reproduce MACE's ASE neighbour list to
1e-05 relative — the fp32 floor, same as the pure-Python self-check. The pair list and shift conventions
of the 2026 interface are therefore confirmed for MACE: half list, `shifts` negated for the mirrored edges.

NVT in GROMACS (v-rescale 300 K, 1 fs, 10 ps, `-nb gpu -ntomp 16`): T = 300.2 ± 10.7 K, **2.84 ns/day, 30.5 ms/step**
on the A5000 — 1.8× the ASE/Python loop (1.55 ns/day) with the same model on the same GPU, because the
integrator, pair list and I/O now live in C++ and only the model forward crosses into libtorch.

**Build friction (all logged in `results/gmx/build-logs/`).**

1. **PyTorch 2.14 headers require C++20**; GROMACS 2026.3 defaults to C++17 → `-DCMAKE_CXX_STANDARD=20`
   (GROMACS honours it). Both the gcc 12 host build and the icpx 2026.1 container build failed identically without it.
2. **NNPot links libpython** (`find_package(Python REQUIRED COMPONENTS Development)`) → needs headers + `libpython.so`
   matching the torch wheel's Python; `-DPython_ROOT_DIR` pointed at uv's CPython 3.12 worked in both environments.
3. The `torch==2.14.1+cu126` wheel's CMake config refuses CUDA < 12.6 even though the driver only needs 12.x
   (driver 535 / CUDA 12.2 runs the result fine via minor-version compatibility) → CUDA 12.6 toolkit for the build only.
4. The pair-cutoff mdp key is `nnpot-pair-cutoff` (module-prefixed); the manual's `pair-cutoff` is silently ignored
   and grompp reports "pair-cutoff is 0.0".
5. The DPC++ build: oneAPI `setvars.sh` dies under `set -u`; source it first.

**XPU tier — MACE in GROMACS on an Intel GPU (B70, 2026-10-15).** `install-xpu` = GROMACS 2026.3 SYCL (DPC++ 2026.1,
Xe2 cluster flags, oneMKL FFT) + libtorch 2.14.1+xpu + this patch, in the container, host kernel 6.18. Same Si512 case,
`GMX_NN_DEVICE=gpu` (→ `torch::kXPU` through the patch):

| NNPot device | ΔE vs ASE/MACE | force RMS diff (rel) | NVT 300 K, 10 ps | ns/day | ms/step |
|---|---|---|---|---|---|
| Arc Pro B70 (XPU) | −0.04 kJ/mol (−0.001 meV/atom) | 6.7e-03 kJ/mol/nm (1.2e-05) | 299.6 ± 10.5 K | **2.93** | 29.5 |
| RTX A5000 (CUDA) | +0.05 kJ/mol | 6.7e-03 kJ/mol/nm (1.2e-05) | 300.2 ± 10.7 K | 2.84 | 30.5 |

GROMACS runs a foundation ML potential on an Intel GPU with the same forces, the same thermostat statistics
and the same step time as on the NVIDIA card. This is, to our knowledge, the first time GROMACS's NNPot
interface has driven a model on a non-NVIDIA/AMD GPU; the patch is 40 lines.

## Operational notes

- `xe` hwmon exposes cumulative energy counters (`energy1_input` = card, `energy2_input` = package, µJ), not an
  instantaneous `power1_input`; `mace_power.py` differentiates the card counter over 0.5 s windows.
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

- Propose upstream, when cleared: GROMACS `pair-shifts` typo + XPU device selection; a MACE docs note that `device="xpu"` works.
- Larger systems / MACE-MP medium; torch.compile on XPU; memory ceiling on the 32 GB B70.
