# GROMACS upstream contributions

Two independent changes against the official GROMACS repository (`gitlab.com/gromacs/gromacs`), formatted with
clang-format 18.1.8 (the version GROMACS CI uses). Submitted 2026-10-17:

| patch | issue | merge request | target |
|---|---|---|---|
| `0001-nnpot-pair-shifts-fix.patch` | [#5753](https://gitlab.com/gromacs/gromacs/-/issues/5753) | [!6370](https://gitlab.com/gromacs/gromacs/-/merge_requests/6370) | `release-2026` |
| `0001-nnpot-intel-gpu-xpu.patch` | [#5754](https://gitlab.com/gromacs/gromacs/-/issues/5754) | [!6372](https://gitlab.com/gromacs/gromacs/-/merge_requests/6372) (Draft) | `main` |

The rest of this file is the verification record that backs them.

## 1. `0001-nnpot-pair-shifts-fix.patch` — bug fix, target branch `release-2026`

`nnpot.cpp` subscribed the module to the pair-list range and the pair-list construction notifications
only when the model inputs included `"atom-pairs"` or the misspelled `"pairs-shifts"`. A model whose
inputs include `pair-shifts` but not `atom-pairs` therefore never got a pair list. The fix uses the
real option name and adds three unit tests to `tests/nnpot.cpp` (range requested for `atom-pairs`,
for `pair-shifts`, none without pair-list inputs) plus a 2026.5 release note.

Validated: CPU build of `release-2026` + this commit with a CPU LibTorch 2.14.1, `nnpot_applied_forces-test`
23/23 pass. Five new tests (two for the setup range, one for no range, two for the run-time pair-list
subscription). EXECUTED against the unpatched `nnpot.cpp`: the two `pair-shifts` tests fail, the other three
pass; with the fix all pass. Tests need no LibTorch (nnpot.cpp is built with GMX_NNPOT=OFF too).

Verification round 4 (2026-10-16, adversarial review of the fix's premise, via GitLab API on current tips):
typo present in v2026.0–v2026.4 and both branch tips, introduced in 948d5ef944 (!5532, 2025-11-06), untouched
since; no prior issue/MR mentions it; `pairs-shifts` is accepted nowhere (torchmodel.cpp rejects it); nothing
validates input combinations, so the path is reachable; practical impact near zero (shifts are indexed by
the pair list, no sensible model requests one without the other) — the issue says so plainly. Open !5764
(dictionary I/O, main) will remove these lines, so the fix targets release-2026 only. Release note moved to
the end of its section, title in the neighbours' style, affected versions stated.

Verification round 5 (2026-10-16, end-to-end on mdrun): a three-input TorchScript model (`repro-pair-shifts/make_model.py`: positions, atomic numbers,
pair-shifts; energy = number of shift rows received) run on the Si512 cell (0.6 nm cutoff) with CPU-only
gmx builds of the release-2026 tip `4cdcf92` and of the fix `2536c4f`:

| binary | model inputs | NN Potential (kJ/mol) = shift rows |
|---|---|---|
| unfixed 4cdcf92 | positions, numbers, pair-shifts | 0 (grompp passes, no warning) |
| fixed 2536c4f | positions, numbers, pair-shifts | 11403 |
| unfixed 4cdcf92 | + atom-pairs (model `make_model2.py`: pairs + 1000·shifts) | 11414403 → 11403 pairs, 11403 shifts |
| fixed 2536c4f | + atom-pairs | 11414403 (unchanged) |

Independent count from conf.gro with minimum image (`count_pairs.py`): 11403 unordered pairs within 0.6 nm.
So with the fix a shifts-only model receives exactly the pair list it should; without it, nothing; and the
atom-pairs path is unchanged. Energies in `repro-pair-shifts/*.xvg` (e-* fixed, eu-* unfixed).

## 2. `0001-nnpot-intel-gpu-xpu.patch` — feature, target branch `main`

Runs NNPot inference on Intel GPUs through LibTorch's XPU backend.

- mdrun passes the `DeviceInformation` of the device assigned to the rank to the MD modules through
  the simulation-setup notifier (new `const DeviceInformation*` notification; nullptr for CPU ranks).
- `NNPotOptions` stores it, `NNPotForceProvider` and `TorchModel` take it; `TorchModel` matches the
  rank's `sycl::device` against the devices LibTorch enumerates (`c10::xpu::get_raw_device`) and runs
  the model on `torch::kXPU` with that index. Nothing changes for CUDA/HIP builds beyond the plumbing.
- Device type and availability are resolved by name instead of hard-coding CUDA; messages explain
  why a GPU cannot be used (LibTorch without XPU support, device not visible to LibTorch). A rank
  without an assigned GPU (grompp's model check, or mdrun with all GPU work on the CPU) uses the
  first XPU LibTorch sees, exactly as the CUDA/HIP paths use the current device. DPC++ builds that
  target NVIDIA/AMD GPUs keep the old behaviour and never see Intel/XPU wording.
- CMake: `GMX_TORCH_XPU` is set when the found LibTorch exports the `torch_xpu` target; a SYCL build
  paired with a LibTorch that lacks it warns at configure time; the LibTorch headers are test-compiled
  with the configured C++ standard (PyTorch >= 2.14 requires C++20) and configure fails with a
  clear message instead of a wall of compiler errors.
- `<torch/xpu.h>` is included only when it exists (`__has_include`), so LibTorch 2.0–2.3 still builds.
- Unit tests: the GPU cases of `nnpotforceprovider.cpp` now also run on Intel devices and pass the
  device to the provider (the "better way to handle device selection" the test comment asked for).
- Docs: reference manual NNPot page, install guide (C++20, Intel GPUs), 2027 major-features note.

Validated on an Arc Pro B70 (SYCL DPC++ 2026.1 build of `main` + this commit, LibTorch 2.14.1+xpu):
`nnpot_applied_forces-test` 18/18 pass with the GPU case on the B70 (main has 18 tests; the 3 new tests live on the fix branch); a 512-atom Si pure-NNP case with
MACE-MP-0 reproduces the ASE/MACE reference forces to 1.2e-05 relative on the GPU; with the GPU
withheld from mdrun, `GMX_NN_DEVICE=gpu` fails with the new message and the automatic mode falls back
to the CPU with a warning.

Review pass 2026-10-16 (fresh-eyes review of the patches + host CPU-tier builds with GCC 12 + B70 re-run,
`main` + this commit). Found and fixed, all now in the patch:
- `GMX_NNPOT=OFF` stub constructor not updated (every no-torch build failed).
- LibTorch header check also linked (fails with GCC; now compile-only via `CMAKE_TRY_COMPILE_TARGET_TYPE`)
  and cached its failure so its own advice did not work (now `unset(... CACHE)` before the check).
- Intel GPU unit-test case ran on SYCL builds with a non-XPU LibTorch and would have failed (gated on
  `GMX_TORCH_XPU`).
- DPC++ builds targeting NVIDIA/AMD got Intel/XPU wording and the CMake warning (vendor check at runtime,
  `GMX_DPCPP_HAVE_INTEL_TARGET` in CMake). `<c10/xpu/XPUFunctions.h>` include guard matches its use site.
- A process without an assigned GPU (grompp's model check, mdrun with all GPU work on the CPU) warned about
  mdrun; it now uses the first XPU, as the CUDA/HIP paths use the current device. Verified on the B70:
  grompp loads the model on the GPU, `mdrun -nb cpu ...` with `GMX_NN_DEVICE=gpu` gives rel 1.19e-05.
- Device matching compares the Intel device UUID (`ext_intel_device_info_uuid`) before falling back to handle
  identity, so a device GROMACS holds through OpenCL still matches LibTorch's Level Zero enumeration. The
  unit tests pass with both backends visible (18/18) and with Level Zero only (18/18).
- Docs (nnpot.rst) now state the two runtime requirements found the hard way: one SYCL runtime in the process
  (GROMACS's oneAPI 2026.1 `libsycl.so.9` build 20260724 and the LibTorch wheel's bundled 2026.1.0.20260617
  copy crash each other's side when mixed: oneAPI-first → null call in LibTorch's oneDNN graph-capture query;
  LibTorch-first → crash in GROMACS's USM free at teardown when the OpenCL device was chosen), and a single
  SYCL backend visible (`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, which GROMACS itself recommends when it sees a
  GPU through two backends), plus `ZE_FLAT_DEVICE_HIERARCHY=FLAT` on multi-tile GPUs (identity/UUID match
  cannot pair GROMACS sub-devices with LibTorch root devices; not testable on the single-tile B70 — say so in
  the MR).

After the fixes: B70 (SYCL build, LibTorch 2.14.1+xpu, LibTorch runtime first, Level Zero only): unit tests
18/18; Si512 MACE-MP-0 forces vs ASE rel 1.19e-05 with mdrun on the GPU and again with all mdrun GPU work on
the CPU; auto mode picks the XPU; with the GPU hidden from the process `GMX_NN_DEVICE=gpu` fails with the
new message. Host: `GMX_GPU=OFF GMX_NNPOT=OFF` builds `gmx` and the full Sphinx manual (no warnings in
touched files); `GMX_GPU=OFF GMX_NNPOT=TORCH` (CPU LibTorch) builds and passes 18/18; `GMX_GPU=CUDA
GMX_NNPOT=TORCH` (LibTorch cu126) builds `gmx` and the test binary (CUDA tests cannot run: no NVIDIA driver on
the current kernel; the `torch::hasCUDA()` assertion they hit is upstream code, identical on unpatched `main`).

Verification round 2 (2026-10-16, after the review fixes):
- Full GROMACS test suites, CPU builds with CPU LibTorch: fix branch 102/102, feature branch 104/104.
- `admin/copyright.sh` check: clean on both branches (headers use the open-ended `YYYY-` form).
- clang-tidy 18.1.8 (CI version) with the in-tree `src/.clang-tidy` on every touched translation unit: every
  warning inside a touched file is on a pre-existing upstream line (git blame = main/release-2026 head); CI
  only reports changed lines, so the gate is clean.
- Rebase check: the fix applies on `main` and the feature cherry-picks on top of it with no conflicts.
- Not possible on this box: CUDA tests (no NVIDIA driver on the 6.18 kernel), HIP and AdaptiveCpp builds.

Verification round 3 (2026-10-16, second cold review of the final patch): no functional defects; fixed in the
patch: Signed-off-by dropped from both commits (not a GROMACS convention); a distinct error when LibTorch sees
a CUDA/HIP device but the GROMACS build has no matching GPU support (the old assert's information, which the
first version had lost); the LibTorch header check is cached keyed on LibTorch version + C++ standard +
compiler instead of re-running on every configure (verified: 0 re-runs on reconfigure); nnpot.rst says the
main rank is what matters; the XPU message names ZE_FLAT_DEVICE_HIERARCHY=FLAT; UUID query guarded by
SYCL_EXT_INTEL_DEVICE_INFO >= 5 like the rest of the tree; a rank with no GPU and no XPU visible gets the
neutral "no matching device" message; install guide no longer cites an unverified LibTorch XPU download.
Re-validated after: B70 unit tests 18/18 (Level Zero only and both backends), Si512 rel 1.19e-05 with mdrun
on the GPU and with mdrun GPU work on the CPU; host NNPot-off + Sphinx, CPU-torch 18/18, CUDA build OK.

Note for the feature MR: `main` carries the same `pairs-shifts` typo in lines the feature patch touches as
context, so the feature branch needs a trivial rebase once the fix is merged up from release-2026.

## Rebuilding the branches

```
git clone https://gitlab.com/gromacs/gromacs.git && cd gromacs
git checkout -b nnpot-pair-shifts-fix release-2026 && git am ../0001-nnpot-pair-shifts-fix.patch
git checkout -b nnpot-xpu main && git am ../0001-nnpot-intel-gpu-xpu.patch
```
Or take the branches `nnpot-pair-shifts-fix` and `nnpot-xpu` from `gitlab.com/DylanHoldren/gromacs`.
