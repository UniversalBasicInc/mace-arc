# GROMACS upstream candidates (NOT submitted)

Two independent changes against the official GROMACS repository (GitLab, `gitlab.com/gromacs/gromacs`),
produced on 2026-10-15, formatted with clang-format 18.1.8 (the version GROMACS CI uses), signed off
as Dylan Holdren. Both carry an `#PLACEHOLDER` issue reference: GROMACS wants an issue first, then a
merge request that references it.

## 1. `0001-nnpot-pair-shifts-fix.patch` — bug fix, target branch `release-2026`

`nnpot.cpp` subscribed the module to the pair-list range and the pair-list construction notifications
only when the model inputs included `"atom-pairs"` or the misspelled `"pairs-shifts"`. A model whose
inputs include `pair-shifts` but not `atom-pairs` therefore never got a pair list. The fix uses the
real option name and adds three unit tests to `tests/nnpot.cpp` (range requested for `atom-pairs`,
for `pair-shifts`, none without pair-list inputs) plus a 2026.5 release note.

Validated: CPU build of `release-2026` + this commit with a CPU LibTorch 2.14.1, `nnpot_applied_forces-test`
21/21 pass. Of the three new tests only `PairShiftsInputRequestsPairlistRange` fails on the unpatched tree;
the other two pin the surrounding behaviour.

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

Review pass 2026-10-16 (fresh-eyes review of the patches + host CPU-tier builds, GCC 12, `main` + this commit)
found and fixed, now in the patch: `GMX_NNPOT=OFF` stub constructor not updated (build failed); LibTorch
header check also linked, failing with GCC (now compile-only via `CMAKE_TRY_COMPILE_TARGET_TYPE`) and
cached its failure so its own advice did not work (now `unset(... CACHE)` before the check); Intel GPU
unit-test case ran on SYCL builds with a non-XPU LibTorch and would have failed (now gated on
`GMX_TORCH_XPU`); DPC++ builds targeting NVIDIA/AMD got Intel/XPU messages and the CMake warning (now
vendor-checked at runtime, `GMX_DPCPP_HAVE_INTEL_TARGET` in CMake); `<c10/xpu/XPUFunctions.h>` include
guard matched its use site; a process without an assigned GPU (grompp) got a warning about mdrun (now uses
the first XPU like CUDA uses the current device); docs state the same-runtime requirement and
`ZE_FLAT_DEVICE_HIERARCHY=FLAT` for multi-tile GPUs (device matching by `sycl::device` identity cannot
match GROMACS sub-devices against LibTorch root devices; not testable on the single-tile B70 — say so in
the MR). After the fixes: `GMX_GPU=OFF GMX_NNPOT=OFF` builds `gmx` and the full Sphinx manual (no warnings
in touched files); `GMX_GPU=OFF GMX_NNPOT=TORCH` (CPU LibTorch 2.14.1) builds and passes 18/18;
`GMX_GPU=CUDA GMX_NNPOT=TORCH` (LibTorch cu126) builds `gmx` and the test binary (CUDA tests cannot run:
no NVIDIA driver on the current kernel; the `torch::hasCUDA()` assertion they hit is upstream code and
identical on unpatched `main`). B70 re-validation: see RESULTS below once recorded.

Note for the feature MR: `main` carries the same `pairs-shifts` typo in lines the feature patch touches as
context, so the feature branch needs a trivial rebase once the fix is merged up from release-2026.

## Before submitting (process, not code)

1. Open two GitLab issues (bug; feature proposal — the NNPot maintainers are the KTH group that wrote
   the interface; a short discussion on the feature issue before the MR is the polite route).
2. Replace `#PLACEHOLDER` in both commit messages with the issue numbers.
3. Push to a fork, open the MRs against `release-2026` and `main`; CI runs clang-format, clang-tidy,
   copyright checks (headers are untouched here) and the unit tests.
4. Mention the validation above and offer the B70 as a test machine; GROMACS CI has no Intel GPU runner
   for the LibTorch path.

## Rebuilding the branches

```
git clone https://gitlab.com/gromacs/gromacs.git && cd gromacs
git checkout -b nnpot-pair-shifts-fix release-2026 && git am ../0001-nnpot-pair-shifts-fix.patch
git checkout -b nnpot-xpu main && git am ../0001-nnpot-intel-gpu-xpu.patch
```
The patch scripts that generated the edits (`../apply_pair_shifts_fix.py`, `../apply_xpu_main.py`) are kept
for reference; the patches are the source of truth.
