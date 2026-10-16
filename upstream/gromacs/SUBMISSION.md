# Submission text (drafts, nothing posted)

## Issue A — bug (for the fix MR → release-2026)

**Title:** NNPot: pair list never requested for models whose inputs include `pair-shifts` but not `atom-pairs`

**Body:**

`src/gromacs/applied_forces/nnpot/nnpot.cpp` subscribes the module to the pair-list range and
pair-list construction notifications only when the model inputs contain `"atom-pairs"` or
`"pairs-shifts"`. The documented option is `pair-shifts` (`nnpot-model-input*`, see
`docs/reference-manual/special/nnpot.rst`), so a model that asks for `pair-shifts` without
`atom-pairs` never gets a pair list and the shifts tensor is empty.

Affects: release-2026 (2026.3 confirmed), main.

Fix: use the real option name; add unit tests that the range is requested for `atom-pairs`,
for `pair-shifts`, and not requested without pair-list inputs. MR follows.

## Issue B — feature (for the XPU MR → main)

**Title:** NNPot: run the LibTorch model on Intel GPUs (XPU) using the device assigned to mdrun

**Body:**

NNPot currently runs its LibTorch model on CUDA when available and on the CPU otherwise. LibTorch
builds with XPU support (PyTorch >= 2.5, Intel GPUs through oneAPI) expose the same device API
under `torch::kXPU`, and the SYCL build of mdrun already owns a `sycl::device` for the rank.

Proposal:

- mdrun passes the `DeviceInformation` of the rank's device to the MD modules through the
  simulation-setup notifier (nullptr on CPU ranks).
- `TorchModel` matches that `sycl::device` against LibTorch's XPU enumeration
  (`c10::xpu::get_raw_device`) and runs the model on the same card mdrun uses.
- Device type and availability resolved by name instead of hard-coded CUDA; explicit messages for
  "no GPU on this rank", "LibTorch built without XPU", "device not visible to LibTorch".
- CMake: `GMX_TORCH_XPU` from the `torch_xpu` target; configure-time warning for SYCL build +
  non-XPU LibTorch; test-compile of the LibTorch headers against the configured C++ standard
  (PyTorch >= 2.14 needs C++20) so the failure is one message, not a compiler wall.
- `<torch/xpu.h>` behind `__has_include` so LibTorch 2.0–2.3 still builds.
- The GPU cases of the NNPot unit tests run on Intel devices too and receive the device from the
  test (addresses the "better way to handle device selection" comment there).

Validated on an Intel Arc Pro B70 with a SYCL DPC++ 2026.1 build of main and LibTorch 2.14.1+xpu:
unit tests pass with the GPU case on the B70; a 512-atom Si pure-NNP case with a published
foundation MLIP reproduces its ASE reference forces to 1.2e-05 relative; with the GPU withheld from
mdrun, `GMX_NN_DEVICE=gpu` fails with the new message and automatic mode falls back to the CPU with
a warning. A CUDA build was compile-checked. No functional change for CUDA/HIP builds.

GROMACS CI has no Intel GPU runner for the LibTorch path; the B70 is available for testing
anything reviewers want run. MR ready; happy to split or reshape it.

## MR descriptions

Same content as the commit messages plus "Fixes #A" / "Refs #B", the validation paragraph above, and
the CI note. Both end with the Claude Code co-author trailer already on the commits.

## Open choice

Issue B and the feature MR say "a published foundation MLIP", not MACE. Naming MACE is more useful
to reviewers; leave it unnamed until the MACE work is cleared for public.
