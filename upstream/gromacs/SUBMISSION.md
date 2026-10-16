# Submission text (drafts, nothing posted)

Written to match how this tracker is actually used: issues follow the repository templates
(`.gitlab/issue_templates/Bug.md`, `Feature.md`); merge-request descriptions in this area are one or
two lines ("Closes #NNNN." plus a sentence). The contribute guide asks for advance discussion of new
functionality, so the feature goes in as an issue first and the MR opens as a **Draft** that points at it.
Tone: plain first person, say what was checked and what was not, no claims beyond the evidence.

## Issue A — bug report (fix MR → release-2026). Template: Bug

**POSTED 2026-10-17 as https://gitlab.com/gromacs/gromacs/-/issues/5753** (label quick action was consumed but no label applied: non-member; maintainers triage). Fix commit amended to `Fixes #5753`, release note `:issue:`5753`` (JARVIS wt-fix, commit ad89e096). Branch push to the fork and the MR still pending.

**Title:** NNPot: models that request `pair-shifts` without `atom-pairs` never receive a pair list

**Summary**

In `src/gromacs/applied_forces/nnpot/nnpot.cpp` (lines 180 and 210 on release-2026), the NNPot
module subscribes to the pair-list range and to the pair-list construction signal only when one of
the model inputs is `atom-pairs` or `pairs-shifts`. The option value is spelled `pair-shifts`
everywhere else: the mdp documentation, `nnpotoptions.cpp`, `nnpotforceprovider.cpp` and
`torchmodel.cpp` (which rejects anything else as an unknown input). So a model that lists
`pair-shifts` without `atom-pairs` is never subscribed. I think this is simply a typo.

To be upfront about the impact: I do not know of a model that asks for the shifts without the pair
list they index, so I would not expect anyone to have hit this in practice. With `atom-pairs` present
the condition short-circuits and everything works. It is a small consistency fix rather than a
reported problem; I am filing it because the configuration is accepted and documented, and the
failure is silent.

**GROMACS version**

Present in 2026.0 through 2026.4 and at the current tips of release-2026 and main. The lines date
from the pair-list input (!5532, #5228) and have not been touched since. I noticed that !5764 on main
replaces this check with a cutoff test as part of the dictionary I/O change, so this is aimed at
release-2026 only.

**Steps to reproduce**

An mdp with `nnpot-model-input1 = atom-positions`, `nnpot-model-input2 = atom-numbers`,
`nnpot-model-input3 = pair-shifts`, a positive `nnpot-pair-cutoff`, and no `atom-pairs` entry. I do
not know of a public model with this input set, so I reproduced it with a three-input TorchScript model
that returns the number of shift rows it received as the energy, on a 512-atom Si cell with a 0.6 nm
cutoff:

```python
class ShiftCounter(torch.nn.Module):
    def forward(self, positions, atom_numbers, pair_shifts):
        return float(pair_shifts.shape[0]) + 0.0 * positions.sum()
```

NN Potential reported by mdrun: 0 on the release-2026 tip, 11403 with the fix, which is the number of
pairs within the cutoff (the same model with `atom-pairs` added reports 11403 pairs and 11403 shifts,
before and after). The merge request carries unit tests on the module's subscriptions rather than this
model.

**What is the current bug behavior?**

grompp passes (its model check feeds a non-empty dummy shift list), mdrun requests no pair-list range
and never subscribes to the pair-list signal, `doPairlist_` stays false, and the model receives an
empty (0, 3) shift tensor at every step with no error or warning.

**What did you expect the correct behavior to be?**

The module requests the pair list for `pair-shifts` the same way it does for `atom-pairs`.

**Possible fixes**

Use the documented spelling in the two conditions. Merge request follows: the two spelling changes,
five small tests in `tests/nnpot.cpp` (range requested for `atom-pairs` and for `pair-shifts`, none
without pair inputs; subscription to the pair-list signal for `pair-shifts`, none without pair inputs),
and a 2026.5 release note. The two `pair-shifts` tests fail on the unpatched tree and pass with the
fix; the tests do not need LibTorch. Related only by area: #5721, !6265.

/label ~Bug

## Issue B — proposal (XPU MR → main). Template: Feature

**Title:** NNPot: run the LibTorch model on Intel GPUs (LibTorch XPU backend)

**Summary**

The NNPot interface runs the model on CUDA (and HIP) when GROMACS is built for those, and on the CPU
otherwise. LibTorch also has an XPU backend for Intel GPUs (PyTorch >= 2.5, through oneAPI), and a SYCL
build of GROMACS already holds a `sycl::device` for the rank. I have a working implementation that lets
the model run on the same Intel GPU mdrun uses, and would like to know whether it is wanted in this form
before polishing it further.

**Use cases**

People with Intel GPUs (Arc, Data Center Max) who want to run MLIPs through GROMACS without the model
falling back to the CPU. Today a SYCL build with LibTorch silently runs inference on the CPU.

**Impact**

No change for CUDA/HIP builds beyond the plumbing (a `DeviceInformation` pointer handed to the MD
modules), and no change for CPU builds. The new code is behind `GMX_SYCL_DPCPP` and a CMake-detected
`GMX_TORCH_XPU`. The maintenance question is the usual one for a new backend: I can test on an
Arc Pro B70 and am willing to keep doing so; CI has no Intel GPU runner for the LibTorch path.

**Detailed description**

- mdrun passes the rank's `DeviceInformation` to the MD modules through the simulation-setup notifier
  (nullptr on ranks without a GPU).
- `TorchModel` matches that device against LibTorch's XPU enumeration (by Intel device UUID, falling
  back to handle identity) and runs on `torch::kXPU` with that index. If the main rank has no GPU
  assigned (or for the grompp model check) it uses the first XPU, mirroring what the CUDA path does
  with the current device.
- Device type and availability are resolved by name instead of hard-coding CUDA, with distinct
  messages for "LibTorch built without XPU support", "device not visible to LibTorch", and (restored
  from the old assert) "LibTorch sees a CUDA/HIP device but this build has no matching GPU support".
- CMake: `GMX_TORCH_XPU` from the `torch_xpu` target; a warning for a DPC++ Intel build paired with a
  non-XPU LibTorch; a configure-time check that the LibTorch headers compile with the configured C++
  standard (PyTorch >= 2.14 needs C++20), cached per LibTorch/standard/compiler.
- `<torch/xpu.h>` is included only if it exists, so LibTorch 2.0–2.3 still builds.
- The GPU cases of the NNPot unit tests run on Intel devices too and receive the device from the test.
- Docs: nnpot.rst, install guide, 2027 major-features note.

**Requirements**

A LibTorch with XPU support, and at run time one SYCL runtime in the process (two different `libsycl`
builds crash either side; I hit this) and a single SYCL backend visible (`ONEAPI_DEVICE_SELECTOR`),
both documented. Multi-tile GPUs: the device match cannot pair GROMACS sub-devices with LibTorch root
devices, so `ZE_FLAT_DEVICE_HIERARCHY=FLAT` is needed there; I could not test that (single-tile card).

**Links/references/implementations**

What I verified: unit tests on the B70 (Level Zero only, and with both backends visible); a 512-atom
Si system with MACE-MP-0 reproduces its ASE forces to 1.2e-5 relative, with mdrun on the GPU and with
mdrun's own GPU work on the CPU; builds with NNPot off, with a CPU-only LibTorch (tests pass), and with
a CUDA LibTorch (compile only — no NVIDIA driver on that machine right now). Not tested: HIP,
AdaptiveCpp. Related: !5099 (device selection), #5532. Draft MR to follow if this is welcome.

/label ~Feature

## MR descriptions (short, like the neighbours)

Fix MR (→ release-2026): `Fixes #5753.` + one sentence: "Use the documented `pair-shifts` spelling when
deciding whether to request the pair list; adds five tests for the subscriptions and a 2026.5 note."

Feature MR (→ main, opened as Draft): `Refs #B.` + "Runs NNPot inference on Intel GPUs through LibTorch
XPU; details and validation in the issue. Needs a trivial rebase once !<fix MR> is merged up from
release-2026. Developed with AI assistance (Claude Code); every change was reviewed and validated by
me on the hardware listed in the issue."

## Choices still open

- Name MACE-MP-0 (as drafted) or write "a published foundation MLIP".
- Whether to also post a short note on the GROMACS developer forum, as the contribute guide suggests
  for new functionality. The issue covers the same ground; a forum post is optional.
