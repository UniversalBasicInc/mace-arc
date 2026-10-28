# GROMACS upstream contributions

Changes against the official GROMACS repository (`gitlab.com/gromacs/gromacs`), formatted with
clang-format 18.1.8 (the version GROMACS CI uses).

| patch | issue | merge request | target | state |
|---|---|---|---|---|
| `0001-nnpot-pair-shifts-fix.patch` | [#5753](https://gitlab.com/gromacs/gromacs/-/issues/5753) | [!6370](https://gitlab.com/gromacs/gromacs/-/merge_requests/6370) | `release-2026` | merged 2026-10-22, ships in 2026.5 |
| `0001-nnpot-intel-gpu-xpu-v2.patch` | [#5754](https://gitlab.com/gromacs/gromacs/-/issues/5754) | [!6372](https://gitlab.com/gromacs/gromacs/-/merge_requests/6372) | `main` | v2 of 2026-10-27, replaces the v1 patch below |
| `0001-nnpot-torch-cxx-standard.patch` | [#5754](https://gitlab.com/gromacs/gromacs/-/issues/5754) | [!6417](https://gitlab.com/gromacs/gromacs/-/merge_requests/6417) | `main` | opened 2026-10-28 |
| `0001-nnpot-intel-gpu-xpu.patch` | #5754 | !6372 as first pushed 2026-10-17 | `main` | superseded by v2, kept as a record |

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

## 2. `0001-nnpot-intel-gpu-xpu-v2.patch` — feature, target branch `main` (!6372, reworked 2026-10-27)

Reviewer feedback on #5754 (Lukas Müllender 10-20, Andrey Alekseenko 10-27): the `DeviceInformation`
notification of v1 should be its own preparatory MR (#5467), and the XPU support should be merged first
without it, selecting the device through `GMX_NN_DEVICE`; C++20 should be applied only to the files that
include the Torch headers. v2 does exactly that, in two commits:

**XPU support (7 files, +91/-10).** `torchmodel.cpp` gains an `#elif GMX_TORCH_XPU` branch next to CUDA and
HIP: when the LibTorch found at configure time exports the `torch_xpu` target, the model runs on the first
Intel GPU LibTorch enumerates, by default and with `GMX_NN_DEVICE=gpu`; `ONEAPI_DEVICE_SELECTOR` or
`ZE_AFFINITY_MASK` restrict which GPUs it sees. Two helpers map the device-type string to `torch::kXPU` or
`torch::kCUDA` and to the matching `is_available()` query; the CUDA/HIP/CPU paths and every message are
unchanged. CMake sets `GMX_TORCH_XPU` (also in `config.h`) and warns when a SYCL build for Intel GPUs meets
a LibTorch without XPU support. The GPU cases of the unit test also run on Intel devices. Docs: NNPot
reference page, install guide, 2027 feature note. Nothing in `NNPotOptions`, `NNPotForceProvider`,
`runner.cpp` or the notifier.

**C++ standard per target (`0001-nnpot-torch-cxx-standard.patch`, 7 files).** LibTorch 2.14 and later
require C++20 (`torch/all.h` moved its `#error`), GROMACS defaults to C++17. Three places include Torch:
`torchmodel.cpp`, Colvars (`colvarcomp_torchann.h` via `colvar.cpp`, compiled into `colvars_library` with
`COLVARS_TORCH`), and `binary_information.cpp`, which only needs `TORCH_VERSION` and now includes
`<torch/version.h>`. Configure `try_compile`s `<torch/torch.h>` with the configured standard and, if that
fails, with C++20 (compile-only, module scanning off in the test project, cached and invalidated with
`gmx_check_if_changed`); `torchmodel.cpp` moves into an object library `nnpot_torch` with that standard
(objects reach `libgromacs` through an interface target, the `colvars_module` pattern) and `colvars_library`
gets the same standard; everything else keeps `CMAKE_CXX_STANDARD`. A compile check and not a version gate
because `TorchConfig.cmake` declares the standard LibTorch was *built* with (`CXX_STANDARD 20` from 2.13 on)
while 2.13 still compiles with C++17. `nnpot_torch` gets the CUDA include directory or `hip::host`
explicitly. Opened as a separate MR because the problem is not Intel-specific and !6372 does not need it
with LibTorch up to 2.13; folding it into !6372 is offered. Deviation stated to the reviewers: for Colvars
the standard applies to the whole object library (two of fifty files include Torch, but the Torch flags
already apply to all of them, and the whole library builds and passes its 16 tests at C++20 in every LibTorch 2.14 row; note
the GROMACS C++20 CI jobs set `GMX_USE_COLVARS=NONE`, so they do not cover Colvars).

**Verification 2026-10-27** (host GCC 12.2; container oneAPI 2026.1 icpx for the Arc Pro B70; every NNPot
row 18/18, host rows also Colvars 16/16):

| build | result |
|---|---|
| C++ MR, host CPU, LibTorch 2.14.1 cpu, default std | probe no@17 yes@20; `nnpot_torch`+`colvars_library` C++20, `libgromacs`+`colvars_objlib` C++17; did not build before |
| C++ MR, host CPU, LibTorch 2.13.0 | probe yes@17, everything C++17 |
| C++ MR, host CPU, LibTorch 2.9.1 (CI version) | probe yes@17, everything C++17, 0 warnings |
| C++ MR, host CPU, LibTorch 2.14.1, `-DCMAKE_CXX_STANDARD=20` | everything C++20 |
| C++ MR, host CUDA 12.6, LibTorch 2.14.1 cu126, default std | builds; same split; CUDA include dir on `nnpot_torch`; tests need a driver (none on this kernel: upstream `torch::hasCUDA()` assert, identical on main) |
| C++ MR, Ninja + icpx, LibTorch 2.14.1 xpu | configure OK, probe no@17 yes@20, `-std=c++20` on torchmodel.cpp |
| XPU MR alone, host CUDA, LibTorch 2.14.1 cu126, C++20 | builds |
| XPU MR alone, B70 SYCL, LibTorch 2.13.0 xpu, C++17 | 18/18 both runtime orders; Si512 MACE-MP-0 auto mode "Using XPU device", forces vs ASE rel 1.19e-05; `GMX_NN_DEVICE=gpu/cpu` OK; GPU hidden: `gpu` fails with "no matching device is available", auto falls back to CPU |
| both MRs, B70 SYCL, LibTorch 2.14.1 xpu, default std | rerun after the review fixes: 18/18 both runtime orders, auto mode XPU, rel 1.19e-05, all modes; torchmodel.cpp and colvar.cpp at C++20, nnpotforceprovider.cpp at C++17 |
| XPU MR, non-SYCL GROMACS (g++ 15) + LibTorch 2.13 xpu wheel | configure without a findable oneAPI compiler: aborts inside the wheel's `Caffe2Config.cmake` ("cannot find the XPU runtime libraries"), hence the install-guide wording; with oneAPI present: builds, 18/18 (no device list, GPU cases skipped), Si512 auto mode "Using XPU device", rel 1.19e-05, `gpu`/`cpu`/hidden-GPU modes correct |
| SYCL Intel build + CPU-only LibTorch | new configure warning fires |
| C++ MR, static libs, GCC 12, LibTorch 2.14.1 via `GMX_NNPOT=AUTO`, `-Werror=dev` (CP2K CI shape, 10-28) | builds `libgromacs.a` with the same split, 18/18 + 16/16; both branches configure clean with `-Werror=dev` |

2026-10-28: Matthias Krack (CP2K) reported on #5754 that the CP2K CI (GCC 15, static, MPI, Spack PyTorch 2.14.1, `GMX_NNPOT=AUTO`) fails in the Colvars Torch header at C++17, the exact bug the C++ MR fixes; CP2K merged `-DGMX_NNPOT=OFF` as a workaround (cp2k/cp2k#6409). His diagnosis reads the torch target's `CXX_STANDARD 20` as the requirement; it is the build standard (2.13 has it too and compiles at C++17); the !6417 description explains this.

Reconfiguring a configured tree re-runs 0 header checks. Observed, not ours: a non-SYCL `GMX_FFT_LIBRARY=mkl`
build with the XPU wheel fails in `gmxManageFFTLibraries.cmake` because LibTorch's CMake already created
`MKL::MKL` (pre-existing interaction; fftpack/FFTW avoid it).

Independent review pass 2026-10-27 (fresh-eyes read of both patches against the reviewers' words, CI YAML and
the LibTorch CMake configs) found and led to: the probe rewritten on `try_compile` with module scanning off
(a Ninja + scanner-less clang would have failed the C++20 probe for the wrong reason); `CMakeConfigureLog.yaml`
instead of the no-longer-written `CMakeError.log` in the error; `gmx_check_if_changed` instead of a hand-rolled
cache key (now also keyed on `Torch_DIR` and `CMAKE_CXX_FLAGS`); explicit CUDA/HIP runtime headers on
`nnpot_torch`; the install guide now says the XPU wheel's CMake needs a oneAPI DPC++ compiler at configure
time even for a non-SYCL GROMACS. Accepted as-is: `torch::xpu::is_available()` runs before `GMX_NN_DEVICE`
is read (parity with `cudaGetDevice`); `GMX_NN_DEVICE=cuda` as deprecated alias also selects the XPU; no
warning for AdaptiveCpp builds targeting Intel GPUs.

Not testable here: HIP/ROCm LibTorch, AdaptiveCpp, MSVC (moot: `torchmodel.cpp` includes `<dlfcn.h>`),
multi-GPU Intel nodes.

### v1 (`0001-nnpot-intel-gpu-xpu.patch`, pushed 2026-10-17, superseded)

Passed the rank's `DeviceInformation` to the module through a new notification and matched the rank's
`sycl::device` against LibTorch's XPU enumeration (UUID, then handle identity). Validated on the B70 and on
host CPU/CUDA builds, three review rounds (details in the git history of this file). Its device matching
and the single-runtime / single-backend findings are what the v2 docs still describe; the notification itself
becomes a preparatory MR for #5467 when the maintainers want it.

## Rebuilding the branches

```
git clone https://gitlab.com/gromacs/gromacs.git && cd gromacs
git checkout -b nnpot-pair-shifts-fix release-2026 && git am ../0001-nnpot-pair-shifts-fix.patch
git checkout -b nnpot-xpu main && git am ../0001-nnpot-intel-gpu-xpu-v2.patch
git checkout -b nnpot-torch-cxx-standard main && git am ../0001-nnpot-torch-cxx-standard.patch
```
Or take the branches `nnpot-pair-shifts-fix`, `nnpot-xpu` and `nnpot-torch-cxx-standard` from
`gitlab.com/DylanHoldren/gromacs`.
