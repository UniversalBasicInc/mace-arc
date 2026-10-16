# Apply the NNPot Intel-GPU (PyTorch XPU) feature to a GROMACS *main* checkout (branch nnpot-xpu).
#   python apply_xpu_main.py <gromacs-git dir>
# Edits: torchmodel.{h,cpp}, nnpotforceprovider.{h,cpp}, nnpot.cpp, nnpotoptions.{h,cpp},
#        mdmodulesnotifiers.h, runner.cpp, gmxManageNNPot.cmake, config.h.cmakein, tests, docs, release notes.
import sys, os, re
R = sys.argv[1]
def edit(path, pairs):
    p = os.path.join(R, path); s = open(p).read()
    for a, b, n in pairs:
        assert s.count(a) == n, (path, a[:70], s.count(a)); s = s.replace(a, b)
    open(p, "w").write(s); print("edited", path)

NN = "src/gromacs/applied_forces/nnpot/"

# ---- 1. CMake: detect XPU support in the found LibTorch; fail early on the C++ standard ----
edit("cmake/gmxManageNNPot.cmake", [(
'''        set(GMX_TORCH ON)
    elseif(GMX_NNPOT STREQUAL "TORCH")''',
'''        # LibTorch headers from PyTorch 2.14 onwards require C++20; detect it at configure
        # time instead of failing deep inside the build.
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
        set(CMAKE_REQUIRED_INCLUDES "${TORCH_INCLUDE_DIRS}")
        check_cxx_source_compiles("#include <torch/torch.h>\\nint main() { return 0; }" TORCH_HEADERS_COMPILE_WITH_CXX_STANDARD)
        unset(CMAKE_REQUIRED_FLAGS)
        unset(CMAKE_REQUIRED_INCLUDES)
        if(NOT TORCH_HEADERS_COMPILE_WITH_CXX_STANDARD)
            message(FATAL_ERROR "The LibTorch ${Torch_VERSION} headers do not compile with C++${CMAKE_CXX_STANDARD}. "
                                "Recent LibTorch versions require C++20; configure with -DCMAKE_CXX_STANDARD=20, "
                                "or use an older LibTorch, or disable the interface with -DGMX_NNPOT=OFF.")
        endif()

        # Intel GPUs: LibTorch built with XPU support (PyTorch >= 2.5) exports the torch_xpu target.
        set(GMX_TORCH_XPU OFF)
        if(TARGET torch_xpu)
            set(GMX_TORCH_XPU ON)
        endif()
        if(GMX_GPU_SYCL AND GMX_SYCL_DPCPP AND NOT GMX_TORCH_XPU AND NOT FIND_TORCH_QUIETLY)
            message(WARNING "GROMACS is built for Intel GPUs with SYCL, but the LibTorch found has no XPU "
                            "support. Neural network potential inference will run on the CPU. Use a LibTorch "
                            "build with XPU support (PyTorch >= 2.5) to run the model on the GPU.")
        endif()

        set(GMX_TORCH ON)
    elseif(GMX_NNPOT STREQUAL "TORCH")''', 1)])

# ---- 2. mdrun hands the module the device it runs on ----
edit("src/gromacs/mdrunutility/mdmodulesnotifiers.h", [
('''struct gmx_mtop_t;
''', '''struct gmx_mtop_t;
struct DeviceInformation;
''', 1),
('''     * \\tparam MDModulesDirectProvider*
''', '''     * \\tparam const DeviceInformation*
     *                              Provides modules with the GPU device used by this rank
     *                              (nullptr when mdrun runs on the CPU only)
     * \\tparam MDModulesDirectProvider*
''', 1),
('''                           MDModulesDirectProvider*,
                           const PlumedInputFilename&>::type simulationSetupNotifier_;''',
'''                           MDModulesDirectProvider*,
                           const PlumedInputFilename&,
                           const DeviceInformation*>::type simulationSetupNotifier_;''', 1)])

edit("src/gromacs/mdrun/runner.cpp", [(
'''        setupNotifier.notify(&mdModuleCoulombDirectProvider);
''', '''        setupNotifier.notify(&mdModuleCoulombDirectProvider);
        setupNotifier.notify(static_cast<const DeviceInformation*>(deviceInfo));
''', 1)])

# ---- 3. NNPot module plumbing: options store the device, force provider and model receive it ----
edit(NN + "nnpotoptions.h", [
('''    void setComm(const MpiComm& /*mpiComm*/);
''', '''    void setComm(const MpiComm& /*mpiComm*/);

    /*! \\brief Store the GPU device mdrun uses on this rank (nullptr for CPU-only runs).
     * \\param[in] deviceInfo device information, or nullptr
     */
    void setDeviceInformation(const DeviceInformation* deviceInfo);
''', 1),
('''    const MpiComm&  mpiComm() const;
''', '''    const MpiComm&  mpiComm() const;
    //! Get the GPU device mdrun uses on this rank, or nullptr
    const DeviceInformation* deviceInformation() const;
''', 1),
('''    const MpiComm* mpiComm_ = nullptr;
''', '''    const MpiComm* mpiComm_ = nullptr;

    //! GPU device used by mdrun on this rank, nullptr when running on the CPU
    const DeviceInformation* deviceInfo_ = nullptr;
''', 1)])
# forward declaration in the options header
s = open(os.path.join(R, NN + "nnpotoptions.h")).read()
assert "class MpiComm;" in s
s = s.replace("class MpiComm;", "class MpiComm;\n}\nstruct DeviceInformation;\nnamespace gmx\n{", 1) if "struct DeviceInformation;" not in s else s
open(os.path.join(R, NN + "nnpotoptions.h"), "w").write(s)

edit(NN + "nnpotoptions.cpp", [(
'''const MpiComm& NNPotOptions::mpiComm() const
''', '''void NNPotOptions::setDeviceInformation(const DeviceInformation* deviceInfo)
{
    deviceInfo_ = deviceInfo;
}

const DeviceInformation* NNPotOptions::deviceInformation() const
{
    return deviceInfo_;
}

const MpiComm& NNPotOptions::mpiComm() const
''', 1)])

edit(NN + "nnpot.cpp", [
('''        notifiers->simulationSetupNotifier_.subscribe(setCommFunction);
''', '''        notifiers->simulationSetupNotifier_.subscribe(setCommFunction);

        // GPU device used by mdrun on this rank, so model inference can run on the same device
        const auto setDeviceFunction = [this](const DeviceInformation* deviceInfo)
        { nnpotOptions_.setDeviceInformation(deviceInfo); };
        notifiers->simulationSetupNotifier_.subscribe(setDeviceFunction);
''', 1),
('''        nnpotForceProvider_ = std::make_unique<NNPotForceProvider>(
                nnpotOptions_.parameters(), nnpotOptions_.logger(), nnpotOptions_.mpiComm());''',
'''        nnpotForceProvider_ = std::make_unique<NNPotForceProvider>(nnpotOptions_.parameters(),
                                                                   nnpotOptions_.logger(),
                                                                   nnpotOptions_.mpiComm(),
                                                                   nnpotOptions_.deviceInformation());''', 1)])

edit(NN + "nnpotforceprovider.h", [
('''class LinkFrontierAtom;
''', '''class LinkFrontierAtom;
}
struct DeviceInformation;
namespace gmx
{
''', 1),
('''    NNPotForceProvider(const NNPotParameters& nnpotParameters, const MDLogger& logger, const MpiComm& mpiComm);''',
'''    NNPotForceProvider(const NNPotParameters&   nnpotParameters,
                       const MDLogger&          logger,
                       const MpiComm&           mpiComm,
                       const DeviceInformation* deviceInfo = nullptr);''', 1)])
edit(NN + "nnpotforceprovider.cpp", [
('''NNPotForceProvider::NNPotForceProvider(const NNPotParameters& nnpotParameters,
                                       const MDLogger&        logger,
                                       const MpiComm&         mpiComm) :''',
'''NNPotForceProvider::NNPotForceProvider(const NNPotParameters&   nnpotParameters,
                                       const MDLogger&          logger,
                                       const MpiComm&           mpiComm,
                                       const DeviceInformation* deviceInfo) :''', 1),
('''        model_ = std::make_shared<TorchModel>(
                params_.modelFileName_, params_.embeddingScheme_, logger_, mpiComm_);''',
'''        model_ = std::make_shared<TorchModel>(
                params_.modelFileName_, params_.embeddingScheme_, logger_, mpiComm_, deviceInfo);''', 1)])

edit(NN + "torchmodel.h", [
('''class LinkFrontierAtom;
''', '''class LinkFrontierAtom;
}
struct DeviceInformation;
namespace gmx
{
''', 1),
('''     * \\param[in] mpiComm handle to MpiComm
     */
    TorchModel(const std::string& filename, NNPotEmbedding embedding, const MDLogger& logger, const MpiComm& mpiComm);''',
'''     * \\param[in] mpiComm handle to MpiComm
     * \\param[in] deviceInfo GPU device used by mdrun on this rank, nullptr for CPU-only runs
     */
    TorchModel(const std::string&       filename,
               NNPotEmbedding           embedding,
               const MDLogger&          logger,
               const MpiComm&           mpiComm,
               const DeviceInformation* deviceInfo = nullptr);''', 1)])

# ---- 4. torchmodel.cpp: XPU device selection ----
edit(NN + "torchmodel.cpp", [
('''#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>
''', '''#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/cuda.h>
#include <torch/script.h>
#include <torch/torch.h>
// Intel GPUs: the XPU C++ API exists from PyTorch 2.4 on; older LibTorch is still supported.
#if __has_include(<torch/xpu.h>)
#    include <torch/xpu.h>
#    define GMX_TORCH_HAVE_XPU_API 1
#else
#    define GMX_TORCH_HAVE_XPU_API 0
#endif
#if GMX_TORCH_XPU && GMX_TORCH_HAVE_XPU_API
#    include <c10/xpu/XPUFunctions.h>
#endif
''', 1),
('''static std::tuple<std::string, std::optional<int>> getMdrunActiveDevice()
{
#if GMX_GPU_CUDA || (GMX_SYCL_ACPP && GMX_ACPP_HAVE_CUDA_TARGET) // AdaptiveCpp uses CUDA Runtime API''',
'''static std::tuple<std::string, std::optional<int>> getMdrunActiveDevice(const DeviceInformation* deviceInfo)
{
    GMX_UNUSED_VALUE(deviceInfo);
#if GMX_GPU_CUDA || (GMX_SYCL_ACPP && GMX_ACPP_HAVE_CUDA_TARGET) // AdaptiveCpp uses CUDA Runtime API''', 1),
('''    return { "hip", activeDevice };
#else
    return { "cpu", std::nullopt };
#endif
}
''', '''    return { "hip", activeDevice };
#elif GMX_SYCL_DPCPP && GMX_TORCH_XPU && GMX_TORCH_HAVE_XPU_API
    // Intel GPUs through LibTorch's XPU backend. There is no process-wide "current device" in
    // SYCL, so the device mdrun was assigned is matched against the devices torch enumerates.
    if (deviceInfo == nullptr || !torch::xpu::is_available())
    {
        return { "xpu", std::nullopt };
    }
    for (c10::DeviceIndex i = 0; i < c10::xpu::device_count(); ++i)
    {
        if (c10::xpu::get_raw_device(i) == deviceInfo->syclDevice)
        {
            return { "xpu", static_cast<int>(i) };
        }
    }
    return { "xpu", std::nullopt };
#elif GMX_SYCL_DPCPP
    // SYCL build for Intel GPUs, but this LibTorch has no XPU support: inference stays on the CPU
    return { "xpu", std::nullopt };
#else
    return { "cpu", std::nullopt };
#endif
}

//! \\brief \\internal torch device type for a device type string from getMdrunActiveDevice()
static torch::DeviceType torchDeviceTypeFor(const std::string& deviceType)
{
    if (deviceType == "xpu")
    {
        return torch::kXPU;
    }
    return torch::kCUDA; // LibTorch addresses HIP devices through the CUDA API
}

//! \\brief \\internal whether LibTorch can see a GPU of the given type
static bool torchGpuAvailable(const std::string& deviceType)
{
    if (deviceType == "xpu")
    {
#if GMX_TORCH_HAVE_XPU_API
        return torch::xpu::is_available();
#else
        return false;
#endif
    }
    return torch::cuda::is_available();
}

//! \\brief \\internal explain why a GPU of the given type is not usable, for log and error messages
static std::string gpuUnavailableReason(const std::string& deviceType)
{
    if (deviceType == "xpu")
    {
#if !GMX_TORCH_XPU
        return "LibTorch was built without XPU (Intel GPU) support; a build with XPU support "
               "(PyTorch >= 2.5) is needed to run the model on the GPU.";
#else
        return "no Intel GPU is visible to LibTorch, or the device mdrun uses is not among the "
               "devices LibTorch enumerates (check ONEAPI_DEVICE_SELECTOR and the Level Zero "
               "driver).";
#endif
    }
    return "no matching device is available.";
}
''', 1),
('''static torch::Device determineDevice(const MDLogger& logger, const MpiComm& mpiComm)
{''', '''static torch::Device determineDevice(const MDLogger& logger, const MpiComm& mpiComm, const DeviceInformation* deviceInfo)
{''', 1),
('''    auto [torchDeviceType, activeDevice] = getMdrunActiveDevice();
''', '''    auto [torchDeviceType, activeDevice] = getMdrunActiveDevice(deviceInfo);
    const bool gpuUsable = torchGpuAvailable(torchDeviceType) && activeDevice.has_value();
''', 1),
('''            if (!torch::cuda::is_available())
            {
                GMX_THROW(InternalError(formatString(
                        "GMX_NN_DEVICE was set to '%s', but no matching device is available.", env)));
            }
            GMX_RELEASE_ASSERT(
                    activeDevice.has_value(),
                    "Active device couldn't be determined. Ensure that Libtorch and GROMACS "
                    "are compiled with matching GPU support.");
            device = torch::Device(torch::kCUDA, activeDevice.value());''',
'''            if (!gpuUsable)
            {
                GMX_THROW(InternalError(formatString("GMX_NN_DEVICE was set to '%s', but %s",
                                                     env,
                                                     gpuUnavailableReason(torchDeviceType).c_str())));
            }
            device = torch::Device(torchDeviceTypeFor(torchDeviceType), activeDevice.value());''', 1),
('''        if (torch::cuda::is_available() && activeDevice.has_value())
        {
            GMX_LOG(logger.info)
                    .asParagraph()
                    .appendText(
                            "GMX_NN_DEVICE environment variable not found."
                            " Using "
                            + toUpperCase(torchDeviceType) + " device.");
            device = torch::Device(torch::kCUDA, activeDevice.value());
        }
        else
        {''', '''        if (gpuUsable)
        {
            GMX_LOG(logger.info)
                    .asParagraph()
                    .appendText(
                            "GMX_NN_DEVICE environment variable not found."
                            " Using "
                            + toUpperCase(torchDeviceType) + " device.");
            device = torch::Device(torchDeviceTypeFor(torchDeviceType), activeDevice.value());
        }
        else if (torchDeviceType == "xpu")
        {
            GMX_LOG(logger.warning)
                    .asParagraph()
                    .appendText(
                            "GMX_NN_DEVICE environment variable not found, and the model cannot run "
                            "on the Intel GPU: "
                            + gpuUnavailableReason(torchDeviceType) + " Using CPU.");
        }
        else
        {''', 1),
('''TorchModel::Impl::Impl(const std::string& fileName,
                       NNPotEmbedding     embedding,
                       const MDLogger&    logger,
                       const MpiComm&     mpiComm) :
    mpiComm_(mpiComm), logger_(logger), embeddingScheme_(embedding), device_(determineDevice(logger, mpiComm))''',
'''TorchModel::Impl::Impl(const std::string&       fileName,
                       NNPotEmbedding           embedding,
                       const MDLogger&          logger,
                       const MpiComm&           mpiComm,
                       const DeviceInformation* deviceInfo) :
    mpiComm_(mpiComm),
    logger_(logger),
    embeddingScheme_(embedding),
    device_(determineDevice(logger, mpiComm, deviceInfo))''', 1)])
# Impl declaration + public ctor forwarding: find them generically
p = os.path.join(R, NN + "torchmodel.cpp"); s = open(p).read()
m = re.search(r'    Impl\(const std::string& fileName,\s*NNPotEmbedding\s+embedding,\s*const MDLogger&\s+logger,\s*const MpiComm&\s+mpiComm\);', s)
assert m, "Impl decl"
s = s[:m.start()] + '''    Impl(const std::string&       fileName,
         NNPotEmbedding           embedding,
         const MDLogger&          logger,
         const MpiComm&           mpiComm,
         const DeviceInformation* deviceInfo);''' + s[m.end():]
m = re.search(r'TorchModel::TorchModel\(const std::string& filename,\s*NNPotEmbedding\s+embedding,\s*const MDLogger&\s+logger,\s*const MpiComm&\s+mpiComm\) :\s*impl_\(std::make_unique<Impl>\(filename, embedding, logger, mpiComm\)\)', s)
assert m, "public ctor"
s = s[:m.start()] + '''TorchModel::TorchModel(const std::string&       filename,
                       NNPotEmbedding           embedding,
                       const MDLogger&          logger,
                       const MpiComm&           mpiComm,
                       const DeviceInformation* deviceInfo) :
    impl_(std::make_unique<Impl>(filename, embedding, logger, mpiComm, deviceInfo))''' + s[m.end():]
open(p, "w").write(s); print("edited torchmodel.cpp ctor/impl")

edit(NN + "torchmodel.cpp", [('#include "gromacs/utility/exceptions.h"\n', '#include "gromacs/utility/basedefinitions.h"\n#include "gromacs/utility/exceptions.h"\n', 1)])

# ---- 5. config.h: GMX_TORCH_XPU ----
edit("src/config.h.cmakein", [('''#cmakedefine01 GMX_TORCH
''', '''#cmakedefine01 GMX_TORCH

/* LibTorch has XPU (Intel GPU) support */
#cmakedefine01 GMX_TORCH_XPU
''', 1)])

# ---- 6. tests: run the GPU cases on Intel devices too, passing the device to the provider ----
edit(NN + "tests/nnpotforceprovider.cpp", [
('''            if (deviceInfo.deviceVendor == DeviceVendor::Nvidia)
            {
                SCOPED_TRACE("Check construction on default NVIDIA GPU");
                auto envGuard = GmxEnvGuard("GMX_NN_DEVICE", "cuda");
                EXPECT_NO_THROW(NNPotForceProvider nnpotForceProvider(params_, logger_, mpiComm_));
                break; // Only test one GPU until we have a better way to hande device selection
            }''',
'''            if (deviceInfo.deviceVendor == DeviceVendor::Nvidia
                || (GMX_SYCL_DPCPP && deviceInfo.deviceVendor == DeviceVendor::Intel))
            {
                SCOPED_TRACE("Check construction on GPU " + std::string(deviceInfo.device_name));
                auto envGuard = GmxEnvGuard("GMX_NN_DEVICE", "gpu");
                EXPECT_NO_THROW(NNPotForceProvider nnpotForceProvider(params_, logger_, mpiComm_, &deviceInfo));
                break; // Only test one GPU
            }''', 1),
('''        if (deviceInfo.deviceVendor == DeviceVendor::Nvidia)
        {
            SCOPED_TRACE("Check calculate forces on default NVIDIA GPU");
            auto envGuard = GmxEnvGuard("GMX_NN_DEVICE", "cuda");
            testCalculateForces(data);
            break; // Only test one GPU until we have a better way to hande device selection
        }''',
'''        if (deviceInfo.deviceVendor == DeviceVendor::Nvidia
            || (GMX_SYCL_DPCPP && deviceInfo.deviceVendor == DeviceVendor::Intel))
        {
            SCOPED_TRACE("Check calculate forces on GPU " + std::string(deviceInfo.device_name));
            auto envGuard = GmxEnvGuard("GMX_NN_DEVICE", "gpu");
            testCalculateForces(data, &deviceInfo);
            break; // Only test one GPU
        }''', 1)])
# testCalculateForces signature: find and extend
p = os.path.join(R, NN + "tests/nnpotforceprovider.cpp"); s = open(p).read()
m = re.search(r'void testCalculateForces\(gmx::test::TestReferenceData& testData\)\s*\{', s)
assert m, "testCalculateForces"
s = s[:m.start()] + 'void testCalculateForces(gmx::test::TestReferenceData& testData, const DeviceInformation* deviceInfo = nullptr)\n    {' + s[m.end():]
# inside it, the provider construction
n = s.count("std::make_unique<NNPotForceProvider>(params_, logger_, mpiComm_));")
assert n == 1, ("construction in testCalculateForces", n)
s = s.replace("std::make_unique<NNPotForceProvider>(params_, logger_, mpiComm_));",
              "std::make_unique<NNPotForceProvider>(params_, logger_, mpiComm_, deviceInfo));", 1)
open(p, "w").write(s); print("edited tests")

# ---- 7. docs ----
edit("docs/reference-manual/special/nnpot.rst", [(
'''If multiple GPUs are available, the active device will be inherited from mdrun.
For ``gpu``, a CUDA/HIP-aware LibTorch installation should be installed, and the
corresponding CUDA/HIP installation should be available in your ``PATH``. ''',
'''If multiple GPUs are available, the active device will be inherited from mdrun.
For ``gpu``, a CUDA/HIP-aware LibTorch installation should be installed, and the
corresponding CUDA/HIP installation should be available in your ``PATH``.
On Intel GPUs, |Gromacs| must be built with SYCL (oneAPI DPC++) and a LibTorch
built with XPU support (PyTorch 2.5 or later); the model then runs on the same
Intel GPU as mdrun. If the LibTorch found at build time has no XPU support, CMake
warns and inference falls back to the CPU. ''', 1)])
edit("docs/install-guide/index.rst", [(
'''The NNP interface is enabled by default when a Libtorch installation is found in the''',
'''Recent Libtorch versions (2.14 and later) require C++20; configure |Gromacs| with
``-DCMAKE_CXX_STANDARD=20`` in that case (CMake reports it when the Libtorch headers
do not compile with the default standard).
To run model inference on Intel GPUs, build |Gromacs| with SYCL (oneAPI DPC++) and
use a Libtorch with XPU support (PyTorch 2.5 or later, for example the ``libtorch``
XPU download or the ``torch`` XPU wheel). The Intel oneAPI runtime libraries that
Libtorch links against must be on the library path at run time.
The NNP interface is enabled by default when a Libtorch installation is found in the''', 1)])
p = os.path.join(R, "docs/release-notes/2027/major/features.rst"); s = open(p).read()
s = s.rstrip("\n") + '''

Neural network potentials can run on Intel GPUs
"""""""""""""""""""""""""""""""""""""""""""""""

The NNPot interface now runs model inference on Intel GPUs when |Gromacs| is
built with SYCL (oneAPI DPC++) and a LibTorch with XPU support (PyTorch 2.5 or
later). The model runs on the GPU assigned to the mdrun rank, which is also what
``GMX_NN_DEVICE=gpu`` selects on CUDA and HIP builds now that mdrun passes its
device to the modules. A LibTorch without XPU support is detected at configure
time and inference falls back to the CPU with a warning.
'''
open(p, "w").write(s); print("release notes")
print("ALL-EDITS-APPLIED")
