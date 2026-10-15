# Apply the NNPot Intel-GPU edits to a GROMACS 2026.3 tree (the source of patches/gromacs-2026.3-nnpot-xpu.patch).
#   python apply_nnpot_xpu.py <gromacs-2026.3 source dir>
import sys, os
root = sys.argv[1]
p = os.path.join(root, "src/gromacs/applied_forces/nnpot/torchmodel.cpp"); s = open(p).read()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a)); s = s.replace(a, b)
rep('#include "torchmodel.h"\n\n#include <dlfcn.h>', '#include "torchmodel.h"\n\n#include <torch/xpu.h>\n\n#include <dlfcn.h>')
rep('''    return { "hip", activeDevice };
#else
    return { "cpu", std::nullopt };
#endif
}
''', '''    return { "hip", activeDevice };
#elif GMX_SYCL_DPCPP
    // Intel GPUs: libtorch's XPU backend (PyTorch >= 2.5). mdrun's SYCL device and torch's XPU
    // enumeration are both in Level Zero order; with one GPU per rank this is device 0.
    if (torch::xpu::is_available())
    {
        return { "xpu", 0 };
    }
    return { "cpu", std::nullopt };
#else
    return { "cpu", std::nullopt };
#endif
}

//! \\brief \\internal torch device type for the string returned by getMdrunActiveDevice()
static torch::DeviceType torchDeviceTypeFor(const std::string& type)
{
    if (type == "xpu")
    {
        return torch::kXPU;
    }
    return torch::kCUDA; // libtorch addresses ROCm/HIP devices as kCUDA
}

//! \\brief \\internal whether libtorch can see a GPU of the given type
static bool torchGpuAvailable(const std::string& type)
{
    if (type == "xpu")
    {
        return torch::xpu::is_available();
    }
    return torch::cuda::is_available();
}
''')
rep('            if (!torch::cuda::is_available())\n', '            if (!torchGpuAvailable(torchDeviceType))\n')
rep('        if (torch::cuda::is_available() && activeDevice.has_value())\n', '        if (torchGpuAvailable(torchDeviceType) && activeDevice.has_value())\n')
rep('device = torch::Device(torch::kCUDA, activeDevice.value());', 'device = torch::Device(torchDeviceTypeFor(torchDeviceType), activeDevice.value());', 2)
open(p, "w").write(s)
p = os.path.join(root, "src/gromacs/applied_forces/nnpot/nnpot.cpp"); s = open(p).read()
assert s.count('"pairs-shifts"') == 2; s = s.replace('"pairs-shifts"', '"pair-shifts"'); open(p, "w").write(s)
print("edited", root)
