/**
 * @file test_accelerator.cpp
 */
#include "test_util.hpp"
#include <cmath>
#include <np/np.hpp>
int main()
{
    using namespace np::accelerator;
    auto cpu = AcceleratorFactory::cpu();
    auto gpu = AcceleratorFactory::gpu();
    auto reram = AcceleratorFactory::reram();
    auto automatic = AcceleratorFactory::auto_select();
    test::check(cpu->name() == "CPU", "CPU");
    test::check(gpu->name() == "GPU", "GPU");
    test::check(reram->name() == "ReRAM-sim", "ReRAM-sim");
    test::check(automatic->name() == "Auto", "Auto");
    auto a = np::eye<float>(2);
    auto b = np::eye<float>(2);
    auto c = cpu->matmul(a, b);
    test::check(c.size() == 4, "accelerator matmul");
    // ReRAM-sim runs the analog crossbar model: close to ideal on this
    // small input (8-bit DAC/ADC), and deterministic across runs.
    auto r1 = reram->matmul(a, b);
    auto r2 = reram->matmul(a, b);
    bool close = r1.size() == 4 && r2.size() == 4;
    for (size_t i = 0; close && i < 4; ++i)
        close = std::abs(r1.data()[i] - c.data()[i]) < 2e-2 && r1.data()[i] == r2.data()[i];
    test::check(close, "ReRAM-sim close + deterministic");
    // AutoAccelerator dispatches and stays consistent across size classes.
    auto big = np::eye<float>(40);
    auto abig = automatic->matmul(big, big);
    test::check(abig.size() == 1600 && std::abs(abig.data()[0] - 1.0f) < 1e-4, "auto dispatch");
    return test::failures() ? 1 : 0;
}
