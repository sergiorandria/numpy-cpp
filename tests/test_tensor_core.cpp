/**
 * @file test_tensor_core.cpp
 */
#include "test_util.hpp"
#include <np/np.hpp>
int main()
{
    using namespace np::tensor;
    auto a = np::eye<float>(2);
    auto b = np::eye<float>(2);
    auto c = matmul_fp8(a, b, 1.0f, 1.0f);
    test::check(std::abs(c(0, 0) - 1) < 1e-3, "tensor matmul_fp8");
    auto cpu = TensorFactory::cpu();
    test::check(cpu->name() == "CPU", "CPU backend");
    auto hop = TensorFactory::hopper();
    test::check(hop->name() == "Hopper-FP8" || hop->name() == "Blackwell-FP4", "Hopper");
    auto amx = TensorFactory::amx();
    test::check(amx->name() == "AMX", "AMX");
    QuantizedTensor<float> qt{a, 0.5f, TensorDtype::FP8};
    auto dq = qt.dequantize();
    test::check(dq.size() == 4, "quantized dequant");
    auto q = quantize(a, 0.5f);
    test::check(q.size() == 4, "quantize");
    return test::failures() ? 1 : 0;
}
