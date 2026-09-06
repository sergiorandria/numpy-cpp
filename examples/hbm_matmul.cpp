/**
 * @example hbm_matmul.cpp
 * HBM/CXL heterogeneous memory + tensor cores
 */
#include <iostream>
#include <np/np.hpp>

int main()
{
    auto a = np::eye<float>(4);
    auto b = np::eye<float>(4);

    // HBM
    auto ha = np::mem::migrate_to_hbm(a);
    auto hb = np::mem::migrate_to_hbm(b);
    auto hc = np::mem::migrate_to_host(ha); // demo round-trip
    std::cout << "HBM " << ha.size() << " " << hc.size() << "\n";

    // Tensor core FP8
    auto c = np::tensor::matmul_fp8(a, b, 1.0f, 1.0f);
    std::cout << "tensor matmul_fp8 (0,0) " << c(0, 0) << "\n";

    // Accelerator Strategy
    auto cpu = np::accelerator::AcceleratorFactory::cpu();
    auto gpu = np::accelerator::AcceleratorFactory::gpu();
    std::cout << cpu->name() << " " << gpu->name() << "\n";
    std::cout << "acc matmul " << cpu->matmul(a, b)(0, 0) << "\n";
    return 0;
}
