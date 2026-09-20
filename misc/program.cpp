#include "../include/np/np.hpp"
#include <numbers>

int main() {
    // Create arrays
    auto a = np::arange<double>(0.0, 10.0, 0.5);  // [0, 0.5, 1.0, ..., 9.5]
    auto b = np::zeros<double>({3, 4});           // 3x4 array of zeros
    auto c = np::ndarray<int>{{1, 2}, {3, 4}};    // 2x2 from initializer list
    a.print();
    b.print();
    c.print();
    // Mathematical operations
    auto x = np::linspace<double>(0.0, 2.0 * std::numbers::pi, 100);
    auto y = np::sin(x);  // Element-wise sine
    x.print(); y.print();
    // Array operations with broadcasting
    auto result = a * 2.0 + b;  // Broadcasting supported
    result.print();
    // Reductions
    //double sum = a.sum();
    //double mean = a.mean();
    //auto col_sums = b.sum(0);  // Sum along axis 0
    
    // Linear algebra
    //auto m1 = np::eye<double>(3);
    //auto m2 = np::ones<double>({3, 3});
    //auto product = np::linalg::matmul(m1, m2);
    
    return 0;
}