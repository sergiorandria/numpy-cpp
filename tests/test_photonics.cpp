/**
 * @file test_photonics.cpp
 */
#include "test_util.hpp"
#include <np/np.hpp>
int main()
{
    using namespace np::photonics;
    auto mesh = PhotonicsFactory::identity(2);
    auto x = np::ndarray<c128>(std::vector<int>{2});
    x[0] = c128(1, 0);
    x[1] = c128(0, 0);
    auto y = mesh.apply(x);
    test::check(std::abs(static_cast<c128>(y[0]).real() - 1) < 1e-9, "photonics identity");
    return test::failures() ? 1 : 0;
}
