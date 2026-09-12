/**
 * @file fft.hpp
 * @brief Fast Fourier Transforms (np::fft).
 *
 * Umbrella header for the whole np::fft module, split over:
 *  - fft/fft_core.hpp   : shared engines (radix-2, Bluestein, twiddle cache)
 *  - fft/fft_1d.hpp     : fft, ifft, rfft, irfft, hfft, ihfft
 *  - fft/fft_nd.hpp     : fftn, ifftn, fft2, ifft2, rfftn, irfftn, rfft2, irfft2
 *  - fft/fft_shift.hpp  : fftfreq, rfftfreq, fftshift, ifftshift
 *
 * The DFT convention matches NumPy: forward X[k] = sum_j x[j] e^(-2*pi*i*j*k/n),
 * with normalization governed by np::fft::Norm (default Backward).
 * Inputs are promoted to std::complex<double>.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_FFT_HPP
#define NP_FFT_HPP

#include "api_macros.hpp"
#include "fft/fft_1d.hpp"
#include "fft/fft_core.hpp"
#include "fft/fft_nd.hpp"
#include "fft/fft_shift.hpp"
#include "pqc.hpp"

namespace np::fft::secure
{
template <typename... Args> NP_NODISCARD inline auto fft(Args &&...args)
{
    auto r = ::np::fft::fft(std::forward<Args>(args)...);
    pqc::ct_barrier();
    return r;
}
template <typename... Args> NP_NODISCARD inline auto ifft(Args &&...args)
{
    auto r = ::np::fft::ifft(std::forward<Args>(args)...);
    pqc::ct_barrier();
    return r;
}
template <typename... Args> NP_NODISCARD inline auto rfft(Args &&...args)
{
    auto r = ::np::fft::rfft(std::forward<Args>(args)...);
    pqc::ct_barrier();
    return r;
}
template <typename... Args> NP_NODISCARD inline auto fftn(Args &&...args)
{
    auto r = ::np::fft::fftn(std::forward<Args>(args)...);
    pqc::ct_barrier();
    return r;
}
} // namespace np::fft::secure

#endif // NP_FFT_HPP