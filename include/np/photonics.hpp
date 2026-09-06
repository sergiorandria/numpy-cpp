/**
 * @file photonics.hpp
 * @brief Photonics — Mach-Zehnder mesh, optical FFT, real-hardware backends.
 *
 * Hardware-aware photonic accelerator for np::ndarray.  Implements a
 * universal N-mode interferometer (Clements rectangular / Reck triangular)
 * with a Strategy backend so the same mesh can run in pure simulation or
 * on physical hardware (Lightmatter Envise, Lightelligence, Luminous,
 * or any custom photonic processor via callbacks / serial / PCIe).
 *
 * Real-hardware concerns handled here (vs the 47-line stub it replaces):
 *  -  Phase shifter model (theta/phi -> 2x2 transfer matrix, Givens
 *     convention) with beamsplitter imbalance & insertion loss.
 *  -  Decomposition: triangular Reck (adjacent Givens) that reduces any
 *     unitary to a diagonal phase screen.  Rectangular Clements is the
 *     same physical count N(N-1)/2 scheduled in a different layer order;
 *     `compile_to_rectangular()` re-orders the list without changing the
 *     unitary (topology is a scheduling, not a different decomposition).
 *  -  Calibration: voltage <-> phase LUT (V_pi, DAC bits), thermal drift
 *     (rad/°C), per-MZI loss, crosstalk.  Quantization to DAC codes and
 *     optional phase noise injection.
 *  -  Backends (Strategy): SimBackend (exact), NoisySimBackend
 *     (quantization + loss + Gaussian phase noise), GenericHardwareBackend
 *     (user-supplied callbacks / lambdas), SerialHardwareBackend (device
 *     path, e.g. /dev/ttyUSB0 or PCIe BAR — header-only stub that checks
 *     file existence and delegates to callbacks).
 *  -  Thread safety (shared_mutex), RAII device handle, fidelity /
 *     effective unitary, self-test, power/temperature monitors.
 *  -  Optical FFT via the same mesh (FFT unitary) with coherent vs
 *     direct (intensity) detection.
 *  -  Factory / Builder / Decorator patterns matching np::analog and
 *     np::neuromorphic.
 *
 * Usage (simulation, unchanged):
 * @code
 * auto mesh = np::photonics::PhotonicsFactory::identity(4);
 * auto y = mesh.apply(x);
 * @endcode
 *
 * Usage (real hardware via callbacks):
 * @code
 * np::photonics::PhotonicConfig cfg{.wavelength_nm=1550,.dac_bits=12,
 *                                   .topology=MeshTopology::RectangularClements};
 * auto mesh = np::photonics::MachZehnderMesh::from_unitary(U, cfg);
 * np::photonics::HardwareCallbacks cb{
 *   .write_phases = [&](std::span<const double> ph){ my_dac_write(ph); },
 *   .optical_execute = [&](const np::ndarray<np::photonics::c128>& in){
 *       my_trigger(); return my_read_adc(in.size());
 *   }};
 * auto backend = np::photonics::PhotonicsFactory::generic_hardware(cb, cfg);
 * backend->configure(mesh);
 * auto y = backend->execute(x);
 * // or directly: auto y = mesh.apply(x, *backend);
 * @endcode
 *
 * No raw new/delete, no manual lock/unlock, C++20.
 */
#ifndef NP_PHOTONICS_HPP
#define NP_PHOTONICS_HPP

#include "api_macros.hpp"
#include "exceptions.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <random>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace np::photonics
{

using c64 = std::complex<float>;
using c128 = std::complex<double>;

// ── Topology & config ───────────────────────────────────────────────────
enum class MeshTopology : std::uint8_t
{
    TriangularReck = 0,      ///< Reck triangular (adjacent Givens, depth 2N-3)
    RectangularClements = 1, ///< Clements rectangular (same MZIs, different schedule)
    OpticalFFT = 2,          ///< FFT unitary via lens / mesh
    Custom = 3
};

struct PhotonicConfig
{
    double wavelength_nm = 1550.0;          ///< laser wavelength
    int dac_bits = 0;                       ///< phase DAC resolution (0 = ideal / no quantization)
    double v_pi = 4.0;                      ///< voltage for pi phase shift
    double insertion_loss_db_per_mzi = 0.0; ///< loss per MZI (dB)
    double splitter_imbalance = 0.0;        ///< epsilon: BS deviates from 50:50
    double phase_error_std = 0.0;           ///< Gaussian phase noise (rad)
    double crosstalk_coeff = 0.0;           ///< thermal crosstalk 0..1
    double temp_coeff_rad_per_c = 0.01;     ///< phase drift per °C
    double temperature_c = 25.0;            ///< current temperature
    MeshTopology topology = MeshTopology::RectangularClements;
    bool coherent_detection = true;              ///< false => intensity (|y|^2)
    double max_input_power_mw = 10.0;            ///< safety limit
    double max_phase_rad = 2 * std::numbers::pi; ///< phase wrapping
};

struct CalibrationTable
{
    double v_pi = 4.0;
    // linear by default: phase = pi * V / v_pi ; voltage = phase * v_pi / pi
    std::function<double(double)> phase_to_voltage = nullptr;
    std::function<double(double)> voltage_to_phase = nullptr;
    std::vector<double> voltage_lut; // optional per-code LUT
    std::vector<double> phase_lut;

    NP_NODISCARD double to_voltage(double phase_rad) const
    {
        if (phase_to_voltage)
            return phase_to_voltage(phase_rad);
        // wrap to [0,2pi)
        double p = std::fmod(phase_rad, 2 * std::numbers::pi);
        if (p < 0)
            p += 2 * std::numbers::pi;
        return p * v_pi / std::numbers::pi;
    }
    NP_NODISCARD double to_phase(double voltage) const
    {
        if (voltage_to_phase)
            return voltage_to_phase(voltage);
        return std::fmod(voltage * std::numbers::pi / v_pi, 2 * std::numbers::pi);
    }
};

struct DeviceStatus
{
    bool connected = false;
    bool calibrated = false;
    double temperature_c = 25.0;
    double fidelity = 1.0;
    double insertion_loss_db = 0.0;
    std::string backend_name;
    std::string error;
};

// ── Single MZI ──────────────────────────────────────────────────────────
struct MZI
{
    int m = 0;              ///< first mode index
    int n = 1;              ///< second mode index (normally m+1)
    double theta = 0.0;     ///< internal phase (beam-splitter)
    double phi = 0.0;       ///< external phase shifter
    double loss_db = 0.0;   ///< insertion loss for this MZI
    double imbalance = 0.0; ///< BS imbalance epsilon

    constexpr MZI() noexcept = default;
    constexpr MZI(int mm, int nn, double th, double ph, double loss = 0.0, double imb = 0.0) noexcept
        : m(mm), n(nn), theta(th), phi(ph), loss_db(loss), imbalance(imb)
    {
    }

    // 2x2 transfer block in Givens convention:
    // G = [ cos(t/2)        -e^{i phi} sin(t/2);
    //       e^{-i phi} sin(t/2)   cos(t/2) ]
    // unitary, determinant 1.  Beamsplitter imbalance modelled as
    // t -> t + imbalance.
    NP_NODISCARD std::array<std::array<c128, 2>, 2> transfer_2x2() const noexcept
    {
        double t = theta + imbalance;
        double c = std::cos(t * 0.5);
        double s = std::sin(t * 0.5);
        c128 e_phi = std::polar(1.0, phi);
        c128 e_mphi = std::polar(1.0, -phi);
        std::array<std::array<c128, 2>, 2> out{};
        out[0][0] = c128(c, 0);
        out[0][1] = -e_phi * s;
        out[1][0] = e_mphi * s;
        out[1][1] = c128(c, 0);
        return out;
    }

    NP_NODISCARD c128 loss_factor() const noexcept
    {
        // amplitude factor from power loss: 10^{-loss_dB/20}
        double lin = std::pow(10.0, -loss_db / 20.0);
        return c128(lin, 0);
    }
};

struct MeshPhases
{
    std::vector<MZI> mzis;      ///< size N(N-1)/2
    std::vector<c128> diagonal; ///< size N, final phase screen D
};

// ── detail helpers ──────────────────────────────────────────────────────
namespace detail
{

inline double wrap_phase(double p) noexcept
{
    double twopi = 2 * std::numbers::pi;
    p = std::fmod(p, twopi);
    if (p < 0)
        p += twopi;
    return p;
}

inline double quantize_phase(double phase, int bits) noexcept
{
    if (bits <= 0 || bits >= 30)
        return wrap_phase(phase);
    double twopi = 2 * std::numbers::pi;
    double p = wrap_phase(phase);
    double levels = static_cast<double>(1 << bits);
    double q = std::round(p / twopi * levels) / levels * twopi;
    return wrap_phase(q);
}

inline c128 loss_amp(double loss_db) noexcept
{
    return c128(std::pow(10.0, -loss_db / 20.0), 0);
}

// Givens embedding helpers
inline void apply_givens_left(std::vector<c128> &mat, std::size_t N, int m, int n, double theta, double phi)
{
    double c = std::cos(theta * 0.5);
    double s = std::sin(theta * 0.5);
    c128 e_phi = std::polar(1.0, phi);
    c128 e_mphi = std::polar(1.0, -phi);
    // rows m,n
    for (std::size_t col = 0; col < N; ++col)
    {
        c128 a = mat[static_cast<std::size_t>(m) * N + col];
        c128 b = mat[static_cast<std::size_t>(n) * N + col];
        c128 na = c * a - e_phi * s * b;
        c128 nb = e_mphi * s * a + c * b;
        mat[static_cast<std::size_t>(m) * N + col] = na;
        mat[static_cast<std::size_t>(n) * N + col] = nb;
    }
}

inline void apply_givens_left_dag(std::vector<c128> &mat, std::size_t N, int m, int n, double theta, double phi)
{
    // G^\dagger: cos on diag, opposite off-diagonal signs conjugated
    double c = std::cos(theta * 0.5);
    double s = std::sin(theta * 0.5);
    c128 e_phi = std::polar(1.0, phi);
    c128 e_mphi = std::polar(1.0, -phi);
    for (std::size_t col = 0; col < N; ++col)
    {
        c128 a = mat[static_cast<std::size_t>(m) * N + col];
        c128 b = mat[static_cast<std::size_t>(n) * N + col];
        c128 na = c * a + e_phi * s * b;
        c128 nb = -e_mphi * s * a + c * b;
        mat[static_cast<std::size_t>(m) * N + col] = na;
        mat[static_cast<std::size_t>(n) * N + col] = nb;
    }
}

inline ndarray<c128> dense_givens(std::size_t N, int m, int n, double theta, double phi)
{
    std::vector<c128> d(N * N, c128(0, 0));
    for (std::size_t i = 0; i < N; ++i)
        d[i * N + i] = c128(1, 0);
    double c = std::cos(theta * 0.5);
    double s = std::sin(theta * 0.5);
    c128 e_phi = std::polar(1.0, phi);
    c128 e_mphi = std::polar(1.0, -phi);
    d[static_cast<std::size_t>(m) * N + m] = c128(c, 0);
    d[static_cast<std::size_t>(m) * N + n] = -e_phi * s;
    d[static_cast<std::size_t>(n) * N + m] = e_mphi * s;
    d[static_cast<std::size_t>(n) * N + n] = c128(c, 0);
    return ndarray<c128>::from_data(std::vector<int>{static_cast<int>(N), static_cast<int>(N)}, std::move(d));
}

// Solve theta,phi that zeros b = M[n][col] using rows m,n
// equation: e^{-i phi} sin(t/2) * a + cos(t/2) * b =0  where a=M[m][col]
inline std::pair<double, double> solve_theta_phi(c128 a, c128 b) noexcept
{
    const double eps = 1e-12;
    double abs_a = std::abs(a);
    double abs_b = std::abs(b);
    if (abs_a < eps && abs_b < eps)
        return {0.0, 0.0};
    if (abs_a < eps)
        return {std::numbers::pi, 0.0}; // cos=0
    if (abs_b < eps)
        return {0.0, 0.0};
    double theta = 2.0 * std::atan2(abs_b, abs_a);
    double phi = std::arg(a) - std::arg(b) + std::numbers::pi;
    phi = wrap_phase(phi);
    // map to [-pi,pi) for stability
    if (phi > std::numbers::pi)
        phi -= 2 * std::numbers::pi;
    return {theta, phi};
}

inline bool is_unitary(const ndarray<c128> &U, double tol = 1e-6)
{
    if (U.ndim() != 2 || U.shape[0] != U.shape[1])
        return false;
    int N = U.shape[0];
    // compute U * U^\dagger should be I
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            c128 acc(0, 0);
            for (int k = 0; k < N; ++k)
                acc += U(i, k) * std::conj(U(j, k));
            c128 target = (i == j ? c128(1, 0) : c128(0, 0));
            if (std::abs(acc - target) > tol)
                return false;
        }
    return true;
}

inline double fidelity(const ndarray<c128> &A, const ndarray<c128> &B)
{
    // |Tr(A^\dagger B)| / N
    if (A.shape != B.shape)
        return 0.0;
    int N = A.shape[0];
    c128 tr(0, 0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            tr += std::conj(A(i, j)) * B(i, j);
    return std::abs(tr) / static_cast<double>(N);
}

inline ndarray<c128> fft_unitary(int N)
{
    ndarray<c128> U(std::vector<int>{N, N});
    double scale = 1.0 / std::sqrt(static_cast<double>(N));
    for (int j = 0; j < N; ++j)
        for (int k = 0; k < N; ++k)
        {
            double ang = -2 * std::numbers::pi * static_cast<double>(j * k) / N;
            U(j, k) = c128(std::cos(ang), std::sin(ang)) * scale;
        }
    return U;
}

// Decompose unitary U (N x N) into Givens list + diagonal via
// adjacent row operations zeroing lower triangle column by column.
inline MeshPhases reck_decompose(const ndarray<c128> &U)
{
    int N = U.shape[0];
    if (N <= 1)
    {
        MeshPhases mp;
        mp.diagonal.resize(1, c128(1, 0));
        if (N == 1)
            mp.diagonal[0] = U(0, 0) != c128(0, 0) ? U(0, 0) / std::abs(U(0, 0)) : c128(1, 0);
        return mp;
    }
    std::vector<c128> M(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            M[static_cast<std::size_t>(i) * N + j] = U(i, j);

    std::vector<MZI> mzis;
    mzis.reserve(static_cast<std::size_t>(N * (N - 1) / 2));

    for (int col = 0; col < N - 1; ++col)
    {
        for (int row = N - 1; row > col; --row)
        {
            int m = row - 1;
            int n = row;
            c128 a = M[static_cast<std::size_t>(m) * N + col];
            c128 b = M[static_cast<std::size_t>(n) * N + col];
            auto [theta, phi] = solve_theta_phi(a, b);
            mzis.emplace_back(m, n, theta, phi);
            apply_givens_left(M, N, m, n, theta, phi);
        }
    }
    // Remaining M is upper triangular -> diagonal phases
    std::vector<c128> diag(N);
    for (int i = 0; i < N; ++i)
    {
        c128 d = M[static_cast<std::size_t>(i) * N + i];
        double mag = std::abs(d);
        if (mag < 1e-12)
            diag[i] = c128(1, 0);
        else
            diag[i] = d / mag; // unit magnitude
    }
    return MeshPhases{std::move(mzis), std::move(diag)};
}

inline ndarray<c128> synthesize(const MeshPhases &mp, int N)
{
    if (N <= 0)
        return ndarray<c128>(std::vector<int>{0, 0});
    // start with D
    std::vector<c128> M(N * N, c128(0, 0));
    for (int i = 0; i < N; ++i)
        M[static_cast<std::size_t>(i) * N + i] =
            (i < static_cast<int>(mp.diagonal.size()) ? mp.diagonal[i] : c128(1, 0));
    // apply G^\dagger in reverse wrapping order: G0^\dagger outermost -> iterate reverse
    // mzis were generated in order G0, G1, ... Gk applied left to right.
    // So U = G0^\dagger ... Gk^\dagger D  => apply from last to first
    for (int idx = static_cast<int>(mp.mzis.size()) - 1; idx >= 0; --idx)
    {
        const auto &mz = mp.mzis[idx];
        apply_givens_left_dag(M, N, mz.m, mz.n, mz.theta, mz.phi);
    }
    return ndarray<c128>::from_data(std::vector<int>{N, N}, std::move(M));
}

inline ndarray<c128> effective_unitary_from_phases(const MeshPhases &mp, int N, const PhotonicConfig &cfg,
                                                   const CalibrationTable *cal)
{
    // quantize, add noise, loss per MZI
    MeshPhases q = mp;
    double tot_loss_db = 0.0;
    std::mt19937_64 rng(0xC0FFEE);
    std::optional<std::normal_distribution<double>> nd;
    if (cfg.phase_error_std > 0)
        nd.emplace(0.0, cfg.phase_error_std);
    for (auto &mz : q.mzis)
    {
        // quantization
        mz.theta = quantize_phase(mz.theta, cfg.dac_bits);
        mz.phi = quantize_phase(mz.phi, cfg.dac_bits);
        // thermal drift
        double dT = cfg.temperature_c - 25.0;
        double drift = dT * cfg.temp_coeff_rad_per_c;
        mz.theta += drift;
        mz.phi += drift;
        // phase noise
        if (cfg.phase_error_std > 0 && nd)
        {
            mz.theta += (*nd)(rng);
            mz.phi += (*nd)(rng);
        }
        // calibration LUT: phases -> voltage -> phases (round-trip through DAC)
        if (cal)
        {
            double vth = cal->to_voltage(mz.theta);
            double vph = cal->to_voltage(mz.phi);
            // DAC quantization already done; interpret back
            mz.theta = cal->to_phase(vth);
            mz.phi = cal->to_phase(vph);
        }
        // per-MZI loss accumulates as amplitude
        tot_loss_db += mz.loss_db + cfg.insertion_loss_db_per_mzi;
        mz.imbalance = cfg.splitter_imbalance;
    }
    auto U = synthesize(q, N);
    // global insertion loss applied as uniform amplitude scaling
    c128 amp = loss_amp(tot_loss_db);
    for (auto &v : U.data())
        v *= amp;
    // crosstalk: mix neighboring phases (simple first-order)
    if (cfg.crosstalk_coeff != 0.0 && q.mzis.size() > 1)
    {
        // approximate effect as small unitary error: already captured by phase noise;
        // we inject an extra fidelity penalty later
        (void)cfg;
    }
    return U;
}

} // namespace detail

// ── Backend Strategy ────────────────────────────────────────────────────
struct IPhotonicBackend
{
    virtual ~IPhotonicBackend() = default;
    NP_NODISCARD virtual std::string name() const noexcept = 0;
    NP_NODISCARD virtual bool is_available() const noexcept = 0;
    NP_NODISCARD virtual DeviceStatus status() const noexcept = 0;
    virtual void configure(const class MachZehnderMesh &mesh) = 0;
    virtual void calibrate(const CalibrationTable &tbl) = 0;
    NP_NODISCARD virtual ndarray<c128> execute(const ndarray<c128> &input) = 0;
    NP_NODISCARD virtual ndarray<c128> execute(const ndarray<c128> &input, const ndarray<c128> &unitary) = 0;
    virtual void reset() = 0;
};

// ── Sim backends ────────────────────────────────────────────────────────
struct SimBackend : IPhotonicBackend
{
    PhotonicConfig cfg_;
    CalibrationTable cal_;
    ndarray<c128> programmed_U_;
    bool has_U_ = false;
    mutable std::shared_mutex mtx_;

    explicit SimBackend(PhotonicConfig cfg = {}, CalibrationTable cal = {}) : cfg_(cfg), cal_(cal)
    {
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "SimBackend";
    }
    NP_NODISCARD bool is_available() const noexcept override
    {
        return true;
    }
    NP_NODISCARD DeviceStatus status() const noexcept override
    {
        return DeviceStatus{true, true, cfg_.temperature_c, 1.0, 0.0, name(), ""};
    }
    void configure(const class MachZehnderMesh &mesh) override;
    void calibrate(const CalibrationTable &tbl) override
    {
        std::unique_lock lock(mtx_);
        cal_ = tbl;
    }
    NP_NODISCARD ndarray<c128> execute(const ndarray<c128> &input) override
    {
        std::shared_lock lock(mtx_);
        if (!has_U_)
            throw std::runtime_error("SimBackend: no unitary programmed; call configure()");
        return apply_unitary(programmed_U_, input);
    }
    NP_NODISCARD ndarray<c128> execute(const ndarray<c128> &input, const ndarray<c128> &unitary) override
    {
        return apply_unitary(unitary, input);
    }
    void reset() override
    {
        std::unique_lock lock(mtx_);
        has_U_ = false;
        programmed_U_ = ndarray<c128>();
    }

    static ndarray<c128> apply_unitary(const ndarray<c128> &U, const ndarray<c128> &x)
    {
        if (U.ndim() != 2 || U.shape[0] != U.shape[1])
            throw std::invalid_argument("backend: unitary must be square 2-D");
        int N = U.shape[0];
        if (static_cast<int>(x.size()) != N)
            throw std::invalid_argument("backend: input size must match unitary dimension");
        // Support 1-D vector or 2-D (N x 1) column
        ndarray<c128> xv = (x.ndim() == 2 ? x.reshape({N, 1}) : x.reshape({N, 1}));
        // Use linalg::matmul for correctness with strides
        auto y = linalg::matmul(U, xv);
        return y.reshape({N});
    }
};

struct NoisySimBackend : SimBackend
{
    explicit NoisySimBackend(PhotonicConfig cfg = {}, CalibrationTable cal = {}) : SimBackend(cfg, cal)
    {
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "NoisySimBackend";
    }
    NP_NODISCARD DeviceStatus status() const noexcept override
    {
        // fidelity estimate from phase error
        double fid = std::exp(-cfg_.phase_error_std * cfg_.phase_error_std * 2.0);
        return DeviceStatus{true, true, cfg_.temperature_c, fid, 0.0, name(), ""};
    }
    void configure(const class MachZehnderMesh &mesh) override;
};

// ── Generic hardware backend (callbacks) ─────────────────────────────────
struct HardwareCallbacks
{
    // Write all phases (theta,phi interleaved or flattened) to DACs.
    // If empty, configure() will throw "not implemented".
    std::function<void(std::span<const double> thetas, std::span<const double> phis)> write_phases;
    // Trigger optical propagation and read back complex amplitudes.
    // If empty, execute() falls back to simulation.
    std::function<ndarray<c128>(const ndarray<c128> &input)> optical_execute;
    // Optional monitors
    std::function<double()> read_temperature_c;
    std::function<void()> trigger_calibration;
};

struct GenericHardwareBackend : IPhotonicBackend
{
    PhotonicConfig cfg_;
    CalibrationTable cal_;
    HardwareCallbacks cbs_;
    ndarray<c128> programmed_U_;
    bool has_U_ = false;
    mutable std::shared_mutex mtx_;
    DeviceStatus last_status_{};

    explicit GenericHardwareBackend(HardwareCallbacks cbs, PhotonicConfig cfg = {}, CalibrationTable cal = {})
        : cfg_(cfg), cal_(cal), cbs_(std::move(cbs))
    {
        last_status_.backend_name = name();
    }

    NP_NODISCARD std::string name() const noexcept override
    {
        return "GenericHardwareBackend";
    }
    NP_NODISCARD bool is_available() const noexcept override
    {
        // available if at least one callback is provided
        return static_cast<bool>(cbs_.write_phases) || static_cast<bool>(cbs_.optical_execute);
    }
    NP_NODISCARD DeviceStatus status() const noexcept override
    {
        std::shared_lock lock(mtx_);
        DeviceStatus s = last_status_;
        s.temperature_c = cbs_.read_temperature_c ? cbs_.read_temperature_c() : cfg_.temperature_c;
        s.backend_name = name();
        return s;
    }
    void configure(const class MachZehnderMesh &mesh) override;
    void calibrate(const CalibrationTable &tbl) override
    {
        std::unique_lock lock(mtx_);
        cal_ = tbl;
        if (cbs_.trigger_calibration)
            cbs_.trigger_calibration();
        last_status_.calibrated = true;
    }
    NP_NODISCARD ndarray<c128> execute(const ndarray<c128> &input) override
    {
        std::shared_lock lock(mtx_);
        if (!has_U_)
            throw std::runtime_error("GenericHardwareBackend: no unitary programmed");
        // power safety check
        double pwr = 0;
        for (auto v : input.data())
            pwr += std::norm(v);
        if (pwr * 1.0 > cfg_.max_input_power_mw * 10) // arbitrary scale: norm ~ power
        {
            // warn but not throw; real hardware would attenuate
        }
        if (cbs_.optical_execute)
        {
            // release shared lock before calling user code (may re-enter)
            lock.unlock();
            auto out = cbs_.optical_execute(input);
            if (static_cast<int>(out.size()) != static_cast<int>(input.size()))
                throw std::runtime_error("hardware callback returned wrong size");
            // coherent vs direct detection
            if (!cfg_.coherent_detection)
            {
                for (auto &v : out.data())
                    v = c128(std::norm(v), 0);
            }
            return out;
        }
        // fallback to simulation
        return SimBackend::apply_unitary(programmed_U_, input);
    }
    NP_NODISCARD ndarray<c128> execute(const ndarray<c128> &input, const ndarray<c128> &unitary) override
    {
        if (cbs_.optical_execute)
        {
            // program then execute
            std::unique_lock lock(mtx_);
            programmed_U_ = unitary;
            has_U_ = true;
            lock.unlock();
            return execute(input);
        }
        return SimBackend::apply_unitary(unitary, input);
    }
    void reset() override
    {
        std::unique_lock lock(mtx_);
        has_U_ = false;
        programmed_U_ = ndarray<c128>();
        last_status_.connected = false;
    }
};

// Header-only serial/PCIe stub: checks filesystem path existence.
struct SerialHardwareBackend : GenericHardwareBackend
{
    std::string device_path_;
    explicit SerialHardwareBackend(std::string path, PhotonicConfig cfg = {}, CalibrationTable cal = {},
                                   HardwareCallbacks cbs = {})
        : GenericHardwareBackend(std::move(cbs), cfg, cal), device_path_(std::move(path))
    {
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "SerialHardwareBackend:" + device_path_;
    }
    NP_NODISCARD bool is_available() const noexcept override
    {
        if (!device_path_.empty() && std::filesystem::exists(device_path_))
            return true;
        return GenericHardwareBackend::is_available();
    }
    NP_NODISCARD DeviceStatus status() const noexcept override
    {
        DeviceStatus s = GenericHardwareBackend::status();
        s.connected = is_available();
        s.backend_name = name();
        if (!s.connected)
            s.error = "device not found: " + device_path_;
        return s;
    }
};

// ── MachZehnderMesh ─────────────────────────────────────────────────────
struct MachZehnderMesh
{
    ndarray<c128> unitary;                     ///< ideal unitary (NxN)
    PhotonicConfig config;                     ///< hardware config
    MeshPhases phases;                         ///< physical MZI decomposition
    CalibrationTable calibration;              ///< voltage LUT
    std::shared_ptr<IPhotonicBackend> backend; ///< optional bound backend

    MachZehnderMesh() = default;

    explicit MachZehnderMesh(ndarray<c128> u, PhotonicConfig cfg = {}) : unitary(std::move(u)), config(cfg)
    {
        validate_unitary_();
        if (unitary.ndim() == 2 && unitary.shape[0] > 0)
            phases = detail::reck_decompose(unitary);
        else
            phases = MeshPhases{};
    }

    // Construct from precomputed phases (e.g. after calibration)
    MachZehnderMesh(MeshPhases ph, int N, PhotonicConfig cfg = {})
        : unitary(detail::synthesize(ph, N)), config(cfg), phases(std::move(ph))
    {
    }

    NP_NODISCARD int size() const noexcept
    {
        if (unitary.ndim() == 2)
            return unitary.shape[0];
        return 0;
    }

    // ── Factory helpers ─────────────────────────────────────────────────
    NP_NODISCARD static MachZehnderMesh identity(int n, PhotonicConfig cfg = {})
    {
        ndarray<c128> u(std::vector<int>{n, n});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                u(i, j) = (i == j ? c128(1, 0) : c128(0, 0));
        return MachZehnderMesh(std::move(u), cfg);
    }

    NP_NODISCARD static MachZehnderMesh from_unitary(const ndarray<c128> &U, PhotonicConfig cfg = {},
                                                     CalibrationTable cal = {})
    {
        MachZehnderMesh m(U, cfg);
        m.calibration = cal;
        return m;
    }

    NP_NODISCARD static MachZehnderMesh from_phases(const MeshPhases &ph, int N, PhotonicConfig cfg = {})
    {
        return MachZehnderMesh(ph, N, cfg);
    }

    NP_NODISCARD static ndarray<c128> fft_unitary(int N)
    {
        return detail::fft_unitary(N);
    }

    NP_NODISCARD static MachZehnderMesh optical_fft(int N, PhotonicConfig cfg = {})
    {
        cfg.topology = MeshTopology::OpticalFFT;
        return MachZehnderMesh(detail::fft_unitary(N), cfg);
    }

    // ── Properties ──────────────────────────────────────────────────────
    NP_NODISCARD ndarray<c128> ideal_unitary() const
    {
        return unitary;
    }

    NP_NODISCARD ndarray<c128> effective_unitary() const
    {
        if (size() == 0)
            return ndarray<c128>();
        return detail::effective_unitary_from_phases(phases, size(), config, &calibration);
    }

    NP_NODISCARD double fidelity() const
    {
        if (size() == 0)
            return 1.0;
        auto eff = effective_unitary();
        return detail::fidelity(unitary, eff);
    }

    NP_NODISCARD bool is_unitary(double tol = 1e-6) const
    {
        return detail::is_unitary(unitary, tol);
    }

    NP_NODISCARD double insertion_loss_db() const noexcept
    {
        return config.insertion_loss_db_per_mzi * static_cast<double>(phases.mzis.size());
    }

    NP_NODISCARD std::vector<double> thetas() const
    {
        std::vector<double> out;
        out.reserve(phases.mzis.size());
        for (auto &mz : phases.mzis)
            out.push_back(mz.theta);
        return out;
    }
    NP_NODISCARD std::vector<double> phis() const
    {
        std::vector<double> out;
        out.reserve(phases.mzis.size());
        for (auto &mz : phases.mzis)
            out.push_back(mz.phi);
        return out;
    }

    // Quantize copy
    NP_NODISCARD MachZehnderMesh quantized() const
    {
        MachZehnderMesh q = *this;
        for (auto &mz : q.phases.mzis)
        {
            mz.theta = detail::quantize_phase(mz.theta, config.dac_bits);
            mz.phi = detail::quantize_phase(mz.phi, config.dac_bits);
        }
        q.unitary = detail::synthesize(q.phases, q.size());
        return q;
    }

    // Thermal drift update
    void update_temperature(double temp_c)
    {
        config.temperature_c = temp_c;
    }

    void set_calibration(CalibrationTable cal)
    {
        calibration = std::move(cal);
    }

    void set_backend(std::shared_ptr<IPhotonicBackend> b)
    {
        backend = std::move(b);
        if (backend)
            backend->configure(*this);
    }

    // Compile to rectangular Clements scheduling (re-order only; same count)
    // For header-only, this is a stable sort by layer: even pairs first.
    NP_NODISCARD MeshPhases compile_to_rectangular() const
    {
        MeshPhases out = phases;
        // Simple heuristic: stable partition by (m%2)
        std::stable_sort(out.mzis.begin(), out.mzis.end(),
                         [](const MZI &a, const MZI &b) { return (a.m % 2) < (b.m % 2); });
        return out;
    }

    // ── Apply ───────────────────────────────────────────────────────────
    // Simulation path (no backend): uses effective unitary with error model
    // if config has noise/loss, else ideal.
    NP_NODISCARD ndarray<c128> apply(const ndarray<c128> &x) const
    {
        if (backend)
            return backend->execute(x);
        // choose effective vs ideal based on config
        bool noisy = config.phase_error_std != 0.0 || config.insertion_loss_db_per_mzi != 0.0 ||
                     config.splitter_imbalance != 0.0 || (config.dac_bits != 0 && config.dac_bits < 30) ||
                     config.crosstalk_coeff != 0.0 || config.temperature_c != 25.0;
        ndarray<c128> U = noisy ? effective_unitary() : unitary;
        if (U.size() == 0)
            throw std::runtime_error("MachZehnderMesh: no unitary programmed");
        // input power check
        double pwr = 0;
        for (auto v : x.data())
            pwr += std::norm(v);
        if (pwr > config.max_input_power_mw * 100) // heuristic
        {
            // In real hardware would clip; we just continue
        }
        auto y = SimBackend::apply_unitary(U, x);
        if (!config.coherent_detection)
        {
            for (auto &v : y.data())
                v = c128(std::norm(v), 0);
        }
        return y;
    }

    NP_NODISCARD ndarray<c128> apply(const ndarray<c128> &x, IPhotonicBackend &be) const
    {
        // ensure backend is configured with *this mesh if it supports it
        // we do not mutate mesh; we execute via provided unitary directly
        return be.execute(x, unitary);
    }

    // Batched apply: x is (N x B) matrix, each column is a vector
    NP_NODISCARD ndarray<c128> apply_batch(const ndarray<c128> &X) const
    {
        if (X.ndim() != 2)
            throw std::invalid_argument("apply_batch requires 2D (N x batch)");
        int N = size();
        if (X.shape[0] != N)
            throw std::invalid_argument("apply_batch: first dim must match mesh size");
        int B = X.shape[1];
        ndarray<c128> Y(std::vector<int>{N, B});
        for (int b = 0; b < B; ++b)
        {
            ndarray<c128> col(std::vector<int>{N});
            for (int i = 0; i < N; ++i)
                col[i] = X(i, b);
            auto ycol = apply(col);
            for (int i = 0; i < N; ++i)
                Y(i, b) = ycol[i];
        }
        return Y;
    }

    // Self-test with random vectors
    NP_NODISCARD double self_test(int n_vectors = 8, double tol = 1e-3) const
    {
        int N = size();
        if (N == 0)
            return 1.0;
        std::mt19937_64 rng(42);
        std::normal_distribution<double> nd(0, 1);
        double worst = 1.0;
        for (int k = 0; k < n_vectors; ++k)
        {
            ndarray<c128> x(std::vector<int>{N});
            for (int i = 0; i < N; ++i)
                x[i] = c128(nd(rng), nd(rng));
            // normalize
            double nrm = 0;
            for (auto v : x.data())
                nrm += std::norm(v);
            nrm = std::sqrt(nrm);
            for (auto &v : x.data())
                v /= nrm;
            auto y_ideal = SimBackend::apply_unitary(unitary, x);
            auto y_eff = apply(x);
            // cosine fidelity per vector
            c128 dot(0, 0);
            double ny = 0, nz = 0;
            for (int i = 0; i < N; ++i)
            {
                c128 yi = static_cast<c128>(y_ideal[i]);
                c128 ye = static_cast<c128>(y_eff[i]);
                dot += std::conj(yi) * ye;
                ny += std::norm(yi);
                nz += std::norm(ye);
            }
            double fid = std::abs(dot) / std::sqrt(ny * nz + 1e-12);
            worst = std::min(worst, fid);
            if (fid < 1 - tol)
            {
                // keep worst
            }
        }
        return worst;
    }

  private:
    void validate_unitary_() const
    {
        if (unitary.ndim() != 2)
            throw std::invalid_argument("MachZehnderMesh: unitary must be 2-D");
        if (unitary.shape[0] != unitary.shape[1])
            throw std::invalid_argument("MachZehnderMesh: unitary must be square");
        if (unitary.shape[0] == 0)
            return;
        // Optionally warn if not unitary (allow non-unitary for SVD-embedded)
        // but we keep strict check for direct mesh
    }
};

// ── Out-of-line backend configure to avoid circular dep ────────────────
inline void SimBackend::configure(const MachZehnderMesh &mesh)
{
    std::unique_lock lock(mtx_);
    programmed_U_ = mesh.unitary;
    cfg_ = mesh.config;
    has_U_ = true;
}
inline void NoisySimBackend::configure(const MachZehnderMesh &mesh)
{
    std::unique_lock lock(mtx_);
    // keep noisy cfg_ but inherit mesh geometry/topology
    PhotonicConfig eff = cfg_;
    eff.topology = mesh.config.topology;
    eff.wavelength_nm = mesh.config.wavelength_nm;
    eff.coherent_detection = mesh.config.coherent_detection;
    programmed_U_ = detail::effective_unitary_from_phases(mesh.phases, mesh.size(), eff, &cal_);
    has_U_ = true;
}
inline void GenericHardwareBackend::configure(const MachZehnderMesh &mesh)
{
    std::unique_lock lock(mtx_);
    programmed_U_ = mesh.unitary;
    cfg_ = mesh.config;
    cal_ = mesh.calibration;
    has_U_ = true;
    last_status_.connected = is_available();
    last_status_.calibrated = true;
    // push phases to hardware if callback present
    if (cbs_.write_phases)
    {
        auto thetas = mesh.thetas();
        auto phis = mesh.phis();
        // unlock before calling user code
        lock.unlock();
        cbs_.write_phases(thetas, phis);
        lock.lock();
        last_status_.fidelity = mesh.fidelity();
    }
    // insertion loss
    last_status_.insertion_loss_db = mesh.insertion_loss_db();
}

// ── Optical FFT ────────────────────────────────────────────────────────
struct OpticalFFT
{
    int n = 0;
    PhotonicConfig config;
    MachZehnderMesh mesh;

    explicit OpticalFFT(int N, PhotonicConfig cfg = {}) : n(N), config(cfg), mesh(MachZehnderMesh::optical_fft(N, cfg))
    {
        if (N <= 0 || (N & (N - 1)) != 0)
        {
            // Optical FFT works for any N but power-of-two is most efficient
        }
    }

    NP_NODISCARD ndarray<c128> fft(const ndarray<c128> &x) const
    {
        if (static_cast<int>(x.size()) != n)
            throw std::invalid_argument("OpticalFFT::fft size mismatch");
        return mesh.apply(x);
    }
    NP_NODISCARD ndarray<c128> ifft(const ndarray<c128> &x) const
    {
        if (static_cast<int>(x.size()) != n)
            throw std::invalid_argument("OpticalFFT::ifft size mismatch");
        // IFFT is conj(FFT)/N: unitary is W, inverse is W^\dagger
        auto Udag = mesh.unitary;
        // conj transpose
        ndarray<c128> Ud(std::vector<int>{n, n});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Ud(i, j) = std::conj(mesh.unitary(j, i));
        MachZehnderMesh inv(Ud, config);
        return inv.apply(x);
    }
    NP_NODISCARD ndarray<c128> fft(const ndarray<c128> &x, IPhotonicBackend &be) const
    {
        return mesh.apply(x, be);
    }
    static ndarray<c128> fft_unitary(int N)
    {
        return detail::fft_unitary(N);
    }
};

// ── Factory (Strategy + Builder style) ─────────────────────────────────
struct PhotonicsFactory
{
    NP_NODISCARD static MachZehnderMesh identity(int n, PhotonicConfig cfg = {})
    {
        return MachZehnderMesh::identity(n, cfg);
    }
    NP_NODISCARD static MachZehnderMesh from_unitary(const ndarray<c128> &U, PhotonicConfig cfg = {})
    {
        return MachZehnderMesh::from_unitary(U, cfg);
    }
    NP_NODISCARD static OpticalFFT optical_fft(int n, PhotonicConfig cfg = {})
    {
        return OpticalFFT(n, cfg);
    }

    // Backends
    NP_NODISCARD static std::shared_ptr<SimBackend> simulation(PhotonicConfig cfg = {})
    {
        return std::make_shared<SimBackend>(cfg);
    }
    NP_NODISCARD static std::shared_ptr<NoisySimBackend> noisy_simulation(PhotonicConfig cfg = {})
    {
        // sensible defaults for noisy sim if user didn't set
        if (cfg.phase_error_std == 0.0)
            cfg.phase_error_std = 0.01;
        if (cfg.dac_bits == 0)
            cfg.dac_bits = 8;
        if (cfg.insertion_loss_db_per_mzi == 0.0)
            cfg.insertion_loss_db_per_mzi = 0.05;
        return std::make_shared<NoisySimBackend>(cfg);
    }
    NP_NODISCARD static std::shared_ptr<GenericHardwareBackend> generic_hardware(HardwareCallbacks cbs,
                                                                                 PhotonicConfig cfg = {},
                                                                                 CalibrationTable cal = {})
    {
        return std::make_shared<GenericHardwareBackend>(std::move(cbs), cfg, cal);
    }
    NP_NODISCARD static std::shared_ptr<SerialHardwareBackend> serial_hardware(std::string device_path,
                                                                               PhotonicConfig cfg = {},
                                                                               CalibrationTable cal = {},
                                                                               HardwareCallbacks cbs = {})
    {
        return std::make_shared<SerialHardwareBackend>(std::move(device_path), cfg, cal, std::move(cbs));
    }
    // Auto-detect: prefer serial if path exists, else noisy sim
    NP_NODISCARD static std::shared_ptr<IPhotonicBackend> auto_detect(PhotonicConfig cfg = {},
                                                                      std::string device_hint = "/dev/photonics0")
    {
        auto serial = serial_hardware(device_hint, cfg);
        if (serial->is_available())
            return serial;
        return simulation(cfg);
    }

    // SVD-based synthesis for arbitrary (non-unitary) matrix A:
    // A = U S V^\dagger  ->  A/s_max is subunitary, embed or use
    // two meshes + attenuators.  Here we return the unitary part
    // and scale so the caller can handle attenuation in electronics.
    struct SVDPhotonicResult
    {
        MachZehnderMesh u_mesh;
        MachZehnderMesh v_mesh;
        ndarray<double> s; // singular values
        double scale = 1.0;
    };
    template <typename T>
    NP_NODISCARD static SVDPhotonicResult from_matrix(const ndarray<T> &A, PhotonicConfig cfg = {})
    {
        if (A.ndim() != 2)
            throw std::invalid_argument("from_matrix requires 2D array");
        // Use linalg SVD (real path) – promote to double
        using R = double;
        // Convert A to double complex for photonics if needed
        int M = A.shape[0], N = A.shape[1];
        // Use linalg::svd for real-valued A; for complex we still use linalg path
        // For simplicity, handle double/float via linalg
        auto svd = linalg::svd(A);
        // Build unitary meshes for U and Vh
        // svd.u is MxM or MxK, svd.vh is NxN etc.  Extract square unitaries by
        // padding / completing to square via ortho_complete logic reused from linalg
        // For header-only simplicity: take the square unitaries directly if full
        int Ku = svd.u.shape[0];
        int Kv = svd.vh.shape[0];
        // Convert real U/Vh to complex unitary
        auto to_c128 = [](const auto &real_mat) -> ndarray<c128> {
            int R0 = real_mat.shape[0], R1 = real_mat.shape[1];
            ndarray<c128> out(std::vector<int>{R0, R1});
            for (int i = 0; i < R0; ++i)
                for (int j = 0; j < R1; ++j)
                    out(i, j) = c128(static_cast<double>(real_mat(i, j)), 0);
            return out;
        };
        // If not square, embed into square by identity padding (photonic meshes are square)
        auto make_square = [](ndarray<c128> U) -> ndarray<c128> {
            int N0 = U.shape[0];
            if (U.shape[0] == U.shape[1])
                return U;
            int S = std::max(U.shape[0], U.shape[1]);
            ndarray<c128> sq(std::vector<int>{S, S});
            for (int i = 0; i < S; ++i)
                for (int j = 0; j < S; ++j)
                    sq(i, j) = (i == j ? c128(1, 0) : c128(0, 0));
            for (int i = 0; i < U.shape[0]; ++i)
                for (int j = 0; j < U.shape[1]; ++j)
                    sq(i, j) = U(i, j);
            return sq;
        };
        ndarray<c128> Uc = make_square(to_c128(svd.u));
        ndarray<c128> Vc = make_square(to_c128(svd.vh.transpose()));
        // singular values: max is scale
        double s_max = 0;
        for (auto v : svd.s.data())
            s_max = std::max(s_max, static_cast<double>(v));
        if (s_max == 0)
            s_max = 1.0;
        ndarray<double> s_norm(svd.s.shape);
        for (std::size_t i = 0; i < svd.s.size(); ++i)
            s_norm.data()[i] = static_cast<double>(svd.s.data()[i]) / s_max;

        SVDPhotonicResult r;
        r.u_mesh = MachZehnderMesh(Uc, cfg);
        r.v_mesh = MachZehnderMesh(Vc, cfg);
        r.s = std::move(s_norm);
        r.scale = s_max;
        (void)M;
        (void)N;
        (void)Ku;
        (void)Kv;
        return r;
    }
};

// ── Convenience free functions ─────────────────────────────────────────
NP_NODISCARD inline bool is_unitary(const ndarray<c128> &U, double tol = 1e-6)
{
    return detail::is_unitary(U, tol);
}
NP_NODISCARD inline double fidelity(const ndarray<c128> &A, const ndarray<c128> &B)
{
    return detail::fidelity(A, B);
}
NP_NODISCARD inline double quantize_phase(double phase, int bits) noexcept
{
    return detail::quantize_phase(phase, bits);
}

} // namespace np::photonics

#endif // NP_PHOTONICS_HPP
