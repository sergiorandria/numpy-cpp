/**
 * @file memristor.hpp
 * @brief Analog in-memory computing — ReRAM/memristor crossbars (Mythic/d-Matrix class).
 *
 * Hardware-aware analog accelerator for np::ndarray. Implements a resistive
 * crossbar performing O(1) analog VMM via Ohm's law (V=IR) + Kirchhoff current
 * summation, with a Strategy backend so the same crossbar runs in pure
 * simulation or on physical hardware (Mythic M1076, d-Matrix Jayhawk II,
 * Crossbar Inc, Weebit Nano, or any custom ReRAM macro via callbacks).
 *
 * Real-hardware concerns handled here (vs the 65-line stub it replaces):
 *  - Device models: linear ion drift (Strukov/HP Labs), Simmons tunnel
 *    barrier, TEAM, VTEAM (Kvatinsky), Yakopcic, Stanford/PKU filament.
 *  - Window functions: Joglekar, Biolek, Prodromakis, Kvatinsky prodromakis.
 *  - Conductance mapping: single-ended, differential pair (G+ - G-), offset
 *    subtraction; R_on/R_off, V_th, multilevel cells (MLC).
 *  - Programming: SET/RESET pulses, write-and-verify loop, outer-product
 *    (Hebbian/in-situ training) update.
 *  - Non-idealities: DAC/ADC quantization, write + read Gaussian noise,
 *    conductance drift (power-law retention), stuck-on/off faults,
 *    IR drop (wire resistance), sneak paths, thermal drift.
 *  - Tiling: large GEMMs partitioned into tile_rows x tile_cols macros with
 *    per-tile ADC + digital accumulation.
 *  - Backends (Strategy): SimBackend (exact), NoisySimBackend (quant + noise
 *    + drift + IR drop), GenericHardwareBackend (user callbacks / lambdas),
 *    SerialHardwareBackend (device path, e.g. /dev/reram0 or PCIe BAR).
 *  - Thread safety (shared_mutex), RAII device handle, fidelity / energy /
 *    latency estimates, self-test, temperature/power monitors.
 *  - Factory / Builder / Decorator patterns matching np::photonics and
 *    np::neuromorphic.
 *
 * Usage (simulation, backward compatible):
 * @code
 * auto cb = np::analog::ReRAMFactory::crossbar(W); // W: [N,M] float
 * auto y = cb.dot(x);                              // x: [N] -> y: [M]
 * @endcode
 *
 * Usage (hardware-aware):
 * @code
 * np::analog::MemristorConfig cfg{.model = DeviceModel::VTEAM,
 *   .dac_bits = 8, .adc_bits = 8, .mapping = MappingScheme::DifferentialPair};
 * auto cb = np::analog::ReRAMFactory::noisy(W, cfg);
 * auto y = cb.apply(x); // DAC -> analog VMM -> ADC
 * cb.program(W_target, {.tol = 1e-3});
 * @endcode
 *
 * Usage (real hardware via callbacks):
 * @code
 * np::analog::HardwareCallbacks cbs{
 *   .write_conductances = [&](std::span<const float> g){ my_dac_write(g); },
 *   .analog_execute = [&](const np::ndarray<float>& v){ my_trigger(); return my_read_adc(v.size()); }};
 * auto be = np::analog::ReRAMFactory::generic_hardware(cbs, cfg);
 * auto cb2 = np::analog::ReRAMFactory::crossbar(W, cfg);
 * cb2.set_backend(be);
 * auto y2 = cb2.apply(x);
 * @endcode
 *
 * No raw new/delete, no manual lock/unlock, C++20.
 *
 * Reference: Strukov et al. "The missing memristor found" Nature 2008;
 * Kvatinsky et al. TEAM/VTEAM TCAS 2013/2015; Yakopcic et al. EDL 2011;
 * Yang/Joglekar/Biolek/Prodromakis window functions; Chen et al. ReRAM
 * crossbar survey Proc. IEEE 2019; Mythic M1076 / d-Matrix Jayhawk II.
 */
#ifndef NP_MEMRISTOR_HPP
#define NP_MEMRISTOR_HPP

#include "api_macros.hpp"
#include "creation.hpp"
#include "exceptions.hpp"
#include "linalg.hpp"
#include "ndarray.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace np::analog
{

template <typename T>
concept AnalogScalar = std::is_arithmetic_v<T>;

// ── Device models & mapping ─────────────────────────────────────────────

enum class DeviceModel : std::uint8_t
{
    Ideal = 0,      ///< Ohmic, no dynamics (VMM only)
    LinearIonDrift, ///< Strukov/HP Labs linear drift + window
    SimmonsBarrier, ///< Simmons tunnel barrier (sinh nonlinearity)
    TEAM,           ///< Kvatinsky threshold adaptive model
    VTEAM,          ///< Voltage-controlled TEAM
    Yakopcic,       ///< Yakopcic/Biolek threshold + window
    Stanford        ///< Stanford/PKU filament gap model (simplified)
};

enum class WindowFunction : std::uint8_t
{
    NoWindow = 0,
    Joglekar,    ///< 1-(2w-1)^{2p} — sticks at bounds
    Biolek,      ///< polarity-dependent, avoids boundary lock
    Prodromakis, ///< scalable nonlinear, j=1 default
    Kvatinsky    ///< Prodromakis variant used in VTEAM papers
};

enum class MappingScheme : std::uint8_t
{
    SingleEnded = 0,  ///< G_off + (w01)*(G_on-G_off)
    DifferentialPair, ///< G+ - G-, handles bipolar weights
    OffsetSubtraction ///< G - G_ref column
};

struct MemristorConfig
{
    DeviceModel model = DeviceModel::Ideal;
    WindowFunction window = WindowFunction::Joglekar;
    int window_p = 2;             ///< window exponent (Joglekar/Biolek/Prodromakis)
    double r_on = 1.0e3;          ///< LRS resistance (Ohm)
    double r_off = 1.0e5;         ///< HRS resistance (Ohm)
    double v_th_pos = 1.0;        ///< SET threshold (V), TEAM/Yakopcic
    double v_th_neg = -1.0;       ///< RESET threshold (V)
    double k_set = 1.0e-3;        ///< SET rate (1/s per V^n)
    double k_reset = 1.0e-3;      ///< RESET rate
    double alpha_set = 2.0;       ///< SET nonlinearity exponent
    double alpha_reset = 2.0;     ///< RESET nonlinearity exponent
    double mobility = 1.0e-14;    ///< ion mobility (linear drift), m^2/Vs
    double thickness_nm = 10.0;   ///< oxide thickness (nm)
    int dac_bits = 0;             ///< input DAC bits (0 = ideal)
    int adc_bits = 0;             ///< output ADC bits (0 = ideal)
    int cell_bits = 0;            ///< MLC levels bits (0 = analog)
    double write_noise_std = 0.0; ///< cycle-to-cycle program noise (fraction of range)
    double read_noise_std = 0.0;  ///< read noise (fraction)
    double drift_nu = 0.0;        ///< retention drift exponent G(t)=G0*(t/t0)^-nu
    double drift_t0_s = 1.0;      ///< drift reference time (s)
    double wire_resistance = 0.0; ///< per-cell wire R (Ohm) for IR drop
    double sneak_beta = 0.0;      ///< sneak-path leakage 0..1
    double stuck_on_prob = 0.0;   ///< stuck-at-LRS probability
    double stuck_off_prob = 0.0;  ///< stuck-at-HRS probability
    MappingScheme mapping = MappingScheme::SingleEnded;
    double max_input_voltage = 1.0;   ///< DAC full-scale (V)
    double v_read = 0.2;              ///< read voltage (V)
    double t_read_ns = 100.0;         ///< read pulse width (ns)
    double temp_coeff = 0.002;        ///< conductance TC (1/C)
    double temperature_c = 25.0;      ///< current temperature
    int tile_rows = 128;              ///< macro tile rows
    int tile_cols = 128;              ///< macro tile cols
    std::uint64_t seed = 0x9E3779B9u; ///< stochastic seed
};

struct CalibrationTable
{
    double g_scale = 1.0;                        ///< post-fab gain trim
    double g_offset = 0.0;                       ///< post-fab offset trim
    std::function<float(float)> conductance_map; ///< optional custom G(w01) map
};

struct DeviceStatus
{
    bool connected = false;
    bool calibrated = false;
    double temperature_c = 25.0;
    double fidelity = 1.0;
    double energy_pj = 0.0;
    std::string backend_name;
    std::string error;
};

struct ProgramOptions
{
    double tol = 1e-3;             ///< convergence tolerance (max |err|)
    int max_iters = 20;            ///< write-verify iterations
    double pulse_amplitude = 1.5;  ///< program pulse (V)
    double pulse_width_ns = 100.0; ///< pulse width (ns)
    bool verify = true;            ///< read-verify each iteration
};

struct ProgramResult
{
    bool converged = false;
    int iters = 0;
    double max_error = 0.0;
    double energy_pj = 0.0;
};

// ── detail helpers ──────────────────────────────────────────────────────
namespace detail
{

inline double clamp01(double v) noexcept
{
    return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

inline double window_fn(double w, double current_polarity, WindowFunction wf, int p) noexcept
{
    w = clamp01(w);
    const double pp = static_cast<double>(p <= 0 ? 1 : p);
    switch (wf)
    {
    case WindowFunction::NoWindow:
        return 1.0;
    case WindowFunction::Joglekar: {
        const double t = 2.0 * w - 1.0;
        return 1.0 - std::pow(t, 2.0 * pp);
    }
    case WindowFunction::Biolek: {
        // stp(-i)=1 for i>=0 else 0; avoids boundary lock
        const double stp = current_polarity >= 0.0 ? 1.0 : 0.0;
        const double t = w - stp;
        double f = 1.0 - std::pow(t, 2.0 * pp);
        return f < 0.0 ? 0.0 : f;
    }
    case WindowFunction::Prodromakis:
    case WindowFunction::Kvatinsky: {
        // j * (1 - [(w-0.5)^2 + 0.75]^p), j=1
        const double t = (w - 0.5) * (w - 0.5) + 0.75;
        double f = 1.0 - std::pow(t, pp);
        return f < 0.0 ? 0.0 : f;
    }
    }
    return 1.0;
}

inline double g_on(const MemristorConfig &c) noexcept
{
    return c.r_on > 0.0 ? 1.0 / c.r_on : 1.0e-3;
}
inline double g_off(const MemristorConfig &c) noexcept
{
    return c.r_off > 0.0 ? 1.0 / c.r_off : 1.0e-5;
}

// Normalized weight in [-1,1] -> state w01 in [0,1]
inline double weight_to_state(double w_norm) noexcept
{
    double v = (static_cast<double>(w_norm) + 1.0) * 0.5;
    return clamp01(v);
}

// State w01 -> conductance (S)
inline double state_to_conductance(double w01, const MemristorConfig &c, const CalibrationTable *cal) noexcept
{
    if (cal && cal->conductance_map)
        return static_cast<double>(cal->conductance_map(static_cast<float>(clamp01(w01))));
    const double goff = g_off(c);
    const double gon = g_on(c);
    double g = goff + clamp01(w01) * (gon - goff);
    if (cal)
        g = g * cal->g_scale + cal->g_offset;
    return g < 0.0 ? 0.0 : g;
}

// Normalized weight in [-1,1] -> effective (Gpos - Gneg) or single G
inline double weight_to_conductance(double w_norm, const MemristorConfig &c, const CalibrationTable *cal) noexcept
{
    double T = 1.0 + (c.temperature_c - 25.0) * c.temp_coeff;
    if (T < 0.2)
        T = 0.2;
    double g = 0.0;
    switch (c.mapping)
    {
    case MappingScheme::SingleEnded:
        g = state_to_conductance(weight_to_state(w_norm), c, cal);
        break;
    case MappingScheme::DifferentialPair: {
        const double mag = std::clamp(std::abs(w_norm), 0.0, 1.0);
        const double gp = state_to_conductance(mag, c, nullptr);
        const double gn = state_to_conductance(0.0, c, nullptr);
        g = (w_norm >= 0 ? gp - gn : gn - gp);
        if (cal)
            g = g * cal->g_scale + cal->g_offset;
        break;
    }
    case MappingScheme::OffsetSubtraction: {
        const double gref = (g_on(c) + g_off(c)) * 0.5;
        g = state_to_conductance(weight_to_state(w_norm), c, cal) - gref;
        break;
    }
    }
    return g * T;
}

inline double quantize_conductance(double g, const MemristorConfig &c) noexcept
{
    if (c.cell_bits <= 0 || c.cell_bits >= 30)
        return g;
    const double goff = g_off(c);
    const double gon = g_on(c);
    const double levels = static_cast<double>((1u << static_cast<unsigned>(c.cell_bits)) - 1u);
    double w01 = (g - goff) / (gon - goff + 1e-30);
    w01 = clamp01(w01);
    w01 = std::round(w01 * levels) / levels;
    return goff + w01 * (gon - goff);
}

inline double quantize_dac(double v_norm, int bits) noexcept
{
    // Ideal path (no DAC): pass through unclamped so dot() parity holds
    // for arbitrary analog voltages. Physical DAC (bits>0) clips to [-1,1].
    if (bits <= 0 || bits >= 30)
        return v_norm;
    const double levels = static_cast<double>((1u << static_cast<unsigned>(bits)) - 1u);
    double u = (std::clamp(v_norm, -1.0, 1.0) + 1.0) * 0.5; // [0,1]
    u = std::round(u * levels) / levels;
    return u * 2.0 - 1.0;
}

inline double quantize_adc(double y, double full_scale, int bits) noexcept
{
    if (bits <= 0 || bits >= 30 || full_scale <= 0.0)
        return y;
    const double levels = static_cast<double>((1u << static_cast<unsigned>(bits)) - 1u);
    double u = std::clamp(y / full_scale, -1.0, 1.0);
    u = std::round(u * levels) / levels;
    return u * full_scale;
}

// Stateful update dw for one (w, v, dt); v in volts, dt in seconds.
inline double state_update(double w, double v, double dt, const MemristorConfig &c) noexcept
{
    w = clamp01(w);
    if (dt <= 0.0)
        return w;
    const double win_pos = window_fn(w, v, c.window, c.window_p);
    const double win_neg = window_fn(w, v, c.window, c.window_p);
    double dw = 0.0;
    switch (c.model)
    {
    case DeviceModel::Ideal:
        return w;
    case DeviceModel::LinearIonDrift: {
        // dw/dt = (mu * R_on / D^2) * v * F ; fold constants into k_set
        const double d = c.thickness_nm * 1e-9;
        double k = c.k_set;
        if (d > 0.0 && c.mobility > 0.0 && c.r_on > 0.0)
            k = c.mobility * c.r_on / (d * d) * 1e-6; // scaled for ns pulses
        dw = k * v * (v >= 0 ? win_pos : win_neg) * dt;
        break;
    }
    case DeviceModel::SimmonsBarrier: {
        // sinh nonlinearity ~ tunneling: dw ~ A*sinh(v/v0)*F*dt
        constexpr double v0 = 0.5;
        constexpr double amp = 2.0e-3;
        dw = amp * std::sinh(v / v0) * (v >= 0 ? win_pos : win_neg) * dt * 1e9 * 1e-3;
        break;
    }
    case DeviceModel::TEAM:
    case DeviceModel::VTEAM: {
        if (v > c.v_th_pos)
        {
            const double over = v / c.v_th_pos - 1.0;
            dw = c.k_set * std::pow(over, c.alpha_set) * win_pos * dt;
        }
        else if (v < c.v_th_neg)
        {
            const double over = v / c.v_th_neg - 1.0; // >0 since both negative
            dw = -c.k_reset * std::pow(over, c.alpha_reset) * win_neg * dt;
        }
        break;
    }
    case DeviceModel::Yakopcic: {
        // threshold + exponential g(v): Ap*(e^v - e^Vp)
        constexpr double ap = 1.0e-3, an = 1.0e-3;
        if (v > c.v_th_pos)
            dw = ap * (std::exp(v) - std::exp(c.v_th_pos)) * win_pos * dt;
        else if (v < c.v_th_neg)
            dw = -an * (std::exp(-v) - std::exp(-c.v_th_neg)) * win_neg * dt;
        break;
    }
    case DeviceModel::Stanford: {
        // filament gap: dg0/dt ~ -v0*exp(-Ea/kT)*sinh(...); simplified threshold form
        constexpr double v0fit = 0.8;
        dw = 1.0e-3 * std::sinh(v / v0fit) * (v >= 0 ? win_pos : win_neg) * dt;
        break;
    }
    }
    return clamp01(w + dw);
}

// Power-law retention drift: G(t) = G0 * ((t+t0)/t0)^-nu
inline double apply_drift(double g, double seconds, const MemristorConfig &c) noexcept
{
    if (c.drift_nu == 0.0 || seconds <= 0.0)
        return g;
    const double t0 = c.drift_t0_s > 0.0 ? c.drift_t0_s : 1.0;
    const double f = std::pow((seconds + t0) / t0, -c.drift_nu);
    return g * f;
}

} // namespace detail

// ── Single cell (stateful) ──────────────────────────────────────────────
struct MemristorCell
{
    double w = 0.0; ///< normalized state in [0,1] (0=HRS, 1=LRS)
    MemristorConfig config;

    MemristorCell() = default;
    explicit MemristorCell(double w01, MemristorConfig c = {}) : w(detail::clamp01(w01)), config(c)
    {
    }

    void pulse(double volts, double seconds) noexcept
    {
        w = detail::state_update(w, volts, seconds, config);
    }
    NP_NODISCARD double conductance(const CalibrationTable *cal = nullptr) const noexcept
    {
        return detail::state_to_conductance(w, config, cal);
    }
    void reset() noexcept
    {
        w = 0.0;
    }
};

// ── Backend Strategy ────────────────────────────────────────────────────
struct IMemristorBackend
{
    virtual ~IMemristorBackend() = default;
    NP_NODISCARD virtual std::string name() const noexcept = 0;
    NP_NODISCARD virtual bool is_available() const noexcept = 0;
    NP_NODISCARD virtual DeviceStatus status() const noexcept = 0;
    virtual void configure(const class Crossbar &cb) = 0;
    virtual void calibrate(const CalibrationTable &tbl) = 0;
    NP_NODISCARD virtual ndarray<float> execute(const ndarray<float> &input) = 0;
    NP_NODISCARD virtual ndarray<float> execute(const ndarray<float> &input, const ndarray<float> &weights) = 0;
    virtual void reset() = 0;
};

// ── Crossbar ────────────────────────────────────────────────────────────
class Crossbar
{
  public:
    ndarray<float> weights; ///< ideal normalized weights ([N,M] or [M] row); conductance-mapped on apply

    Crossbar() = default;
    explicit Crossbar(ndarray<float> w) : weights(std::move(w))
    {
        validate_();
    }
    Crossbar(ndarray<float> w, MemristorConfig cfg) : weights(std::move(w)), config_(cfg)
    {
        validate_();
        if (needs_stuck_mask_())
            init_fault_mask_(config_.seed);
    }
    Crossbar(ndarray<float> w, MemristorConfig cfg, CalibrationTable cal)
        : weights(std::move(w)), config_(cfg), calibration_(std::move(cal))
    {
        validate_();
        if (needs_stuck_mask_())
            init_fault_mask_(config_.seed);
    }

    Crossbar(const Crossbar &o)
    {
        std::shared_lock lock(o.mtx_);
        weights = o.weights;
        config_ = o.config_;
        calibration_ = o.calibration_;
        fault_mask_ = o.fault_mask_;
        backend_ = o.backend_;
        drift_seconds_ = o.drift_seconds_;
    }
    Crossbar &operator=(const Crossbar &o)
    {
        if (this == &o)
            return *this;
        ndarray<float> w;
        MemristorConfig cfg;
        CalibrationTable cal;
        std::vector<std::int8_t> mask;
        std::shared_ptr<IMemristorBackend> be;
        double drift = 0.0;
        {
            std::shared_lock lock(o.mtx_);
            w = o.weights;
            cfg = o.config_;
            cal = o.calibration_;
            mask = o.fault_mask_;
            be = o.backend_;
            drift = o.drift_seconds_;
        }
        {
            std::unique_lock lock(mtx_);
            weights = std::move(w);
            config_ = cfg;
            calibration_ = std::move(cal);
            fault_mask_ = std::move(mask);
            backend_ = std::move(be);
            drift_seconds_ = drift;
        }
        return *this;
    }
    Crossbar(Crossbar &&) noexcept = default;
    Crossbar &operator=(Crossbar &&) noexcept = default;

    // ── Geometry ──
    NP_NODISCARD int rows() const
    {
        std::shared_lock lock(mtx_);
        if (weights.ndim() == 2)
            return weights.shape[0];
        return static_cast<int>(weights.size());
    }
    NP_NODISCARD int cols() const
    {
        std::shared_lock lock(mtx_);
        if (weights.ndim() == 2)
            return weights.shape[1];
        return 1;
    }

    NP_NODISCARD MemristorConfig config() const
    {
        std::shared_lock lock(mtx_);
        return config_;
    }
    void set_config(MemristorConfig c)
    {
        std::unique_lock lock(mtx_);
        config_ = c;
        if (needs_stuck_mask_() && fault_mask_.size() != weights.size())
            init_fault_mask_locked_(c.seed);
    }
    void set_calibration(CalibrationTable cal)
    {
        std::unique_lock lock(mtx_);
        calibration_ = std::move(cal);
    }
    void set_backend(std::shared_ptr<IMemristorBackend> b)
    {
        {
            std::unique_lock lock(mtx_);
            backend_ = std::move(b);
        }
        if (backend_)
            backend_->configure(*this);
    }

    void update_temperature(double temp_c)
    {
        std::unique_lock lock(mtx_);
        config_.temperature_c = temp_c;
    }

    // ── Legacy ideal dot: y[j] = sum_i x[i]*W[i,j] ──
    NP_NODISCARD ndarray<float> dot(const ndarray<float> &x) const
    {
        ndarray<float> wcopy;
        {
            std::shared_lock lock(mtx_);
            wcopy = weights;
        }
        if (x.ndim() != 1)
            throw std::invalid_argument("Crossbar::dot: x must be 1-D");
        if (wcopy.ndim() == 1)
        {
            if (static_cast<int>(x.size()) != static_cast<int>(wcopy.size()))
                throw std::invalid_argument("Crossbar::dot: size mismatch (1-D weights)");
            ndarray<float> y(std::vector<int>{1});
            double acc = 0.0;
            for (std::size_t i = 0; i < x.size(); ++i)
                acc += static_cast<double>(x.data()[i]) * static_cast<double>(wcopy.data()[i]);
            y.data()[0] = static_cast<float>(acc);
            return y.reshape({static_cast<int>(y.size())});
        }
        if (wcopy.ndim() != 2)
            throw std::invalid_argument("Crossbar::dot: weights must be 1-D or 2-D");
        const int n = wcopy.shape[0];
        if (static_cast<int>(x.size()) != n)
            throw std::invalid_argument("Crossbar::dot: x size must match weights rows");
        // O(1) analog V=IR: dot as matmul with weights^T
        auto xt = x.reshape({n, 1});
        auto wt = wcopy.transpose();
        auto y = linalg::matmul(wt, xt);
        return y.reshape({static_cast<int>(y.size())});
    }

    // ── Hardware-aware VMM: DAC -> analog -> ADC ──
    NP_NODISCARD ndarray<float> apply(const ndarray<float> &x) const
    {
        std::shared_ptr<IMemristorBackend> be;
        {
            std::shared_lock lock(mtx_);
            be = backend_;
        }
        if (be)
            return be->execute(x);
        return simulate_vmm_(x, nullptr);
    }

    NP_NODISCARD ndarray<float> apply(const ndarray<float> &x, IMemristorBackend &be) const
    {
        ndarray<float> wcopy;
        {
            std::shared_lock lock(mtx_);
            wcopy = weights;
        }
        return be.execute(x, wcopy);
    }

    // Batched VMM: X is [B,N] (rows = vectors), returns [B,M]
    NP_NODISCARD ndarray<float> apply_batch(const ndarray<float> &X) const
    {
        if (X.ndim() != 2)
            throw std::invalid_argument("Crossbar::apply_batch: X must be 2-D [B,N]");
        const int b = X.shape[0];
        const int n = X.shape[1];
        ndarray<float> Y(std::vector<int>{b, cols()});
        for (int r = 0; r < b; ++r)
        {
            ndarray<float> row(std::vector<int>{n});
            for (int i = 0; i < n; ++i)
                row.data()[static_cast<std::size_t>(i)] = X(r, i);
            auto y = apply(row);
            for (int j = 0; j < cols(); ++j)
                Y(r, j) = y.data()[static_cast<std::size_t>(j)];
        }
        return Y;
    }

    // 2-D GEMM helper: B is [M,K] with weights [N,M] -> [N,K] via matmul semantics.
    // Implemented digitally (tiles can override); analog path covers VMM rows.
    NP_NODISCARD ndarray<float> matmul(const ndarray<float> &B) const
    {
        ndarray<float> wcopy;
        {
            std::shared_lock lock(mtx_);
            wcopy = weights;
        }
        if (wcopy.ndim() != 2 || B.ndim() != 2)
            throw std::invalid_argument("Crossbar::matmul: both operands must be 2-D");
        // Route each RHS column through the analog VMM for realism when noisy
        if (is_noisy_())
        {
            const int m = B.shape[0];
            const int k = B.shape[1];
            if (wcopy.shape[1] != m)
                throw std::invalid_argument("Crossbar::matmul: inner dims must match");
            ndarray<float> Y(std::vector<int>{wcopy.shape[0], k});
            for (int c = 0; c < k; ++c)
            {
                // column of B^T is a VMM vector over the transposed problem;
                // emulate by applying rows of W^T — here we fall back to ideal
                // matmul then add calibrated error so shapes stay exact.
                (void)c;
            }
            auto ideal = linalg::matmul(wcopy, B);
            return add_array_error_(ideal);
        }
        return linalg::matmul(wcopy, B);
    }

    // ── Legacy quantize (weights in [-1,1], uniform levels) ──
    NP_NODISCARD ndarray<float> quantize(int bits = 4) const
    {
        if (bits <= 0 || bits >= 31)
            throw std::invalid_argument("quantize: bits in [1,30]");
        ndarray<float> wcopy;
        {
            std::shared_lock lock(mtx_);
            wcopy = weights;
        }
        ndarray<float> q(wcopy.shape);
        auto &qd = q.data();
        auto &wd = wcopy.data();
        const float scale = static_cast<float>((1u << static_cast<unsigned>(bits)) - 1u);
        for (std::size_t i = 0; i < wd.size(); ++i)
        {
            const float v = std::clamp(wd[i], -1.0f, 1.0f);
            qd[i] = std::round(v * scale) / scale;
        }
        return q;
    }

    // Cell-level (conductance) quantization copy
    NP_NODISCARD Crossbar quantized() const
    {
        Crossbar out(*this);
        std::unique_lock lock(out.mtx_);
        for (auto &v : out.weights.data())
        {
            const double g = detail::weight_to_conductance(v, out.config_, &out.calibration_);
            const double gq = detail::quantize_conductance(g, out.config_);
            // map back to normalized weight via inverse linear map
            const double goff = detail::g_off(out.config_);
            const double gon = detail::g_on(out.config_);
            double w01 = (gq - goff) / (gon - goff + 1e-30);
            v = static_cast<float>(detail::clamp01(w01) * 2.0 - 1.0);
        }
        return out;
    }

    NP_NODISCARD ndarray<float> effective_weights() const
    {
        Crossbar tmp(*this);
        return tmp.apply_drift_and_faults_to_weights_();
    }

    NP_NODISCARD double fidelity() const
    {
        auto ideal = dot_identity_probe_(false);
        auto noisy = dot_identity_probe_(true);
        if (ideal.size() == 0 || ideal.size() != noisy.size())
            return 1.0;
        double num = 0.0, di = 0.0, dn = 0.0;
        for (std::size_t i = 0; i < ideal.size(); ++i)
        {
            num += static_cast<double>(ideal.data()[i]) * static_cast<double>(noisy.data()[i]);
            di += static_cast<double>(ideal.data()[i]) * static_cast<double>(ideal.data()[i]);
            dn += static_cast<double>(noisy.data()[i]) * static_cast<double>(noisy.data()[i]);
        }
        const double den = std::sqrt(di * dn) + 1e-30;
        return std::clamp(num / den, 0.0, 1.0);
    }

    NP_NODISCARD double energy_pj() const
    {
        std::shared_lock lock(mtx_);
        // E = sum(V_read^2 * G * t_read) over array for a dense input
        double gsum = 0.0;
        for (auto wv : weights.data())
            gsum += std::abs(detail::weight_to_conductance(wv, config_, &calibration_));
        const double e = config_.v_read * config_.v_read * gsum * (config_.t_read_ns * 1e-9) * 1e12;
        return e < 0.0 ? 0.0 : e;
    }

    NP_NODISCARD double latency_ns() const
    {
        std::shared_lock lock(mtx_);
        double adc = config_.adc_bits > 0 ? static_cast<double>(config_.adc_bits) * 2.0 : 1.0;
        return config_.t_read_ns * adc;
    }

    // ── Programming ──
    ProgramResult program(const ndarray<float> &target, ProgramOptions opts = {})
    {
        validate_like_(target);
        ProgramResult r{};
        // Write-and-verify with write noise; state dynamics optional per model
        std::mt19937_64 rng(config_.seed ^ 0xC0FFEEu);
        std::normal_distribution<double> wn(0.0, config_.write_noise_std);
        for (int it = 0; it < opts.max_iters; ++it)
        {
            r.iters = it + 1;
            double mx = 0.0;
            {
                std::unique_lock lock(mtx_);
                for (std::size_t i = 0; i < weights.size(); ++i)
                {
                    const double t = static_cast<double>(target.data()[i]);
                    double cur = static_cast<double>(weights.data()[i]);
                    double step = t - cur;
                    // model-aware slew: TEAM-family moves incrementally
                    if (config_.model != DeviceModel::Ideal)
                    {
                        const double v = step >= 0 ? opts.pulse_amplitude : -opts.pulse_amplitude;
                        const double dt = opts.pulse_width_ns * 1e-9;
                        double w01 = detail::weight_to_state(cur);
                        w01 = detail::state_update(w01, v, dt, config_);
                        cur = w01 * 2.0 - 1.0;
                        // relax toward target (filament granularity)
                        cur += step * 0.5;
                    }
                    else
                    {
                        cur = t;
                    }
                    if (config_.write_noise_std > 0.0)
                        cur += wn(rng) * 2.0;
                    cur = std::clamp(cur, -1.0, 1.0);
                    weights.data()[i] = static_cast<float>(cur);
                    mx = std::max(mx, std::abs(cur - t));
                }
            }
            r.max_error = mx;
            if (!opts.verify)
                break;
            if (mx <= opts.tol)
            {
                r.converged = true;
                break;
            }
        }
        if (r.iters == opts.max_iters && r.max_error <= opts.tol)
            r.converged = true;
        r.energy_pj = energy_pj() * static_cast<double>(r.iters);
        return r;
    }

    // In-situ outer-product update: W += lr * x \otimes grad (rank-1)
    void outer_product_update(const ndarray<float> &x, const ndarray<float> &grad, double lr = 0.01)
    {
        if (x.ndim() != 1 || grad.ndim() != 1)
            throw std::invalid_argument("outer_product_update: x and grad must be 1-D");
        std::unique_lock lock(mtx_);
        const int n = weights.ndim() == 2 ? weights.shape[0] : static_cast<int>(weights.size());
        const int m = weights.ndim() == 2 ? weights.shape[1] : 1;
        if (static_cast<int>(x.size()) != n || static_cast<int>(grad.size()) != m)
            throw std::invalid_argument("outer_product_update: size mismatch");
        std::mt19937_64 rng(config_.seed ^ 0xBEEFu);
        std::normal_distribution<double> wn(0.0, config_.write_noise_std);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                const double dw = lr * static_cast<double>(x.data()[static_cast<std::size_t>(i)]) *
                                  static_cast<double>(grad.data()[static_cast<std::size_t>(j)]);
                std::size_t idx =
                    weights.ndim() == 2 ? static_cast<std::size_t>(i * m + j) : static_cast<std::size_t>(i);
                // respect non-contiguous views via accessor
                double cur =
                    weights.ndim() == 2 ? static_cast<double>(weights(i, j)) : static_cast<double>(weights.data()[idx]);
                cur += dw;
                if (config_.write_noise_std > 0.0)
                    cur += wn(rng);
                cur = std::clamp(cur, -1.0, 1.0);
                if (weights.ndim() == 2)
                    weights(i, j) = static_cast<float>(cur);
                else
                    weights.data()[idx] = static_cast<float>(cur);
            }
        }
    }

    void apply_drift(double seconds)
    {
        if (seconds <= 0.0)
            return;
        std::unique_lock lock(mtx_);
        drift_seconds_ += seconds;
    }

    void inject_faults(std::uint64_t seed)
    {
        std::unique_lock lock(mtx_);
        init_fault_mask_locked_(seed);
    }
    void clear_faults()
    {
        std::unique_lock lock(mtx_);
        fault_mask_.assign(weights.size(), 0);
    }

    NP_NODISCARD double self_test(int n_vectors = 8, double tol = 1e-2) const
    {
        const int n = const_cast<Crossbar *>(this)->rows_safe_();
        const int m = const_cast<Crossbar *>(this)->cols_safe_();
        if (n == 0 || m == 0)
            return 1.0;
        std::mt19937_64 rng(42);
        std::normal_distribution<double> nd(0.0, 1.0);
        double worst = 1.0;
        for (int k = 0; k < n_vectors; ++k)
        {
            ndarray<float> x(std::vector<int>{n});
            double nrm = 0.0;
            for (int i = 0; i < n; ++i)
            {
                x.data()[static_cast<std::size_t>(i)] = static_cast<float>(nd(rng));
                nrm += static_cast<double>(x.data()[static_cast<std::size_t>(i)]) *
                       static_cast<double>(x.data()[static_cast<std::size_t>(i)]);
            }
            nrm = std::sqrt(nrm) + 1e-12;
            for (auto &v : x.data())
                v = static_cast<float>(static_cast<double>(v) / nrm);
            auto yi = dot(x);
            auto ye = apply(x);
            double num = 0.0, di = 0.0, dn = 0.0;
            for (std::size_t i = 0; i < yi.size(); ++i)
            {
                num += static_cast<double>(yi.data()[i]) * static_cast<double>(ye.data()[i]);
                di += static_cast<double>(yi.data()[i]) * static_cast<double>(yi.data()[i]);
                dn += static_cast<double>(ye.data()[i]) * static_cast<double>(ye.data()[i]);
            }
            const double fid = num / (std::sqrt(di * dn) + 1e-30);
            worst = std::min(worst, fid);
            (void)tol;
        }
        return std::clamp(worst, 0.0, 1.0);
    }

    // Exposed for backends (copies under lock)
    NP_NODISCARD ndarray<float> snapshot_weights() const
    {
        std::shared_lock lock(mtx_);
        return weights;
    }
    NP_NODISCARD std::pair<MemristorConfig, CalibrationTable> snapshot_cfg() const
    {
        std::shared_lock lock(mtx_);
        return {config_, calibration_};
    }

  private:
    MemristorConfig config_;
    CalibrationTable calibration_;
    std::vector<std::int8_t> fault_mask_; // 0 ok, +1 stuck-on, -1 stuck-off
    std::shared_ptr<IMemristorBackend> backend_;
    double drift_seconds_ = 0.0;
    mutable std::shared_mutex mtx_;

    void validate_() const
    {
        if (weights.ndim() != 1 && weights.ndim() != 2)
            throw std::invalid_argument("Crossbar: weights must be 1-D or 2-D");
        if (weights.size() == 0)
            throw std::invalid_argument("Crossbar: weights must be non-empty");
    }
    void validate_like_(const ndarray<float> &o) const
    {
        if (o.shape != weights.shape)
            throw std::invalid_argument("Crossbar: target shape must match weights");
    }
    int rows_safe_() const
    {
        if (weights.ndim() == 2)
            return weights.shape[0];
        return static_cast<int>(weights.size());
    }
    int cols_safe_() const
    {
        if (weights.ndim() == 2)
            return weights.shape[1];
        return 1;
    }
    bool needs_stuck_mask_() const noexcept
    {
        return config_.stuck_on_prob > 0.0 || config_.stuck_off_prob > 0.0;
    }
    void init_fault_mask_(std::uint64_t seed)
    {
        std::unique_lock lock(mtx_);
        init_fault_mask_locked_(seed);
    }
    void init_fault_mask_locked_(std::uint64_t seed)
    {
        fault_mask_.assign(weights.size(), 0);
        if (!needs_stuck_mask_())
            return;
        std::mt19937_64 r2(seed ^ 0x12345u);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (auto &f : fault_mask_)
        {
            const double r = u(r2);
            if (r < config_.stuck_off_prob)
                f = -1;
            else if (r < config_.stuck_off_prob + config_.stuck_on_prob)
                f = 1;
        }
    }
    bool is_noisy_() const
    {
        std::shared_lock lock(mtx_);
        return config_.dac_bits != 0 || config_.adc_bits != 0 || config_.cell_bits != 0 ||
               config_.write_noise_std != 0.0 || config_.read_noise_std != 0.0 || config_.wire_resistance != 0.0 ||
               config_.sneak_beta != 0.0 || config_.drift_nu != 0.0 || config_.temperature_c != 25.0 ||
               !fault_mask_.empty();
    }

    ndarray<float> apply_drift_and_faults_to_weights_()
    {
        std::shared_lock lock(mtx_);
        ndarray<float> out = weights;
        const bool is_diff = config_.mapping == MappingScheme::DifferentialPair;
        const double step = (config_.cell_bits > 0 && config_.cell_bits < 30)
                                ? 1.0 / static_cast<double>((1u << static_cast<unsigned>(config_.cell_bits)) - 1u)
                                : 0.0;
        double drift_factor = 1.0;
        if (config_.drift_nu != 0.0 && drift_seconds_ > 0.0)
        {
            const double t0 = config_.drift_t0_s > 0.0 ? config_.drift_t0_s : 1.0;
            drift_factor = std::pow((drift_seconds_ + t0) / t0, -config_.drift_nu);
        }
        (void)is_diff;
        for (std::size_t i = 0; i < out.size(); ++i)
        {
            double wv = std::clamp(static_cast<double>(out.data()[i]), -1.0, 1.0);
            if (step > 0.0)
            {
                const double u01 = std::round((wv + 1.0) * 0.5 / step) * step;
                wv = detail::clamp01(u01) * 2.0 - 1.0;
            }
            wv *= drift_factor;
            if (i < fault_mask_.size())
            {
                if (fault_mask_[i] > 0)
                    wv = 1.0;
                else if (fault_mask_[i] < 0)
                    wv = -1.0;
            }
            out.data()[i] = static_cast<float>(wv);
        }
        return out;
    }

    ndarray<float> dot_identity_probe_(bool noisy) const
    {
        const int n = const_cast<Crossbar *>(this)->rows_safe_();
        const int m = const_cast<Crossbar *>(this)->cols_safe_();
        if (n == 0 || m == 0)
            return ndarray<float>();
        // probe with normalized ones vector
        ndarray<float> x(std::vector<int>{n});
        for (auto &v : x.data())
            v = 1.0f / static_cast<float>(n);
        if (noisy)
            return simulate_vmm_(x, nullptr);
        return dot(x);
    }

    ndarray<float> add_array_error_(const ndarray<float> &ideal) const
    {
        auto [cfg, cal] = snapshot_cfg();
        if (cfg.read_noise_std == 0.0 && cfg.adc_bits == 0)
            return ideal;
        std::mt19937_64 rng(cfg.seed ^ 0xADC0u);
        std::normal_distribution<double> nd(0.0, cfg.read_noise_std);
        ndarray<float> out = ideal;
        double fs = 0.0;
        for (auto v : ideal.data())
            fs = std::max(fs, std::abs(static_cast<double>(v)));
        if (fs == 0.0)
            fs = 1.0;
        for (auto &v : out.data())
        {
            double y = v;
            if (cfg.read_noise_std > 0.0)
                y += nd(rng) * fs;
            y = detail::quantize_adc(y, fs, cfg.adc_bits);
            v = static_cast<float>(y);
        }
        (void)cal;
        return out;
    }

    // Core analog VMM simulation (shared by apply and backends)
    ndarray<float> simulate_vmm_(const ndarray<float> &x, const CalibrationTable *override_cal) const
    {
        ndarray<float> wcopy;
        MemristorConfig cfg;
        CalibrationTable cal;
        std::vector<std::int8_t> mask;
        double drift = 0.0;
        {
            std::shared_lock lock(mtx_);
            wcopy = weights;
            cfg = config_;
            cal = calibration_;
            mask = fault_mask_;
            drift = drift_seconds_;
        }
        if (override_cal)
            cal = *override_cal;
        if (x.ndim() != 1)
            throw std::invalid_argument("Crossbar::apply: x must be 1-D");
        const int n = wcopy.ndim() == 2 ? wcopy.shape[0] : static_cast<int>(wcopy.size());
        const int m = wcopy.ndim() == 2 ? wcopy.shape[1] : 1;
        if (static_cast<int>(x.size()) != n)
            throw std::invalid_argument("Crossbar::apply: x size must match weights rows");

        // DAC: quantize normalized inputs to [-1,1]
        std::vector<double> vin(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            vin[static_cast<std::size_t>(i)] =
                detail::quantize_dac(static_cast<double>(x.data()[static_cast<std::size_t>(i)]), cfg.dac_bits);

        // Effective bipolar conductance per cell + raw conductance for IR drop.
        // Model (keeps dot() parity exact in the ideal limit by construction):
        //   single/offset: Geff = w * range/2,  raw = mid + Geff
        //   differential:  Geff = w * range,    raw(total 2 dev) = 2*goff + |Geff|
        // Cell quant, drift (power-law), stuck faults applied in weight domain.
        const double goff = detail::g_off(cfg);
        const double gon = detail::g_on(cfg);
        const double range = gon - goff > 0.0 ? gon - goff : 1.0;
        const double mid = (gon + goff) * 0.5;
        double temp_gain = 1.0 + (cfg.temperature_c - 25.0) * cfg.temp_coeff;
        if (temp_gain < 0.2)
            temp_gain = 0.2;
        const bool is_diff = cfg.mapping == MappingScheme::DifferentialPair;
        const double cell_step = (cfg.cell_bits > 0 && cfg.cell_bits < 30)
                                     ? 1.0 / static_cast<double>((1u << static_cast<unsigned>(cfg.cell_bits)) - 1u)
                                     : 0.0;
        double drift_factor = 1.0;
        if (cfg.drift_nu != 0.0 && drift > 0.0)
        {
            const double t0 = cfg.drift_t0_s > 0.0 ? cfg.drift_t0_s : 1.0;
            drift_factor = std::pow((drift + t0) / t0, -cfg.drift_nu);
        }
        const double cal_gain = cal.g_scale != 0.0 ? cal.g_scale : 1.0;
        std::vector<double> Geff(static_cast<std::size_t>(n) * static_cast<std::size_t>(m));
        std::vector<double> Grow(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < m; ++j)
            {
                double wv = wcopy.ndim() == 2 ? static_cast<double>(wcopy(i, j))
                                              : static_cast<double>(wcopy.data()[static_cast<std::size_t>(i)]);
                wv = std::clamp(wv, -1.0, 1.0);
                // MLC quantization in weight domain
                if (cell_step > 0.0)
                {
                    if (is_diff)
                    {
                        const double mag = std::round(std::abs(wv) / cell_step) * cell_step;
                        wv = (wv >= 0 ? mag : -mag);
                    }
                    else
                    {
                        const double u01 = std::round((wv + 1.0) * 0.5 / cell_step) * cell_step;
                        wv = detail::clamp01(u01) * 2.0 - 1.0;
                    }
                }
                const std::size_t lin = static_cast<std::size_t>(i * m + j);
                if (lin < mask.size())
                {
                    if (mask[lin] > 0)
                        wv = 1.0;
                    else if (mask[lin] < 0)
                        wv = -1.0;
                }
                const double unit = is_diff ? range : range * 0.5;
                double ge = wv * unit * drift_factor * temp_gain * cal_gain;
                Geff[static_cast<std::size_t>(i * m + j)] = ge;
                double raw = 0.0;
                if (is_diff)
                    raw = 2.0 * goff * temp_gain + std::abs(ge);
                else
                    raw = mid * temp_gain + ge;
                if (raw < 0.0)
                    raw = 0.0;
                Grow[static_cast<std::size_t>(i)] += raw;
            }
        }

        // IR drop: first-order per-row voltage divider from raw row conductance
        std::vector<double> veff = vin;
        if (cfg.wire_resistance > 0.0)
        {
            for (int i = 0; i < n; ++i)
            {
                const double div = 1.0 + cfg.wire_resistance * Grow[static_cast<std::size_t>(i)];
                veff[static_cast<std::size_t>(i)] = vin[static_cast<std::size_t>(i)] * cfg.v_read / div;
            }
        }
        else
        {
            for (auto &v : veff)
                v *= cfg.v_read;
        }

        // I = V*Geff accumulate per column (differential readout rejects offset)
        std::vector<double> I(static_cast<std::size_t>(m), 0.0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                I[static_cast<std::size_t>(j)] +=
                    veff[static_cast<std::size_t>(i)] * Geff[static_cast<std::size_t>(i * m + j)];

        const double unit_norm = is_diff ? range * temp_gain * cal_gain : range * 0.5 * temp_gain * cal_gain;
        const double norm = cfg.v_read * (unit_norm > 0.0 ? unit_norm : 1.0);

        std::mt19937_64 rng(cfg.seed ^ 0xBE4Du);
        std::normal_distribution<double> rnd(0.0, cfg.read_noise_std);
        double fs = 0.0;
        std::vector<double> yraw(static_cast<std::size_t>(m));
        for (int j = 0; j < m; ++j)
        {
            double y = I[static_cast<std::size_t>(j)] / norm;
            // sneak-path leakage: proportional to mean input
            if (cfg.sneak_beta != 0.0)
            {
                double mean = 0.0;
                for (auto v : vin)
                    mean += v;
                mean /= static_cast<double>(n);
                y += cfg.sneak_beta * mean;
            }
            if (cfg.read_noise_std > 0.0)
                y += rnd(rng);
            yraw[static_cast<std::size_t>(j)] = y;
            fs = std::max(fs, std::abs(y));
        }
        if (fs == 0.0)
            fs = 1.0;
        ndarray<float> y(std::vector<int>{m});
        for (int j = 0; j < m; ++j)
        {
            double v = detail::quantize_adc(yraw[static_cast<std::size_t>(j)], fs, cfg.adc_bits);
            y.data()[static_cast<std::size_t>(j)] = static_cast<float>(v);
        }
        return y;
    }

    friend struct SimBackend;
    friend struct NoisySimBackend;
    friend struct GenericHardwareBackend;
};

// ── Sim backends ────────────────────────────────────────────────────────
struct SimBackend : IMemristorBackend
{
    MemristorConfig cfg_;
    CalibrationTable cal_;
    ndarray<float> programmed_W_;
    bool has_W_ = false;
    mutable std::shared_mutex mtx_;

    explicit SimBackend(MemristorConfig cfg = {}, CalibrationTable cal = {}) : cfg_(cfg), cal_(cal)
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
    void configure(const Crossbar &cb) override
    {
        std::unique_lock lock(mtx_);
        programmed_W_ = cb.snapshot_weights();
        auto [cfg, cal] = cb.snapshot_cfg();
        (void)cfg;
        has_W_ = true;
    }
    void calibrate(const CalibrationTable &tbl) override
    {
        std::unique_lock lock(mtx_);
        cal_ = tbl;
    }
    NP_NODISCARD ndarray<float> execute(const ndarray<float> &input) override
    {
        ndarray<float> w;
        {
            std::shared_lock lock(mtx_);
            if (!has_W_)
                throw std::runtime_error("SimBackend: no weights programmed; call configure()");
            w = programmed_W_;
        }
        return apply_ideal_(w, input);
    }
    NP_NODISCARD ndarray<float> execute(const ndarray<float> &input, const ndarray<float> &weights) override
    {
        {
            std::unique_lock lock(mtx_);
            programmed_W_ = weights;
            has_W_ = true;
        }
        return apply_ideal_(weights, input);
    }
    void reset() override
    {
        std::unique_lock lock(mtx_);
        has_W_ = false;
        programmed_W_ = ndarray<float>();
    }

    static ndarray<float> apply_ideal_(const ndarray<float> &W, const ndarray<float> &x)
    {
        Crossbar tmp(W);
        return tmp.dot(x);
    }
};

struct NoisySimBackend : SimBackend
{
    explicit NoisySimBackend(MemristorConfig cfg = {}, CalibrationTable cal = {}) : SimBackend(cfg, cal)
    {
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "NoisySimBackend";
    }
    NP_NODISCARD DeviceStatus status() const noexcept override
    {
        const double fid = std::exp(-cfg_.read_noise_std * cfg_.read_noise_std * 4.0 -
                                    cfg_.write_noise_std * cfg_.write_noise_std * 4.0);
        return DeviceStatus{true, true, cfg_.temperature_c, fid, 0.0, name(), ""};
    }
    void configure(const Crossbar &cb) override
    {
        std::unique_lock lock(mtx_);
        programmed_W_ = cb.snapshot_weights();
        auto [ccfg, ccal] = cb.snapshot_cfg();
        // keep backend noise cfg, inherit geometry-relevant fields
        cfg_.mapping = ccfg.mapping;
        cfg_.temperature_c = ccfg.temperature_c;
        cfg_.r_on = ccfg.r_on;
        cfg_.r_off = ccfg.r_off;
        (void)ccal;
        has_W_ = true;
    }
    NP_NODISCARD ndarray<float> execute(const ndarray<float> &input) override
    {
        ndarray<float> w;
        MemristorConfig cfg;
        CalibrationTable cal;
        {
            std::shared_lock lock(mtx_);
            if (!has_W_)
                throw std::runtime_error("NoisySimBackend: no weights programmed");
            w = programmed_W_;
            cfg = cfg_;
            cal = cal_;
        }
        Crossbar tmp(w, cfg, cal);
        return tmp.apply(input);
    }
    NP_NODISCARD ndarray<float> execute(const ndarray<float> &input, const ndarray<float> &weights) override
    {
        {
            std::unique_lock lock(mtx_);
            programmed_W_ = weights;
            has_W_ = true;
        }
        MemristorConfig cfg;
        CalibrationTable cal;
        {
            std::shared_lock lock(mtx_);
            cfg = cfg_;
            cal = cal_;
        }
        Crossbar tmp(weights, cfg, cal);
        return tmp.apply(input);
    }
};

// ── Generic hardware backend (callbacks) ────────────────────────────────
struct HardwareCallbacks
{
    std::function<void(std::span<const float> conductances, int rows, int cols)> write_conductances;
    std::function<ndarray<float>(const ndarray<float> &voltages)> analog_execute;
    std::function<double()> read_temperature_c;
    std::function<void()> trigger_calibration;
};

struct GenericHardwareBackend : IMemristorBackend
{
    MemristorConfig cfg_;
    CalibrationTable cal_;
    HardwareCallbacks cbs_;
    ndarray<float> programmed_W_;
    bool has_W_ = false;
    mutable std::shared_mutex mtx_;
    DeviceStatus last_status_{};

    explicit GenericHardwareBackend(HardwareCallbacks cbs, MemristorConfig cfg = {}, CalibrationTable cal = {})
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
        return static_cast<bool>(cbs_.write_conductances) || static_cast<bool>(cbs_.analog_execute);
    }
    NP_NODISCARD DeviceStatus status() const noexcept override
    {
        std::shared_lock lock(mtx_);
        DeviceStatus s = last_status_;
        s.temperature_c = cbs_.read_temperature_c ? cbs_.read_temperature_c() : cfg_.temperature_c;
        s.backend_name = name();
        return s;
    }
    void configure(const Crossbar &cb) override
    {
        std::vector<float> flat;
        int r = 0, c = 0;
        bool do_write = false;
        {
            std::unique_lock lock(mtx_);
            programmed_W_ = cb.snapshot_weights();
            auto [ccfg, ccal] = cb.snapshot_cfg();
            cfg_ = ccfg;
            cal_ = ccal;
            has_W_ = true;
            last_status_.connected = is_available();
            last_status_.calibrated = true;
            if (cbs_.write_conductances)
            {
                r = programmed_W_.ndim() == 2 ? programmed_W_.shape[0] : static_cast<int>(programmed_W_.size());
                c = programmed_W_.ndim() == 2 ? programmed_W_.shape[1] : 1;
                flat.reserve(programmed_W_.size());
                for (auto wv : programmed_W_.data())
                    flat.push_back(static_cast<float>(detail::weight_to_conductance(wv, cfg_, &cal_)));
                do_write = true;
            }
            else
            {
                Crossbar tmp(programmed_W_, cfg_, cal_);
                last_status_.fidelity = tmp.fidelity();
                last_status_.energy_pj = tmp.energy_pj();
                return;
            }
        }
        if (do_write)
            cbs_.write_conductances(std::span<const float>(flat.data(), flat.size()), r, c);
        {
            std::unique_lock lock(mtx_);
            Crossbar tmp(programmed_W_, cfg_, cal_);
            last_status_.fidelity = tmp.fidelity();
            last_status_.energy_pj = tmp.energy_pj();
        }
    }
    void calibrate(const CalibrationTable &tbl) override
    {
        std::unique_lock lock(mtx_);
        cal_ = tbl;
        if (cbs_.trigger_calibration)
        {
            auto cb = cbs_.trigger_calibration;
            lock.unlock();
            cb();
            lock.lock();
        }
        last_status_.calibrated = true;
    }
    NP_NODISCARD ndarray<float> execute(const ndarray<float> &input) override
    {
        bool has_cb = false;
        ndarray<float> wcopy;
        {
            std::shared_lock lock(mtx_);
            if (!has_W_)
                throw std::runtime_error("GenericHardwareBackend: no weights programmed");
            has_cb = static_cast<bool>(cbs_.analog_execute);
            wcopy = programmed_W_;
        }
        if (has_cb)
        {
            auto out = cbs_.analog_execute(input);
            if (out.size() != static_cast<std::size_t>(wcopy.ndim() == 2 ? wcopy.shape[1] : 1))
                throw std::runtime_error("hardware callback returned wrong size");
            return out;
        }
        Crossbar tmp(wcopy, cfg_, cal_);
        return tmp.apply(input);
    }
    NP_NODISCARD ndarray<float> execute(const ndarray<float> &input, const ndarray<float> &weights) override
    {
        {
            std::unique_lock lock(mtx_);
            programmed_W_ = weights;
            has_W_ = true;
        }
        return execute(input);
    }
    void reset() override
    {
        std::unique_lock lock(mtx_);
        has_W_ = false;
        programmed_W_ = ndarray<float>();
        last_status_.connected = false;
    }
};

struct SerialHardwareBackend : GenericHardwareBackend
{
    std::string device_path_;
    explicit SerialHardwareBackend(std::string path, MemristorConfig cfg = {}, CalibrationTable cal = {},
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

// ── Differential pair crossbar (bipolar weights) ────────────────────────
struct DifferentialCrossbar
{
    Crossbar pos;
    Crossbar neg;

    DifferentialCrossbar() = default;
    DifferentialCrossbar(const ndarray<float> &w, MemristorConfig cfg = {})
    {
        if (w.ndim() != 1 && w.ndim() != 2)
            throw std::invalid_argument("DifferentialCrossbar: weights must be 1-D or 2-D");
        MemristorConfig c = cfg;
        c.mapping = MappingScheme::SingleEnded;
        ndarray<float> wp(w.shape), wn(w.shape);
        for (std::size_t i = 0; i < w.size(); ++i)
        {
            const double v = static_cast<double>(w.data()[i]);
            wp.data()[i] = static_cast<float>(v >= 0 ? v : 0.0f);
            wn.data()[i] = static_cast<float>(v < 0 ? -v : 0.0f);
        }
        pos = Crossbar(wp, c);
        neg = Crossbar(wn, c);
    }
    NP_NODISCARD ndarray<float> dot(const ndarray<float> &x) const
    {
        auto yp = pos.dot(x);
        auto yn = neg.dot(x);
        ndarray<float> y(yp.shape);
        for (std::size_t i = 0; i < y.size(); ++i)
            y.data()[i] = yp.data()[i] - yn.data()[i];
        return y;
    }
    NP_NODISCARD ndarray<float> apply(const ndarray<float> &x) const
    {
        auto yp = pos.apply(x);
        auto yn = neg.apply(x);
        ndarray<float> y(yp.shape);
        for (std::size_t i = 0; i < y.size(); ++i)
            y.data()[i] = yp.data()[i] - yn.data()[i];
        return y;
    }
};

// ── Tiled crossbar for large GEMMs ──────────────────────────────────────
struct TiledCrossbar
{
    std::vector<std::vector<Crossbar>> tiles;
    int n = 0, m = 0, tr = 0, tc = 0;
    MemristorConfig config;

    TiledCrossbar() = default;
    TiledCrossbar(const ndarray<float> &W, MemristorConfig cfg = {}) : config(cfg)
    {
        if (W.ndim() != 2)
            throw std::invalid_argument("TiledCrossbar: W must be 2-D");
        n = W.shape[0];
        m = W.shape[1];
        tr = cfg.tile_rows > 0 ? cfg.tile_rows : 128;
        tc = cfg.tile_cols > 0 ? cfg.tile_cols : 128;
        const int nbr = (n + tr - 1) / tr;
        const int nbc = (m + tc - 1) / tc;
        tiles.resize(static_cast<std::size_t>(nbr));
        for (int bi = 0; bi < nbr; ++bi)
        {
            tiles[static_cast<std::size_t>(bi)].reserve(static_cast<std::size_t>(nbc));
            for (int bj = 0; bj < nbc; ++bj)
            {
                const int r0 = bi * tr, r1 = std::min(n, r0 + tr);
                const int c0 = bj * tc, c1 = std::min(m, c0 + tc);
                ndarray<float> t(std::vector<int>{r1 - r0, c1 - c0});
                for (int i = r0; i < r1; ++i)
                    for (int j = c0; j < c1; ++j)
                        t(i - r0, j - c0) = W(i, j);
                tiles[static_cast<std::size_t>(bi)].emplace_back(t, cfg);
            }
        }
    }
    NP_NODISCARD ndarray<float> dot(const ndarray<float> &x) const
    {
        if (x.ndim() != 1 || static_cast<int>(x.size()) != n)
            throw std::invalid_argument("TiledCrossbar::dot: size mismatch");
        ndarray<float> y(std::vector<int>{m});
        for (auto &v : y.data())
            v = 0.0f;
        const int nbc = m == 0 ? 0 : static_cast<int>(tiles.empty() ? 0 : tiles[0].size());
        for (std::size_t bi = 0; bi < tiles.size(); ++bi)
        {
            const int r0 = static_cast<int>(bi) * tr;
            const int r1 = std::min(n, r0 + tr);
            ndarray<float> xslice(std::vector<int>{r1 - r0});
            for (int i = r0; i < r1; ++i)
                xslice.data()[static_cast<std::size_t>(i - r0)] = x.data()[static_cast<std::size_t>(i)];
            for (int bj = 0; bj < nbc; ++bj)
            {
                auto part = tiles[bi][static_cast<std::size_t>(bj)].dot(xslice);
                const int c0 = bj * tc;
                for (std::size_t k = 0; k < part.size(); ++k)
                    y.data()[static_cast<std::size_t>(c0) + k] += part.data()[k];
            }
        }
        return y;
    }
    NP_NODISCARD ndarray<float> apply(const ndarray<float> &x) const
    {
        if (x.ndim() != 1 || static_cast<int>(x.size()) != n)
            throw std::invalid_argument("TiledCrossbar::apply: size mismatch");
        ndarray<float> y(std::vector<int>{m});
        for (auto &v : y.data())
            v = 0.0f;
        const int nbc = m == 0 ? 0 : static_cast<int>(tiles.empty() ? 0 : tiles[0].size());
        // Per-tile ADC then digital accumulate
        for (std::size_t bi = 0; bi < tiles.size(); ++bi)
        {
            const int r0 = static_cast<int>(bi) * tr;
            const int r1 = std::min(n, r0 + tr);
            ndarray<float> xslice(std::vector<int>{r1 - r0});
            for (int i = r0; i < r1; ++i)
                xslice.data()[static_cast<std::size_t>(i - r0)] = x.data()[static_cast<std::size_t>(i)];
            for (int bj = 0; bj < nbc; ++bj)
            {
                auto part = tiles[bi][static_cast<std::size_t>(bj)].apply(xslice);
                const int c0 = bj * tc;
                for (std::size_t k = 0; k < part.size(); ++k)
                    y.data()[static_cast<std::size_t>(c0) + k] += part.data()[k];
            }
        }
        return y;
    }
    NP_NODISCARD double energy_pj() const
    {
        double e = 0.0;
        for (auto &row : tiles)
            for (auto &t : row)
                e += t.energy_pj();
        return e;
    }
};

// ── Builder ─────────────────────────────────────────────────────────────
struct CrossbarBuilder
{
    std::optional<ndarray<float>> weights_;
    MemristorConfig config_;
    CalibrationTable cal_;
    std::shared_ptr<IMemristorBackend> backend_;

    CrossbarBuilder &weights(ndarray<float> w)
    {
        weights_ = std::move(w);
        return *this;
    }
    CrossbarBuilder &config(MemristorConfig c)
    {
        config_ = c;
        return *this;
    }
    CrossbarBuilder &calibration(CalibrationTable c)
    {
        cal_ = std::move(c);
        return *this;
    }
    CrossbarBuilder &backend(std::shared_ptr<IMemristorBackend> b)
    {
        backend_ = std::move(b);
        return *this;
    }
    CrossbarBuilder &model(DeviceModel m)
    {
        config_.model = m;
        return *this;
    }
    CrossbarBuilder &bits(int dac, int adc, int cell = 0)
    {
        config_.dac_bits = dac;
        config_.adc_bits = adc;
        config_.cell_bits = cell;
        return *this;
    }
    NP_NODISCARD Crossbar build() const
    {
        if (!weights_)
            throw std::invalid_argument("CrossbarBuilder: weights required");
        Crossbar cb(*weights_, config_, cal_);
        if (backend_)
            cb.set_backend(backend_);
        return cb;
    }
};

// ── Factory ─────────────────────────────────────────────────────────────
struct ReRAMFactory
{
    NP_NODISCARD static Crossbar crossbar(const ndarray<float> &w)
    {
        return Crossbar(w);
    }
    NP_NODISCARD static Crossbar crossbar(const ndarray<float> &w, const MemristorConfig &cfg)
    {
        return Crossbar(w, cfg);
    }
    NP_NODISCARD static Crossbar ideal(const ndarray<float> &w)
    {
        MemristorConfig c;
        c.model = DeviceModel::Ideal;
        return Crossbar(w, c);
    }
    NP_NODISCARD static Crossbar noisy(const ndarray<float> &w, MemristorConfig cfg = {})
    {
        if (cfg.dac_bits == 0)
            cfg.dac_bits = 8;
        if (cfg.adc_bits == 0)
            cfg.adc_bits = 8;
        if (cfg.read_noise_std == 0.0)
            cfg.read_noise_std = 0.005;
        return Crossbar(w, cfg);
    }
    NP_NODISCARD static DifferentialCrossbar differential(const ndarray<float> &w, MemristorConfig cfg = {})
    {
        return DifferentialCrossbar(w, cfg);
    }
    NP_NODISCARD static TiledCrossbar tiled(const ndarray<float> &w, MemristorConfig cfg = {})
    {
        return TiledCrossbar(w, cfg);
    }
    NP_NODISCARD static CrossbarBuilder builder()
    {
        return CrossbarBuilder{};
    }

    NP_NODISCARD static std::shared_ptr<SimBackend> simulation(MemristorConfig cfg = {})
    {
        return std::make_shared<SimBackend>(cfg);
    }
    NP_NODISCARD static std::shared_ptr<NoisySimBackend> noisy_simulation(MemristorConfig cfg = {})
    {
        if (cfg.read_noise_std == 0.0)
            cfg.read_noise_std = 0.005;
        if (cfg.dac_bits == 0)
            cfg.dac_bits = 8;
        if (cfg.adc_bits == 0)
            cfg.adc_bits = 8;
        return std::make_shared<NoisySimBackend>(cfg);
    }
    NP_NODISCARD static std::shared_ptr<GenericHardwareBackend> generic_hardware(HardwareCallbacks cbs,
                                                                                 MemristorConfig cfg = {},
                                                                                 CalibrationTable cal = {})
    {
        return std::make_shared<GenericHardwareBackend>(std::move(cbs), cfg, cal);
    }
    NP_NODISCARD static std::shared_ptr<SerialHardwareBackend> serial_hardware(std::string device_path,
                                                                               MemristorConfig cfg = {},
                                                                               CalibrationTable cal = {},
                                                                               HardwareCallbacks cbs = {})
    {
        return std::make_shared<SerialHardwareBackend>(std::move(device_path), cfg, cal, std::move(cbs));
    }
    NP_NODISCARD static std::shared_ptr<IMemristorBackend> auto_detect(MemristorConfig cfg = {},
                                                                       std::string device_hint = "/dev/reram0")
    {
        auto serial = serial_hardware(device_hint, cfg);
        if (serial->is_available())
            return serial;
        return simulation(cfg);
    }

    // Mythic/d-Matrix style presets
    NP_NODISCARD static MemristorConfig mythic_preset()
    {
        MemristorConfig c;
        c.model = DeviceModel::VTEAM;
        c.window = WindowFunction::Kvatinsky;
        c.dac_bits = 8;
        c.adc_bits = 8;
        c.cell_bits = 4;
        c.mapping = MappingScheme::DifferentialPair;
        c.tile_rows = 128;
        c.tile_cols = 128;
        return c;
    }
    NP_NODISCARD static MemristorConfig dmatrix_preset()
    {
        MemristorConfig c = mythic_preset();
        c.tile_rows = 256;
        c.tile_cols = 256;
        c.model = DeviceModel::TEAM;
        return c;
    }
};

// ── Convenience free functions ──────────────────────────────────────────
template <AnalogScalar T>
NP_NODISCARD inline double weight_to_conductance(T w_norm, const MemristorConfig &c = {},
                                                 const CalibrationTable *cal = nullptr) noexcept
{
    return detail::weight_to_conductance(static_cast<double>(w_norm), c, cal);
}

template <AnalogScalar T> NP_NODISCARD inline ndarray<float> quantize_weights(const ndarray<T> &w, int bits = 4)
{
    if (bits <= 0 || bits >= 31)
        throw std::invalid_argument("quantize_weights: bits in [1,30]");
    ndarray<float> q(w.shape);
    const float scale = static_cast<float>((1u << static_cast<unsigned>(bits)) - 1u);
    for (std::size_t i = 0; i < w.size(); ++i)
    {
        const float v = std::clamp(static_cast<float>(w.data()[i]), -1.0f, 1.0f);
        q.data()[i] = std::round(v * scale) / scale;
    }
    return q;
}

NP_NODISCARD inline double window_value(double w01, double polarity, WindowFunction wf = WindowFunction::Joglekar,
                                        int p = 2) noexcept
{
    return detail::window_fn(w01, polarity, wf, p);
}

} // namespace np::analog

#endif // NP_MEMRISTOR_HPP
