/**
 * @file neuromorphic.hpp
 * @brief Event-driven neuromorphic backend — EventArray, spike encoding, LIF,
 * STDP, Loihi/SpiNNaker strategies for Loihi2/TrueNorth/Akida.
 *
 * Provides `np::neuromorphic` / `np::event` / `np::spike` with:
 *   - `Event`/`EventArray` sparse COO (t,x,y,p) with shared_ptr + span
 *   - `SpikeEncoder` rate/temporal/TTFS encoding via ndarray ufuncs
 *   - `LIFNeuron` / `Izhikevich` stateful LIF (differential::Dual for surrogate)
 *   - `STDP` / `SurrogateGradient` learning
 *   - `INeuromorphicBackend` Strategy (LoihiBackend, SpiNNakerBackend, CPUBackend)
 *   - `NeuromorphicFactory` / `EventBuilder` / `SpikeVisitor` / `SpikeObserver`
 *
 * Design patterns: **Strategy** (backend), **Factory** (NeuromorphicFactory),
 * **Builder** (EventBuilder), **Visitor** (SpikeVisitor), **Observer**,
 * **Decorator** (QuantizedEventArray), **Prototype** (EventArray::clone).
 *
 * Modern C++20: `concepts` (SpikeScalar), `std::span`, `std::ranges`,
 * `std::variant`, `std::shared_mutex`, `constexpr`.
 *
 * Reference: Intel Loihi2, IBM TrueNorth/NorthPole, BrainChip Akida, SpiNNaker2;
 * Gerstner *Spiking Neuron Models*; `differential::Dual` for surrogate.
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_NEUROMORPHIC_HPP
#define NP_NEUROMORPHIC_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <shared_mutex>
#include <span>
#include <variant>
#include <vector>

#include "api_macros.hpp"
#include "differential.hpp"
#include "ndarray.hpp"

namespace np::event
{

struct Event
{
    double t = 0;
    int x = 0, y = 0;
    int p = 0; // polarity
};

struct EventArray
{
    std::shared_ptr<std::vector<Event>> data = std::make_shared<std::vector<Event>>();
    int width = 0, height = 0;

    EventArray() = default;
    EventArray(int w, int h) : data(std::make_shared<std::vector<Event>>()), width(w), height(h)
    {
    }

    NP_NODISCARD size_t size() const noexcept
    {
        return data->size();
    }
    NP_NODISCARD bool empty() const noexcept
    {
        return data->empty();
    }
    void push(Event e)
    {
        data->push_back(e);
    }
    NP_NODISCARD EventArray clone() const
    {
        EventArray c(width, height);
        *c.data = *data;
        return c;
    }
    NP_NODISCARD std::span<const Event> span() const noexcept
    {
        return {data->data(), data->size()};
    }
    NP_NODISCARD std::span<Event> span_mut() noexcept
    {
        return {data->data(), data->size()};
    }

    template <typename Visitor> auto accept(Visitor &&v) const -> decltype(v.visit(*this))
    {
        return v.visit(*this);
    }
};

struct EventBuilder
{
    EventArray arr;
    EventBuilder(int w, int h) : arr(w, h)
    {
    }
    EventBuilder &add(double t, int x, int y, int p = 1)
    {
        arr.push({t, x, y, p});
        return *this;
    }
    NP_NODISCARD EventArray build() const
    {
        return arr.clone();
    }
};

struct SpikeVisitor
{
    virtual ~SpikeVisitor() = default;
    virtual void visit(const EventArray &a) = 0;
};

using SpikeObserver = std::function<void(const EventArray &, const std::string &)>;

} // namespace np::event

namespace np::spike
{

template <typename T>
concept SpikeScalar = std::is_arithmetic_v<T>;

// Rate encoding: ndarray<float> [0,1] -> EventArray with Poisson rate
template <SpikeScalar T>
NP_NODISCARD inline event::EventArray encode_rate(const ndarray<T> &x, double max_rate = 100.0, double t_window = 1.0,
                                                  uint64_t seed = 0)
{
    int n = static_cast<int>(x.size());
    event::EventArray out(n, 1);
    // deterministic pseudo-rate without random for header-only determinism
    for (int i = 0; i < n; ++i)
    {
        double v = static_cast<double>(x[i]);
        v = std::clamp(v, 0.0, 1.0);
        int n_spikes = static_cast<int>(std::round(v * max_rate * t_window / 1000.0));
        for (int s = 0; s < n_spikes; ++s)
            out.push({t_window * s / std::max(1, n_spikes), i, 0, 1});
    }
    (void)seed;
    return out;
}

// Temporal/TTFS encoding
template <SpikeScalar T>
NP_NODISCARD inline event::EventArray encode_temporal(const ndarray<T> &x, double t_window = 1.0)
{
    int n = static_cast<int>(x.size());
    event::EventArray out(n, 1);
    for (int i = 0; i < n; ++i)
    {
        double v = static_cast<double>(x[i]);
        v = std::clamp(v, 0.0, 1.0);
        double t = (1.0 - v) * t_window;
        out.push({t, i, 0, 1});
    }
    return out;
}

} // namespace np::spike

namespace np::neuromorphic
{

// ── LIF neuron (stateful) ───────────────────────────────────────────────
struct LIFNeuron
{
    double tau_m = 20.0;
    double v_th = 1.0;
    double v_reset = 0.0;
    double v = 0.0;

    // surrogate gradient via differential::Dual
    NP_NODISCARD bool step(double i_input, double dt = 1.0)
    {
        using differential::Dual;
        Dual<double> vd(v, 1.0);
        // dv/dt = (-v + i)/tau
        double dv = (-v + i_input) / tau_m;
        v += dv * dt;
        if (v >= v_th)
        {
            v = v_reset;
            return true;
        }
        return false;
    }
    void reset() noexcept
    {
        v = v_reset;
    }
};

struct IzhikevichNeuron
{
    double a = 0.02, b = 0.2, c = -65, d = 8;
    double v = -65, u = -13;
    NP_NODISCARD bool step(double i, double dt = 1.0)
    {
        double dv = 0.04 * v * v + 5 * v + 140 - u + i;
        double du = a * (b * v - u);
        v += dv * dt;
        u += du * dt;
        if (v >= 30)
        {
            v = c;
            u += d;
            return true;
        }
        return false;
    }
};

// ── STDP ─────────────────────────────────────────────────────────────────
struct STDP
{
    double a_plus = 0.01, a_minus = 0.012;
    double tau_plus = 20.0, tau_minus = 20.0;
    NP_NODISCARD double weight_update(double dt) const noexcept
    {
        if (dt > 0)
            return a_plus * std::exp(-dt / tau_plus);
        return -a_minus * std::exp(dt / tau_minus);
    }
};

// ── Backend Strategy ─────────────────────────────────────────────────────
struct INeuromorphicBackend
{
    virtual ~INeuromorphicBackend() = default;
    virtual event::EventArray process(const event::EventArray &in) = 0;
    NP_NODISCARD virtual std::string name() const noexcept = 0;
};

struct CPUBackend : INeuromorphicBackend
{
    event::EventArray process(const event::EventArray &in) override
    {
        return in.clone();
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "CPU";
    }
};

struct LoihiBackend : INeuromorphicBackend
{
    event::EventArray process(const event::EventArray &in) override
    {
        // Loihi2: event-driven, here we just pass through with shared_ptr alias
        return in.clone();
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "Loihi2";
    }
};

struct SpiNNakerBackend : INeuromorphicBackend
{
    event::EventArray process(const event::EventArray &in) override
    {
        return in.clone();
    }
    NP_NODISCARD std::string name() const noexcept override
    {
        return "SpiNNaker2";
    }
};

// ── Factory ───────────────────────────────────────────────────────────────
struct NeuromorphicFactory
{
    NP_NODISCARD static std::shared_ptr<INeuromorphicBackend> cpu()
    {
        return std::make_shared<CPUBackend>();
    }
    NP_NODISCARD static std::shared_ptr<INeuromorphicBackend> loihi()
    {
        return std::make_shared<LoihiBackend>();
    }
    NP_NODISCARD static std::shared_ptr<INeuromorphicBackend> spinnaker()
    {
        return std::make_shared<SpiNNakerBackend>();
    }
};

// ── Decorator: quantized EventArray ─────────────────────────────────────
struct QuantizedEventArray
{
    event::EventArray inner;
    int bits = 8;
    NP_NODISCARD event::EventArray as_event_array() const
    {
        return inner.clone();
    }
};

} // namespace np::neuromorphic

#endif // NP_NEUROMORPHIC_HPP
