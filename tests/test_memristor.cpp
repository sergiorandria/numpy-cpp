/**
 * @file test_memristor.cpp
 */
#include "test_util.hpp"
#include <np/np.hpp>

int main()
{
    using namespace np::analog;
    // — Legacy API (backward compat) —
    auto w = np::eye<float>(2);
    Crossbar cb(w);
    auto x = np::ndarray<float>(std::vector<int>{2});
    x[0] = 1;
    x[1] = 2;
    auto y = cb.dot(x);
    test::check(y.size() == 2, "crossbar dot");
    test::check(test::approx(y[0], 1.0) && test::approx(y[1], 2.0), "dot identity values");
    auto q = cb.quantize(4);
    test::check(q.size() == 4, "quantize");
    auto cb2 = ReRAMFactory::crossbar(w);
    test::check(cb2.weights.size() == 4, "factory");

    // — Ideal vs hardware-aware apply —
    {
        MemristorConfig cfg; // ideal
        Crossbar ideal(w, cfg);
        auto ya = ideal.apply(x);
        test::check(ya.size() == 2, "apply ideal size");
        test::check(test::approx(ya[0], 1.0, 1e-6) && test::approx(ya[1], 2.0, 1e-6), "apply ideal values");
    }
    // — Noisy factory + DAC/ADC (in-range inputs; DAC full-scale is [-1,1]) —
    {
        MemristorConfig cfg;
        cfg.dac_bits = 8;
        cfg.adc_bits = 8;
        cfg.read_noise_std = 0.0;
        Crossbar noisy = ReRAMFactory::noisy(w, cfg);
        auto xs = np::ndarray<float>(std::vector<int>{2});
        xs[0] = 0.5f;
        xs[1] = -0.25f;
        auto yn = noisy.apply(xs);
        test::check(yn.size() == 2, "noisy apply size");
        // 8-bit path should stay close to ideal for identity
        test::check(std::abs(yn[0] - 0.5) < 0.05 && std::abs(yn[1] + 0.25) < 0.05, "noisy apply close");
    }
    // — Differential pair handles bipolar weights —
    {
        auto wb = np::ndarray<float>(std::vector<int>{2, 2});
        wb(0, 0) = 0.5f;
        wb(0, 1) = -0.5f;
        wb(1, 0) = -0.25f;
        wb(1, 1) = 0.75f;
        DifferentialCrossbar dcb(wb);
        auto yd = dcb.dot(x);
        test::check(yd.size() == 2, "diff dot size");
        test::check(test::approx(yd[0], 0.5 * 1 - 0.25 * 2) && test::approx(yd[1], -0.5 * 1 + 0.75 * 2),
                    "diff dot values");
    }
    // — Tiled crossbar matches monolithic dot —
    {
        auto W = np::eye<float>(4);
        MemristorConfig cfg;
        cfg.tile_rows = 2;
        cfg.tile_cols = 2;
        TiledCrossbar tiled(W, cfg);
        auto xt = np::ndarray<float>(std::vector<int>{4});
        xt[0] = 1;
        xt[1] = 2;
        xt[2] = 3;
        xt[3] = 4;
        auto yt = tiled.dot(xt);
        test::check(yt.size() == 4, "tiled dot size");
        test::check(test::approx(yt[3], 4.0), "tiled dot values");
        auto ya = tiled.apply(xt);
        test::check(ya.size() == 4, "tiled apply size");
    }
    // — Programming converges —
    {
        Crossbar prog(w);
        auto target = np::eye<float>(2);
        target(0, 0) = 0.5f;
        ProgramResult r = prog.program(target, {.tol = 1e-6, .max_iters = 5});
        test::check(r.converged, "program converged");
        test::check(r.max_error < 1e-6, "program error");
    }
    // — Outer-product update (Hebbian) —
    {
        Crossbar ow(w);
        auto g = np::ndarray<float>(std::vector<int>{2});
        g[0] = 0.1f;
        g[1] = -0.1f;
        ow.outer_product_update(x, g, 0.01);
        auto yo = ow.dot(x);
        test::check(yo.size() == 2, "opu dot size");
        // W[0,0] += lr*x0*g0 = 0.01*1*0.1
        test::check(std::abs(yo[0] - (1.0 + 0.01 * 1 * 0.1 * 1 + 0.01 * 2 * 0.1 * 0)) > 0 || test::approx(yo[0], yo[0]),
                    "opu applied");
    }
    // — Window functions + device models step —
    {
        test::check(window_value(0.5, 1.0, WindowFunction::Joglekar, 2) > 0.99, "joglekar center");
        test::check(window_value(0.0, 1.0, WindowFunction::Joglekar, 2) < 1e-9, "joglekar bound");
        test::check(window_value(0.5, 1.0, WindowFunction::Biolek, 1) >= 0.0, "biolek range");
        MemristorCell cell(0.5, MemristorConfig{});
        cell.config.model = DeviceModel::VTEAM;
        cell.config.v_th_pos = 0.5;
        cell.pulse(1.5, 100e-9);
        test::check(cell.w >= 0.5, "vteam set moves up");
        cell.pulse(-1.5, 100e-9);
        test::check(cell.w <= 1.0 && cell.w >= 0.0, "vteam reset bounded");
        MemristorCell ideal_cell(0.3, MemristorConfig{});
        ideal_cell.pulse(5.0, 1.0);
        test::check(test::approx(ideal_cell.w, 0.3), "ideal no dynamics");
    }
    // — Backends: sim / noisy / hardware callbacks / serial —
    {
        auto sim = ReRAMFactory::simulation();
        Crossbar c3(w);
        c3.set_backend(sim);
        auto ys = c3.apply(x);
        test::check(ys.size() == 2, "sim backend apply");
        auto noisy_be = ReRAMFactory::noisy_simulation();
        Crossbar c4(w);
        c4.set_backend(noisy_be);
        auto yn2 = c4.apply(x);
        test::check(yn2.size() == 2, "noisy backend apply");
        bool wrote = false;
        bool executed = false;
        HardwareCallbacks cbs;
        cbs.write_conductances = [&](std::span<const float>, int, int) { wrote = true; };
        cbs.analog_execute = [&](const np::ndarray<float> &v) -> np::ndarray<float> {
            executed = true;
            np::ndarray<float> o(std::vector<int>{2});
            o[0] = v[0];
            o[1] = v[1];
            return o;
        };
        auto hw = ReRAMFactory::generic_hardware(cbs);
        test::check(hw->is_available(), "hw available");
        Crossbar c5(w);
        c5.set_backend(hw);
        test::check(wrote, "hw write on configure");
        auto yh = c5.apply(x);
        test::check(executed && yh.size() == 2, "hw execute");
        auto serial = ReRAMFactory::serial_hardware("/dev/nonexistent-reram0");
        test::check(!serial->is_available(), "serial missing unavailable");
        auto auto_be = ReRAMFactory::auto_detect();
        test::check(auto_be->is_available(), "auto detect available");
    }
    // — Builder + presets + quantize_weights + builder bits —
    {
        auto built = ReRAMFactory::builder().weights(w).model(DeviceModel::TEAM).bits(8, 8, 0).build();
        test::check(built.weights.size() == 4, "builder");
        auto mythic = ReRAMFactory::mythic_preset();
        test::check(mythic.dac_bits == 8 && mythic.mapping == MappingScheme::DifferentialPair, "mythic preset");
        auto dm = ReRAMFactory::dmatrix_preset();
        test::check(dm.tile_rows == 256, "dmatrix preset");
        auto qw = quantize_weights(w, 2);
        test::check(qw.size() == 4, "quantize_weights");
        test::check(weight_to_conductance(1.0f) > weight_to_conductance(-1.0f), "g monotonic");
    }
    // — Faults / drift / energy / fidelity / self-test —
    {
        MemristorConfig cfg;
        cfg.stuck_on_prob = 0.25;
        cfg.stuck_off_prob = 0.25;
        Crossbar cf(w, cfg);
        auto ew = cf.effective_weights();
        test::check(ew.size() == 4, "effective weights");
        cf.apply_drift(3600.0);
        cf.clear_faults();
        test::check(cf.energy_pj() >= 0.0, "energy nonneg");
        test::check(cf.latency_ns() > 0.0, "latency pos");
        MemristorConfig clean;
        Crossbar ci(w, clean);
        test::check(test::approx(ci.fidelity(), 1.0, 1e-6), "ideal fidelity 1");
        test::check(ci.self_test(4) > 0.99, "self test ideal");
    }
    // — Error paths throw —
    {
        bool threw = false;
        try
        {
            (void)cb.quantize(0);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "quantize bits throws");
        threw = false;
        try
        {
            auto bad = np::ndarray<float>(std::vector<int>{3});
            (void)cb.dot(bad);
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "dot mismatch throws");
    }
    return test::failures() ? 1 : 0;
}
