/**
 * @file test_datetime.cpp
 * @brief Tests for datetime.hpp (busday, datetime_as_string, datetime_data, NaT).
 */
#include "test_util.hpp"
#include <np/datetime.hpp>

#include <chrono>

int main()
{
    using namespace np::datetime;
    using sys_days = std::chrono::sys_days;

    // ── weekmask parsing (NumPy 3 forms) ──────────────────────────────────
    {
        auto m1 = parse_weekmask("1111100");
        test::check(m1[0] && m1[4] && !m1[5] && !m1[6], "weekmask 1111100");
        auto m2 = parse_weekmask("Mon Tue Wed Thu Fri");
        test::check(m2 == m1, "weekmask Mon Tue ...");
        auto m3 = parse_weekmask("MonTue Wed  Thu \t Fri");
        test::check(m3 == m1, "weekmask MonTue Wed Thu Fri whitespace");
        auto m4 = parse_weekmask("1111100");
        auto m5 = parse_weekmask(std::vector<int>{1, 1, 1, 1, 1, 0, 0});
        test::check(m5 == m4, "weekmask vector<int>");
        test::check(weekmask_to_string(m1) == "1111100", "weekmask_to_string");
    }

    // ── busdaycalendar normalization ──────────────────────────────────────
    {
        auto d = datetime64_from_string("2023-01-15");
        std::vector<sys_days> hol = {d, NaT, d};
        auto cal = busdaycalendar("1111100", hol);
        test::check(cal.holidays.size() == 1 && cal.holidays[0] == d, "calendar normalize NaT/dedup");
        auto cal2 = busdaycalendar("Mon Tue Wed Thu Fri", hol);
        test::check(cal2.weekmask == parse_weekmask("1111100"), "calendar abbr weekmask");
    }

    // ── is_busday (sys_days, ndarray, NaT, holidays NaT ignored) ─────────
    {
        auto mon = datetime64_from_string("2023-01-09"); // Mon
        auto sat = datetime64_from_string("2023-01-14"); // Sat
        auto sun = datetime64_from_string("2023-01-15"); // Sun
        test::check(is_busday(mon) == true, "is_busday Mon true");
        test::check(is_busday(sat) == false, "is_busday Sat false");
        test::check(is_busday(NaT) == false, "is_busday NaT false");

        // abbreviation weekmask
        test::check(is_busday(mon, "Mon Tue Wed Thu Fri") == true, "is_busday abbr");
        test::check(is_busday(sat, "Mon Tue Wed Thu Fri") == false, "is_busday abbr Sat");

        // ndarray contiguous
        auto arr = arange(mon, datetime64_from_string("2023-01-16"));
        auto ib = is_busday(arr);
        test::check(ib.size() == 7, "is_busday arr size");
        test::check(ib.at(0) == true && ib.at(5) == false && ib.at(6) == false, "is_busday arr values");

        // holidays NaT ignored, holiday Monday makes false
        std::vector<sys_days> hol = {mon, NaT};
        test::check(is_busday(mon, "1111100", hol) == false, "is_busday holiday NaT ignored");
        (void)sun;
    }

    // ── busday_offset (roll rules, NaT, week-jump) ────────────────────────
    {
        auto sat = datetime64_from_string("2023-01-14");
        auto mon = datetime64_from_string("2023-01-16");
        // forward
        auto fwd = busday_offset(sat, 0, "forward");
        test::check(fwd == mon, "busday_offset forward Sat->Mon");
        // nat roll returns NaT
        auto nat = busday_offset(sat, 0, "nat");
        test::check(nat == NaT, "busday_offset nat");
        // NaT input propagates
        test::check(busday_offset(NaT, 5) == NaT, "busday_offset NaT input");
        // backward
        auto sun = datetime64_from_string("2023-01-15");
        auto bwd = busday_offset(sun, 0, "backward");
        test::check(bwd == datetime64_from_string("2023-01-13"), "busday_offset backward Sun->Fri");
        // offset
        auto off1 = busday_offset(datetime64_from_string("2023-01-09"), 1);
        test::check(off1 == datetime64_from_string("2023-01-10"), "busday_offset +1");
        auto offNeg = busday_offset(mon, -1);
        test::check(offNeg == datetime64_from_string("2023-01-13"), "busday_offset -1 Mon->Fri");
        // broadcast
        auto dates = arange(datetime64_from_string("2023-01-09"), datetime64_from_string("2023-01-11"));
        np::ndarray<std::int64_t> offs(std::vector<int>{2});
        offs.data()[0] = 1;
        offs.data()[1] = 2;
        auto out = busday_offset(dates, offs);
        test::check(out.size() == 2, "busday_offset broadcast size");
    }

    // ── busday_count (week arithmetic, NaT, holidays) ─────────────────────
    {
        auto b = datetime64_from_string("2023-01-09");
        auto e = datetime64_from_string("2023-01-16");
        test::check(busday_count(b, e) == 5, "busday_count Mon->Mon 5");
        test::check(busday_count(e, b) == -5, "busday_count negative");
        test::check(busday_count(b, b) == 0, "busday_count zero");
        test::check(busday_count(NaT, e) == 0, "busday_count NaT");
        test::check(busday_count(b, NaT) == 0, "busday_count NaT 2");
        // abbreviation
        test::check(busday_count(b, e, "Mon Tue Wed Thu Fri") == 5, "busday_count abbr");
        // holidays
        std::vector<sys_days> hol = {datetime64_from_string("2023-01-10")}; // Tue
        test::check(busday_count(b, e, "1111100", hol) == 4, "busday_count holiday");
    }

    // ── datetime_as_string (unit, timezone, NaT, casting) ─────────────────
    {
        auto d = datetime64_from_string("2023-01-15");
        test::check(datetime_as_string(d, "D") == "2023-01-15", "as_string D");
        test::check(datetime_as_string(d, "Y") == "2023", "as_string Y");
        test::check(datetime_as_string(d, "M") == "2023-01", "as_string M");
        test::check(datetime_as_string(d, "h") == "2023-01-15T00", "as_string h");
        test::check(datetime_as_string(d, "s") == "2023-01-15T00:00:00", "as_string s");
        test::check(datetime_as_string(d, "ms") == "2023-01-15T00:00:00.000", "as_string ms");
        test::check(datetime_as_string(NaT) == "NaT", "as_string NaT");
        test::check(datetime_as_string(d, "D", "UTC") == "2023-01-15Z", "as_string UTC");
        test::check(datetime_as_string(d, "D", "local") == "2023-01-15+0000", "as_string local");
        // casting safe should throw when going to finer
        bool threw = false;
        try
        {
            (void)datetime_as_string(d, "s", "naive", "safe");
        }
        catch (std::invalid_argument &)
        {
            threw = true;
        }
        test::check(threw, "as_string casting safe throws");

        // ndarray
        auto arr = arange(d, datetime64_from_string("2023-01-17"));
        auto strs = datetime_as_string(arr, "D");
        test::check(strs.size() == 2 && strs.at(0) == "2023-01-15" && strs.at(1) == "2023-01-16", "as_string ndarray");

        // int64_t arr with NaT sentinel
        np::ndarray<std::int64_t> iarr(std::vector<int>{2});
        iarr.data()[0] = 0; // 1970-01-01
        iarr.data()[1] = std::numeric_limits<std::int64_t>::min();
        auto sarr = datetime_as_string(iarr, "D");
        test::check(sarr.at(1) == "NaT", "as_string int64 NaT");
    }

    // ── datetime_data (count parsing) ─────────────────────────────────────
    {
        auto p1 = datetime_data("datetime64[D]");
        test::check(p1.first == "D" && p1.second == 1, "datetime_data D");
        auto p2 = datetime_data("timedelta64[25s]");
        test::check(p2.first == "s" && p2.second == 25, "datetime_data 25s");
        auto p3 = datetime_data("datetime64[10ms]");
        test::check(p3.first == "ms" && p3.second == 10, "datetime_data 10ms");
        auto p4 = datetime_data("timedelta64[ns]");
        test::check(p4.first == "ns" && p4.second == 1, "datetime_data ns");
        auto p5 = datetime_data("generic");
        test::check(p5.first == "generic" && p5.second == 1, "datetime_data generic");
        auto p6 = datetime_data("datetime64[  5  us ]");
        test::check(p6.first == "us" && p6.second == 5, "datetime_data spaced");
    }

    // ── helpers: datetime64_from_string, arange, add/subtract, isnat ──────
    {
        auto d = datetime64_from_string("2023-01-09");
        test::check(datetime_as_string(d) == "2023-01-09", "from_string");
        test::check(isnat(NaT) == true, "isnat NaT");
        test::check(isnat(d) == false, "isnat false");
        test::check(isnat(std::string("NaT")) == true, "isnat string");
        test::check(isnat(std::int64_t{std::numeric_limits<std::int64_t>::min()}) == true, "isnat int64");

        auto ar = arange(d, datetime64_from_string("2023-01-12"));
        test::check(ar.size() == 3 && ar.at(0) == d, "arange size");
        auto ar2 = arange(d, datetime64_from_string("2023-01-13"), std::int64_t{2});
        test::check(ar2.size() == 2, "arange step 2");

        auto ad = add(d, 3);
        test::check(ad == datetime64_from_string("2023-01-12"), "add");
        test::check(add(NaT, 3) == NaT, "add NaT");

        auto b = datetime64_from_string("2023-01-12");
        test::check(subtract(b, d) == 3, "subtract");
        test::check(subtract(NaT, d) == std::numeric_limits<std::int64_t>::min(), "subtract NaT");

        // array add/subtract broadcast
        auto arr = arange(d, datetime64_from_string("2023-01-11"));
        np::ndarray<std::int64_t> deltas(std::vector<int>{2});
        deltas.data()[0] = 1;
        deltas.data()[1] = 1;
        auto added = add(arr, deltas);
        test::check(added.at(0) == datetime64_from_string("2023-01-10"), "add arr");

        // isnat array
        np::ndarray<sys_days> narr(std::vector<int>{2});
        narr.data()[0] = d;
        narr.data()[1] = NaT;
        auto inm = isnat(narr);
        test::check(inm.at(0) == false && inm.at(1) == true, "isnat arr");

        // normalize_holidays
        std::vector<sys_days> hol = {d, NaT, d, b};
        auto norm = normalize_holidays(hol);
        test::check(norm.size() == 2, "normalize_holidays size");
    }

    return test::failures() ? 1 : 0;
}
