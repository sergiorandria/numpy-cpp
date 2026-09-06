/**
 * @file test_chararray.cpp
 * @brief Test suite for numpy.char.chararray class (deprecated).
 */
#include <iostream>
#include <string>
#include <vector>

#include "np/char.hpp"
#include "np/creation.hpp"
#include "test_util.hpp"

int main()
{
    using namespace np;
    using namespace np::ch;

    std::cout << "Testing numpy.char.chararray class (deprecated)...\n";

    /* Test construction */
    {
        auto arr = array(std::vector<std::string>{"hello", "world"});
        chararray ca(arr);

        test::check(ca.array().size() == 2, "chararray: construction");
        test::check(ca.array().data()[0] == "hello", "chararray: access 1");
        test::check(ca.array().data()[1] == "world", "chararray: access 2");
    }

    /* Test method chaining */
    {
        auto arr = array(std::vector<std::string>{"  HELLO  ", "  WORLD  "});
        chararray ca(arr);

        auto result = ca.strip().lower().capitalize();
        test::check(result.array().data()[0] == "Hello", "chararray: chaining 1");
        test::check(result.array().data()[1] == "World", "chararray: chaining 2");
    }

    /* Test string methods */
    {
        auto arr = array(std::vector<std::string>{"hello", "world"});
        chararray ca(arr);

        auto upper_ca = ca.upper();
        test::check(upper_ca.array().data()[0] == "HELLO", "chararray: upper");

        auto cap_ca = ca.capitalize();
        test::check(cap_ca.array().data()[0] == "Hello", "chararray: capitalize");

        auto rep_ca = ca.replace("l", "L");
        test::check(rep_ca.array().data()[0] == "heLLo", "chararray: replace");
    }

    /* Test information methods */
    {
        auto arr = array(std::vector<std::string>{"hello", "world"});
        chararray ca(arr);

        auto lengths = ca.str_len();
        test::check(lengths.data()[0] == 5, "chararray: str_len 1");
        test::check(lengths.data()[1] == 5, "chararray: str_len 2");

        auto finds = ca.find("l");
        test::check(finds.data()[0] == 2, "chararray: find");

        auto counts = ca.count("l");
        test::check(counts.data()[0] == 2, "chararray: count");
    }

    /* Test testing methods */
    {
        auto arr = array(std::vector<std::string>{"abc", "ABC", "123"});
        chararray ca(arr);

        auto alpha = ca.isalpha();
        test::check(alpha.data()[0] == true, "chararray: isalpha true");
        test::check(alpha.data()[2] == false, "chararray: isalpha false");

        auto lower_check = ca.islower();
        test::check(lower_check.data()[0] == true, "chararray: islower true");
        test::check(lower_check.data()[1] == false, "chararray: islower false");
    }

    /* Test padding methods */
    {
        auto arr = array(std::vector<std::string>{"abc"});
        chararray ca(arr);

        auto centered = ca.center(7, '*');
        test::check(centered.array().data()[0] == "**abc**", "chararray: center");

        auto ljusted = ca.ljust(5, '-');
        test::check(ljusted.array().data()[0] == "abc--", "chararray: ljust");

        auto rjusted = ca.rjust(5, '-');
        test::check(rjusted.array().data()[0] == "--abc", "chararray: rjust");
    }

    /* Test implicit conversion */
    {
        auto arr = array(std::vector<std::string>{"test"});
        chararray ca(arr);

        /* Implicit conversion to ndarray<std::string> */
        ndarray<std::string> arr2 = ca;
        test::check(arr2.data()[0] == "test", "chararray: implicit conversion");
    }

    /* Test new methods - encode/decode/translate */
    {
        auto arr = array(std::vector<std::string>{"hello", "world"});
        chararray ca(arr);

        auto encoded = ca.encode("utf-8");
        test::check(encoded.array().data()[0] == "hello", "chararray: encode (no-op)");

        auto decoded = ca.decode("utf-8");
        test::check(decoded.array().data()[0] == "hello", "chararray: decode (no-op)");

        std::string table(256, '\0');
        for (int i = 0; i < 256; ++i)
            table[i] = static_cast<char>(i);
        table[static_cast<unsigned char>('h')] = 'H';
        auto translated = ca.translate(table);
        test::check(translated.array().data()[0] == "Hello", "chararray: translate");
    }

    /* Test split methods – grouped per element */
    {
        auto arr = array(std::vector<std::string>{"hello world", "foo bar baz"});
        chararray ca(arr);

        auto splits = ca.split(" ");
        test::check(splits.size() == 2, "chararray: split groups");
        test::check(splits[0].size() == 2 && splits[1].size() == 3, "chararray: split per-element");

        auto rsplits = ca.rsplit(" ");
        test::check(rsplits.size() == 2, "chararray: rsplit groups");

        auto arr2 = array(std::vector<std::string>{"line1\nline2"});
        chararray ca2(arr2);
        auto lines = ca2.splitlines();
        test::check(lines.size() == 1 && lines[0].size() == 2, "chararray: splitlines grouped");
    }

    /* Test partition methods */
    {
        auto arr = array(std::vector<std::string>{"hello-world"});
        chararray ca(arr);

        auto parts = ca.partition("-");
        test::check(parts.size() == 3, "chararray: partition size");
        test::check(parts.data()[0] == "hello", "chararray: partition before");
        test::check(parts.data()[1] == "-", "chararray: partition sep");
        test::check(parts.data()[2] == "world", "chararray: partition after");

        auto rparts = ca.rpartition("-");
        test::check(rparts.size() == 3, "chararray: rpartition size");
    }

    /* Test join method */
    {
        auto sep = array(std::vector<std::string>{"-"});
        chararray ca_sep(sep);

        auto seq = array(std::vector<std::string>{"a b c"});
        auto joined = ca_sep.join(seq);
        test::check(joined.data()[0] == "a- -b- -c", "chararray: join");
    }

    /* Test additional boolean methods */
    {
        auto arr = array(std::vector<std::string>{"123", "hello"});
        chararray ca(arr);

        auto decimal = ca.isdecimal();
        test::check(decimal.data()[0] == true, "chararray: isdecimal true");
        test::check(decimal.data()[1] == false, "chararray: isdecimal false");

        auto numeric = ca.isnumeric();
        test::check(numeric.data()[0] == true, "chararray: isnumeric true");
        test::check(numeric.data()[1] == false, "chararray: isnumeric false");
    }

    /* Test index and rindex methods */
    {
        auto arr = array(std::vector<std::string>{"hello", "world"});
        chararray ca(arr);

        auto indices = ca.index("l");
        test::check(indices.data()[0] == 2, "chararray: index");

        auto rindices = ca.rindex("l");
        test::check(rindices.data()[0] == 3, "chararray: rindex");
    }

    /* Test properties */
    {
        auto arr = array(std::vector<std::string>{"a", "b", "c"});
        chararray ca(arr);

        test::check(ca.size() == 3, "chararray: size()");
        test::check(ca.ndim() == 1, "chararray: ndim()");
        test::check(ca.shape()[0] == 3, "chararray: shape()");
    }

    std::cout << "All chararray tests completed.\n";
    std::cout << "NOTE: chararray is deprecated in NumPy 2.5\n";
    std::cout << "      Use ndarray<std::string> with np::ch functions instead.\n";

    return test::failures() ? 1 : 0;
}
