/**
 * @file test_char.cpp
 * @brief Test suite for numpy.char module (string operations).
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

    std::cout << "Testing numpy.char module...\n";

    // --- String Operations ---

    // Test add (concatenation)
    {
        auto a = array(std::vector<std::string>{"hello", "foo"});
        auto b = array(std::vector<std::string>{" world", "bar"});
        auto result = add(a, b);
        test::check(result.data()[0] == "hello world", "add: concatenation 1");
        test::check(result.data()[1] == "foobar", "add: concatenation 2");
    }

    // Test multiply (repetition)
    {
        auto a = array(std::vector<std::string>{"abc", "x"});
        ndarray<int> counts = empty<int>(std::vector<int>{2});
        counts.data() = {3, 5};
        auto result = multiply(a, counts);
        test::check(result.data()[0] == "abcabcabc", "multiply: repetition 1");
        test::check(result.data()[1] == "xxxxx", "multiply: repetition 2");
    }

    // Test capitalize
    {
        auto a = array(std::vector<std::string>{"hello world", "PYTHON"});
        auto result = capitalize(a);
        test::check(result.data()[0] == "Hello world", "capitalize: first char upper");
        test::check(result.data()[1] == "Python", "capitalize: rest lower");
    }

    // Test lower
    {
        auto a = array(std::vector<std::string>{"HELLO", "WoRlD"});
        auto result = lower(a);
        test::check(result.data()[0] == "hello", "lower: all lowercase 1");
        test::check(result.data()[1] == "world", "lower: all lowercase 2");
    }

    // Test upper
    {
        auto a = array(std::vector<std::string>{"hello", "world"});
        auto result = upper(a);
        test::check(result.data()[0] == "HELLO", "upper: all uppercase 1");
        test::check(result.data()[1] == "WORLD", "upper: all uppercase 2");
    }

    // Test strip
    {
        auto a = array(std::vector<std::string>{"  hello  ", "\tworld\n"});
        auto result = strip(a);
        test::check(result.data()[0] == "hello", "strip: remove whitespace");
        test::check(result.data()[1] == "world", "strip: remove tabs/newlines");
    }

    // Test lstrip
    {
        auto a = array(std::vector<std::string>{"  hello", "  world  "});
        auto result = lstrip(a);
        test::check(result.data()[0] == "hello", "lstrip: remove left");
        test::check(result.data()[1] == "world  ", "lstrip: keep right");
    }

    // Test rstrip
    {
        auto a = array(std::vector<std::string>{"hello  ", "  world  "});
        auto result = rstrip(a);
        test::check(result.data()[0] == "hello", "rstrip: remove right");
        test::check(result.data()[1] == "  world", "rstrip: keep left");
    }

    // Test swapcase
    {
        auto a = array(std::vector<std::string>{"Hello", "WoRlD"});
        auto result = swapcase(a);
        test::check(result.data()[0] == "hELLO", "swapcase: swap 1");
        test::check(result.data()[1] == "wOrLd", "swapcase: swap 2");
    }

    // Test title
    {
        auto a = array(std::vector<std::string>{"hello world", "python programming"});
        auto result = title(a);
        test::check(result.data()[0] == "Hello World", "title: title case 1");
        test::check(result.data()[1] == "Python Programming", "title: title case 2");
    }

    // Test center
    {
        auto a = array(std::vector<std::string>{"abc", "x"});
        auto result = center(a, 7, '*');
        test::check(result.data()[0] == "**abc**", "center: pad 1");
        test::check(result.data()[1] == "***x***", "center: pad 2");
    }

    // Test ljust
    {
        auto a = array(std::vector<std::string>{"abc", "xy"});
        auto result = ljust(a, 5, '-');
        test::check(result.data()[0] == "abc--", "ljust: left justify 1");
        test::check(result.data()[1] == "xy---", "ljust: left justify 2");
    }

    // Test rjust
    {
        auto a = array(std::vector<std::string>{"abc", "xy"});
        auto result = rjust(a, 5, '-');
        test::check(result.data()[0] == "--abc", "rjust: right justify 1");
        test::check(result.data()[1] == "---xy", "rjust: right justify 2");
    }

    // Test zfill
    {
        auto a = array(std::vector<std::string>{"42", "-42"});
        auto result = zfill(a, 5);
        test::check(result.data()[0] == "00042", "zfill: zero pad 1");
        test::check(result.data()[1] == "-0042", "zfill: zero pad with sign");
    }

    // Test replace
    {
        auto a = array(std::vector<std::string>{"hello world", "foo bar foo"});
        auto result = replace(a, "o", "0");
        test::check(result.data()[0] == "hell0 w0rld", "replace: all occurrences");
        test::check(result.data()[1] == "f00 bar f00", "replace: multiple");

        auto result2 = replace(a, "o", "0", 1);
        test::check(result2.data()[0] == "hell0 world", "replace: maxcount 1");
    }

    // --- Comparison Functions ---

    // Test equal
    {
        auto a = array(std::vector<std::string>{"abc", "def"});
        auto b = array(std::vector<std::string>{"abc", "xyz"});
        auto result = equal(a, b);
        test::check(result.data()[0] == true, "equal: true case");
        test::check(result.data()[1] == false, "equal: false case");
    }

    // Test not_equal
    {
        auto a = array(std::vector<std::string>{"abc", "def"});
        auto b = array(std::vector<std::string>{"abc", "xyz"});
        auto result = not_equal(a, b);
        test::check(result.data()[0] == false, "not_equal: false case");
        test::check(result.data()[1] == true, "not_equal: true case");
    }

    // Test less
    {
        auto a = array(std::vector<std::string>{"abc", "xyz"});
        auto b = array(std::vector<std::string>{"def", "uvw"});
        auto result = less(a, b);
        test::check(result.data()[0] == true, "less: true case");
        test::check(result.data()[1] == false, "less: false case");
    }

    // Test greater
    {
        auto a = array(std::vector<std::string>{"xyz", "abc"});
        auto b = array(std::vector<std::string>{"abc", "def"});
        auto result = greater(a, b);
        test::check(result.data()[0] == true, "greater: true case");
        test::check(result.data()[1] == false, "greater: false case");
    }

    // --- Information Functions ---

    // Test count
    {
        auto a = array(std::vector<std::string>{"aaa", "ababa"});
        auto result = count(a, "a");
        test::check(result.data()[0] == 3, "count: multiple occurrences");
        test::check(result.data()[1] == 3, "count: non-overlapping");
    }

    // Test find
    {
        auto a = array(std::vector<std::string>{"hello", "world"});
        auto result = find(a, "l");
        test::check(result.data()[0] == 2, "find: first occurrence");
        test::check(result.data()[1] == 3, "find: found");

        auto result2 = find(a, "z");
        test::check(result2.data()[0] == -1, "find: not found");
    }

    // Test rfind
    {
        auto a = array(std::vector<std::string>{"hello", "world"});
        auto result = rfind(a, "l");
        test::check(result.data()[0] == 3, "rfind: last occurrence");
        test::check(result.data()[1] == 3, "rfind: found");
    }

    // Test startswith
    {
        auto a = array(std::vector<std::string>{"hello", "world"});
        auto result = startswith(a, "he");
        test::check(result.data()[0] == true, "startswith: true case");
        test::check(result.data()[1] == false, "startswith: false case");
    }

    // Test endswith
    {
        auto a = array(std::vector<std::string>{"hello", "world"});
        auto result = endswith(a, "lo");
        test::check(result.data()[0] == true, "endswith: true case");
        test::check(result.data()[1] == false, "endswith: false case");
    }

    // Test str_len
    {
        auto a = array(std::vector<std::string>{"abc", "hello"});
        auto result = str_len(a);
        test::check(result.data()[0] == 3, "str_len: length 1");
        test::check(result.data()[1] == 5, "str_len: length 2");
    }

    // --- Testing Functions ---

    // Test isalpha
    {
        auto a = array(std::vector<std::string>{"abc", "abc123", ""});
        auto result = isalpha(a);
        test::check(result.data()[0] == true, "isalpha: all alpha");
        test::check(result.data()[1] == false, "isalpha: mixed");
        test::check(result.data()[2] == false, "isalpha: empty");
    }

    // Test isdigit
    {
        auto a = array(std::vector<std::string>{"123", "12a", ""});
        auto result = isdigit(a);
        test::check(result.data()[0] == true, "isdigit: all digits");
        test::check(result.data()[1] == false, "isdigit: mixed");
        test::check(result.data()[2] == false, "isdigit: empty");
    }

    // Test isalnum
    {
        auto a = array(std::vector<std::string>{"abc123", "abc-123", ""});
        auto result = isalnum(a);
        test::check(result.data()[0] == true, "isalnum: all alnum");
        test::check(result.data()[1] == false, "isalnum: with hyphen");
        test::check(result.data()[2] == false, "isalnum: empty");
    }

    // Test islower
    {
        auto a = array(std::vector<std::string>{"abc", "Abc", "123"});
        auto result = islower(a);
        test::check(result.data()[0] == true, "islower: all lower");
        test::check(result.data()[1] == false, "islower: mixed case");
        test::check(result.data()[2] == false, "islower: no cased chars");
    }

    // Test isupper
    {
        auto a = array(std::vector<std::string>{"ABC", "Abc", "123"});
        auto result = isupper(a);
        test::check(result.data()[0] == true, "isupper: all upper");
        test::check(result.data()[1] == false, "isupper: mixed case");
        test::check(result.data()[2] == false, "isupper: no cased chars");
    }

    // Test isspace
    {
        auto a = array(std::vector<std::string>{"   ", " a ", ""});
        auto result = isspace(a);
        test::check(result.data()[0] == true, "isspace: all spaces");
        test::check(result.data()[1] == false, "isspace: with char");
        test::check(result.data()[2] == false, "isspace: empty");
    }

    // Test istitle
    {
        auto a = array(std::vector<std::string>{"Hello World", "hello world", "HELLO"});
        auto result = istitle(a);
        test::check(result.data()[0] == true, "istitle: title case");
        test::check(result.data()[1] == false, "istitle: not title");
        test::check(result.data()[2] == false, "istitle: all caps");
    }

    // --- Split/Join Functions ---

    // Test expandtabs
    {
        auto a = array(std::vector<std::string>{"a\tb", "x\ty"});
        auto result = expandtabs(a, 4);
        test::check(result.data()[0] == "a   b", "expandtabs: tab to spaces");
        test::check(result.data()[1] == "x   y", "expandtabs: multiple");
    }

    // Test partition
    {
        auto a = array(std::vector<std::string>{"a-b-c"});
        auto result = partition(a, "-");
        test::check(result.data()[0] == "a", "partition: before");
        test::check(result.data()[1] == "-", "partition: sep");
        test::check(result.data()[2] == "b-c", "partition: after");
    }

    // Test rpartition
    {
        auto a = array(std::vector<std::string>{"a-b-c"});
        auto result = rpartition(a, "-");
        test::check(result.data()[0] == "a-b", "rpartition: before");
        test::check(result.data()[1] == "-", "rpartition: sep");
        test::check(result.data()[2] == "c", "rpartition: after");
    }

    // Test split – now grouped per element
    {
        auto a = array(std::vector<std::string>{"a b c"});
        auto result = split(a);
        test::check(result.size() == 1, "split: one input -> one group");
        test::check(result[0].size() == 3, "split: whitespace count");
        test::check(result[0].data()[0] == "a", "split: part 1");
        test::check(result[0].data()[1] == "b", "split: part 2");
        test::check(result[0].data()[2] == "c", "split: part 3");

        auto a2 = array(std::vector<std::string>{"a,b,c"});
        auto result2 = split(a2, ",");
        test::check(result2[0].data()[0] == "a", "split: sep part 1");
        test::check(result2[0].data()[1] == "b", "split: sep part 2");
        test::check(result2[0].data()[2] == "c", "split: sep part 3");

        // Multi-element grouping: 2 strings -> 2 groups
        auto a3 = array(std::vector<std::string>{"a b", "c d"});
        auto result3 = split(a3);
        test::check(result3.size() == 2, "split: two inputs -> two groups");
        test::check(result3[0].data()[0] == "a" && result3[1].data()[0] == "c", "split: per-element grouping");
    }

    // Test splitlines – grouped
    {
        auto a = array(std::vector<std::string>{"a\nb\nc"});
        auto result = splitlines(a);
        test::check(result.size() == 1, "splitlines: one group");
        test::check(result[0].size() == 3, "splitlines: line count");
        test::check(result[0].data()[0] == "a", "splitlines: line 1");
        test::check(result[0].data()[1] == "b", "splitlines: line 2");
        test::check(result[0].data()[2] == "c", "splitlines: line 3");
    }

    // --- Creation Functions ---

    // Test array creation
    {
        auto a = array(std::vector<std::string>{"hello", "world"});
        test::check(a.size() == 2, "array: size");
        test::check(a.data()[0] == "hello", "array: element 1");
        test::check(a.data()[1] == "world", "array: element 2");
    }

    // Test asarray
    {
        auto a = array(std::vector<std::string>{"test"});
        auto b = asarray(a);
        test::check(b.data()[0] == "test", "asarray: conversion");
    }

    std::cout << "All char module tests completed.\n";
    return test::failures() ? 1 : 0;
}
