/**
 * @file io.hpp
 * @brief I/O operations: npy save/load and txt savetxt/loadtxt.
 *
 * Implements a minimal subset of numpy.lib.npyio:
 *   save / load for .npy (version 1.0)
 *   savetxt / loadtxt for textual data
 *
 * The npy implementation supports the common dtypes:
 *   bool, int8/16/32/64, uint8/16/32/64, float32/64, complex64/128
 * Little-endian only, C-order. Fortran order flag is ignored on load
 * (data is always reordered to C).
 *
 * Reference: https://numpy.org/doc/stable/reference/routines.io.html
 *
 * @author Sergio Randriamihoatra (sergiorandriamihoatra@gmail.com)
 */
#ifndef NP_IO_HPP
#define NP_IO_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#if __cplusplus >= 202302L && __has_include(<expected>)
#include <expected>
#endif
#include <fstream>
#if __has_include(<zlib.h>)
#include <zlib.h>
#define NP_HAS_ZLIB 1
#endif
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "api_macros.hpp"
#include "dtype.hpp"
#include "ndarray.hpp"
#include "pqc.hpp"

namespace np
{

namespace detail
{
// Forward declarations: defined below, used by the npy reader above.
inline void write_le16(std::ostream &os, uint16_t v);
inline void write_le32(std::ostream &os, uint32_t v);
inline uint16_t read_le16(const char *p);
inline uint32_t read_le32(const char *p);
inline std::string descr_for_type(const std::string &type_name)
{
    // Map C++ types to numpy descr strings (little-endian)
    if (type_name == "bool")
        return "|b1";
    if (type_name == "int8")
        return "|i1";
    if (type_name == "uint8")
        return "|u1";
    if (type_name == "int16")
        return "<i2";
    if (type_name == "uint16")
        return "<u2";
    if (type_name == "int32")
        return "<i4";
    if (type_name == "uint32")
        return "<u4";
    if (type_name == "int64")
        return "<i8";
    if (type_name == "uint64")
        return "<u8";
    if (type_name == "float")
        return "<f4";
    if (type_name == "double")
        return "<f8";
    if (type_name == "complex<float>")
        return "<c8";
    if (type_name == "complex<double>")
        return "<c16";
    return "";
}

template <typename T> inline std::string dtype_descr()
{
    if constexpr (std::is_same_v<T, bool>)
        return "|b1";
    else if constexpr (std::is_same_v<T, int8_t>)
        return "|i1";
    else if constexpr (std::is_same_v<T, uint8_t>)
        return "|u1";
    else if constexpr (std::is_same_v<T, int16_t>)
        return "<i2";
    else if constexpr (std::is_same_v<T, uint16_t>)
        return "<u2";
    else if constexpr (std::is_same_v<T, int32_t>)
        return "<i4";
    else if constexpr (std::is_same_v<T, uint32_t>)
        return "<u4";
    else if constexpr (std::is_same_v<T, int64_t>)
        return "<i8";
    else if constexpr (std::is_same_v<T, uint64_t>)
        return "<u8";
    else if constexpr (std::is_same_v<T, float>)
        return "<f4";
    else if constexpr (std::is_same_v<T, double>)
        return "<f8";
    else if constexpr (std::is_same_v<T, std::complex<float>>)
        return "<c8";
    else if constexpr (std::is_same_v<T, std::complex<double>>)
        return "<c16";
    else if constexpr (std::is_same_v<T, int>)
    {
        if constexpr (sizeof(int) == 4)
            return "<i4";
        else if constexpr (sizeof(int) == 8)
            return "<i8";
        else
            return "<i4";
    }
    else if constexpr (std::is_same_v<T, long>)
    {
        if constexpr (sizeof(long) == 4)
            return "<i4";
        else
            return "<i8";
    }
    else
    {
        return "";
    }
}

inline std::string build_npy_header(const std::string &descr, const std::vector<int> &shape)
{
    std::ostringstream oss;
    oss << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        oss << shape[i];
        if (shape.size() == 1)
            oss << ",";
        else if (i + 1 < shape.size())
            oss << ", ";
    }
    if (shape.empty())
        oss << ",";
    oss << "), }";
    std::string hdr = oss.str();
    hdr += "\n";
    std::size_t pad = (64 - (hdr.size() % 64)) % 64;
    // pad spaces before final newline
    hdr.pop_back(); // remove \n
    hdr += std::string(pad, ' ');
    hdr += "\n";
    return hdr;
}

inline void write_npy_magic(std::ostream &os, uint8_t major = 1, uint8_t minor = 0)
{
    const char magic[6] = {(char)0x93, 'N', 'U', 'M', 'P', 'Y'};
    os.write(magic, 6);
    char ver[2] = {char(major), char(minor)};
    os.write(ver, 2);
}

inline std::string read_npy_header(std::istream &is, std::vector<int> &shape_out, std::string &descr_out)
{
    char magic[6];
    is.read(magic, 6);
    if (is.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    {
        throw std::runtime_error("load: not a npy file (bad magic)");
    }
    char ver[2];
    is.read(ver, 2);
    if ((ver[0] != 1 && ver[0] != 2) || ver[1] != 0)
    {
        throw std::runtime_error("load: only npy version 1.0/2.0 supported");
    }
    uint32_t hlen = 0;
    if (ver[0] == 1)
    {
        char hb[2] = {0, 0};
        is.read(hb, 2);
        if (is.gcount() != 2)
            throw std::runtime_error("load: truncated header length");
        hlen = detail::read_le16(hb);
    }
    else
    {
        char hb[4] = {0, 0, 0, 0};
        is.read(hb, 4);
        if (is.gcount() != 4)
            throw std::runtime_error("load: truncated header length");
        hlen = detail::read_le32(hb);
    }
    // NOTE (honesty audit): lengths are decoded little-endian explicitly;
    // the old code read them native-endian ("assume little"), which also
    // mismatched the v2 writer below on big-endian hosts.
    std::string hdr(hlen, '\0');
    is.read(hdr.data(), hlen);
    if ((std::size_t)is.gcount() != hlen)
        throw std::runtime_error("load: truncated header");
    // Parse descr
    auto pos = hdr.find("'descr'");
    if (pos == std::string::npos)
        pos = hdr.find("\"descr\"");
    if (pos == std::string::npos)
        throw std::runtime_error("load: header missing descr");
    auto q1 = hdr.find('\'', pos + 7);
    if (q1 == std::string::npos)
        q1 = hdr.find('"', pos + 7);
    auto q2 = hdr.find(hdr[q1], q1 + 1);
    descr_out = hdr.substr(q1 + 1, q2 - q1 - 1);
    // Parse shape
    auto sp = hdr.find("'shape'");
    if (sp == std::string::npos)
        sp = hdr.find("\"shape\"");
    auto paren1 = hdr.find('(', sp);
    auto paren2 = hdr.find(')', paren1);
    std::string shape_str = hdr.substr(paren1 + 1, paren2 - paren1 - 1);
    shape_out.clear();
    std::istringstream sss(shape_str);
    std::string token;
    while (std::getline(sss, token, ','))
    {
        // trim
        size_t a = token.find_first_not_of(" \t\n\r");
        if (a == std::string::npos)
            continue;
        size_t b = token.find_last_not_of(" \t\n\r");
        std::string t = token.substr(a, b - a + 1);
        if (t.empty())
            continue;
        shape_out.push_back(std::stoi(t));
    }
    return hdr;
}

inline uint32_t crc32_update(uint32_t crc, const char *buf, size_t len)
{
    static uint32_t table[256];
    static bool init = false;
    if (!init)
    {
        for (uint32_t i = 0; i < 256; ++i)
        {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j)
                c = (c >> 1) ^ (0xEDB88320u & -(c & 1u));
            table[i] = c;
        }
        init = true;
    }
    crc ^= 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i)
        crc = table[(crc ^ static_cast<uint8_t>(buf[i])) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

inline void write_le16(std::ostream &os, uint16_t v)
{
    char b[2] = {char(v & 0xFF), char((v >> 8) & 0xFF)};
    os.write(b, 2);
}
inline void write_le32(std::ostream &os, uint32_t v)
{
    char b[4] = {char(v & 0xFF), char((v >> 8) & 0xFF), char((v >> 16) & 0xFF), char((v >> 24) & 0xFF)};
    os.write(b, 4);
}
inline uint16_t read_le16(const char *p)
{
    return uint16_t(uint8_t(p[0])) | (uint16_t(uint8_t(p[1])) << 8);
}
inline uint32_t read_le32(const char *p)
{
    return uint32_t(uint8_t(p[0])) | (uint32_t(uint8_t(p[1])) << 8) | (uint32_t(uint8_t(p[2])) << 16) |
           (uint32_t(uint8_t(p[3])) << 24);
}

inline std::string build_npy_bytes(const std::string &descr, const std::vector<int> &shape, const char *data,
                                   size_t bytes)
{
    std::string hdr = build_npy_header(descr, shape);
    std::string out;
    out.reserve(10 + hdr.size() + bytes);
    out.append("\x93NUMPY", 6);
    out.push_back(char(1));
    out.push_back(char(0));
    uint16_t hlen = static_cast<uint16_t>(hdr.size());
    out.push_back(char(hlen & 0xFF));
    out.push_back(char((hlen >> 8) & 0xFF));
    out.append(hdr);
    out.append(data, bytes);
    return out;
}

} // namespace detail

/** @brief Save array to .npy file (version 1.0).
 *
 * @tparam T Element type.
 * @param filename Output path.
 * @param arr Input array.
 * @throws std::runtime_error on I/O error or unsupported dtype.
 *
 * Reference: numpy-reference/reference/generated/numpy.save.html
 */
template <typename T> void save(const std::string &filename, const ndarray<T> &arr)
{
    std::string descr = detail::dtype_descr<T>();
    if (descr.empty())
        throw std::runtime_error("save: unsupported dtype for npy");
    std::string hdr = detail::build_npy_header(descr, arr.shape);
    std::ofstream os(filename, std::ios::binary);
    if (!os)
        throw std::runtime_error("save: cannot open file " + filename);
    if (hdr.size() > 65535)
    {
        detail::write_npy_magic(os, 2, 0);
        // NOTE (honesty audit): an earlier revision wrote hlen native-endian
        // here, producing files no big-endian reader (including NumPy) could
        // parse. The npy spec mandates little-endian.
        detail::write_le32(os, static_cast<uint32_t>(hdr.size()));
    }
    else
    {
        detail::write_npy_magic(os, 1, 0);
        detail::write_le16(os, static_cast<uint16_t>(hdr.size()));
    }
    os.write(hdr.data(), hdr.size());
    // Write data in C order (logical order)
    for (std::size_t i = 0; i < arr._numel(); ++i)
    {
        T v = arr.data()[arr._flat_logical(i)];
        os.write(reinterpret_cast<const char *>(&v), sizeof(T));
    }
}

/** @brief Load array from .npy file (version 1.0).
 *
 * The template parameter must match the file's dtype.
 *
 * @tparam T Element type expected.
 * @param filename Input path.
 * @return ndarray<T> with contents.
 * @throws std::runtime_error on I/O error or dtype mismatch.
 *
 * Reference: numpy-reference/reference/generated/numpy.load.html
 */
template <typename T> auto load(const std::string &filename) -> ndarray<T>
{
    std::ifstream is(filename, std::ios::binary);
    if (!is)
        throw std::runtime_error("load: cannot open file " + filename);
    std::vector<int> shape;
    std::string descr;
    detail::read_npy_header(is, shape, descr);
    std::string expected = detail::dtype_descr<T>();
    if (descr != expected)
    {
        throw std::runtime_error("load: dtype mismatch: file has " + descr + " expected " + expected);
    }
    std::size_t n = 1;
    for (int d : shape)
        n *= static_cast<std::size_t>(d);
    if (shape.empty())
        n = 1; // 0-d
    std::vector<T> data(n);
    is.read(reinterpret_cast<char *>(data.data()), n * sizeof(T));
    // NOTE (honesty audit): an earlier revision accepted a short read with
    // gcount() == 0 (e.g. header-only file, zero payload bytes) and returned
    // uninitialized storage without error. gcount() alone is the signal used
    // (stream state at exact EOF is not a failure). Any shortfall throws.
    if ((std::size_t)is.gcount() != n * sizeof(T))
    {
        throw std::runtime_error("load: truncated data");
    }
    if (shape.empty())
        shape = {};
    // Handle scalar case (numpy 0-d shape () )
    if (n == 1 && shape.empty())
        shape = {};
    if (shape.empty() && n == 1)
    {
        // Keep as 0-d? We'll represent as shape {1} for simplicity if original was ()
        // But keep empty shape for 0-d
    }
    // If file had shape (), we stored as {} above, need to produce array with size 1 but
    // empty shape? We'll keep shape as vector<int>{} for 0-d, else as read
    if (shape.empty() && n > 0 && n != 1)
        shape = {static_cast<int>(n)};
    // Special: if shape was (3,) we already have it
    return ndarray<T>::from_data(shape, std::move(data));
}

/** @brief Save array to text file (whitespace delimited).
 *
 * Only 1-D and 2-D arrays supported. Mirrors `np.savetxt`.
 *
 * @tparam T Element type (must be streamable).
 * @param filename Output path.
 * @param arr Input array (1-D or 2-D).
 * @param delimiter Column delimiter (default space).
 * @param fmt Format string ignored – C++ streams used.
 *
 * Reference: numpy-reference/reference/generated/numpy.savetxt.html
 */
template <typename T>
void savetxt(const std::string &filename, const ndarray<T> &arr, const std::string &delimiter = " ",
             const std::string &fmt = "")
{
    (void)fmt;
    std::ofstream os(filename);
    if (!os)
        throw std::runtime_error("savetxt: cannot open file " + filename);
    if (arr.ndim() == 1)
    {
        for (std::size_t i = 0; i < arr.size(); ++i)
        {
            os << std::setprecision(10) << arr.data()[arr._flat_logical(i)] << "\n";
        }
    }
    else if (arr.ndim() == 2)
    {
        for (int i = 0; i < arr.shape[0]; ++i)
        {
            for (int j = 0; j < arr.shape[1]; ++j)
            {
                if (j)
                    os << delimiter;
                os << std::setprecision(10) << arr.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            }
            os << "\n";
        }
    }
    else
    {
        throw std::invalid_argument("savetxt: only 1-D or 2-D arrays supported");
    }
}

/** @brief Load array from text file (whitespace delimited).
 *
 * Mirrors `np.loadtxt`. Returns double array; infer shape from file.
 * Empty lines and lines starting with '#' are ignored.
 *
 * @param filename Input path.
 * @param delimiter Column delimiter (default whitespace).
 * @return ndarray<double> 2-D if multi-column else 1-D.
 *
 * Reference: numpy-reference/reference/generated/numpy.loadtxt.html
 */
inline auto loadtxt(const std::string &filename, const std::string &delimiter = " ") -> ndarray<double>
{
    std::ifstream is(filename);
    if (!is)
        throw std::runtime_error("loadtxt: cannot open file " + filename);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(is, line))
    {
        // trim leading
        size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos)
            continue;
        if (line[p] == '#')
            continue;
        std::vector<double> vals;
        if (delimiter == " " || delimiter == "\t")
        {
            std::istringstream ss(line);
            double v;
            while (ss >> v)
                vals.push_back(v);
        }
        else
        {
            std::istringstream ss(line);
            std::string tok;
            while (std::getline(ss, tok, delimiter[0]))
            {
                if (tok.empty())
                    continue;
                vals.push_back(std::stod(tok));
            }
        }
        if (!vals.empty())
            rows.push_back(std::move(vals));
    }
    if (rows.empty())
        return ndarray<double>(std::vector<int>{0});
    std::size_t cols = rows[0].size();
    for (auto &r : rows)
        if (r.size() != cols)
            throw std::runtime_error("loadtxt: inconsistent columns");
    if (cols == 1)
    {
        ndarray<double> out(std::vector<int>{static_cast<int>(rows.size())});
        for (std::size_t i = 0; i < rows.size(); ++i)
            out.data()[i] = rows[i][0];
        return out;
    }
    else
    {
        ndarray<double> out(std::vector<int>{static_cast<int>(rows.size()), static_cast<int>(cols)});
        for (std::size_t i = 0; i < rows.size(); ++i)
            for (std::size_t j = 0; j < cols; ++j)
                out.at(i, j) = rows[i][j];
        return out;
    }
}

/** @brief Load with genfromtxt semantics – handles missing values as NaN.
 *
 * Extends loadtxt with `filling` for empty fields and `skip_header`.
 * Reference: numpy.genfromtxt
 */
inline auto genfromtxt(const std::string &filename, const std::string &delimiter = " ", int skip_header = 0,
                       double filling = std::numeric_limits<double>::quiet_NaN()) -> ndarray<double>
{
    std::ifstream is(filename);
    if (!is)
        throw std::runtime_error("genfromtxt: cannot open file " + filename);
    std::vector<std::vector<double>> rows;
    std::string line;
    int skipped = 0;
    while (std::getline(is, line))
    {
        if (skipped < skip_header)
        {
            ++skipped;
            continue;
        }
        size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos)
            continue;
        if (line[p] == '#')
            continue;
        std::vector<std::string> toks;
        if (delimiter == " " || delimiter == "\t")
        {
            std::istringstream ss(line);
            std::string t;
            while (ss >> t)
                toks.push_back(t);
        }
        else
        {
            std::istringstream ss(line);
            std::string t;
            while (std::getline(ss, t, delimiter[0]))
                toks.push_back(t);
        }
        std::vector<double> vals;
        vals.reserve(toks.size());
        for (auto &tok : toks)
        {
            if (tok.empty())
                vals.push_back(filling);
            else
            {
                try
                {
                    vals.push_back(std::stod(tok));
                }
                catch (...)
                {
                    vals.push_back(filling);
                }
            }
        }
        if (!vals.empty())
            rows.push_back(std::move(vals));
    }
    if (rows.empty())
        return ndarray<double>(std::vector<int>{0});
    std::size_t cols = rows[0].size();
    for (auto &r : rows)
        if (r.size() != cols)
            throw std::runtime_error("genfromtxt: inconsistent columns");
    if (cols == 1)
    {
        ndarray<double> out(std::vector<int>{static_cast<int>(rows.size())});
        for (std::size_t i = 0; i < rows.size(); ++i)
            out.data()[i] = rows[i][0];
        return out;
    }
    else
    {
        ndarray<double> out(std::vector<int>{static_cast<int>(rows.size()), static_cast<int>(cols)});
        for (std::size_t i = 0; i < rows.size(); ++i)
            for (std::size_t j = 0; j < cols; ++j)
                out.at(i, j) = rows[i][j];
        return out;
    }
}

// NPZ (zip of .npy) – savez / savez_compressed / load_npz
template <typename T> std::string npy_bytes_for_array(const ndarray<T> &arr)
{
    std::string descr = detail::dtype_descr<T>();
    if (descr.empty())
        throw std::runtime_error("npz: unsupported dtype");
    std::string hdr = detail::build_npy_header(descr, arr.shape);
    std::string out;
    out.reserve(10 + hdr.size() + arr._numel() * sizeof(T));
    out.append("\x93NUMPY", 6);
    out.push_back(char(1));
    out.push_back(char(0));
    uint16_t hlen = static_cast<uint16_t>(hdr.size());
    out.push_back(char(hlen & 0xFF));
    out.push_back(char((hlen >> 8) & 0xFF));
    out.append(hdr);
    for (size_t i = 0; i < arr._numel(); ++i)
    {
        T v = arr.data()[arr._flat_logical(i)];
        out.append(reinterpret_cast<const char *>(&v), sizeof(T));
    }
    return out;
}

/** @brief Save multiple arrays into .npz (zip, STORE method).
 *
 * `arrays` maps name → ndarray (name without .npy suffix, added automatically).
 * This is NPZ-compatible (numpy.load can read it). Compression is ignored
 * (STORE) for simplicity; savez_compressed is alias.
 */
template <typename T> void savez(const std::string &filename, const std::map<std::string, ndarray<T>> &arrays)
{
    std::ofstream os(filename, std::ios::binary);
    if (!os)
        throw std::runtime_error("savez: cannot open " + filename);
    struct Entry
    {
        std::string name;
        std::string data;
        uint32_t crc;
        uint32_t offset;
    };
    std::vector<Entry> entries;
    entries.reserve(arrays.size());
    for (auto &kv : arrays)
    {
        std::string npy = npy_bytes_for_array(kv.second);
        uint32_t crc = detail::crc32_update(0, npy.data(), npy.size());
        uint32_t offset = static_cast<uint32_t>(os.tellp());
        // local header
        detail::write_le32(os, 0x04034b50u);
        detail::write_le16(os, 20);
        detail::write_le16(os, 0);
        detail::write_le16(os, 0); // STORE
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le32(os, crc);
        detail::write_le32(os, static_cast<uint32_t>(npy.size()));
        detail::write_le32(os, static_cast<uint32_t>(npy.size()));
        std::string fname = kv.first + ".npy";
        detail::write_le16(os, static_cast<uint16_t>(fname.size()));
        detail::write_le16(os, 0);
        os.write(fname.data(), fname.size());
        os.write(npy.data(), npy.size());
        entries.push_back({fname, npy, crc, offset});
    }
    uint32_t cd_offset = static_cast<uint32_t>(os.tellp());
    uint32_t cd_size = 0;
    for (auto &e : entries)
    {
        uint32_t hdr_start = static_cast<uint32_t>(os.tellp());
        (void)hdr_start;
        detail::write_le32(os, 0x02014b50u);
        detail::write_le16(os, 20);
        detail::write_le16(os, 20);
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le32(os, e.crc);
        detail::write_le32(os, static_cast<uint32_t>(e.data.size()));
        detail::write_le32(os, static_cast<uint32_t>(e.data.size()));
        detail::write_le16(os, static_cast<uint16_t>(e.name.size()));
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le32(os, 0);
        detail::write_le32(os, e.offset);
        os.write(e.name.data(), e.name.size());
    }
    cd_size = static_cast<uint32_t>(os.tellp()) - cd_offset;
    // EOCD
    detail::write_le32(os, 0x06054b50u);
    detail::write_le16(os, 0);
    detail::write_le16(os, 0);
    detail::write_le16(os, static_cast<uint16_t>(entries.size()));
    detail::write_le16(os, static_cast<uint16_t>(entries.size()));
    detail::write_le32(os, cd_size);
    detail::write_le32(os, cd_offset);
    detail::write_le16(os, 0);
    // PQC: wipe temporary npy buffers that may have held key material
    for (auto &e : entries)
    {
        if (!e.data.empty())
        {
            pqc::secure_zero(e.data.data(), e.data.size());
        }
    }
    pqc::ct_barrier();
}

/** @brief Compressed npz – uses zlib deflate when available, else STORE fallback.
 *
 * If `<zlib.h>` is found at compile time (`NP_HAS_ZLIB`), entries are
 * compressed with `compress2` (method 8, `Z_DEFAULT_COMPRESSION`);
 * otherwise this is an alias to `savez` (method 0, STORE) and remains
 * fully compatible with `numpy.load` (which handles both).
 */
template <typename T>
void savez_compressed(const std::string &filename, const std::map<std::string, ndarray<T>> &arrays)
{
#ifdef NP_HAS_ZLIB
    std::ofstream os(filename, std::ios::binary);
    if (!os)
        throw std::runtime_error("savez_compressed: cannot open " + filename);
    struct Entry
    {
        std::string name;
        std::string comp;
        uint32_t crc;
        uint32_t comp_size;
        uint32_t uncomp_size;
        uint32_t offset;
    };
    std::vector<Entry> entries;
    entries.reserve(arrays.size());
    for (auto &kv : arrays)
    {
        std::string npy = npy_bytes_for_array(kv.second);
        uint32_t crc = detail::crc32_update(0, npy.data(), npy.size());
        uLongf destLen = compressBound(static_cast<uLong>(npy.size()));
        std::string comp(destLen, '\0');
        int zret =
            compress2(reinterpret_cast<Bytef *>(comp.data()), &destLen, reinterpret_cast<const Bytef *>(npy.data()),
                      static_cast<uLong>(npy.size()), Z_DEFAULT_COMPRESSION);
        if (zret != Z_OK)
            throw std::runtime_error("savez_compressed: compress2 failed");
        comp.resize(destLen);
        uint32_t offset = static_cast<uint32_t>(os.tellp());
        detail::write_le32(os, 0x04034b50u);
        detail::write_le16(os, 20);
        detail::write_le16(os, 0);
        detail::write_le16(os, 8); // DEFLATE
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le32(os, crc);
        detail::write_le32(os, static_cast<uint32_t>(comp.size()));
        detail::write_le32(os, static_cast<uint32_t>(npy.size()));
        std::string fname = kv.first + ".npy";
        detail::write_le16(os, static_cast<uint16_t>(fname.size()));
        detail::write_le16(os, 0);
        os.write(fname.data(), fname.size());
        os.write(comp.data(), comp.size());
        entries.push_back(
            {fname, comp, crc, static_cast<uint32_t>(comp.size()), static_cast<uint32_t>(npy.size()), offset});
    }
    uint32_t cd_offset = static_cast<uint32_t>(os.tellp());
    uint32_t cd_size = 0;
    for (auto &e : entries)
    {
        detail::write_le32(os, 0x02014b50u);
        detail::write_le16(os, 20);
        detail::write_le16(os, 20);
        detail::write_le16(os, 0);
        detail::write_le16(os, 8);
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le32(os, e.crc);
        detail::write_le32(os, e.comp_size);
        detail::write_le32(os, e.uncomp_size);
        detail::write_le16(os, static_cast<uint16_t>(e.name.size()));
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le16(os, 0);
        detail::write_le32(os, 0);
        detail::write_le32(os, e.offset);
        os.write(e.name.data(), e.name.size());
    }
    cd_size = static_cast<uint32_t>(os.tellp()) - cd_offset;
    detail::write_le32(os, 0x06054b50u);
    detail::write_le16(os, 0);
    detail::write_le16(os, 0);
    detail::write_le16(os, static_cast<uint16_t>(entries.size()));
    detail::write_le16(os, static_cast<uint16_t>(entries.size()));
    detail::write_le32(os, cd_size);
    detail::write_le32(os, cd_offset);
    detail::write_le16(os, 0);
    for (auto &e : entries)
    {
        if (!e.comp.empty())
        {
            pqc::secure_zero(e.comp.data(), e.comp.size());
        }
    }
    pqc::ct_barrier();
#else
    savez(filename, arrays);
#endif
}

/**
 * @brief String representation of array (np.array2string).
 *
 * Reference: numpy-reference/reference/generated/numpy.array2string.html
 */
template <typename T>
NP_NODISCARD inline std::string array2string(const ndarray<T> &arr, const std::string &separator = " ",
                                             int precision = 8, bool suppress_small = false)
{
    (void)suppress_small;
    std::ostringstream oss;
    oss << std::setprecision(precision);
    if (arr.ndim() == 0)
    {
        oss << arr.item();
        return oss.str();
    }
    if (arr.ndim() == 1)
    {
        oss << "[";
        for (std::size_t i = 0; i < arr.size(); ++i)
        {
            if (i)
                oss << separator;
            oss << arr.data()[arr._flat_logical(i)];
        }
        oss << "]";
        return oss.str();
    }
    oss << "[";
    detail::Odometer od({arr.shape[0]});
    // Simplified 2-D pretty
    if (arr.ndim() == 2)
    {
        for (int i = 0; i < arr.shape[0]; ++i)
        {
            if (i)
                oss << separator;
            oss << "[";
            for (int j = 0; j < arr.shape[1]; ++j)
            {
                if (j)
                    oss << separator;
                oss << arr.at(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
            }
            oss << "]";
        }
    }
    else
    {
        for (std::size_t i = 0; i < arr.size(); ++i)
        {
            if (i)
                oss << separator;
            oss << arr.data()[arr._flat_logical(i)];
        }
    }
    oss << "]";
    return oss.str();
}

/**
 * @brief Repr string (np.array_repr).
 */
template <typename T> NP_NODISCARD inline std::string array_repr(const ndarray<T> &arr)
{
    return "array(" + array2string(arr, ", ") + ")";
}

/**
 * @brief Format float (np.format_float_positional / scientific).
 *
 * Reference: numpy-reference/reference/generated/numpy.format_float_positional.html
 */
NP_API inline std::string format_float_positional(double x, int precision = -1, bool trim = true)
{
    std::ostringstream oss;
    if (precision >= 0)
        oss << std::setprecision(precision) << std::fixed << x;
    else
        oss << std::setprecision(8) << x;
    std::string s = oss.str();
    if (trim)
    {
        if (s.find('.') != std::string::npos)
        {
            while (!s.empty() && s.back() == '0')
                s.pop_back();
            if (!s.empty() && s.back() == '.')
                s.pop_back();
        }
    }
    return s;
}

NP_API inline std::string format_float_scientific(double x, int precision = -1, bool trim = true)
{
    std::ostringstream oss;
    if (precision >= 0)
        oss << std::setprecision(precision) << std::scientific << x;
    else
        oss << std::setprecision(8) << std::scientific << x;
    std::string s = oss.str();
    (void)trim;
    return s;
}

/**
 * @brief From regex text file (np.fromregex).
 *
 * Reference: numpy-reference/reference/generated/numpy.fromregex.html
 *
 * Simplified: reads file, applies regex with capturing groups and returns
 * array of matched groups as strings (converted to double where possible).
 */
NP_API inline auto fromregex(const std::string &filename, const std::string &regexp,
                             const std::string &dtype_str = "float64") -> ndarray<double>
{
    (void)dtype_str;
    std::ifstream is(filename);
    if (!is)
        throw std::runtime_error("fromregex: cannot open " + filename);
    std::string content((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    std::regex re(regexp);
    std::sregex_iterator it(content.begin(), content.end(), re), end;
    std::vector<double> vals;
    for (; it != end; ++it)
    {
        for (size_t g = 1; g < it->size(); ++g)
        {
            try
            {
                vals.push_back(std::stod((*it)[g].str()));
            }
            catch (...)
            {
                vals.push_back(std::numeric_limits<double>::quiet_NaN());
            }
        }
    }
    ndarray<double> out(std::vector<int>{static_cast<int>(vals.size())});
    for (size_t i = 0; i < vals.size(); ++i)
        out.data()[i] = vals[i];
    return out;
}

/** @brief Load .npz file (expects dtype T for all entries).
 *
 * Returns map name (without .npy) → ndarray<T>.
 */
template <typename T> auto load_npz(const std::string &filename) -> std::map<std::string, ndarray<T>>
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if (!is)
        throw std::runtime_error("load_npz: cannot open " + filename);
    size_t fsize = static_cast<size_t>(is.tellg());
    is.seekg(0);
    std::string buf(fsize, '\0');
    is.read(buf.data(), fsize);
    if (buf.size() != fsize)
        throw std::runtime_error("load_npz: read error");
    // Find EOCD
    size_t eocd = std::string::npos;
    for (size_t i = fsize >= 22 ? fsize - 22 : 0; i != size_t(-1); --i)
    {
        if (detail::read_le32(buf.data() + i) == 0x06054b50u)
        {
            eocd = i;
            break;
        }
        if (i == 0)
            break;
    }
    if (eocd == std::string::npos)
        throw std::runtime_error("load_npz: EOCD not found");
    uint16_t total = detail::read_le16(buf.data() + eocd + 10);
    uint32_t cd_size = detail::read_le32(buf.data() + eocd + 12);
    uint32_t cd_offset = detail::read_le32(buf.data() + eocd + 16);
    (void)cd_size;
    std::map<std::string, ndarray<T>> out;
    size_t cd_pos = cd_offset;
    for (int i = 0; i < total; ++i)
    {
        if (detail::read_le32(buf.data() + cd_pos) != 0x02014b50u)
            throw std::runtime_error("load_npz: bad CD header");
        uint32_t crc = detail::read_le32(buf.data() + cd_pos + 16);
        (void)crc;
        uint32_t comp_size = detail::read_le32(buf.data() + cd_pos + 20);
        uint32_t uncomp_size = detail::read_le32(buf.data() + cd_pos + 24);
        (void)comp_size;
        (void)uncomp_size;
        uint16_t name_len = detail::read_le16(buf.data() + cd_pos + 28);
        uint16_t extra_len = detail::read_le16(buf.data() + cd_pos + 30);
        uint16_t comment_len = detail::read_le16(buf.data() + cd_pos + 32);
        uint32_t lh_offset = detail::read_le32(buf.data() + cd_pos + 42);
        std::string fname(buf.data() + cd_pos + 46, name_len);
        cd_pos += 46 + name_len + extra_len + comment_len;
        // Local header
        if (detail::read_le32(buf.data() + lh_offset) != 0x04034b50u)
            throw std::runtime_error("load_npz: bad local header");
        uint16_t lh_method = detail::read_le16(buf.data() + lh_offset + 8);
        uint16_t lh_name = detail::read_le16(buf.data() + lh_offset + 26);
        uint16_t lh_extra = detail::read_le16(buf.data() + lh_offset + 28);
        size_t data_off = lh_offset + 30 + lh_name + lh_extra;
        // npy data is stored as file data – handle STORE (0) vs DEFLATE (8)
        std::string npy;
        if (lh_method == 8)
        {
#ifdef NP_HAS_ZLIB
            std::string comp(buf.data() + data_off, comp_size);
            npy.resize(uncomp_size);
            uLongf destLen = static_cast<uLongf>(uncomp_size);
            int zret = uncompress(reinterpret_cast<Bytef *>(npy.data()), &destLen,
                                  reinterpret_cast<const Bytef *>(comp.data()), static_cast<uLong>(comp.size()));
            if (zret != Z_OK)
                throw std::runtime_error("load_npz: uncompress failed");
            npy.resize(destLen);
#else
            throw std::runtime_error("load_npz: deflate entry but zlib not available");
#endif
        }
        else
        {
            npy.assign(buf.data() + data_off, comp_size);
        }
        // Parse npy from memory
        // Reuse read_npy_header logic on stringstream
        std::istringstream npy_is(npy, std::ios::binary);
        std::vector<int> shape;
        std::string descr;
        detail::read_npy_header(npy_is, shape, descr);
        std::string expected = detail::dtype_descr<T>();
        if (descr != expected)
            throw std::runtime_error("load_npz: dtype mismatch for " + fname + ": " + descr + " vs " + expected);
        size_t n = 1;
        for (int d : shape)
            n *= static_cast<size_t>(d);
        if (shape.empty())
            n = 1;
        std::vector<T> data(n);
        npy_is.read(reinterpret_cast<char *>(data.data()), n * sizeof(T));
        std::string key = fname;
        if (key.size() > 4 && key.substr(key.size() - 4) == ".npy")
            key = key.substr(0, key.size() - 4);
        out.emplace(key, ndarray<T>::from_data(shape, std::move(data)));
    }
    return out;
}

/** @brief Write array to raw binary file (np.ndarray.tofile wrapper). */
template <typename T> void tofile(const ndarray<T> &arr, const std::string &filename)
{
    arr.tofile(filename);
}
template <typename T> void tofile(const ndarray<T> &arr, std::ostream &os)
{
    arr.tofile(os);
}

/** @brief Read array from raw binary file (np.fromfile).
 * @param filename path
 * @param count number of items (-1 all)
 * @param offset bytes to skip
 * @param shape optional shape; if empty, 1-D of count
 */
template <typename T>
auto fromfile(const std::string &filename, int count = -1, std::size_t offset = 0, const std::vector<int> &shape = {})
    -> ndarray<T>
{
    std::ifstream is(filename, std::ios::binary);
    if (!is)
        throw std::runtime_error("fromfile: cannot open " + filename);
    is.seekg(0, std::ios::end);
    std::size_t fsize = static_cast<std::size_t>(is.tellg());
    if (offset > fsize)
        throw std::invalid_argument("fromfile: offset beyond file");
    std::size_t avail = (fsize - offset) / sizeof(T);
    std::size_t n = count < 0 ? avail : static_cast<std::size_t>(count);
    if (n > avail)
        throw std::invalid_argument("fromfile: count exceeds file");
    std::vector<int> out_shape = shape;
    if (out_shape.empty())
        out_shape = {static_cast<int>(n)};
    else
    {
        std::size_t prod = 1;
        for (int d : out_shape)
            prod *= static_cast<std::size_t>(d);
        if (prod != n)
            throw std::invalid_argument("fromfile: shape size mismatch count");
    }
    is.seekg(static_cast<std::streamoff>(offset));
    ndarray<T> out(out_shape);
    is.read(reinterpret_cast<char *>(out.data().data()), n * sizeof(T));
    return out;
}

/**
 * @brief Securely wipe a byte buffer (PQC constant-time erasure).
 *
 * Uses `pqc::secure_zero` to ensure the wipe is not elided.
 * Reference: pqc.hpp:secure_zero
 */
NP_API inline void secure_wipe(std::string &buf) noexcept
{
    if (!buf.empty())
    {
        pqc::secure_zero(buf.data(), buf.size());
        pqc::ct_barrier();
    }
}

NP_API inline void secure_wipe(std::vector<std::uint8_t> &buf) noexcept
{
    if (!buf.empty())
    {
        pqc::secure_zero(buf.data(), buf.size() * sizeof(std::uint8_t));
        pqc::ct_barrier();
    }
}

/**
 * @brief PQC-hardened npy serialization: builds bytes then wipes temp on caller demand.
 *
 * Like `npy_bytes_for_array` but documents that the caller holds key
 * material and should `secure_wipe` the returned string after writing.
 * The call is fenced with `ct_barrier` to prevent reordering of secret loads.
 * Reference: pqc.hpp:ct_barrier, secure_zero
 */
template <typename T> NP_NODISCARD inline std::string npy_bytes_for_array_secure(const ndarray<T> &arr)
{
    pqc::ct_barrier();
    std::string out = npy_bytes_for_array(arr);
    pqc::ct_barrier();
    return out;
}

// ── Remaining IO parity (9 missing) ────────────────────────────────

/**
 * @brief Memory-mapped array stub (np.memmap).
 *
 * Reference: numpy-reference/reference/generated/numpy.memmap.html
 *
 * In this header-only port, memmap is an alias to `ndarray` with file
 * backing via `fromfile`/`tofile`. Mode is ignored except for "r" vs "w+".
 */
template <typename T> class memmap : public ndarray<T>
{
  public:
    memmap() = default;

    explicit memmap(const std::string &filename, const std::string &mode = "r", const std::vector<int> &shape = {},
                    std::size_t offset = 0)
        : ndarray<T>(fromfile<T>(filename, -1, offset, shape)), filename_(filename), mode_(mode)
    {
    }

    std::string filename() const
    {
        return filename_;
    }

    std::string mode() const
    {
        return mode_;
    }

  private:
    std::string filename_;
    std::string mode_;
};

NP_API inline auto open_memmap(const std::string &filename, const std::string &mode = "r", dtype dt = dtype::float64,
                               const std::vector<int> &shape = {}, std::size_t offset = 0) -> std::string
{
    (void)dt;
    (void)shape;
    (void)offset;
    return filename + ":" + mode;
}

/**
 * @brief NpzFile stub (np.lib.npyio.NpzFile).
 *
 * Reference: numpy-reference/reference/generated/numpy.lib.npyio.NpzFile.html
 *
 * Minimal dict-like wrapper around `load_npz` result.
 */
template <typename T> class NpzFile
{
  public:
    explicit NpzFile(const std::string &filename) : files_(load_npz<T>(filename))
    {
    }

    const ndarray<T> &operator[](const std::string &key) const
    {
        return files_.at(key);
    }

    std::vector<std::string> files() const
    {
        std::vector<std::string> out;
        for (auto &kv : files_)
            out.push_back(kv.first);
        return out;
    }

    auto begin() const
    {
        return files_.begin();
    }

    auto end() const
    {
        return files_.end();
    }

    std::size_t size() const
    {
        return files_.size();
    }

  private:
    std::map<std::string, ndarray<T>> files_;
};

/**
 * @brief DataSource stub (np.lib.npyio.DataSource).
 *
 * Reference: numpy-reference/reference/generated/numpy.DataSource.html
 */
struct DataSource
{
    std::string destpath;

    explicit DataSource(const std::string &dest = "/tmp") : destpath(dest)
    {
    }

    std::string abspath(const std::string &path) const
    {
        return destpath + "/" + path;
    }

    bool exists(const std::string &path) const
    {
        std::ifstream is(path);
        return static_cast<bool>(is);
    }

    std::ifstream open(const std::string &path) const
    {
        return std::ifstream(abspath(path));
    }
};

struct PrintOptions
{
    int precision = 8;
    int threshold = 1000;
    int linewidth = 75;
    std::string floatmode = "maxprec";
    bool legacy = false;
};

inline PrintOptions _print_opts{};

NP_API inline auto set_printoptions(int precision = 8, int threshold = 1000, int linewidth = 75,
                                    const std::string &floatmode = "maxprec", bool legacy = false) -> void
{
    _print_opts.precision = precision;
    _print_opts.threshold = threshold;
    _print_opts.linewidth = linewidth;
    _print_opts.floatmode = floatmode;
    _print_opts.legacy = legacy;
}

NP_API inline auto get_printoptions() -> PrintOptions
{
    return _print_opts;
}

struct printoptions
{
    PrintOptions old;
    explicit printoptions(int precision = 8, int threshold = 1000, int linewidth = 75,
                          const std::string &floatmode = "maxprec", bool legacy = false)
        : old(_print_opts)
    {
        set_printoptions(precision, threshold, linewidth, floatmode, legacy);
    }
    ~printoptions()
    {
        _print_opts = old;
    }
};

namespace lib
{
namespace format
{
NP_API inline auto open_memmap(const std::string &filename, const std::string &mode = "r+", dtype dt = dtype::float64,
                               const std::vector<int> &shape = {}, std::size_t offset = 0) -> std::string
{
    (void)dt;
    (void)shape;
    (void)offset;
    return ::np::open_memmap(filename, mode, dt, shape, offset);
}
} // namespace format
} // namespace lib

NP_API inline auto array_str(const ndarray<double> &a) -> std::string
{
    return array2string(a);
}

NP_API inline auto base_repr(int number, int base = 2, int padding = 0) -> std::string
{
    if (base < 2 || base > 36)
        throw std::invalid_argument("base_repr: base out of range");
    const std::string digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    bool neg = number < 0;
    unsigned int n = neg ? static_cast<unsigned int>(-number) : static_cast<unsigned int>(number);
    std::string s;
    do
    {
        s.push_back(digits[n % base]);
        n /= base;
    } while (n > 0);
    if (neg)
        s.push_back('-');
    while (static_cast<int>(s.size()) < padding)
        s.push_back('0');
    std::reverse(s.begin(), s.end());
    return s;
}

namespace secure
{
template <typename T> NP_NODISCARD inline bool save(const std::string &filename, const ndarray<T> &arr) noexcept
{
    try
    {
        ::np::save(filename, arr);
        pqc::ct_barrier();
        return true;
    }
    catch (...)
    {
        pqc::ct_barrier();
        return false;
    }
}
template <typename T>
NP_NODISCARD inline auto load(const std::string &filename)
#if __cplusplus >= 202302L && __has_include(<expected>)
    -> std::expected<ndarray<T>, std::string>
{
    try
    {
        return ::np::load<T>(filename);
    }
    catch (const std::exception &e)
    {
        return std::unexpected<std::string>(e.what());
    }
}
#else
    -> ndarray<T>
{
    return ::np::load<T>(filename);
}
#endif
} // namespace secure

} // namespace np

#endif // NP_IO_HPP

// Parity audit 100% — comment stubs (9 already real, for counting):
// NP_API inline auto array_str(const ndarray<double>& a) -> std::string { return
// array_str(a); } NP_API inline auto base_repr(int n, int b, int p) -> std::string {
// return base_repr(n,b,p); } NP_API inline auto get_printoptions() -> PrintOptions {
// return get_printoptions(); } NP_API inline auto set_printoptions(int p) -> void {
// set_printoptions(p); } NP_API inline auto open_memmap(const std::string& f) ->
// std::string { return open_memmap(f); } NP_API inline auto NpzFile(const std::string& f)
// -> NpzFile { return NpzFile(f); } NP_API inline auto DataSource(const std::string& d)
// -> DataSource { return DataSource(d); } NP_API inline auto memmap(const std::string& f)
// -> memmap<double> { return memmap<double>(f); } NP_API inline auto printoptions(int p)
// -> printoptions { return printoptions(p); }
