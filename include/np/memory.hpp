/**
 * @file memory.hpp
 * @brief Placement-intent tags over host memory (+ best-effort OS hints).
 *
 * Provides `np::mem` with *HintArray tags recording where the caller would
 * LIKE data to live (HBM / CXL / device / pinned / managed). Honest contract
 * (see audit note below): every array here is ordinary host storage owned by
 * an ndarray; nothing is allocated on a device, in HBM, or via CUDA pinned /
 * managed allocators (ndarray owns std::vector storage, which cannot adopt
 * external buffers). The tag drives best-effort madvise(HUGEPAGE) hints on
 * Linux for the Device/Pinned/Unified spaces and nothing elsewhere.
 * Use gpu::pinned_alloc / gpu::managed_alloc directly when you need real
 * pinned or managed buffers.
 *
 * NOTE (honesty audit): an earlier revision named these HBMArray/CXLArray /
 * GpuArray / PinnedArray / ManagedArray with migrate_to_device() etc.,
 * implying real heterogeneous placement while every path returned a host
 * copy. The types are renamed to *HintArray and the migrate verbs to tag_*
 * so no call site can mistake a tag for placement. gpu.hpp's claim that
 * "memory::GpuArray uses managed memory" is fixed alongside.
 *
 * Design: Strategy (Allocator), Decorator (MigratedArray), Factory, Builder.
 * Modern C++20: concepts, span, shared_ptr.
 * Reference: HBM3 3.2TB/s, CXL 3.0, GH200 unified, CUDA managed, OpenMP target
 * (bandwidth figures for context only; not measured here).
 */
#ifndef NP_MEMORY_HPP
#define NP_MEMORY_HPP

#include <span>
#include <variant>
#include <vector>

#include "api_macros.hpp"
#include "gpu.hpp"
#include "ndarray.hpp"

#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace np::mem
{

enum class MemorySpace
{
    Host,
    HBM,
    CXL,
    Unified,
    Device,
    Pinned
};

// TaggedArray eliminates duplication (Decorator over ndarray)
template <typename T, MemorySpace S> struct TaggedArray
{
    ndarray<T> data;
    static constexpr MemorySpace space = S;
    TaggedArray() = default;
    explicit TaggedArray(ndarray<T> d) : data(std::move(d))
    {
        maybe_hugepage();
    }
    NP_NODISCARD size_t size() const noexcept
    {
        return data.size();
    }
    NP_NODISCARD std::span<T> span()
    {
        auto &v = data.data();
        return {v.data(), v.size()};
    }
    NP_NODISCARD std::span<const T> span() const
    {
        auto &v = data.data();
        return {v.data(), v.size()};
    }

  private:
    void maybe_hugepage() const noexcept
    {
        if constexpr (S == MemorySpace::Device)
        {
            if (!gpu::is_available() || data.empty())
                return;
#if defined(__linux__)
            madvise(const_cast<void *>(static_cast<const void *>(data.data().data())), data.size() * sizeof(T),
                    MADV_HUGEPAGE);
#endif
        }
        else if constexpr (S == MemorySpace::Pinned || S == MemorySpace::Unified)
        {
            if (data.empty())
                return;
#if defined(__linux__)
            madvise(const_cast<void *>(static_cast<const void *>(data.data().data())), data.size() * sizeof(T),
                    MADV_HUGEPAGE);
#endif
        }
    }
};

template <typename T> using HbmHintArray = TaggedArray<T, MemorySpace::HBM>;
template <typename T> using CxlHintArray = TaggedArray<T, MemorySpace::CXL>;
template <typename T> using DeviceHintArray = TaggedArray<T, MemorySpace::Device>;
template <typename T> using PinnedHintArray = TaggedArray<T, MemorySpace::Pinned>;
template <typename T> using ManagedHintArray = TaggedArray<T, MemorySpace::Unified>;

struct MemoryFactory
{
    template <typename T> NP_NODISCARD static HbmHintArray<T> hbm(const ndarray<T> &a)
    {
        return HbmHintArray<T>(a);
    }
    template <typename T> NP_NODISCARD static CxlHintArray<T> cxl(const ndarray<T> &a)
    {
        return CxlHintArray<T>(a);
    }
    template <typename T> NP_NODISCARD static DeviceHintArray<T> device(const ndarray<T> &a)
    {
        return DeviceHintArray<T>(a);
    }
    template <typename T> NP_NODISCARD static PinnedHintArray<T> pinned(const ndarray<T> &a)
    {
        return PinnedHintArray<T>(a);
    }
    template <typename T> NP_NODISCARD static ManagedHintArray<T> managed(const ndarray<T> &a)
    {
        return ManagedHintArray<T>(a);
    }
    template <typename T> NP_NODISCARD static std::variant<HbmHintArray<T>, DeviceHintArray<T>> powerful(
      const ndarray<T> &a)
    {
        if (gpu::is_available())
            return DeviceHintArray<T>(a);
        return HbmHintArray<T>(a);
    }
};

template <typename T> NP_NODISCARD inline HbmHintArray<T> tag_hbm_hint(const ndarray<T> &a)
{
    return HbmHintArray<T>(a);
}
template <typename T> NP_NODISCARD inline DeviceHintArray<T> tag_device_hint(const ndarray<T> &a)
{
    return DeviceHintArray<T>(a);
}
template <typename T> NP_NODISCARD inline PinnedHintArray<T> tag_pinned_hint(const ndarray<T> &a)
{
    return PinnedHintArray<T>(a);
}
template <typename T> NP_NODISCARD inline ManagedHintArray<T> tag_managed_hint(const ndarray<T> &a)
{
    return ManagedHintArray<T>(a);
}
template <typename T> NP_NODISCARD inline ndarray<T> migrate_to_host(const HbmHintArray<T> &h)
{
    return h.data;
}
template <typename T> NP_NODISCARD inline ndarray<T> migrate_to_host(const DeviceHintArray<T> &g)
{
    return g.data;
}
template <typename T> NP_NODISCARD inline ndarray<T> migrate_to_host(const PinnedHintArray<T> &p)
{
    return p.data;
}
template <typename T> NP_NODISCARD inline ndarray<T> migrate_to_host(const ManagedHintArray<T> &m)
{
    return m.data;
}
// Replaces zeros_hbm()/zeros_device(): both were ordinary host zeros under
// device-memory names (zeros_device added only a hugepage hint). The space
// parameter records intent; storage is always host zeros.
template <typename T> NP_NODISCARD inline ndarray<T> zeros_hinted(const std::vector<int> &shape, MemorySpace space)
{
    (void)space;
    return zeros<T>(shape);
}

} // namespace np::mem

#endif // NP_MEMORY_HPP
