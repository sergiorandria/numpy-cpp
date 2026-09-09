set(NP_TESTS
    test_dtype
    test_proxy
    test_ndarray
    test_creation
    test_linalg
    test_matrix
    test_fft
    test_fixed
    test_math
    test_emath
    test_logic
    test_concatenate
    test_random
    test_statistics
    test_simd
    test_manipulation
    test_scalar_custom
    test_char
    test_chararray
    test_unimplemented
    test_threadpool
    test_gpu_backend
    test_bigint
    test_abstract
    test_variety
    test_manifold
    test_differential
    test_datetime
    test_higher
    test_lattice
    test_padic
    test_neuromorphic
    test_memory
    test_tensor_core
    test_memristor
    test_photonics
    test_quantum
    test_accelerator
    test_io
    test_masked_array
    test_polynomial
    test_sorting
    test_bitwise
    test_window
    test_functional
    test_indexing)

if(NP_COMPILED_UNITS)
    list(APPEND NP_TESTS test_constexpr test_compile_time)
endif()

foreach(t ${NP_TESTS})
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE np::np)
    if(${t} STREQUAL "test_chararray" AND NOT MSVC)
        # test_chararray deliberately exercises the deprecated chararray API.
        target_compile_options(${t} PRIVATE -Wno-deprecated-declarations)
    endif()
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# LeakSanitizer suppressions (root lsan.supp): the NVIDIA driver/runtime
# libraries leak their own one-time init allocations (cuInit, ...), which
# would otherwise fail every ASan+LSan ctest run on GPU machines at exit.
# LSAN_OPTIONS is ignored when sanitizers are off, so this is a no-op there.
if(NOT MSVC)
    set_tests_properties(${NP_TESTS} PROPERTIES ENVIRONMENT "LSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/lsan.supp")
endif()

# Standalone micro-benchmark for math.hpp ufunc paths. Not run by
# ctest; invoke directly: ./build/tests/bench_math
add_executable(bench_math bench_math.cpp)
target_link_libraries(bench_math PRIVATE np::np)
if(NOT MSVC)
    # -O2 for meaningful timings; -mavx so the np::simd AVX kernels
    # (NP_SIMD_AVX) are compiled and dispatched by the generic
    # add/mul/div_vectorized wrappers.
    target_compile_options(bench_math PRIVATE -O2 -mavx)
endif()

# Hardware benchmark (HBM/tensor/neuromorphic/padic/lattice) — not in ctest
add_executable(bench_hardware bench_hardware.cpp)
target_link_libraries(bench_hardware PRIVATE np::np)
if(NOT MSVC)
    target_compile_options(bench_hardware PRIVATE -O2 -mavx)
endif()
