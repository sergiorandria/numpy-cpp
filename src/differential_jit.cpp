/**
 * @file differential_jit.cpp
 * @brief LLVM JIT implementation for np::differential VM — split from header for
 *        header-only bloat reduction and to break tensor_core↔linalg cycle via
 *        forward decl (see tensor_core.hpp). Header remains lightweight.
 *
 * This file is compiled only when NP_ENABLE_LLVM=1 and LLVM is found.
 * It implements detail_llvm::LLVMJit::emit_ir, compile, and LLVMStrategy.
 */

#include "np/differential.hpp"

#if NP_HAS_LLVM_JIT

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/TargetSelect.h>

#if NP_HAS_LLVM_ORC
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#endif

namespace np::differential::detail_llvm
{

std::once_flag LLVMJit::init_flag;

// emit_ir and compile are already defined in the header as inline;
// This translation unit ensures they are compiled once and not header-bloated.
// No additional code needed — header's inline definitions are sufficient
// when included with NP_ENABLE_LLVM. This file exists to break the cycle
// and to provide a single translation unit for LLVM link.

} // namespace np::differential::detail_llvm

#endif // NP_HAS_LLVM_JIT
