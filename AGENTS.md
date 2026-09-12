#AGENTS.md — Modern C++ Directives

Instructions for any AI agent (or human) writing, reviewing, or refactoring C++ code in this repository. Follow these rules by default; deviate only when the codebase's existing conventions clearly require it, and say so.


## 1. Language Standard


- Target **C++20** by default. Use **C++23** features only if the project's `CMakeLists.txt` / toolchain already sets `CXX_STANDARD 23` and the compiler supports it (GCC 13+, Clang 16+, MSVC 19.34+).
- Never write C++98/03-style code (no raw `typedef`, no C-style casts, no `NULL`).
- Prefer standard library facilities over hand-rolled equivalents or third-party utility libraries unless the project already depends on one (e.g. Boost, Abseil, fmt).


## 2. Memory & Resource Management


- **No raw `new` / `delete`.** Use RAII everywhere.
- Ownership:
  - `std::unique_ptr<T>` — default choice for owned, exclusive resources.
  - `std::shared_ptr<T>` — only when shared ownership is a real requirement (not a substitute for design).
  - `std::weak_ptr<T>` — to break cycles / observe without owning.
  - Prefer value semantics and containers over pointers whenever lifetime doesn't need to outlive the enclosing scope.
- Non-owning references: pass `T&`, `const T&`, or `std::span<T>` / `std::string_view` instead of raw pointers where possible.
- No manual `malloc`/`free` in new code.
- Every resource (file handle, mutex lock, socket, GPU handle) must be wrapped in an RAII type.


## 3. Types & Correctness


- `const`-correct by default: mark variables, parameters, and member functions `const` unless mutation is required.
- Prefer `constexpr` over `const` for compile-time-known values; prefer `consteval`/`constinit` (C++20) where semantics call for it.
- Use `auto` for local variable deduction when it improves readability (iterators, lambdas, long template types);
do **not** use `auto` where it hides an important type at a glance (e.g. public API return types, numeric conversions).
- Use scoped `enum class`, never unscoped `enum`.
- Use `std::optional<T>` instead of sentinel values (`-1`, `nullptr`, magic strings) to represent "no value."
- Use `std::variant<...>` + `std::visit` instead of manual tagged unions or inheritance-based polymorphism when the set of alternatives is closed.
- Prefer `std::span<T>` over `(pointer, size)` pairs, and `std::string_view` over `const std::string&` for read-only string params.
- Use structured bindings (`auto [a, b] = ...`) instead of `.first`/`.second` or manual unpacking.
- Use Concepts (C++20) to constrain templates instead of SFINAE or unconstrained templates with unclear requirements.


## 4. Error Handling


- Use exceptions for truly exceptional, unrecoverable-at-the-call-site errors.
- For expected, recoverable failure paths (parse errors, "not found," validation), prefer `std::expected<T, E>` (C++23) or a project-standard `Result<T, E>` type over exceptions or error codes.
- Never use error codes returned through output parameters in new code.
- Never swallow exceptions silently (`catch (...) {}` with no action is forbidden). Log or rethrow.
- Mark functions `noexcept` when they genuinely cannot throw — this affects optimization and container behavior (e.g. `std::vector` move semantics).


## 5. Functions, Classes & API Design


- Follow the **Rule of Zero**: don't declare special member functions unless the class manages a resource directly. Let RAII members handle it.
- If you must manage a resource directly, follow the **Rule of Five** (or `= delete` copy/move explicitly).
- Mark single-argument constructors `explicit` unless implicit conversion is intentional and documented.
- Prefer free functions over static member functions for stateless utilities.
- Prefer composition over inheritance. Use inheritance only for genuine "is-a" polymorphism with virtual dispatch;
mark base destructors `virtual` (or classes `final`).- Pass small trivially
    - copyable types by value;
pass large / non - trivial types by `const&`; use `&&` for sink parameters that will be moved-from.
- Return by value and rely on RVO/move semantics — don't manually optimize with output parameters unless profiling proves it necessary.


## 6. Modules, Headers & Build


- If the toolchain supports C++20 modules and the project has adopted them, prefer modules for new components. Otherwise use traditional headers with `#pragma once`.
- Keep headers minimal: forward-declare where possible, include only what you use (IWYU principle).
- One class/component per translation unit pair (`.hpp`/`.cpp`) unless components are tightly coupled and small.
- Use namespaces to scope project code;
avoid `using namespace std;` in headers (acceptable, sparingly, inside `.cpp` function scope only).


## 7. Concurrency


- Prefer `std::jthread` (C++20) over `std::thread` — it joins automatically and supports cooperative cancellation via `std::stop_token`.
- Protect shared state with RAII locks (`std::lock_guard`, `std::scoped_lock`, `std::unique_lock`) — never call `mutex.lock()`/`unlock()` manually.
- Prefer message-passing / task-based designs (queues, futures, `std::async`) over shared mutable state where feasible.
- Use `std::atomic<T>` for lock-free simple shared counters/flags; don't hand-roll atomics with volatile.


## 8. Algorithms & Containers


- Prefer `<algorithm>` and `<ranges>` (C++20) over hand-written loops: `std::ranges::sort`, `std::ranges::find`, range-based pipelines with `|` views.
- Prefer range-based `for` loops over index-based loops unless the index itself is needed.
- Choose containers deliberately: `std::vector` by default, `std::array` for fixed-size stack data, `std::unordered_map`/`std::map` based on ordering needs, `std::deque` only when front/back growth is required.
- Reserve capacity (`.reserve()`) when the final size is known ahead of a loop that grows a container.


## 9. Formatting & Style


- Format with **clang-format**; commit a `.clang-format` file (LLVM or Google base style, project's choice) and never hand-format against it.
- Naming: pick one convention and apply it consistently (e.g. `snake_case` for variables/functions, `PascalCase` for types, `SCREAMING_SNAKE_CASE` for macros/constants). Match whatever the existing codebase already uses — don't introduce a second convention.
- Keep functions short and single-purpose; extract helpers rather than nesting deeply.
- Avoid macros for anything expressible as a `constexpr` function, template, or `enum class`. Reserve macros for conditional compilation and header guards only.


## 10. Tooling & Static Analysis


Run before considering any change complete:


- `clang-format` — formatting.
- `clang-tidy` — static analysis (enable at minimum: `bugprone-*`, `modernize-*`, `performance-*`, `cppcoreguidelines-*`).
- **Sanitizers** in debug/test builds: `-fsanitize=address,undefined` (ASan+UBSan) at minimum; TSan for concurrent code.
- Treat compiler warnings as errors in CI (`-Wall -Wextra -Wpedantic -Werror` on GCC/Clang, `/W4 /WX` on MSVC).
- Prefer CMake as the build system, with `FetchContent` or a package manager (vcpkg/Conan) for dependencies rather than vendoring or system-wide installs.


## 11. Testing


- Every new function/class with non-trivial logic gets a unit test.
- Use an established framework already in the project (GoogleTest, Catch2, doctest); don't introduce a second framework.
- Tests must be deterministic and independent — no shared mutable global state between tests.
- Prefer property-style/table-driven tests for functions with many input classes.


## 12. What to Avoid


- Raw `new`/`delete`, C-style arrays for dynamic data, C-style casts (`(int)x`) — use `static_cast`/`dynamic_cast`/`reinterpret_cast`/`const_cast` explicitly.
- `using namespace std;` at file/header scope.
- Output parameters as the primary way to return data.
- Deep inheritance hierarchies and multiple inheritance (except interface-only mixins).
- Global mutable state.
- Manual index math where an iterator, range, or `std::span` would do.
- Silent narrowing conversions — use `{}`-initialization, which errors on narrowing.


## 13. Commit Hygiene


- Keep commits scoped to one logical change.
- Run formatter + linter + tests before proposing a change as done.
- Document *why*, not *what*, in comments — the code should already say what it does.
