# ============================================================================
# NCA — Nano-Core Architecture
# Root CMakeLists.txt
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(NCA VERSION 0.1.0 LANGUAGES CXX)

# ── C++20 (MANDATORY — LibTorch may try to inject C++17, we override) ──────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── LibTorch ───────────────────────────────────────────────────────────────
set(NCA_LIBTORCH_DIR "C:/libtorch" CACHE PATH "Path to LibTorch installation")
list(APPEND CMAKE_PREFIX_PATH "${NCA_LIBTORCH_DIR}")
find_package(Torch REQUIRED)

# LibTorch's TorchConfig.cmake may inject /std:c++17 into CMAKE_CXX_FLAGS.
# Strip it — we enforce C++20.
string(REPLACE "/std:c++17" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-std=c++17" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# Re-assert after stripping
set(CMAKE_CXX_STANDARD 20)

# ── MSVC Flags ─────────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(
        /W4             # High warning level
        /permissive-    # Strict standards conformance
        /Zc:__cplusplus # Report correct __cplusplus value (202002L for C++20)
        /Zc:preprocessor # Standards-conforming preprocessor
        /utf-8          # Source and execution charset
        /EHsc           # Standard C++ exception handling
    )
    # Disable noisy warnings that flood with LibTorch headers
    add_compile_options(
        /wd4244  # conversion from 'type1' to 'type2', possible loss of data
        /wd4267  # conversion from 'size_t' to 'type', possible loss of data
        /wd4996  # deprecated declarations
        /wd4100  # unreferenced formal parameter (common in interfaces)
    )
endif()

# ── Global include path ────────────────────────────────────────────────────
# All headers are referenced relative to nn/ root:
#   #include "core/simd/dispatch.hpp"
#   #include "config/model_config.hpp"
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ── Subdirectories ─────────────────────────────────────────────────────────
add_subdirectory(core)
add_subdirectory(tests)

# ── Summary ────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════╗")
message(STATUS "║  NCA — Nano-Core Architecture v${PROJECT_VERSION}     ║")
message(STATUS "╠══════════════════════════════════════════╣")
message(STATUS "║  C++ Standard : ${CMAKE_CXX_STANDARD}                      ║")
message(STATUS "║  LibTorch     : ${TORCH_INSTALL_PREFIX}")
message(STATUS "║  Compiler     : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "╚══════════════════════════════════════════╝")
message(STATUS "")
