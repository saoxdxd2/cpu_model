cmake_minimum_required(VERSION 3.20)
project(nca_tests)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Provide LibTorch library directory
set(LIBTORCH_LIB_DIR "${TORCH_INSTALL_PREFIX}/lib")

# ── Auto-Detect Profiler Suite ────────────────────────────────────────────────
add_executable(test_auto test_auto.cpp)
target_include_directories(test_auto PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_auto PRIVATE nca_core "${TORCH_LIBRARIES}")
target_compile_options(test_auto PRIVATE /arch:AVX512 /O2 /fp:fast)
# ─────────────────────────────────────────────────────────────────────────────
