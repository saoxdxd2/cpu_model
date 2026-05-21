cmake_minimum_required(VERSION 3.20)
project(nca_tests)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(LIBTORCH_LIB_DIR "${TORCH_INSTALL_PREFIX}/lib")

# ── Zero-Maintenance Auto-Discovery Profiler ──────────────────────────────────
find_package(Python3 REQUIRED)
file(GLOB_RECURSE CORE_HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/core/*.hpp")

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_auto_generated.cpp
    COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/autogenerate.py
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/autogenerate.py
        ${CORE_HEADERS}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating C++ auto-profiler..."
)

add_executable(test_auto ${CMAKE_CURRENT_SOURCE_DIR}/test_auto_generated.cpp)
target_include_directories(test_auto PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_auto PRIVATE nca_core "${TORCH_LIBRARIES}")
target_compile_options(test_auto PRIVATE /arch:AVX512 /O2 /fp:fast)

# ── Spectral Logic Tests (v7.0) ───────────────────────────────────────────────
add_executable(test_spectral ${CMAKE_CURRENT_SOURCE_DIR}/test_spectral.cpp)
target_include_directories(test_spectral PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_spectral PRIVATE nca_core "${TORCH_LIBRARIES}")
target_compile_options(test_spectral PRIVATE /arch:AVX512 /O2 /fp:fast)

# ── Robustness Verification ───────────────────────────────────────────────────
add_executable(test_robustness ${CMAKE_CURRENT_SOURCE_DIR}/test_robustness.cpp)
target_include_directories(test_robustness PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_robustness PRIVATE nca_core "${TORCH_LIBRARIES}")
target_compile_options(test_robustness PRIVATE /arch:AVX512 /O2 /fp:fast)

# ── Final Integrated Verification ─────────────────────────────────────────────
add_executable(test_final ${CMAKE_CURRENT_SOURCE_DIR}/test_final.cpp)
target_include_directories(test_final PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_final PRIVATE nca_core "${TORCH_LIBRARIES}")
target_compile_options(test_final PRIVATE /arch:AVX512 /O2 /fp:fast)
