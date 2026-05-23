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

add_executable(test_gup test_gup.cpp)
target_include_directories(test_gup PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_gup PRIVATE nca_core nca_deploy agentic_env "${TORCH_LIBRARIES}")
target_compile_options(test_gup PRIVATE /arch:AVX512 /O2 /fp:fast)

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

# ── Transformer Killer Benchmark ──────────────────────────────────────────────
add_executable(test_transformer_killer ${CMAKE_CURRENT_SOURCE_DIR}/test_transformer_killer.cpp)
target_include_directories(test_transformer_killer PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_transformer_killer PRIVATE nca_core "${TORCH_LIBRARIES}")
target_compile_options(test_transformer_killer PRIVATE /arch:AVX512 /O2 /fp:fast)

# ── Cognitive Integrity Benchmark ─────────────────────────────────────────────
add_executable(test_cognitive ${CMAKE_CURRENT_SOURCE_DIR}/test_cognitive.cpp)
target_include_directories(test_cognitive PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_cognitive PRIVATE nca_core "${TORCH_LIBRARIES}")
target_compile_options(test_cognitive PRIVATE /arch:AVX512 /O2 /fp:fast)

# ── Throughput & Speed Projection ─────────────────────────────────────────────
add_executable(test_speed ${CMAKE_CURRENT_SOURCE_DIR}/test_speed.cpp)
target_include_directories(test_speed PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_speed PRIVATE nca_core "${TORCH_LIBRARIES}")
target_compile_options(test_speed PRIVATE /arch:AVX512 /O2 /fp:fast)

# ── External Training Bridge Test ─────────────────────────────────────────────
add_executable(test_torch_integration ${CMAKE_CURRENT_SOURCE_DIR}/test_torch_integration.cpp)
target_include_directories(test_torch_integration PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_torch_integration PRIVATE nca_core "${TORCH_LIBRARIES}")
target_compile_options(test_torch_integration PRIVATE /arch:AVX512 /O2 /fp:fast)

add_executable(test_full_response test_full_response.cpp)
target_include_directories(test_full_response PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_full_response PRIVATE nca_core agentic_env "${TORCH_LIBRARIES}")

add_executable(test_visual_reasoning test_visual_reasoning.cpp)
target_include_directories(test_visual_reasoning PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_visual_reasoning PRIVATE nca_core nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_agentic_cli test_agentic_cli.cpp)
target_include_directories(test_agentic_cli PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_agentic_cli PRIVATE nca_core agentic_env "${TORCH_LIBRARIES}")

add_executable(test_silicon_writing test_silicon_writing.cpp)
target_include_directories(test_silicon_writing PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_silicon_writing PRIVATE nca_core agentic_env nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_multi_agent_cost test_multi_agent_cost.cpp)
target_include_directories(test_multi_agent_cost PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_multi_agent_cost PRIVATE nca_core agentic_env nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_silicon_swarm test_silicon_swarm.cpp)
target_include_directories(test_silicon_swarm PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_silicon_swarm PRIVATE nca_core agentic_env nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_intelligence_audit test_intelligence_audit.cpp)
target_include_directories(test_intelligence_audit PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_intelligence_audit PRIVATE nca_core agentic_env nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_memory_compression test_memory_compression.cpp)
target_include_directories(test_memory_compression PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_memory_compression PRIVATE nca_core agentic_env nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_freedom_refactor test_freedom_refactor.cpp)
target_include_directories(test_freedom_refactor PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_freedom_refactor PRIVATE nca_core agentic_env nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_memory_localization test_memory_localization.cpp)
target_include_directories(test_memory_localization PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_memory_localization PRIVATE nca_core agentic_env nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_silicon_ui test_silicon_ui.cpp)
target_include_directories(test_silicon_ui PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_silicon_ui PRIVATE nca_deploy nca_core)

add_executable(test_silicon_indexer test_silicon_indexer.cpp)
target_include_directories(test_silicon_indexer PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_silicon_indexer PRIVATE nca_core agentic_env nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_zero_error_integration test_zero_error_integration.cpp)
target_include_directories(test_zero_error_integration PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_zero_error_integration PRIVATE nca_deploy nca_core nca_encoding agentic_env "${TORCH_LIBRARIES}")

add_executable(test_stability_audit test_stability_audit.cpp)
target_include_directories(test_stability_audit PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_stability_audit PRIVATE nca_core agentic_env nca_encoding "${TORCH_LIBRARIES}")

add_executable(test_final_hardening test_final_hardening.cpp)
target_include_directories(test_final_hardening PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/..)
target_link_libraries(test_final_hardening PRIVATE nca_deploy nca_core nca_encoding agentic_env "${TORCH_LIBRARIES}")

target_compile_options(test_full_response PRIVATE /arch:AVX512 /O2 /fp:fast)
target_compile_options(test_visual_reasoning PRIVATE /arch:AVX512 /O2 /fp:fast)
