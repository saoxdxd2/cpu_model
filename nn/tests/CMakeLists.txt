# ============================================================================
# NCA — Tests
# tests/CMakeLists.txt
# ============================================================================

# ── Phase 0: Build verification + CPU detection ───────────────────────────
add_executable(test_phase0 test_phase0.cpp)
target_link_libraries(test_phase0 PRIVATE nca_core ${TORCH_LIBRARIES})

if(WIN32)
    # torch_cpu.dll and other dependencies must be in PATH or next to .exe
    add_custom_command(TARGET test_phase0 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env
            "PATH=${NCA_LIBTORCH_DIR}/lib;$ENV{PATH}"
            ${CMAKE_COMMAND} -E echo "LibTorch DLLs available via PATH"
    )
endif()

# ── Phase 1: RMSNorm AMI Test ─────────────────────────────────────────────
add_executable(test_phase1_rmsnorm test_phase1_rmsnorm.cpp)
if(MSVC)
    target_compile_options(test_phase1_rmsnorm PRIVATE /arch:AVX512)
endif()
target_link_libraries(test_phase1_rmsnorm PRIVATE nca_core ${TORCH_LIBRARIES})

# ── Phase 2: Activations AMI Test ─────────────────────────────────────────
add_executable(test_phase2_activations test_phase2_activations.cpp)
if(MSVC)
    target_compile_options(test_phase2_activations PRIVATE /arch:AVX512)
endif()
target_link_libraries(test_phase2_activations PRIVATE nca_core ${TORCH_LIBRARIES})

# ── Phase 3: VNNI MX Math Test ────────────────────────────────────────────
add_executable(test_phase3_vnni test_phase3_vnni.cpp)
if(MSVC)
    target_compile_options(test_phase3_vnni PRIVATE /arch:AVX512)
endif()
target_link_libraries(test_phase3_vnni PRIVATE nca_core ${TORCH_LIBRARIES})

# ── Profiler Tool ─────────────────────────────────────────────────────────
add_executable(test_profiling test_profiling.cpp)
if(MSVC)
    target_compile_options(test_profiling PRIVATE /arch:AVX512)
endif()
target_link_libraries(test_profiling PRIVATE nca_core)

# ── Phase 4: GLR Backbone ─────────────────────────────────────────────────
add_executable(test_phase4_glr test_phase4_glr.cpp)
if(MSVC)
    target_compile_options(test_phase4_glr PRIVATE /arch:AVX512)
endif()
target_link_libraries(test_phase4_glr PRIVATE nca_core)
