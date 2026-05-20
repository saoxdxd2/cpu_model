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
    # AVX-512 intrinsic usage requires the architecture flag in MSVC 
    # to guarantee __m512 registers are emitted properly.
    target_compile_options(test_phase1_rmsnorm PRIVATE /arch:AVX512)
endif()
target_link_libraries(test_phase1_rmsnorm PRIVATE nca_core ${TORCH_LIBRARIES})
