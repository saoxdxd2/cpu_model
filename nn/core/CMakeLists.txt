# ============================================================================
# NCA — Core math library
# core/CMakeLists.txt
# ============================================================================

add_library(nca_core STATIC
    normalization.cpp
    activations.cpp
    
    # ── Backbone Layers ──
    layers/glr.cpp
    layers/ssm.cpp
    layers/sla.cpp
    layers/halting.cpp

    # ── Vision Layers ──
    vision/scanner.cpp
    vision/spectral_pruner.cpp

    # ── Spectral Logic (v7.0) ──
    spectral/fwht.cpp
    spectral/kronecker_rls.cpp
    spectral/spectral_logic.cpp

    # ── Execution ──
    execution/route_planner.cpp
    execution/importance.cpp
    execution/multimodal_engine.cpp
    execution/silicon_memory.cpp
    execution/silicon_automation.cpp
    execution/wavefront_router.cpp

    # ── Linalg ──

    # ── SIMD Kernels ──
    simd/dispatch.cpp
    simd/avx2_kernels.cpp
    simd/avx512_kernels.cpp
)

target_include_directories(nca_core PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(nca_core PUBLIC
    ${TORCH_LIBRARIES}
    nca_encoding
)

if(MSVC)
    target_compile_options(nca_core PRIVATE "/arch:AVX512")
    set_source_files_properties(simd/avx2_kernels.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
    target_compile_options(nca_core PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512vnni)
endif()
