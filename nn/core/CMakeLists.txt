# ============================================================================
# NCA — Core math library
# core/CMakeLists.txt
# ============================================================================

add_library(nca_core STATIC
    # ── SIMD dispatch (no special arch flags — just CPUID queries) ──
    simd/dispatch.cpp

    # ── Isolated SIMD Kernels ──
    simd/avx2_kernels.cpp
    simd/avx512_kernels.cpp
    simd/avx512_vnni.cpp

    # ── Core Math ──
    normalization.cpp
    activations.cpp
    
    # ── Linear Algebra ──
    linalg/mx_linear.cpp

    # ── Backbone Layers ──
    layers/glr.cpp
    layers/ssm.cpp
    layers/sla.cpp
    layers/mlp.cpp
    layers/halting.cpp

    # ── Vision Layers ──
    vision/scanner.cpp

    # ── Execution ──
    execution/route_planner.cpp
)

target_include_directories(nca_core PUBLIC
    ${CMAKE_SOURCE_DIR}  # Enables: #include "core/simd/dispatch.hpp"
)

target_link_libraries(nca_core PUBLIC
    ${TORCH_LIBRARIES}
)

# ── SIMD compilation flags for kernel files ──────────────────────────────
if(MSVC)
    # Apply AVX2/AVX512 to the whole core library to avoid TU-mismatches and ensure
    # that the dispatcher always has access to the fast paths.
    target_compile_options(nca_core PRIVATE "/arch:AVX512")
    
    # Specific kernels still keep their flags for clarity, though redundant now
    set_source_files_properties(simd/avx2_kernels.cpp
        PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    set_source_files_properties(simd/avx512_kernels.cpp simd/avx512_vnni.cpp
        PROPERTIES COMPILE_FLAGS "/arch:AVX512")
else()
    target_compile_options(nca_core PRIVATE -mavx512f -mavx512bw -mavx512vl -mavx512vnni)
endif()
