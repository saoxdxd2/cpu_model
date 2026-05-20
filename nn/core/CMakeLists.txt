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

    # ── Core Math ──
    normalization.cpp
    
    # Future phases will add:
    # linalg/gemm.cpp
    # linalg/sparse_gemm.cpp
    # linalg/tucker_decomp.cpp
    # activations.cpp
)

target_include_directories(nca_core PUBLIC
    ${CMAKE_SOURCE_DIR}  # Enables: #include "core/simd/dispatch.hpp"
)

target_link_libraries(nca_core PUBLIC
    ${TORCH_LIBRARIES}
)

# ── SIMD compilation flags for kernel files ──────────────────────────────
if(MSVC)
    set_source_files_properties(simd/avx2_kernels.cpp
        PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    set_source_files_properties(simd/avx512_kernels.cpp
        PROPERTIES COMPILE_FLAGS "/arch:AVX512")
endif()
