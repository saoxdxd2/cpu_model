cmake_minimum_required(VERSION 3.20)
project(NCA_Project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Global Dependencies (LibTorch)
set(NCA_LIBTORCH_DIR "C:/libtorch" CACHE PATH "Path to LibTorch")
list(APPEND CMAKE_PREFIX_PATH "${NCA_LIBTORCH_DIR}")
find_package(Torch REQUIRED)

# 2. Sub-Modules
# Note: We build in a specific order to satisfy dependencies
add_subdirectory(encoding)
add_subdirectory(nn)
add_subdirectory(agentic_env)
add_subdirectory(training)
add_subdirectory(deployment)

# 3. Global Configuration
if(MSVC)
    add_compile_options(/W4 /arch:AVX512 /Oi /Ot /Ox /GS-)
endif()
