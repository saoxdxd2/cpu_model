cmake_minimum_required(VERSION 3.18)
project(nca_training)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
find_package(Torch REQUIRED)

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../nn
    ${TORCH_INCLUDE_DIRS}
)

# NCA Libraries (assuming they are built as part of the parent project)
# We might need to link against nca_core and agentic_env
link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../nn/build/core/Release
    ${CMAKE_CURRENT_SOURCE_DIR}/../agentic_env/build/Release
)

add_executable(offline_trainer offline_trainer.cpp weight_adapter.cpp deep_translator.cpp)
add_executable(fidelity_test fidelity_test.cpp weight_adapter.cpp deep_translator.cpp)

target_link_libraries(offline_trainer 
    nca_core
    agentic_env
    ${TORCH_LIBRARIES}
)

target_link_libraries(fidelity_test
    nca_core
    agentic_env
    ${TORCH_LIBRARIES}
)

if(MSVC)
    target_compile_options(offline_trainer PRIVATE /arch:AVX512 /Oi /Ot)
endif()
