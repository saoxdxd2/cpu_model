cmake_minimum_required(VERSION 3.18)
project(nca_encoding)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../nn
)

add_library(nca_encoding STATIC
    state_encoder.cpp
    vision_encoder.cpp
    tokenizer.cpp
    saliency_writer.cpp
    memory_probe.cpp
)

add_executable(test_encoding test_encoding.cpp)
target_link_libraries(test_encoding nca_encoding)

if(MSVC)
    target_compile_options(nca_encoding PRIVATE /arch:AVX512 /Oi /Ot /W4)
endif()
