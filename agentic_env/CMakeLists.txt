cmake_minimum_required(VERSION 3.20)
project(nca_agentic_env)

# Enforce C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force high performance flags depending on compiler
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX2 /fp:fast")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -ffast-math")
endif()

add_library(agentic_env STATIC
    vec_env.cpp
    replay_memory.cpp
    tactical_env.cpp
    sim_loop.cpp
    vscode_env.cpp
)
target_include_directories(agentic_env PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../nn
)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE agentic_env)

add_executable(test_tactical test_tactical.cpp)
target_link_libraries(test_tactical PRIVATE agentic_env)
