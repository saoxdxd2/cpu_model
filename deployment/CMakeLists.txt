cmake_minimum_required(VERSION 3.18)
project(nca_deployment)

set(CMAKE_CXX_STANDARD 20)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../nn
)

add_library(nca_deploy SHARED
    nca_deploy.cpp
)
target_compile_definitions(nca_deploy PRIVATE NCA_DEPLOY_EXPORT)

target_link_libraries(nca_deploy nca_core)

if(MSVC)
    target_link_libraries(nca_deploy user32.lib gdi32.lib ws2_32.lib)
    target_compile_options(nca_deploy PRIVATE /arch:AVX512 /Oi /Ot)
endif()
