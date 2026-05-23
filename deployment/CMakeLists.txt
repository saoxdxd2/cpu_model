cmake_minimum_required(VERSION 3.18)
project(nca_deployment)

set(CMAKE_CXX_STANDARD 20)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../nn
)

add_library(nca_deploy SHARED
    nca_deploy.cpp
    peripheral_bridge.cpp
    creative_bridge.cpp
    aether_socket.cpp
)
target_compile_definitions(nca_deploy PRIVATE NCA_DEPLOY_EXPORT)

add_executable(test_deployment test_deployment.cpp)
target_link_libraries(test_deployment nca_deploy)

add_executable(test_chatbot test_chatbot.cpp)
target_link_libraries(test_chatbot nca_deploy nca_encoding)

add_executable(test_alphabet test_alphabet.cpp)
target_link_libraries(test_alphabet nca_deploy nca_encoding)

add_executable(test_ide_agent test_ide_agent.cpp)
target_link_libraries(test_ide_agent nca_deploy nca_encoding)

add_executable(test_creative_gen test_creative_gen.cpp)
target_link_libraries(test_creative_gen nca_deploy)

add_executable(test_autonomous_dev test_autonomous_dev.cpp)
target_link_libraries(test_autonomous_dev nca_deploy nca_encoding agentic_env)

add_executable(aether_gateway aether_gateway.cpp peripheral_bridge.cpp)
target_link_libraries(aether_gateway nca_deploy nca_encoding)

target_link_libraries(nca_deploy nca_core)

if(MSVC)
    target_link_libraries(nca_deploy user32.lib gdi32.lib ws2_32.lib)
    target_compile_options(nca_deploy PRIVATE /arch:AVX512 /Oi /Ot)
endif()
