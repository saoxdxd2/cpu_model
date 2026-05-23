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

# NCA Libraries (using target names from unified build)
set(NCA_LIBS nca_core agentic_env nca_encoding ${TORCH_LIBRARIES})

add_executable(offline_trainer 
    offline_trainer.cpp 
    weight_adapter.cpp 
    deep_translator.cpp
)
target_link_libraries(offline_trainer PRIVATE ${NCA_LIBS})

add_executable(fidelity_test 
    fidelity_test.cpp 
    weight_adapter.cpp 
    deep_translator.cpp
)
target_link_libraries(fidelity_test PRIVATE ${NCA_LIBS})

add_executable(ai_ide_trainer 
    ai_ide_trainer.cpp 
    code_streamer.cpp 
    weight_adapter.cpp 
    deep_translator.cpp
)
target_link_libraries(ai_ide_trainer PRIVATE ${NCA_LIBS})

add_executable(bit_trainer 
    bit_trainer.cpp 
    code_streamer.cpp 
    weight_adapter.cpp 
    deep_translator.cpp
)
target_link_libraries(bit_trainer PRIVATE ${NCA_LIBS})

add_executable(visual_trainer 
    visual_trainer.cpp 
    weight_adapter.cpp 
    deep_translator.cpp
)
target_link_libraries(visual_trainer PRIVATE ${NCA_LIBS})

add_executable(ppo_trainer_test 
    ppo_trainer.cpp 
    weight_adapter.cpp 
    deep_translator.cpp
)
target_link_libraries(ppo_trainer_test PRIVATE ${NCA_LIBS})

if(MSVC)
    target_compile_options(offline_trainer PRIVATE /arch:AVX512 /Oi /Ot)
    target_compile_options(ai_ide_trainer PRIVATE /arch:AVX512 /Oi /Ot)
    target_compile_options(bit_trainer PRIVATE /arch:AVX512 /Oi /Ot)
    target_compile_options(visual_trainer PRIVATE /arch:AVX512 /Oi /Ot)
    target_compile_options(ppo_trainer_test PRIVATE /arch:AVX512 /Oi /Ot)
endif()
