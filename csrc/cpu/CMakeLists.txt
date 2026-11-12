# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.25)
project(FlagOSBgeCPUBridge LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter REQUIRED)
if(NOT Torch_DIR)
  execute_process(COMMAND "${Python_EXECUTABLE}" -c
    "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE _torch_prefix OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  list(APPEND CMAKE_PREFIX_PATH "${_torch_prefix}")
endif()
find_package(Torch REQUIRED)
set(FLAGGEMS_ARM_RAW_INCLUDE "" CACHE PATH "FlagGems Arm tle_raw/csrc directory")
set(FLAGGEMS_ARM_RAW_LIBRARY "" CACHE FILEPATH "Matching I8MM or SME FFN Raw library")
if(NOT EXISTS "${FLAGGEMS_ARM_RAW_INCLUDE}/raw_dispatch.h" OR
   NOT EXISTS "${FLAGGEMS_ARM_RAW_LIBRARY}")
  message(FATAL_ERROR "Provide the matching FlagGems Raw headers and compiled library")
endif()
add_library(vllm_fl_bge_cpu SHARED encoder_attention.cpp)
target_compile_features(vllm_fl_bge_cpu PRIVATE cxx_std_20)
target_include_directories(vllm_fl_bge_cpu PRIVATE
  "${FLAGGEMS_ARM_RAW_INCLUDE}" ${TORCH_INCLUDE_DIRS})
separate_arguments(_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(vllm_fl_bge_cpu PRIVATE ${_torch_flags} -O3 -ffp-contract=off)
target_link_libraries(vllm_fl_bge_cpu PRIVATE ${TORCH_LIBRARIES} "${FLAGGEMS_ARM_RAW_LIBRARY}")
if(APPLE)
  set_target_properties(vllm_fl_bge_cpu PROPERTIES INSTALL_RPATH "@loader_path")
else()
  set_target_properties(vllm_fl_bge_cpu PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
install(TARGETS vllm_fl_bge_cpu LIBRARY DESTINATION vllm_fl/lib)
