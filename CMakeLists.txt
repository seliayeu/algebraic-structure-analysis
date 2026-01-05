cmake_minimum_required(VERSION 3.20.0)

project(algebraic-structure-analysis LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
# set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
# set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
# set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
# set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")

include(AddLLVM)
include(TableGen)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/externals/llvm-project)
include_directories(${PROJECT_BINARY_DIR})

add_subdirectory(tools)
add_subdirectory(lib)
