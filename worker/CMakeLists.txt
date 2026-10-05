cmake_minimum_required(VERSION 3.20)
project(batchserve_worker CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Lets editors (clangd) find include paths: build/compile_commands.json.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(SANITIZE "Build with AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
# CUDA execution provider. Written but UNTESTED: no GPU was available.
option(WITH_CUDA "Compile the --provider=cuda code path" OFF)

# --- Find dependencies, Homebrew first ---------------------------------------
# find_package searches CMAKE_PREFIX_PATH before it looks at prefixes derived from
# $PATH. On this machine $PATH contains /opt/anaconda3/bin, whose protobuf (3.20)
# would otherwise win and clash with Homebrew's libprotobuf. Putting Homebrew
# first makes protoc and libprotobuf come from the same package.
find_program(BREW_EXECUTABLE brew)
if(BREW_EXECUTABLE)
  execute_process(COMMAND ${BREW_EXECUTABLE} --prefix
                  OUTPUT_VARIABLE BREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(PREPEND CMAKE_PREFIX_PATH ${BREW_PREFIX})
endif()

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
message(STATUS "protobuf ${Protobuf_VERSION} from ${Protobuf_DIR}")
message(STATUS "gRPC ${gRPC_VERSION} from ${gRPC_DIR}")

# ONNX Runtime has no CMake config in the official tarballs (used in Docker), so
# look for the header and library by hand. ONNXRUNTIME_ROOT overrides the default.
if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
  set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()
if(NOT ONNXRUNTIME_ROOT AND BREW_EXECUTABLE)
  execute_process(COMMAND ${BREW_EXECUTABLE} --prefix onnxruntime
                  OUTPUT_VARIABLE ONNXRUNTIME_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
endif()
# Homebrew puts the header in include/onnxruntime/, the tarball in include/.
find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS ${ONNXRUNTIME_ROOT} PATH_SUFFIXES include include/onnxruntime)
find_library(ORT_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT} PATH_SUFFIXES lib)
if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found. Run `brew install onnxruntime` or "
                      "pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()
message(STATUS "ONNX Runtime: ${ORT_LIBRARY}")

# --- Our own code gets strict warnings; third-party code does not ------------
function(worker_target_options target)
  target_compile_options(${target} PRIVATE -Wall -Wextra)
  if(SANITIZE)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()

# --- Pure code: no ONNX Runtime, no gRPC -------------------------------------
# Decode, letterbox, NMS, postprocess. Kept separate so its tests build and run fast.
add_library(worker_core STATIC src/image.cpp)
target_include_directories(worker_core PUBLIC src)
# SYSTEM: silences warnings that come from stb_image.h.
target_include_directories(worker_core SYSTEM PRIVATE third_party)
worker_target_options(worker_core)

# --- Tests -------------------------------------------------------------------
enable_testing()
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

add_executable(worker_tests
  tests/decode_test.cpp
  tests/letterbox_test.cpp)
target_link_libraries(worker_tests PRIVATE worker_core GTest::gtest_main)
# Tests locate models/ and testdata/ through this, so they work from any cwd.
target_compile_definitions(worker_tests PRIVATE WORKER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
worker_target_options(worker_tests)
gtest_discover_tests(worker_tests)
