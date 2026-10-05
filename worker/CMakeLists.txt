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
add_library(worker_core STATIC src/image.cpp src/postprocess.cpp)
target_include_directories(worker_core PUBLIC src)
# SYSTEM: silences warnings that come from stb_image.h.
target_include_directories(worker_core SYSTEM PRIVATE third_party)
worker_target_options(worker_core)

# --- C++ stubs generated from proto/ at build time ---------------------------
# The generated files are never committed. protoc and the gRPC plugin come from the
# SAME packages we link against, referenced as CMake targets rather than found on
# $PATH: that is what avoids the anaconda protoc 3.20 vs Homebrew libprotobuf clash.
set(PROTO_ROOT ${CMAKE_SOURCE_DIR}/../proto)
set(GEN_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GEN_DIR})

# Runs protoc for the given .proto files (paths relative to `root`). Sets
# ${out_var} to the generated .cc files. `with_grpc` also generates the service stubs.
function(worker_generate_proto out_var root with_grpc)
  set(srcs)
  set(grpc_args)
  foreach(proto ${ARGN})
    string(REGEX REPLACE "\\.proto$" "" base ${proto})
    list(APPEND srcs ${GEN_DIR}/${base}.pb.cc ${GEN_DIR}/${base}.pb.h)
    if(with_grpc)
      list(APPEND srcs ${GEN_DIR}/${base}.grpc.pb.cc ${GEN_DIR}/${base}.grpc.pb.h)
    endif()
  endforeach()
  if(with_grpc)
    set(grpc_args --grpc_out=${GEN_DIR}
                  --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>)
  endif()
  set(proto_paths)
  foreach(proto ${ARGN})
    list(APPEND proto_paths ${root}/${proto})
  endforeach()
  add_custom_command(
    OUTPUT ${srcs}
    COMMAND $<TARGET_FILE:protobuf::protoc> -I ${root} --cpp_out=${GEN_DIR} ${grpc_args} ${ARGN}
    DEPENDS ${proto_paths} protobuf::protoc
    COMMENT "protoc: ${ARGN}")
  set(${out_var} ${srcs} PARENT_SCOPE)
endfunction()

worker_generate_proto(WORKER_PROTO_SRCS ${PROTO_ROOT} OFF batchserve/v1/detector.proto)
worker_generate_proto(WORKER_GRPC_SRCS ${PROTO_ROOT} ON batchserve/v1/worker.proto)
add_library(worker_proto STATIC ${WORKER_PROTO_SRCS} ${WORKER_GRPC_SRCS})
# SYSTEM: generated headers are not ours, so don't warn about them.
target_include_directories(worker_proto SYSTEM PUBLIC ${GEN_DIR})
target_link_libraries(worker_proto PUBLIC gRPC::grpc++ gRPC::grpc++_reflection protobuf::libprotobuf)

# --- gRPC server -------------------------------------------------------------
add_library(worker_server STATIC src/server.cpp)
target_link_libraries(worker_server PUBLIC worker_proto worker_core)
worker_target_options(worker_server)

# --- ONNX Runtime wrapper ----------------------------------------------------
add_library(worker_detector STATIC src/detector.cpp)
target_include_directories(worker_detector SYSTEM PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(worker_detector PUBLIC worker_core ${ORT_LIBRARY})
worker_target_options(worker_detector)

# --- Executable ----------------------------------------------------------------
add_executable(worker src/main.cpp)
target_link_libraries(worker PRIVATE worker_server worker_detector)
worker_target_options(worker)

# --- Tests -------------------------------------------------------------------
enable_testing()
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# Health-check client stubs for the tests only (see tests/proto/.../health.proto).
worker_generate_proto(HEALTH_GRPC_SRCS ${CMAKE_SOURCE_DIR}/tests/proto ON grpc/health/v1/health.proto)
add_library(health_proto STATIC ${HEALTH_GRPC_SRCS})
target_include_directories(health_proto SYSTEM PUBLIC ${GEN_DIR})
target_link_libraries(health_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)

add_executable(worker_tests
  tests/decode_test.cpp
  tests/letterbox_test.cpp
  tests/nms_test.cpp
  tests/postprocess_test.cpp
  tests/detector_test.cpp
  tests/server_test.cpp)
target_link_libraries(worker_tests PRIVATE worker_detector worker_server health_proto GTest::gtest_main)
# Tests locate models/ and testdata/ through this, so they work from any cwd.
target_compile_definitions(worker_tests PRIVATE WORKER_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
worker_target_options(worker_tests)
gtest_discover_tests(worker_tests)
