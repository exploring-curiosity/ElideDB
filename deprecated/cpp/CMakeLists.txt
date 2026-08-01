cmake_minimum_required(VERSION 3.24)
project(streetdex VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  # Default to a build that is honest for benchmarks but still debuggable.
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

option(STREETDEX_BUILD_TESTS "Build Catch2 tests" ON)
option(STREETDEX_BUILD_PYTHON "Build pybind11 bindings" OFF)

# ---- Dependencies -----------------------------------------------------------
include(FetchContent)

# FFmpeg linked directly (not shelled out): we need seek-to-keyframe and
# decode-only-these-bytes discipline, which the CLI cannot express.
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat libavcodec libavutil libswscale)

# tl::expected — C++20 has no std::expected; error handling is expected<T,E>
# everywhere except the CLI boundary (per coding standards).
FetchContent_Declare(tl_expected
  GIT_REPOSITORY https://github.com/TartanLlama/expected.git
  GIT_TAG v1.1.0
  GIT_SHALLOW TRUE)
set(EXPECTED_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tl_expected)

# nlohmann/json — ingest + manifest only; never on the query hot path.
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)
FetchContent_MakeAvailable(nlohmann_json)

if(STREETDEX_BUILD_TESTS)
  FetchContent_Declare(catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(catch2)
endif()

# ---- libstreetdex -----------------------------------------------------------
add_library(streetdex STATIC
  src/streetdex/core/mmap_file.cpp
  src/streetdex/metrics/metrics.cpp
  src/streetdex/format/sdx_writer.cpp
  src/streetdex/format/sdx_reader.cpp
  src/streetdex/video/sfi_builder.cpp
  src/streetdex/video/sfi_reader.cpp
  src/streetdex/video/decoder.cpp
  src/streetdex/ingest/time_json.cpp
  src/streetdex/ingest/clock_model.cpp
  src/streetdex/ingest/wav.cpp
  src/streetdex/ingest/dataset.cpp
  src/streetdex/catalog/manifest.cpp
  src/streetdex/align/resample.cpp
  src/streetdex/query/planner.cpp
  src/streetdex/index/vector_index.cpp
  src/streetdex/index/bptree.cpp
  src/streetdex/query/semantic.cpp
)
target_include_directories(streetdex PUBLIC src)
target_compile_options(streetdex PRIVATE -Wall -Wextra -Werror)
# Accelerate's modern CBLAS headers (the pre-13.3 interface is deprecated).
target_compile_definitions(streetdex PRIVATE ACCELERATE_NEW_LAPACK)
target_link_libraries(streetdex
  PUBLIC tl::expected nlohmann_json::nlohmann_json PkgConfig::FFMPEG
  PUBLIC "-framework Accelerate")

# ---- CLI --------------------------------------------------------------------
add_executable(sdx src/cli/main.cpp)
target_compile_options(sdx PRIVATE -Wall -Wextra -Werror)
target_link_libraries(sdx PRIVATE streetdex)

# ---- Tests ------------------------------------------------------------------
if(STREETDEX_BUILD_TESTS)
  add_executable(streetdex_tests
    tests/test_smoke.cpp
    tests/test_sdx_format.cpp
    tests/test_sdx_property.cpp
    tests/test_time_json.cpp
    tests/test_clock_model.cpp
    tests/test_sfi.cpp
    tests/test_align.cpp
    tests/test_wav.cpp
    tests/test_planner.cpp
    tests/test_vector_index.cpp
    tests/test_bptree.cpp
  )
  target_compile_options(streetdex_tests PRIVATE -Wall -Wextra -Werror)
  target_link_libraries(streetdex_tests PRIVATE streetdex Catch2::Catch2WithMain)
  enable_testing()
  add_test(NAME unit COMMAND streetdex_tests)
endif()

# ---- pybind11 bindings (optional) ------------------------------------------
if(STREETDEX_BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(pystreetdex bindings/pystreetdex.cpp)
  target_link_libraries(pystreetdex PRIVATE streetdex)
endif()
