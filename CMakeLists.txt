cmake_minimum_required(VERSION 3.26.4)

project(modern-osqp-cpp
  LANGUAGES
    C
    CXX
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_FLAGS_DEBUG "-Og -g")
set(CMAKE_CXX_FLAGS_DEBUG "-Og -g")

set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")

set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

set(CMAKE_COMPILE_WARNING_AS_ERROR True)

add_compile_options(
    -pipe
    -fno-plt
    -fexceptions
    -fstack-clash-protection
    -fcf-protection
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

set(CPM_SOURCE_CACHE third_party)
set(CPM_USE_LOCAL_PACKAGES True)

add_compile_definitions(
    SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE
    DOCTEST_CONFIG_SUPER_FAST_ASSERTS
)

include(CTest)
enable_testing()

include(cmake/CPM.cmake)
include(cmake/third_party.cmake)
include(cmake/utils.cmake)

include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)
add_subdirectory(tests)
