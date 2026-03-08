cmake_minimum_required (VERSION 3.10)

project(RaBitQLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "-Wall -Ofast -Wextra -march=native -fpic -fopenmp -ftree-vectorize -fexceptions")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt")
endif()

add_subdirectory(sample)

add_library(rabitq_headers INTERFACE)

target_include_directories(rabitq_headers INTERFACE ${PROJECT_SOURCE_DIR}/include)

# Testing
option(RABITQ_BUILD_TESTS "Build tests" OFF)

if(RABITQ_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
