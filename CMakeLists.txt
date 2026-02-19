cmake_minimum_required(VERSION 3.28) # Versions on Ubuntu LTS

# Combinatorial Optimization with Decision Diagrams
project(CODD VERSION 1.0 LANGUAGES CXX)

# Compilation
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# C++
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS " -Wall")
set(CMAKE_CXX_FLAGS_DEBUG          " -O0 -g -fno-omit-frame-pointer -fconcepts-diagnostics-depth=10")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O2 -g -fno-omit-frame-pointer -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELEASE        " -O3 -DNDEBUG -march=native")

# Dependencies
include(FetchContent)
FetchContent_Declare(
        cxxopts
        URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.3.1.tar.gz
)
FetchContent_Declare(
        fmt
        URL https://github.com/fmtlib/fmt/archive/refs/tags/12.1.0.tar.gz
)
FetchContent_MakeAvailable(cxxopts fmt)

# Core sources
include_directories(src)

# GPU Friendly Library
include_directories(gfl)

# Information
message(STATUS "BUILD TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ COMPILER: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ FLAGS: ${CXX_COMPILE_OPTIONS}")

# Auxiliary functions
function(add_executables DIR FILTER LIBS)
    string(TOUPPER "${DIR}" DIR_UPPER)
    set(DIR_MSG "${DIR_UPPER}: ")
    file(GLOB DIR_SRC "${PROJECT_SOURCE_DIR}/${DIR}/${FILTER}")
    foreach (EXE_SRC ${DIR_SRC})
        get_filename_component(EXE_NAME ${EXE_SRC} NAME_WE)
        set(DIR_MSG "${DIR_MSG} ${EXE_NAME}")
        add_executable(${EXE_NAME} ${EXE_SRC})
        foreach (LIB ${LIBS})
            target_link_libraries(${EXE_NAME} ${LIB})
        endforeach()
    endforeach()
    message(STATUS ${DIR_MSG})
endfunction()

# C++
add_executables("examples" "*.cpp" "cxxopts;fmt::fmt")