cmake_minimum_required(VERSION 3.28)

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
set(CMAKE_CXX_FLAGS_DEBUG          " -O0 -g -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O2 -g -fno-omit-frame-pointer -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        " -O3                            -DNDEBUG -march=native")
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
string(REGEX REPLACE "  +" " " CXX_COMPILE_OPTIONS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UPPER}}")

# Dependencies
include(FetchContent)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        master
)
FetchContent_MakeAvailable(cxxopts)

# Information
message(STATUS "BUILD TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ COMPILER: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ FLAGS: ${CXX_COMPILE_OPTIONS}")

# CODD
file(GLOB CODD_SRC "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(codd ${CODD_SRC})
target_include_directories(codd PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_include_directories(codd PUBLIC "${PROJECT_SOURCE_DIR}/gfl")

# Executables
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
add_executables("tests" "*.cpp" codd)
add_executables("examples" "*.cpp" codd)

# CUDA
add_executables("examples_gpu" "*.cpp" "codd;cxxopts")