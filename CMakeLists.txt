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
set(CMAKE_CXX_FLAGS_DEBUG          " -O0 -g -fno-omit-frame-pointer -fconcepts-diagnostics-depth=10")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " -O2 -g -fno-omit-frame-pointer -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE        " -O3                            -DNDEBUG -march=native")
string(TOUPPER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_UPPER)
string(REGEX REPLACE "  +" " " CXX_COMPILE_OPTIONS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${BUILD_TYPE_UPPER}}")

# CUDA
option(ENABLE_GPU "Enable GPU support" OFF)
if (ENABLE_GPU)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_FLAGS " --extended-lambda")
    set(CMAKE_CUDA_FLAGS_DEBUG          " -O0 -g -G")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -O2    -lineinfo -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE        " -O3              -DNDEBUG")
    # Forward C++ options
    separate_arguments(CXX_COMPILE_OPTIONS_TMP UNIX_COMMAND "${CXX_COMPILE_OPTIONS}")
    list(JOIN CXX_COMPILE_OPTIONS_TMP "," CXX_COMPILE_OPTIONS_TMP)
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=${CXX_COMPILE_OPTIONS_TMP}")
    string(REGEX REPLACE "  +" " " CUDA_COMPILE_OPTIONS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${BUILD_TYPE_UPPER}}")
    string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress  68")
    string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 186")
    string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 830")
    string(APPEND CMAKE_CUDA_FLAGS " --diag-suppress 550")
endif()

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
if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA COMPILER: ${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION} with host compiler ${CMAKE_CUDA_HOST_COMPILER_ID} ${CMAKE_CUDA_HOST_COMPILER_VERSION}")
    message(STATUS "CUDA ARCHITECTURE: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA COMPILER FLAGS: ${CUDA_COMPILE_OPTIONS}")
endif()

# CODD
file(GLOB CODD_SRC "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_library(codd ${CODD_SRC})
target_include_directories(codd PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_include_directories(codd PUBLIC "${PROJECT_SOURCE_DIR}/gfl")
if(CMAKE_CUDA_COMPILER)
    target_link_libraries(codd PRIVATE CUDA::cudart)
    target_include_directories(codd PUBLIC "${PROJECT_SOURCE_DIR}/src_gpu")
    set_source_files_properties("src/dd.cpp" PROPERTIES LANGUAGE CUDA)
    set_source_files_properties("src/searchRelaxedFirst.cpp" PROPERTIES LANGUAGE CUDA)
    set_source_files_properties("src/searchRestrictedFirst.cpp" PROPERTIES LANGUAGE CUDA)
    set_source_files_properties("src/searchRestrictedFirstThreaded.cpp" PROPERTIES LANGUAGE CUDA)
    set_source_files_properties("src/searchRestrictedOnly.cpp" PROPERTIES LANGUAGE CUDA)
    set_source_files_properties("src/searchRestrictedOnlyNoQ.cpp" PROPERTIES LANGUAGE CUDA)
endif()

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
if(CMAKE_CUDA_COMPILER)
    add_executables("examples_gpu" "*.cu" "codd;cxxopts")
endif()