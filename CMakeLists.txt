cmake_minimum_required(VERSION 3.8)

project(C++DDOpt VERSION 1.0)

# ------------------------------------------------------------------------
# Let's go for Debug by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# ------------------------------------------------------------------------
set(CMAKE_VERBOSE_MAKEFILE False)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(USE_CXX_EXCEPTIONS "Enable C++ exception support" ON)

#set(CMAKE_CXX_COMPILER "g++-14")
set(CMAKE_CXX_COMPILER "c++")
set(CMAKE_CXX_FLAGS "-Wall -Wsign-compare")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -D_NDEBUG -DNDEBUG -Wno-unused-lambda-capture")  #[[ comment  ]]

message("FR   : ${CMAKE_CXX_FLAGS_RELEASE}")
message("Build: ${CMAKE_BUILD_TYPE}")

#This is the full list of exapmles for miniCPP		
################################################################################
file(GLOB libs "src/*.cpp")
message("Lib: ${libs}")
add_library(ddOpt
  ${libs}
)

include_directories(src)
file(GLOB files "*.cpp")
message("${files}")

# ----------------------------------------------------------------------
# Library

foreach (file ${files})
	get_filename_component(base ${file} NAME_WE)
	get_filename_component(dir ${file} DIRECTORY)
#	message("BASE    IS: [${base}]")
	message("HEADERS IS: [${PROJECT_SOURCE_DIR}/src]")
	target_include_directories(${base} PRIVATE ${PROJECT_SOURCE_DIR}/src)
	add_executable(${base} ${dir}/${base}.cpp)
	target_link_libraries(${base} ddOpt)
	add_dependencies(${base} ddOpt)
endforeach()


message("------------------------------------------------------------")
file(GLOB tests "test/*cpp")
message("Tests: ${tests}")

foreach (file ${tests})
	get_filename_component(base ${file} NAME_WE)
	get_filename_component(dir ${file} DIRECTORY)
	message("BASE IS: [${base}]")
	message("HEADERS IS: [${PROJECT_SOURCE_DIR}/src]")
	message("ARG: [$dir/${base}.cpp]")
	add_executable(${base} ${dir}/${base}.cpp)
	target_include_directories(${base} PRIVATE ${PROJECT_SOURCE_DIR}/src)
	target_link_libraries(${base} ddOpt)
	add_dependencies(${base} ddOpt)  
endforeach()

message("------------------------------------------------------------")
file(GLOB examples "examples/*cpp")
message("Examples: ${examples}")

foreach (file ${examples})
	get_filename_component(base ${file} NAME_WE)
	get_filename_component(dir ${file} DIRECTORY)
#	message("BASE IS: [${base}]")
	add_executable(${base} ${dir}/${base}.cpp)
	target_include_directories(${base} PRIVATE ${PROJECT_SOURCE_DIR}/src)
	target_link_libraries(${base} ddOpt)
	add_dependencies(${base} ddOpt)  
endforeach()

