cmake_minimum_required(VERSION 3.5)

project(pacxx-gstorm-eval)
set(PACXX_DIR $ENV{PACXX_PATH})
#set(CMAKE_MODULE_PATH ${PACXX_DIR}/lib/cmake/pacxx)
set(CMAKE_CXX_STANDARD 14)

set(GSTORM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gstorm/" CACHE PATH "Path to gstom")
set(RANGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/range-v3/" CACHE PATH "Path to range-v3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wno-invalid-noreturn")

option(PACXX_EXPERIMENTAL_BACKEND "Use the experimental CPU backend" OFF)

if (PACXX_EXPERIMENTAL_BACKEND)
    add_compile_options(-DUSE_EXPERIMENTAL_BACKEND)
endif(PACXX_EXPERIMENTAL_BACKEND)

file(GLOB _ALL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_FILE ${_ALL_FILES})
  if (IS_DIRECTORY ${_FILE})
      if (EXISTS ${_FILE}/CMakeLists.txt)
          add_subdirectory(${_FILE})
      endif()
  endif()
endforeach()
