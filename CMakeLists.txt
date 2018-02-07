cmake_minimum_required(VERSION 3.5)

project(pacxx-gstorm-eval)
set(CMAKE_CXX_STANDARD 14)

set(GSTORM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/gstorm/" CACHE PATH "Path to gstom")
set(RANGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/range-v3/" CACHE PATH "Path to range-v3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wno-invalid-noreturn")

file(GLOB _ALL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_FILE ${_ALL_FILES})
  if (IS_DIRECTORY ${_FILE})
    if (NOT ${_FILE} STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/range-v3" AND NOT ${_FILE} STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/gstorm") #ignore range and gstorm libs
      if (EXISTS ${_FILE}/CMakeLists.txt)
          add_subdirectory(${_FILE})
      endif()
    endif()
  endif()
endforeach()
