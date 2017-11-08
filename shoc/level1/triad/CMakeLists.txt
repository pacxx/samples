cmake_minimum_required(VERSION 3.5)
project(triad)

set(CMAKE_MODULE_PATH ${PACXX_DIR}/lib/cmake/pacxx)

find_package(PACXX REQUIRED)
option(PACXX_EXPERIMENTAL_BACKEND "Use the experimental CPU backend" OFF)
if (PACXX_EXPERIMENTAL_BACKEND)
    add_compile_options(-DUSE_EXPERIMENTAL_BACKEND)
endif (PACXX_EXPERIMENTAL_BACKEND)

include_directories(${PACXX_INCLUDE_DIRECTORY} ${PACXX_INCLUDE_DIRECTORY}/pacxx
        ${GSTORM_DIR}/include ${RANGE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/../../common)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Triad.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/OptionParser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/Option.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/ResultDatabase.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

add_pacxx_to_target(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR} ${SOURCE_FILES})

