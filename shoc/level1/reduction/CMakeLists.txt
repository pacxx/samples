cmake_minimum_required(VERSION 3.5)
project(reduction)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../common)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Reduction.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/OptionParser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/Option.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../common/ResultDatabase.cpp
        )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
