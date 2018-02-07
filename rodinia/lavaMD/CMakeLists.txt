cmake_minimum_required(VERSION 3.5)
project(lavaMD)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})