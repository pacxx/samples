cmake_minimum_required(VERSION 3.5)
project(dot)

set(CMAKE_MODULE_PATH ${PACXX_DIR}/lib/cmake/pacxx)

find_package(PACXX REQUIRED)

include_directories(${PACXX_INCLUDE_DIRECTORY} ${PACXX_INCLUDE_DIRECTORY}/pacxx 
  ${GSTORM_DIR}/include ${RANGE_DIR}/include)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

add_pacxx_to_target(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR} ${SOURCE_FILES})

