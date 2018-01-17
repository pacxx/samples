cmake_minimum_required(VERSION 3.5)
project(unit_test)

set(CMAKE_MODULE_PATH ${PACXX_DIR}/lib/cmake/pacxx)

enable_testing()
find_package(PACXX REQUIRED)
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${PACXX_INCLUDE_DIRECTORY} ${PACXX_INCLUDE_DIRECTORY}/pacxx
        ${GSTORM_DIR}/include ${RANGE_DIR}/include)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

add_pacxx_to_target(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR} ${SOURCE_FILES})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME} PUBLIC ${GTEST_LIBRARIES} PUBLIC Threads::Threads)

add_test(AllTests ${PROJECT_NAME})
