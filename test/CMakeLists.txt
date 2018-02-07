cmake_minimum_required(VERSION 3.5)
project(unit_test)

enable_testing()
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME} PUBLIC ${GTEST_LIBRARIES} PUBLIC Threads::Threads)

add_test(AllTests ${PROJECT_NAME})
