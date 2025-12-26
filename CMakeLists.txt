cmake_minimum_required(VERSION 3.31)
project(sv_gmmubm)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libvoicefeat-install")
find_package(libvoicefeat CONFIG REQUIRED)

add_executable(sv_gmmubm main.cpp)
target_link_libraries(sv_gmmubm PRIVATE libvoicefeat::libvoicefeat)
