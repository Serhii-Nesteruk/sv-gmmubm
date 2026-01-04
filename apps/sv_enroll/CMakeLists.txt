cmake_minimum_required(VERSION 3.31)

project(sv_enroll LANGUAGES CXX)

add_executable(sv_enroll
        main.cpp
)

target_compile_features(sv_enroll PRIVATE cxx_std_20)

target_compile_features(sv_enroll PRIVATE cxx_std_20)

target_link_libraries(sv_enroll
        PRIVATE
        libsv
        libvoicefeat::libvoicefeat
)
target_include_directories(sv_enroll PRIVATE
        "${libsv_include_DIR}/"
)
