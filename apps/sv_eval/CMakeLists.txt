cmake_minimum_required(VERSION 3.31)

project(sv_eval LANGUAGES CXX)

add_executable(sv_eval
        main.cpp
)

target_compile_features(sv_eval PRIVATE cxx_std_20)

target_compile_features(sv_eval PRIVATE cxx_std_20)

target_link_libraries(sv_eval
        PRIVATE
        libsv
        libvoicefeat::libvoicefeat
)
target_include_directories(sv_eval PRIVATE
        "${libsv_include_DIR}/"
)
