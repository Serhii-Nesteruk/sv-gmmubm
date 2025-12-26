cmake_minimum_required(VERSION 3.31)

project(sv_train_ubm LANGUAGES CXX)

add_executable(sv_train_ubm
        main.cpp
)

target_compile_features(sv_train_ubm PRIVATE cxx_std_20)

target_compile_features(sv_train_ubm PRIVATE cxx_std_20)

target_link_libraries(sv_train_ubm
        PRIVATE
        libsv
        libvoicefeat::libvoicefeat
)
target_include_directories(sv_train_ubm PRIVATE
    "${libsv_include_DIR}/"
)
