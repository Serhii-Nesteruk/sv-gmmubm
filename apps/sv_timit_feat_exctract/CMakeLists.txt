cmake_minimum_required(VERSION 3.31)

project(sv_timit_feat_exctract LANGUAGES CXX)

add_executable(sv_timit_feat_exctract
        main.cpp
)

target_compile_features(sv_timit_feat_exctract PRIVATE cxx_std_20)

target_compile_features(sv_timit_feat_exctract PRIVATE cxx_std_20)

target_link_libraries(sv_timit_feat_exctract
        PRIVATE
        libsv
        libvoicefeat::libvoicefeat
)
target_include_directories(sv_timit_feat_exctract PRIVATE
    "${libsv_include_DIR}/"
)
