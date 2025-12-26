add_library(libsv STATIC
        src/io/feature_serdes.cpp
)

target_include_directories(libsv
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(libsv
        PUBLIC
        libvoicefeat::libvoicefeat
)

target_compile_features(libsv PRIVATE cxx_std_20)
