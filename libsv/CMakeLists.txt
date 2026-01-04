add_library(libsv STATIC
        src/io/feature_serdes.cpp
        src/gmm/gmm_ubm_trainer.cpp
        src/gmm/gmm_model_serdes.cpp
        src/gmm/bw_stats_accumulator.cpp
        src/gmm/map_adaptor.cpp
        src/gmm/scorer.cpp
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
