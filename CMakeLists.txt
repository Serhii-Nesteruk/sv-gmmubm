cmake_minimum_required(VERSION 3.31)
project(sv_gmmubm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(libsv_include_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libsv/include")
set(CMAKE_PREFIX_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/libvoicefeat-install"
        ${CMAKE_PREFIX_PATH}
)

find_package(libvoicefeat CONFIG REQUIRED)

add_subdirectory(libsv)
add_subdirectory(apps/sv_timit_feat_exctract)
add_subdirectory(apps/sv_train_ubm)
add_subdirectory(apps/sv_enroll)
add_subdirectory(apps/sv_eval)
