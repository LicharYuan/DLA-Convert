cmake_minimum_required(VERSION 3.16)
project(dla_converter LANGUAGES CXX )

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall --std=c++17 -O3 -fPIC" )
set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -O3 -fPIC" )
set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")

add_subdirectory(dla_converter)
