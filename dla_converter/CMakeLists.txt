
message("cmake source dir: " ${CMAKE_SOURCE_DIR})
message("cmake current dir: " ${CMAKE_CURRENT_LIST_DIR})

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
message(${CMAKE_MODULE_PATH}  " cmake module")
set(CUDLA_LIB /usr/local/cuda/targets/aarch64-linux/lib/libcudla.so)

find_package(TensorRT REQUIRED)

message(${CMAKE_MODULE_PATH})
message(${CMAKE_MODULE_PATH})

include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)



find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})
message("CUDA INCLUDE: ${CUDA_INCLUDE_DIRS}")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
add_library(utils ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp)
target_link_libraries(utils ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

add_library(trt ${CMAKE_CURRENT_LIST_DIR}/src/trt.cpp)
target_link_libraries(trt ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} utils)

add_library(dla ${CMAKE_CURRENT_LIST_DIR}/src/dla.cpp)
target_link_libraries(dla ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} ${CUDLA_LIB} trt)

add_executable(dla_int8 ${CMAKE_CURRENT_LIST_DIR}/src/to_dla_int8.cpp)
target_link_libraries(dla_int8  ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} ${CUDLA_LIB} dla)
