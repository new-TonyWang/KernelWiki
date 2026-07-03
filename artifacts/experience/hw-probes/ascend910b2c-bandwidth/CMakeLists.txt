cmake_minimum_required(VERSION 3.16.0)
project(ascend_910b_bandwidth)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)

set(SOC_VERSION "${SOC_VERSION}" CACHE STRING "Ascend SoC version")
set(RUN_MODE "npu" CACHE STRING "Run mode")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN package path")

if(NOT ASCEND_CANN_PACKAGE_PATH)
    message(FATAL_ERROR "ASCEND_CANN_PACKAGE_PATH is empty; source CANN set_env.sh first")
endif()

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist under ${ASCEND_CANN_PACKAGE_PATH}")
endif()

include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(bw_kernels STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/bandwidth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/cube_bandwidth.cpp
)

ascendc_include_directories(bw_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

add_executable(910b_bandwidth
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_include_directories(910b_bandwidth PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include
)

target_link_directories(910b_bandwidth PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/lib64
)

target_link_libraries(910b_bandwidth PRIVATE
    bw_kernels
    ascendcl
    pthread
    dl
)
