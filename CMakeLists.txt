cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(ZeroCopyInference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# --- Platform Detection ---
set(PLATFORM_JETSON OFF)
set(PLATFORM_PC OFF)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS ">> Detected ARM64 Architecture (Likely Jetson)")
    add_compile_definitions(PLATFORM_JETSON)
    set(PLATFORM_JETSON ON)
    # Jetson specific CUDA flags (Xavier/Orin use sm_87 for Orin, sm_72 for Xavier)
    # Orin Nano is Ampere architecture (sm_87)
    set(CMAKE_CUDA_ARCHITECTURES 87)
else()
    message(STATUS ">> Detected x86_64 Architecture (PC)")
    add_compile_definitions(PLATFORM_PC)
    set(PLATFORM_PC ON)

    # Auto-detect GPU on PC
    #set(CMAKE_CUDA_ARCHITECTURES native)

    # Default to "86" (RTX 3000) if 'native' detection isn't possible (like in Docker)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
endif()

# --- Dependencies ---
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 1. CLI11 for CLI parsing
FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.1.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(CLI11)


# --- ONNX Runtime (Definition) ---
if(PLATFORM_JETSON)
    message(STATUS ">> Architecture: aarch64. Searching for system-installed ONNX Runtime...")
    # Search for Headers
    find_path(ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
        PATHS
        /usr/include/onnxruntime
        /usr/local/include/onnxruntime
        /opt/onnxruntime/include
        DOC "Path to ONNX Runtime headers"
    )
    # Search for Library (.so)
    find_library(ORT_LIB NAMES onnxruntime
        PATHS
        /usr/lib/aarch64-linux-gnu
        /usr/local/lib
        /opt/onnxruntime/lib
        DOC "Path to ONNX Runtime library"
    )

    if(ORT_INCLUDE_DIR AND ORT_LIB)
        message(STATUS "Found ONNX Runtime: ${ORT_LIB}")
        # Create the same IMPORTED target so the rest of the script works unchanged
        add_library(onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ORT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
        )
    else()
        message(FATAL_ERROR "Could not find ONNX Runtime on this Jetson. Please install it or set CMAKE_PREFIX_PATH.")
    endif()

# PC (Windows/Linux x64): Download Pre-built Binaries
else()
    message(STATUS ">> Architecture: x86_64. Downloading pre-built ONNX Runtime...")

    if(WIN32)
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-gpu-1.19.2.zip")
        set(ORT_LIB_FILE "onnxruntime.lib")
        set(ORT_DLL_FILE "onnxruntime.dll")
    else()
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz")
        set(ORT_LIB_FILE "libonnxruntime.so.1.19.2")
        set(ORT_DLL_FILE "libonnxruntime.so.1.19.2")
    endif()

    FetchContent_Declare(
      onnxruntime_bin
      URL ${ORT_URL}
      DOWNLOAD_EXTRACT_TIMESTAMP true
    )
    FetchContent_MakeAvailable(onnxruntime_bin)

    add_library(onnxruntime SHARED IMPORTED)

    # Define Properties
    if(WIN32)
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${onnxruntime_bin_SOURCE_DIR}/lib/${ORT_LIB_FILE}"
            IMPORTED_LOCATION "${onnxruntime_bin_SOURCE_DIR}/lib/${ORT_DLL_FILE}"
            INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_bin_SOURCE_DIR}/include"
        )
    else()
        set_target_properties(onnxruntime PROPERTIES
            IMPORTED_LOCATION "${onnxruntime_bin_SOURCE_DIR}/lib/${ORT_LIB_FILE}"
            INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_bin_SOURCE_DIR}/include"
        )
    endif()

endif()

# CUDA & TensorRT
#find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(NVINFER_LIB nvinfer)
find_library(NVPARSERS_LIB nvonnxparser)

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "TensorRT (nvinfer) not found.")
endif()

if(WIN32)
    # On Windows, vcpkg handles the magic automatically via the toolchain file.
    find_package(FFmpeg REQUIRED)
    set(PLATFORM_LIBS FFmpeg::FFmpeg)
else()
    # On Linux, use PkgConfig
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
        libavcodec libavformat libavutil
    )
    set(PLATFORM_LIBS PkgConfig::FFMPEG)
endif()

# --- Sources ---
file(GLOB_RECURSE SOURCES src/*.cpp src/*.cu include/*.h)

file(GLOB_RECURSE TESTS src/*.cpp src/*.cu include/*.h tests/*.cpp test/*.cu tests/*.h)
list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# --- Target ---
add_executable(ZeroCopyInference ${SOURCES})
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(ZeroCopyInferenceTests ${TESTS})
set(TESTS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/tests)

macro(configure_target target_name includedir)
  set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
  target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)

  # -Xnvlink passes the -w flag to the device linker, silencing incompatible library warnings
  target_link_options(${target_name} PRIVATE $<DEVICE_LINK:-Xnvlink -w>)

  target_include_directories(${target_name} PRIVATE
      "${includedir}"
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${onnxruntime_INCLUDE_DIRS}
      ${CUDAToolkit_INCLUDE_DIRS}
  )

  if (PLATFORM_PC)
    # Set properties based on OS
    if(WIN32)
        # Windows DLL Copy Rule
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${onnxruntime_bin_SOURCE_DIR}/lib/${ORT_DLL_FILE}"
            $<TARGET_FILE_DIR:${target_name}>
        )
    else()
        # Linux PC: Copy .so AND provider .so
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${onnxruntime_bin_SOURCE_DIR}/lib/libonnxruntime_providers_shared.so"
            "${onnxruntime_bin_SOURCE_DIR}/lib/libonnxruntime_providers_cuda.so"
            "${onnxruntime_bin_SOURCE_DIR}/lib/libonnxruntime_providers_tensorrt.so" # Optional if using TRT
            $<TARGET_FILE_DIR:${target_name}>
        )
    endif()
  endif()

  # Consolidated Linking
  target_link_libraries(${target_name} PRIVATE
      onnxruntime
      ${PLATFORM_LIBS}
#      ${OpenCV_LIBS}
      CUDA::cudart
      CUDA::cuda_driver
      CUDA::nvjpeg
#      CUDA::nppig
#      CUDA::nppc
      ${NVINFER_LIB}
      ${NVPARSERS_LIB}
      CLI11::CLI11
  )

endmacro()

configure_target(ZeroCopyInference ${INCLUDE_DIR})

configure_target(ZeroCopyInferenceTests "${TESTS_INCLUDE_DIR}")

target_link_libraries(ZeroCopyInferenceTests PRIVATE GTest::gtest_main GTest::gmock_main)

# Standard definitions
target_compile_definitions(ZeroCopyInference PRIVATE -DUNICODE)
