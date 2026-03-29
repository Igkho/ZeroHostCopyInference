cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(ZeroCopyInference LANGUAGES CXX CUDA)

# Resolve FetchContent timestamp warnings by enforcing the NEW policy behavior
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

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
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 1. CLI11 for CLI parsing
FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.1.zip
)
FetchContent_MakeAvailable(CLI11)

FetchContent_Declare(
  yaml-cpp
  URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.zip
)
# Suppress yaml-cpp tests/tools to speed up build
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# --- ONNX Runtime (FetchContent for ALL Platforms) ---
message(STATUS ">> Downloading pre-built ONNX Runtime...")

if(PLATFORM_JETSON)
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-aarch64-1.19.2.tgz")
    set(ORT_LIB_FILE "libonnxruntime.so.1.19.2")
elseif(WIN32)
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-win-x64-gpu-1.19.2.zip")
    set(ORT_LIB_FILE "onnxruntime.lib")
    set(ORT_DLL_FILE "onnxruntime.dll")
else()
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-x64-gpu-1.19.2.tgz")
    set(ORT_LIB_FILE "libonnxruntime.so.1.19.2")
endif()

FetchContent_Declare(
    onnxruntime_bin
    URL ${ORT_URL}
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

# CUDA & TensorRT
find_package(CUDAToolkit REQUIRED)
find_library(NVINFER_LIB nvinfer)
find_library(NVPARSERS_LIB nvonnxparser)

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "TensorRT (nvinfer) not found.")
endif()

# Hardware MMAPI is the default. Software CUDA is an explicit fallback.
option(USE_JETSON_CUDA_JPEG "Fallback to CUDA SMs (NVJPEG) instead of Hardware Silicon on Jetson" OFF)

# Only fetch and link FFmpeg if building on PC and route the correct JPEG backend per platform
if(PLATFORM_PC)
    if(WIN32)
        # On Windows, vcpkg handles the magic automatically via the toolchain file.
        find_package(FFmpeg REQUIRED)
        set(PLATFORM_LIBS FFmpeg::FFmpeg CUDA::nvjpeg)
    else()
        # On Linux, use PkgConfig
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET libavcodec libavformat libavutil)
        set(PLATFORM_LIBS PkgConfig::FFMPEG CUDA::nvjpeg)
    endif()
else()
    # On Jetson UMA, bypass FFmpeg completely
    message(STATUS ">> Skipping FFmpeg dependencies.")
    if (USE_JETSON_CUDA_JPEG)
        set(PLATFORM_LIBS CUDA::nvjpeg)
    else()
        # Explicitly search for nvjpeg ONLY in the tegra directory
        find_library(TEGRA_NVJPEG_LIB nvjpeg
            PATHS /usr/lib/aarch64-linux-gnu/tegra
            NO_DEFAULT_PATH)
        if (NOT TEGRA_NVJPEG_LIB)
            message(FATAL_ERROR "Could not find hardware libnvjpeg.so in Tegra directory!")
        endif()
        set(PLATFORM_LIBS ${TEGRA_NVJPEG_LIB} nvbufsurface libnvbufsurftransform.so EGL GLESv2 v4l2)
    endif()
endif()

# --- Sources ---
file(GLOB_RECURSE SOURCES src/*.cpp src/*.cu include/*.h)

file(GLOB_RECURSE TESTS src/*.cpp src/*.cu include/*.h tests/*.cpp test/*.cu tests/*.h)
list(REMOVE_ITEM TESTS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

# Strip FFmpeg source implementation from Jetson build
if(PLATFORM_JETSON)
    # FFmpeg is always removed on Jetson
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/FFmpegSource.cpp")
    list(REMOVE_ITEM TESTS "${CMAKE_CURRENT_SOURCE_DIR}/src/FFmpegSource.cpp")

    if (USE_JETSON_CUDA_JPEG)
        message(STATUS ">> Jetson Backend: CUDA SM NVJPEG (Software Fallback)")

        # 1. Pass a #define to C++ so main.cpp knows to load NVJpeg classes
        add_compile_definitions(USE_JETSON_CUDA_JPEG)

        # 2. Strip the MMAPI classes from the build
        list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSource.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSink.cpp")
        list(REMOVE_ITEM TESTS "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSource.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSink.cpp")
        list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSink.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSink.cpp")
        list(REMOVE_ITEM TESTS "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSink.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSink.cpp")
    else()
        message(STATUS ">> Jetson Backend: MMAPI Hardware NVJPG (Default)")

        # 1. Strip the CUDA classes from the build
        list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/NVJpegSource.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/NVJpegSink.cpp")
        list(REMOVE_ITEM TESTS "${CMAKE_CURRENT_SOURCE_DIR}/src/NVJpegSource.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/NVJpegSink.cpp")
        list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/NVJpegSink.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/NVJpegSink.cpp")
        list(REMOVE_ITEM TESTS "${CMAKE_CURRENT_SOURCE_DIR}/src/NVJpegSink.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/NVJpegSink.cpp")

        # 2. Add MMAPI dependencies
        set(MMAPI_CLASS_DIR /usr/src/jetson_multimedia_api/samples/common/classes)
        set(MMAPI_SOURCES
            ${MMAPI_CLASS_DIR}/NvJpegDecoder.cpp
            ${MMAPI_CLASS_DIR}/NvJpegEncoder.cpp
            ${MMAPI_CLASS_DIR}/NvElement.cpp
            ${MMAPI_CLASS_DIR}/NvElementProfiler.cpp
            ${MMAPI_CLASS_DIR}/NvLogging.cpp
            ${MMAPI_CLASS_DIR}/NvBuffer.cpp
        )
        list(APPEND SOURCES ${MMAPI_SOURCES})
        list(APPEND TESTS ${MMAPI_SOURCES})
    endif()
endif()

# Strip MMAPI source implementation from PC build
if(PLATFORM_PC)
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSource.cpp")
    list(REMOVE_ITEM TESTS "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSource.cpp")
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSink.cpp")
    list(REMOVE_ITEM TESTS "${CMAKE_CURRENT_SOURCE_DIR}/src/MMAPIJpegSink.cpp")
endif()


# --- Target ---
add_executable(ZeroCopyInference ${SOURCES})
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(ZeroCopyInferenceTests ${TESTS})
set(TESTS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/tests)

macro(configure_target target_name includedir)
    set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # # Explicitly bind the RPATH to the target if we are on Jetson Hardware Silicon Mode
    # if(PLATFORM_JETSON AND NOT USE_JETSON_CUDA_JPEG)
    #     set_target_properties(${target_name} PROPERTIES BUILD_RPATH "/usr/lib/aarch64-linux-gnu/tegra")
    #     set_target_properties(${target_name} PROPERTIES INSTALL_RPATH "/usr/lib/aarch64-linux-gnu/tegra")
    # endif()

    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>)
    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)

    target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

#    target_compile_definitions(${target_name} PRIVATE ENABLE_DEBUG_SYNC)

    # -Xnvlink passes the -w flag to the device linker, silencing incompatible library warnings
    target_link_options(${target_name} PRIVATE $<DEVICE_LINK:-Xnvlink -w>)

    # Force legacy RPATH so LD_LIBRARY_PATH cannot hijack libnvjpeg.so
    target_link_options(${target_name} PRIVATE "-Wl,--disable-new-dtags")

    target_include_directories(${target_name} PRIVATE
        "${includedir}"
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${onnxruntime_INCLUDE_DIRS}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    if (PLATFORM_JETSON AND NOT USE_JETSON_CUDA_JPEG)
        target_include_directories(${target_name} PRIVATE
            /usr/src/jetson_multimedia_api/include
            /usr/src/jetson_multimedia_api/include/libjpeg-8b
            ${MMAPI_CLASS_DIR}
        )
        target_link_directories(${target_name} PRIVATE
            /usr/lib/aarch64-linux-gnu/tegra
        )
    endif()

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
        CUDA::cudart
        CUDA::cuda_driver
        yaml-cpp
        ${NVINFER_LIB}
        ${NVPARSERS_LIB}
        CLI11::CLI11
    )
endmacro()

configure_target(ZeroCopyInference ${INCLUDE_DIR})

configure_target(ZeroCopyInferenceTests "${TESTS_INCLUDE_DIR}")

target_link_libraries(ZeroCopyInferenceTests PRIVATE GTest::gtest_main GTest::gmock_main)

option(FETCH_TEST_DATA "Download test frames during build" ON)

if(FETCH_TEST_DATA)
    message(STATUS ">> Configuring frames data bootstrapping target...")

    # Fetch Frames Data Target
    add_custom_target(fetch_frames ALL
        COMMAND bash ${CMAKE_SOURCE_DIR}/download_frames_data.sh
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Verifying frames data environment..."
    )
    # Force executables to wait for the download to finish!
    add_dependencies(ZeroCopyInference fetch_frames)
endif()

# Standard definitions
target_compile_definitions(ZeroCopyInference PRIVATE -DUNICODE)
