cmake_minimum_required(VERSION 3.20)
project(industry_picking LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ──────────────────────────────────────────────
# Optional CUDA support
# ──────────────────────────────────────────────
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(CUDA_AVAILABLE)
    message(STATUS "CUDA found — GPU acceleration enabled")
else()
    message(WARNING "CUDA not found — building CPU-only fallback")
endif()

# ──────────────────────────────────────────────
# Find required packages
# ──────────────────────────────────────────────
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm REQUIRED)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(realsense2 REQUIRED realsense2)
find_package(yaml-cpp REQUIRED)

# fmt and spdlog — try find_package, fall back to pkg-config
find_package(fmt QUIET)
if(NOT fmt_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FMT REQUIRED fmt)
endif()

find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SPDLOG REQUIRED spdlog)
endif()

# ──────────────────────────────────────────────
# Source files
# ──────────────────────────────────────────────
set(CPP_SOURCES
    src/main.cpp
    src/camera.cpp
    src/gl_viewer.cpp
    src/pipeline.cpp
    src/registration.cpp
    src/robot.cpp
    src/segmentation.cpp
    src/gpu_impl.cpp
)

set(CUDA_SOURCES "")
if(CMAKE_CUDA_COMPILER)
    set(CUDA_SOURCES
        cuda/depth_processing.cu
        cuda/pointcloud.cu
        cuda/icp.cu
    )
endif()

# ──────────────────────────────────────────────
# Executable
# ──────────────────────────────────────────────
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${realsense2_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    Eigen3::Eigen
    glfw
    GLEW::GLEW
    glm::glm
    ${realsense2_LIBRARIES}
    yaml-cpp::yaml-cpp
)

# Link fmt
if(fmt_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${FMT_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${FMT_LIBRARIES})
endif()

# Link spdlog
if(spdlog_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog)
else()
    target_include_directories(${PROJECT_NAME} PRIVATE ${SPDLOG_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${SPDLOG_LIBRARIES})
endif()

# OpenGL
find_package(OpenGL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenGL::GL)

# CUDA-specific settings
if(CMAKE_CUDA_COMPILER)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "60;70;75;80;86;89;90"
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})
endif()

# Pthreads for std::thread
find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

# ──────────────────────────────────────────────
# Copy shaders and config to build directory
# ──────────────────────────────────────────────
file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/config  DESTINATION ${CMAKE_BINARY_DIR})

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Eigen3: ${EIGEN3_VERSION_STRING}")
// Edit
// Edit
