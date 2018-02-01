cmake_minimum_required (VERSION 2.6)
find_package(CUDA)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}--expt-extended-lambda
--gpu-architecture=compute_52 --gpu-code=compute_52,sm_61,sm_70 -use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -fPIC -O2")

set(PYBIND11_PYTHON_VERSION 2.7)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("external/moderngpu/src")
include_directories("external/cub")
include_directories("source")
add_subdirectory("external/pybind11")


cuda_add_library(gols SHARED
	source/gols.cu
  source/caching_context.cu
)
CUDA_ADD_CUBLAS_TO_TARGET(gols)
target_link_libraries(gols ${CUDA_cusolver_LIBRARY})


pybind11_add_module(_gols
                    source/pygols.cxx)
target_link_libraries(_gols gols)

cuda_add_executable(play
                    play/example.cxx)
target_link_libraries(play gols)

