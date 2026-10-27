cmake_minimum_required(VERSION 3.16)
project(ising_hpc CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
add_compile_options(-Wall -Wextra)

find_package(OpenMP REQUIRED)
find_package(MPI)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(ising src/ising.cpp)
target_link_libraries(ising PRIVATE OpenMP::OpenMP_CXX)

add_executable(wolff src/wolff.cpp)

if(MPI_CXX_FOUND)
  add_executable(ising_mpi src/ising.cpp)
  target_compile_definitions(ising_mpi PRIVATE ISING_MPI)
  target_link_libraries(ising_mpi PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
else()
  message(WARNING "MPI not found: ising_mpi will not be built")
endif()

enable_testing()
add_executable(test_philox tests/test_philox.cpp)
add_executable(test_exact tests/test_exact.cpp)
target_link_libraries(test_exact PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME philox_known_answers COMMAND test_philox)
add_test(NAME exact_enumeration_4x4 COMMAND test_exact)
add_test(NAME backend_consistency COMMAND bash ${CMAKE_SOURCE_DIR}/tests/check_backends.sh ${CMAKE_BINARY_DIR})
