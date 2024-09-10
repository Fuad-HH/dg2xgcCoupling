cmake_minimum_required(VERSION 3.20)
project(dg2xgcCoupling LANGUAGES CXX)
message(STATUS "CMake version: ${CMAKE_VERSION}")

enable_testing()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")
find_package(Omega_h REQUIRED)
message(STATUS "Omega_h found at ${Omega_h_DIR}")
find_package(MPI REQUIRED)
find_package(pcms REQUIRED)

# create lib with node2cell.cpp and .hpp
add_library(node2cell_lib node2cell.cpp node2cell.hpp)
target_link_libraries(node2cell_lib PUBLIC Omega_h::omega_h MPI::MPI_CXX pcms::pcms)

add_executable(node2cell main.cpp)
target_link_libraries(node2cell PRIVATE node2cell_lib)


