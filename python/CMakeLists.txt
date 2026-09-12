cmake_minimum_required(VERSION 3.20)
project(numpy_cpp_python CXX)

find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(pybind11 GIT_REPOSITORY https://github.com/pybind/pybind11.git GIT_TAG v2.13.6)
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(numpy_cpp numpy_cpp.cpp)
target_include_directories(numpy_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(numpy_cpp PRIVATE numpy-cpp::numpy-cpp)
set_target_properties(numpy_cpp PROPERTIES CXX_STANDARD 20)

# Test import
add_test(NAME python_import COMMAND python3 -c "import numpy_cpp; print(numpy_cpp.__doc__)")
