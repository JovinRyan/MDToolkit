cmake_minimum_required(VERSION 3.18)

project(mdtoolkit_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    cpp_IO

    bindings.cpp

    io/read_file.cpp
)

target_include_directories(cpp_IO PRIVATE include)

install(
    TARGETS cpp_IO
    LIBRARY DESTINATION MDToolkit
)