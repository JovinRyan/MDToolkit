cmake_minimum_required(VERSION 3.18)

project(MDToolkit LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cpp_IO
    MDToolkit/cpp/bindings.cpp
    MDToolkit/cpp/src/initialize.cpp
)

target_include_directories(cpp_IO PRIVATE
    MDToolkit/cpp/include
)

target_compile_features(cpp_IO PRIVATE cxx_std_17)

install(
    TARGETS cpp_IO
    DESTINATION MDToolkit
)