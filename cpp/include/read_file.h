#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <indicators/progress_bar.hpp>

namespace py = pybind11;

py::object c_lammps_dump_file_to_simulation(const std::string& filepath, std::unordered_map<int, std::string> element_type_mapping);