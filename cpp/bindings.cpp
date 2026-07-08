#include <pybind11/pybind11.h>

#include "read_file.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_IO, m)
{
    m.def(
        "c_lammps_dump_file_to_simulation",
        &c_lammps_dump_file_to_simulation,
        py::arg("filepath"),
        py::arg("element_type_mapping")
    );
}