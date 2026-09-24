#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "initialize.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_IO, m)
{
    m.def(
        "_cpp_initialize",
        &find_offsets,
        "Find byte offsets of LAMMPS frame headers"
    );
}