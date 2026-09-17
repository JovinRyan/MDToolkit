#include "data.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<std::streamoff> c_initialize(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()){
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::streamoff> offsets;
    std::string line;

    while (true)
    {
        const auto position = file.tellg();

        if (!std::getline(file, line))
            break;

        if (line == "ITEM: TIMESTEP")
            offsets.push_back(position);
    }

    return offsets;
}

PYBIND11_MODULE(cpp_IO, m)
{
    m.doc() = "MDToolkit C++ IO utilities";

    m.def(
        "c_initialize",
        &c_initialize,
        "Find the byte offsets of LAMMPS frames"
    );
}