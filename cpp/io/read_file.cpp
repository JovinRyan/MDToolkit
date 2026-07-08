#include "read_file.h"


py::object c_lammps_dump_file_to_simulation(const std::string& filepath, std::unordered_map<int, std::string> element_type_mapping)
{
    std::ifstream file(filepath);

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    
    py::module_ objects = py::module_::import("MDToolkit.data.objects");

    py::object Atom = objects.attr("Atom");
    py::object Molecule = objects.attr("Molecule");
    py::object StructuredSystem = objects.attr("StructuredSystem");
    py::object Simulation = objects.attr("Simulation");

    std::string line;
    std::vector<std::string> dump_fields;
    std::vector<int> timesteps;
    std::vector<int> atom_counts;
    std::unordered_map<std::string, double> box_dims;
    py::list frames;

    int frame_idx = 0;

    file.seekg(0, std::ios::end);
    const auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    indicators::ProgressBar bar{
        indicators::option::BarWidth{50},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::ShowPercentage{true},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true}
    };

    while (std::getline(file, line))
    {
        if (line.rfind("ITEM: TIMESTEP", 0) == 0) {
            std::getline(file, line);
            timesteps.push_back(std::stoi(line));
        };

        if (line.rfind("ITEM: NUMBER OF ATOMS", 0) == 0) {
            std::getline(file, line);
            atom_counts.push_back(std::stoi(line));
        };

        if (line.rfind("ITEM: BOX BOUNDS", 0) == 0) {
            std::getline(file, line);
            std::stringstream ss_x(line);
            double min_x, max_x;
            ss_x >> min_x >> max_x;

            std::getline(file, line);
            std::stringstream ss_y(line);
            double min_y, max_y;
            ss_y >> min_y >> max_y;

            std::getline(file, line);
            std::stringstream ss_z(line);
            double min_z, max_z;
            ss_z >> min_z >> max_z;

            box_dims = {
                {"min_x", min_x},
                {"max_x", max_x},
                {"min_y", min_y},
                {"max_y", max_y},
                {"min_z", min_z},
                {"max_z", max_z}
            };

        };

        if (line.rfind("ITEM: ATOMS", 0) == 0) {
            line.erase(0, 12);

            std::istringstream iss(line);
            dump_fields = std::vector<std::string>(
                std::istream_iterator<std::string>{iss},
                std::istream_iterator<std::string>()
            );

            std::unordered_map<std::string, int> field_map;

            for (int j = 0; j < dump_fields.size(); j++) {
                field_map[dump_fields[j]] = j;
            }

            std::vector<std::string> elemental_property_keys;

            for (const auto& field : dump_fields) {
                if (field != "id" &&
                    field != "mol" &&
                    field != "type" &&
                    field != "q" &&
                    field != "x" &&
                    field != "y" &&
                    field != "z")
                {
                    elemental_property_keys.push_back(field);
                }
            }

            std::unordered_map<int, std::vector<py::object>> molecule_atoms;

            for (int i = 0; i < atom_counts[frame_idx]; i++) {

                std::getline(file, line);

                std::stringstream atom_stream(line);

                std::string value;

                int id;
                int mol;
                int type;

                double q;
                double x;
                double y;
                double z;

                std::unordered_map<std::string, double> elemental_properties;

                for (int j = 0; j < dump_fields.size(); j++) {
                    atom_stream >> value;

                    if (j == field_map["id"])
                        id = std::stoi(value);

                    else if (j == field_map["mol"])
                        mol = std::stoi(value);

                    else if (j == field_map["type"])
                        type = std::stoi(value);

                    else if (j == field_map["q"])
                        q = std::stod(value);

                    else if (j == field_map["x"])
                        x = std::stod(value);

                    else if (j == field_map["y"])
                        y = std::stod(value);

                    else if (j == field_map["z"])
                        z = std::stod(value);
                    
                    else 
                        elemental_properties[dump_fields[j]] = std::stod(value);
                }

                py::object atom = Atom(
                    id,
                    element_type_mapping[type],                    // replace with type_mapping[type] later
                    std::vector<double>{x, y, z},
                    q,
                    py::cast(elemental_properties),
                    elemental_property_keys
                );

                molecule_atoms[mol].push_back(atom);

            };

            std::vector<py::object> molecules;
            molecules.reserve(molecule_atoms.size());

            for (auto& [mol_id, atoms] : molecule_atoms)
            {
                molecules.push_back(
                    Molecule(
                        mol_id,
                        "ABC",
                        py::cast(atoms)
                    )
                );
            }

            frames.append(
            StructuredSystem(
                py::cast(molecules),
                py::cast(box_dims)
            )
        );

        bar.set_progress(
            static_cast<size_t>(
                100.0 * static_cast<double>(file.tellg()) /
                static_cast<double>(file_size)
            )
        );

        frame_idx++;

        };
    };

    return Simulation(frames, py::cast(timesteps), py::cast(atom_counts));
}