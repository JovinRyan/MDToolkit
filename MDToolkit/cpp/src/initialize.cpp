#include "initialize.h"

#include <fstream>
#include <string_view>
#include <vector>

std::vector<size_t> find_offsets(const std::string& filepath)
{
    constexpr std::string_view needle = "ITEM: TIMESTEP";
    constexpr size_t BUFFER_SIZE = 64 * 1024;

    std::vector<size_t> offsets;

    std::ifstream file(filepath, std::ios::binary);

    if (!file)
        return offsets;

    std::vector<char> buffer(BUFFER_SIZE);

    size_t file_offset = 0;

    while (true)
    {
        file.seekg(file_offset, std::ios::beg);
        file.read(buffer.data(), BUFFER_SIZE);

        size_t bytes_read = file.gcount();

        if (bytes_read == 0)
            break;

        std::string_view chunk(buffer.data(), bytes_read);

        size_t position = 0;

        while ((position = chunk.find(needle, position)) !=
               std::string_view::npos)
        {
            if (position == 0 || chunk[position - 1] == '\n')
                offsets.push_back(file_offset + position);

            ++position;
        }

        if (bytes_read < BUFFER_SIZE)
            break;

        file_offset += BUFFER_SIZE - needle.size();
    }

    return offsets;
}